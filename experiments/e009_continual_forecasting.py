"""e009 — continual learning on quantum time-series forecasting (provided datasets).

Topic: use time-series data and a quantum/hybrid temporal model for sequential learning;
measure catastrophic forgetting; apply continual-learning methods to balance retention and
adaptation. This is the regression counterpart to the classification study (e005/e007), on the
provided quantum/physics forecasting series (Peng & Chen, arXiv:2605.06734).

One recurrent data-reuploading quantum forecaster is trained on a sequence of forecasting tasks
(e.g. narma_5 -> damped_shm -> bessel_j2). We track each task's test NMSE every epoch and
compare:
  naive  : sequential fine-tuning, no protection (forgetting baseline)
  l2     : soft L2 anchoring to previous task optima
  ewc    : soft anchoring weighted by the empirical (MSE/Gauss-Newton) Fisher diagonal
  replay : a small balanced buffer of earlier-task samples mixed into the loss

Forgetting = increase in an earlier task's test NMSE from its phase end to the run end
(lower NMSE is better, so a positive increase is forgetting).

Run:
    python experiments/e009_continual_forecasting.py --tasks narma_5 damped_shm bessel_j2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_data import load_task_sequence  # noqa: E402
from src.e009_qtsf import (  # noqa: E402
    init_weights, make_forecaster, make_state_forecaster, nmse, predict, temporal_qfi_diag,
)

RESULTS = ROOT / "results"
METHODS = ("naive", "l2", "ewc", "qewc", "replay")


def _flat(circ_w, head_w):
    return pnp.concatenate([circ_w.flatten(), head_w.flatten()])


def empirical_fisher(qnode, circ_w, head_w, X):
    """Diagonal empirical Fisher for the MSE head: mean_n (d pred_n / d theta_i)^2."""
    def preds(c, h):
        return predict(qnode, c, h, X)
    jc, jh = qml.jacobian(preds, argnums=(0, 1))(circ_w, head_w)
    jac = np.concatenate([np.asarray(jc).reshape(len(X), -1),
                          np.asarray(jh).reshape(len(X), -1)], axis=1)
    return np.mean(jac ** 2, axis=0)


def train_method(method, tasks, *, n_layers, seq_len, lr, epochs, lam, buffer_size,
                 seed, qfi_samples=16, ansatz=None, verbose=False):
    ansatz = ansatz or {}   # {"entangler": ..., "encoding": ...} for the gate-count ablation
    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=n_layers, seq_len=seq_len, **ansatz)
    state_qnode = (make_state_forecaster(4, n_layers, seq_len, **ansatz)
                   if method == "qewc" else None)
    n_circ = int(np.prod(cs))
    cw, hw = init_weights(cs, hs, seed=seed)
    opt = qml.AdamOptimizer(lr)
    anchors = []          # (theta_star_flat, fisher_flat) for l2/ewc
    replay_X, replay_y = [], []   # buffers for replay
    rng = np.random.default_rng(seed)
    history = []

    def snap(epoch, phase):
        history.append({"epoch": epoch, "phase": phase,
                        "nmse": {t.name: nmse(qnode, cw, hw, t.X_test, t.y_test) for t in tasks},
                        "train_nmse": {t.name: nmse(qnode, cw, hw, t.X_train, t.y_train)
                                       for t in tasks}})

    snap(0, 0)
    for phase, task in enumerate(tasks, start=1):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)
        Rx = pnp.array(np.concatenate(replay_X), requires_grad=False) if replay_X else None
        Ry = pnp.array(np.concatenate(replay_y), requires_grad=False) if replay_y else None

        def cost(c, h, Xtr=Xtr, ytr=ytr, Rx=Rx, Ry=Ry):
            loss = pnp.mean((predict(qnode, c, h, Xtr) - ytr) ** 2)
            if method in ("l2", "ewc", "qewc") and anchors:
                theta = _flat(c, h)
                for ts, F in anchors:
                    loss = loss + 0.5 * lam * pnp.sum(F * (theta - ts) ** 2)
            if method == "replay" and Rx is not None:
                loss = loss + pnp.mean((predict(qnode, c, h, Rx) - Ry) ** 2)
            return loss

        for _ in range(epochs):
            (cw, hw), _ = opt.step_and_cost(cost, cw, hw)
            snap(history[-1]["epoch"] + 1, phase)
        if verbose:
            cur = history[-1]["nmse"]
            print(f"    [{method:7s}] after {task.name}: "
                  + " ".join(f"{t.name}={cur[t.name]:.3f}" for t in tasks), flush=True)

        if phase < len(tasks):   # consolidate / fill buffer for protecting this task
            theta_star = np.asarray(_flat(cw, hw))
            # Fisher importances are normalized to unit mean so lam is comparable across
            # methods and only the STRUCTURE (relative per-parameter weighting) differs.
            def _norm(F):
                F = np.asarray(F, float)
                return F / (F.mean() + 1e-12)

            if method == "l2":
                anchors.append((theta_star, np.ones(theta_star.size)))
            elif method == "ewc":
                anchors.append((theta_star, _norm(empirical_fisher(qnode, cw, hw, task.X_train))))
            elif method == "qewc":
                # QFI on the quantum weights; classical empirical Fisher on the classical head
                # (QFI is undefined there) -> isolates quantum-vs-classical Fisher on the qubits.
                qfi = temporal_qfi_diag(state_qnode, cw, task.X_train, n_samples=qfi_samples,
                                        seed=seed)
                head_f = empirical_fisher(qnode, cw, hw, task.X_train)[n_circ:]
                anchors.append((theta_star, _norm(np.concatenate([qfi, head_f]))))
            elif method == "replay":
                idx = rng.choice(len(task.X_train), size=min(buffer_size, len(task.X_train)),
                                 replace=False)
                replay_X.append(task.X_train[idx])
                replay_y.append(task.y_train[idx])

    names = [t.name for t in tasks]
    end = {names[i]: history[(i + 1) * epochs]["nmse"][names[i]] for i in range(len(tasks))}
    final = {names[i]: history[-1]["nmse"][names[i]] for i in range(len(tasks))}
    forg = {names[i]: round(final[names[i]] - end[names[i]], 4) for i in range(len(tasks) - 1)}
    return {
        "method": method, "lam": lam,
        "final_nmse": {k: round(final[k], 4) for k in final},
        "forgetting_nmse": forg,
        "avg_earlier_forgetting": round(float(np.mean(list(forg.values()))), 4),
        "retention_earlier_nmse": round(float(np.mean([final[names[i]] for i in range(len(tasks) - 1)])), 4),
        "plasticity_final_nmse": round(float(final[names[-1]]), 4),
        "avg_final_nmse": round(float(np.mean(list(final.values()))), 4),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_continual.json")
    args = ap.parse_args()

    tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
    print(f"e009 seed={args.seed}: {args.tasks} (seq_len={args.seq_len})")
    t0 = time.perf_counter()
    results = {}
    for m in METHODS:
        results[m] = train_method(m, tasks, n_layers=args.layers, seq_len=args.seq_len,
                                  lr=args.lr, epochs=args.epochs_per_task, lam=args.lam,
                                  buffer_size=args.buffer_size, seed=args.seed, verbose=True)
    out = {"experiment": "e009_continual_forecasting", "seed": args.seed, "tasks": args.tasks,
           "epochs_per_task": args.epochs_per_task, "methods": results,
           "train_time_sec": round(time.perf_counter() - t0, 1)}
    print("\n=== e009 (lower NMSE = better; forgetting = NMSE increase on earlier tasks) ===")
    for m in METHODS:
        r = results[m]
        print(f"  {m:7s} retention(old NMSE)={r['retention_earlier_nmse']:.3f} "
              f"plasticity(new NMSE)={r['plasticity_final_nmse']:.3f} "
              f"forgetting={r['avg_earlier_forgetting']:+.3f} avg_final={r['avg_final_nmse']:.3f}")
    RESULTS.mkdir(exist_ok=True)
    outp = args.output if args.output.is_absolute() else ROOT / args.output
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
