"""e007 H2 — is adaptive (event-triggered) control better than always-on consolidation?

Compares five methods on the MNIST -> Fashion-MNIST -> SPT/ATF sequence:
  sequential  : no protection (forgetting baseline)
  qewc        : always-on global-QFI penalty (Hsu)
  l2          : always-on L2 anchoring (tests whether QEWC ~ L2 under isotropic QFI)
  gradclip    : fixed per-step norm clip (simple control)
  adaptive    : Adaptive Norm Trust Region (event-triggered, calibrated per-task budget)

Reports stability (earlier-task retention), plasticity (final-task accuracy), and the
intervention fraction. Run:
    python experiments/e007_h2.py --seed 42
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

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_consolidation import EWC, quantum_fisher_diag  # noqa: E402
from src.e005_softmax import accuracy, bce_loss, make_softmax_qnode  # noqa: E402
from src.e007_adaptive import AdaptiveTrustRegion, calibrate_step_budget, clip_step  # noqa: E402
from src.phase_data import load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

N_QUBITS = 4
RESULTS = ROOT / "results"
METHODS = ("sequential", "qewc", "l2", "gradclip", "adaptive")
CAL_SCALES = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)


def train_method(method, tasks, *, clf, qfi_qnode, wshape, lr, epochs, lam_qewc, lam_l2,
                 seed, verbose, eps_scale=1.0):
    keys = [f"task{i+1}" for i in range(len(tasks))]
    reg = EWC(lam_qewc if method == "qewc" else lam_l2) if method in ("qewc", "l2") else None
    guard = AdaptiveTrustRegion() if method == "adaptive" else None
    clip_c = {"v": None}
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(wshape),
                        requires_grad=True)
    history, n_steps, n_interv = [], 0, 0

    def snap(epoch, phase):
        history.append({"epoch": epoch, "phase": phase,
                        "test_acc": {k: accuracy(clf, weights, t.X_test, t.y_test)
                                     for k, t in zip(keys, tasks)}})

    snap(0, 0)
    for phase, task in enumerate(tasks, start=1):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)
        opt = qml.AdamOptimizer(lr)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase):
            base = bce_loss(clf, W, Xtr, ytr)
            return base + reg.penalty(W.flatten(), phase) if reg is not None else base

        for _ in range(epochs):
            wb = np.asarray(weights).copy()
            weights = opt.step(cost, weights)
            delta = np.asarray(weights) - wb
            n_steps += 1
            if method == "adaptive":
                proj, hit = guard.project(np.asarray(weights).ravel())
                if hit:
                    n_interv += 1
                    weights = pnp.array(proj.reshape(wshape), requires_grad=True)
            elif method == "gradclip" and clip_c["v"] is not None:
                d, hit = clip_step(delta.ravel(), clip_c["v"])
                if hit:
                    n_interv += 1
                    weights = pnp.array(wb + d.reshape(wshape), requires_grad=True)
            snap(history[-1]["epoch"] + 1, phase)

        if phase < len(tasks):  # consolidate / calibrate for protecting this task later
            theta_star = np.asarray(weights)
            if method == "qewc":
                reg.consolidate(theta_star.ravel(),
                                quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                                    n_samples=32, seed=seed))
            elif method == "l2":
                reg.consolidate(theta_star.ravel(), np.ones(int(np.prod(wshape))))
            elif method in ("adaptive", "gradclip"):
                eps = calibrate_step_budget(
                    lambda th: accuracy(clf, pnp.array(th, requires_grad=False),
                                        task.X_test, task.y_test),
                    theta_star, wshape, scales=CAL_SCALES, seed=seed)
                if method == "adaptive":
                    guard.add(theta_star, eps * eps_scale)
                elif clip_c["v"] is None:
                    clip_c["v"] = float(np.sqrt(eps))
        if verbose:
            a = history[-1]["test_acc"]
            print(f"    [{method:10s}] after T{phase}: "
                  + " ".join(f"{k}={a[k]:.2f}" for k in keys), flush=True)

    end = {i + 1: history[(i + 1) * epochs]["test_acc"][keys[i]] for i in range(len(tasks))}
    final = {i + 1: history[-1]["test_acc"][keys[i]] for i in range(len(tasks))}
    forg = {i + 1: round(end[i + 1] - final[i + 1], 4) for i in range(len(tasks) - 1)}
    return {
        "method": method,
        "retention_earlier_tasks": round(float(np.mean([final[1], final[2]])), 4),
        "plasticity_final_task": round(float(final[len(tasks)]), 4),
        "avg_forgetting": round(float(np.mean(list(forg.values()))), 4),
        "final_acc": {f"task{i}": round(final[i], 4) for i in final},
        "forgetting_per_task": {f"task{i}": forg[i] for i in forg},
        "intervention_fraction": round(n_interv / max(n_steps, 1), 4),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--lam-l2", type=float, default=0.72)  # ~ lam_qewc * mean(QFI diag)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=RESULTS / "e007_h2.json")
    args = ap.parse_args()

    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train, n_test=args.n_test,
                            seed=args.seed)
    t3 = load_spt_atf(n_train=args.n_train, n_test=args.n_test, n_qubits=N_QUBITS, seed=args.seed)
    tasks = [t1, t2, t3]
    clf, wshape = make_softmax_qnode(N_QUBITS, args.layers)
    qfi_qnode, _ = make_qnode(N_QUBITS, args.layers)

    print(f"H2 seed={args.seed}: {[t.name for t in tasks]}")
    t0 = time.perf_counter()
    results = {}
    for m in METHODS:
        results[m] = train_method(m, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape,
                                  lr=args.lr, epochs=args.epochs_per_task,
                                  lam_qewc=args.lam_qewc, lam_l2=args.lam_l2,
                                  seed=args.seed, verbose=True)
    out = {"experiment": "e007_h2_adaptive", "seed": args.seed, "tasks": [t.name for t in tasks],
           "epochs_per_task": args.epochs_per_task, "methods": results,
           "train_time_sec": round(time.perf_counter() - t0, 1)}
    print("\n=== H2 (retention / plasticity / intervention) ===")
    for m in METHODS:
        r = results[m]
        print(f"  {m:10s} retention={r['retention_earlier_tasks']:.3f} "
              f"plasticity={r['plasticity_final_task']:.3f} "
              f"avg_forget={r['avg_forgetting']:.3f} interv={r['intervention_fraction']:.2f}")
    outp = args.output if args.output.is_absolute() else ROOT / args.output
    RESULTS.mkdir(exist_ok=True)
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
