"""e007 — quantum-state vs function-space trust-region continual learning (built on QEWC / PR#5).

Compares soft always-on consolidation against explicit trust regions on the 3-task sequence
(MNIST -> Fashion-MNIST -> SPT/ATF):

  sequential : no protection
  qewc       : soft, always-on global-QFI penalty (Hsu)
  l2         : soft, always-on L2 anchoring
  tr_qfi     : QFI state trust region  R^Q_j = δ^T F_Q δ <= B   (representation preservation)
  tr_cfi     : CFI function trust region R^C_j = δ^T F_C δ <= B  (readout/KL -> forgetting; primary)

Physics: bounding QFI state drift is a *sufficient* forgetting safeguard but over-broad (it
guards readout-irrelevant state DoF); the CFI trust region bounds the old-task predictive
distribution directly (~ KL), so it is the physically-aligned object. The guard uses the actual
optimizer step (Adam-aware) and scales it to the budget boundary (src/e007_bbqewc.py).

Run one config:
    python experiments/e007_bbqewc.py --method tr_cfi --budget 0.02 --seed 42
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
from src.e005_softmax import accuracy, bce_loss, classical_fisher_diag, make_softmax_qnode  # noqa: E402
from src.e007_bbqewc import BuresBudget  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"
METHODS = ("sequential", "qewc", "l2", "tr_qfi", "tr_cfi")


def train_method(method, tasks, *, clf, qfi_qnode, wshape, lr, epochs, lam, budget, seed,
                 qfi_samples=32, verbose=False):
    keys = [f"task{i+1}" for i in range(len(tasks))]
    reg = EWC(lam) if method in ("qewc", "l2") else None
    guard = BuresBudget(budget) if method in ("tr_qfi", "tr_cfi") else None
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
            n_steps += 1
            if guard is not None:
                beta, hit = guard.scale_step(wb.ravel(), (np.asarray(weights) - wb).ravel())
                if hit:
                    n_interv += 1
                    weights = pnp.array(wb + beta * (np.asarray(weights) - wb),
                                        requires_grad=True)
            snap(history[-1]["epoch"] + 1, phase)

        if phase < len(tasks):
            theta_star = np.asarray(weights)
            if method == "qewc":
                reg.consolidate(theta_star.ravel(),
                                quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                                    n_samples=qfi_samples, seed=seed))
            elif method == "l2":
                reg.consolidate(theta_star.ravel(), np.ones(int(np.prod(wshape))))
            elif method == "tr_qfi":
                guard.add(theta_star.ravel(),
                          quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                              n_samples=qfi_samples, seed=seed))
            elif method == "tr_cfi":
                guard.add(theta_star.ravel(),
                          classical_fisher_diag(clf, weights, task.X_train, task.y_train))
        if verbose:
            a = history[-1]["test_acc"]
            print(f"    [{method:8s}] after T{phase}: "
                  + " ".join(f"{k}={a[k]:.2f}" for k in keys), flush=True)

    end = {i + 1: history[(i + 1) * epochs]["test_acc"][keys[i]] for i in range(len(tasks))}
    final = {i + 1: history[-1]["test_acc"][keys[i]] for i in range(len(tasks))}
    forg = {i + 1: round(end[i + 1] - final[i + 1], 4) for i in range(len(tasks) - 1)}
    return {
        "method": method, "lam": lam, "budget": budget,
        "retention_earlier_tasks": round(float(np.mean([final[1], final[2]])), 4),
        "plasticity_final_task": round(float(final[len(tasks)]), 4),
        "avg_forgetting": round(float(np.mean(list(forg.values()))), 4),
        "final_acc": {f"task{i}": round(final[i], 4) for i in final},
        "intervention_fraction": round(n_interv / max(n_steps, 1), 4),
        "history": history,
    }


def build(seed, layers=20, n_train=800, n_test=200):
    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    clf, wshape = make_softmax_qnode(N_QUBITS, layers)
    qfi_qnode, _ = make_qnode(N_QUBITS, layers)
    return [t1, t2, t3], clf, qfi_qnode, wshape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, default="tr_cfi")
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tasks, clf, qfi_qnode, wshape = build(args.seed, args.layers, args.n_train, args.n_test)
    t0 = time.perf_counter()
    r = train_method(args.method, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape,
                     lr=0.02, epochs=args.epochs_per_task, lam=args.lam, budget=args.budget,
                     seed=args.seed, verbose=True)
    print(f"\n  {args.method}: retention={r['retention_earlier_tasks']:.3f} "
          f"plasticity={r['plasticity_final_task']:.3f} interv={r['intervention_fraction']:.2f} "
          f"({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
