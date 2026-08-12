"""e007 (H1) — early-warning test: does R predict old-task forgetting before accuracy drops?

Train Task 1 (MNIST) -> theta*, take its diagonal QFI. Then train Task 2 (Fashion)
sequentially with NO protection, and at every step log the update's QFI displacement R in
MNIST's geometry alongside the old task's loss, mean correct-class probability, and accuracy.

Two questions:
  (a) Does R correlate with the subsequent old-task loss increase? (Pearson / Spearman)
  (b) Do R and the old-task probability move BEFORE accuracy collapses? (pre-collapse signal)

Run:
    python experiments/e007_h1_earlywarning.py --epochs 40 --seed 42
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.e005_softmax import _log_likelihood, accuracy, bce_loss, make_softmax_qnode  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"
N_QUBITS = 4


def _mean_correct_prob(qnode, w, X, y) -> float:
    return float(np.mean(np.exp(np.asarray(_log_likelihood(qnode, w, X, y)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--qfi-samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train,
                                  n_test=args.n_test, seed=args.seed)
    clf, wshape = make_softmax_qnode(N_QUBITS, args.layers)
    qfi_qnode, _ = make_qnode(N_QUBITS, args.layers)

    # Task 1 (MNIST) -> theta*.
    w = pnp.array(0.01 * np.random.default_rng(args.seed).standard_normal(wshape),
                  requires_grad=True)
    X1 = pnp.array(task1.X_train, requires_grad=False)
    y1 = pnp.array(task1.y_train, requires_grad=False)
    opt = qml.AdamOptimizer(args.lr)
    print(f"Pretraining on {task1.name} for {args.pretrain_epochs} epochs ...")
    t0 = time.perf_counter()
    for _ in range(args.pretrain_epochs):
        w = opt.step(lambda W: bce_loss(clf, W, X1, y1), w)
    fisher1 = quantum_fisher_diag(qfi_qnode, w, task1.X_train, n_samples=args.qfi_samples,
                                  seed=args.seed)
    print(f"  theta* ready; MNIST test acc = "
          f"{accuracy(clf, w, task1.X_test, task1.y_test):.3f} ({time.perf_counter()-t0:.0f}s)")

    # Sequential training on Task 2 (Fashion), logging every step.
    X2 = pnp.array(task2.X_train, requires_grad=False)
    y2 = pnp.array(task2.y_train, requires_grad=False)
    opt2 = qml.AdamOptimizer(args.lr)
    hist = []
    L_old_prev = float(bce_loss(clf, w, task1.X_test, task1.y_test))
    print(f"Training on {task2.name} (no protection), logging R vs old-task drift ...")
    for epoch in range(1, args.epochs + 1):
        w_before = np.asarray(w).copy()
        w = opt2.step(lambda W: bce_loss(clf, W, X2, y2), w)
        delta = (np.asarray(w) - w_before).ravel()
        R = float(np.sum(fisher1 * delta**2))
        step_norm2 = float(np.sum(delta**2))  # control: plain Euclidean step size
        L_old = float(bce_loss(clf, w, task1.X_test, task1.y_test))
        hist.append({
            "epoch": epoch, "R": R, "step_norm2": step_norm2,
            "old_loss": L_old, "delta_old_loss": L_old - L_old_prev,
            "old_prob": _mean_correct_prob(clf, w, task1.X_test, task1.y_test),
            "old_acc": accuracy(clf, w, task1.X_test, task1.y_test),
            "new_acc": accuracy(clf, w, task2.X_test, task2.y_test),
        })
        L_old_prev = L_old
        if epoch % 5 == 0 or epoch == 1:
            h = hist[-1]
            print(f"  ep {epoch:3d}  R={h['R']:.4f}  old_loss={h['old_loss']:.3f} "
                  f"old_prob={h['old_prob']:.3f} old_acc={h['old_acc']:.3f} "
                  f"new_acc={h['new_acc']:.3f}", flush=True)

    R = np.array([h["R"] for h in hist])
    dn = np.array([h["step_norm2"] for h in hist])
    dL = np.array([h["delta_old_loss"] for h in hist])
    # R_t vs same-step old-loss increase, and vs next-step (lag 1).
    pear = float(pearsonr(R, dL)[0])
    spear = float(spearmanr(R, dL)[0])
    pear_lag = float(pearsonr(R[:-1], dL[1:])[0])
    # CONTROL: plain step size ||dtheta||^2 as a predictor, and the QFI anisotropy.
    pear_stepnorm = float(pearsonr(dn, dL)[0])
    partial_R = float(pearsonr(R, dL)[0])  # placeholder; partial corr computed below
    lam = np.asarray(fisher1, dtype=float)
    eff_rank = float((lam.sum() ** 2) / np.sum(lam**2))  # participation ratio
    qfi_spread = {"min": round(float(lam.min()), 4), "max": round(float(lam.max()), 4),
                  "mean": round(float(lam.mean()), 4),
                  "effective_rank": round(eff_rank, 1), "n_weights": int(lam.size),
                  "anisotropy_max_over_min": round(float(lam.max() / max(lam.min(), 1e-9)), 2)}

    # Pre-collapse gap: first epoch old_prob drops below 0.6 vs first epoch old_acc drops below 0.9.
    prob = np.array([h["old_prob"] for h in hist])
    acc = np.array([h["old_acc"] for h in hist])
    def first_below(arr, thr):
        idx = np.where(arr < thr)[0]
        return int(idx[0] + 1) if len(idx) else None
    prob_drop_ep = first_below(prob, 0.6)
    acc_drop_ep = first_below(acc, 0.9)

    result = {
        "experiment": "e007_h1_earlywarning",
        "tasks": [task1.name, task2.name], "n_qubits": N_QUBITS, "layers": args.layers,
        "seed": args.seed, "epochs": args.epochs,
        "corr_R_vs_delta_old_loss": {"pearson": round(pear, 4), "spearman": round(spear, 4),
                                     "pearson_lag1": round(pear_lag, 4)},
        "control_stepnorm_vs_delta_old_loss": {"pearson": round(pear_stepnorm, 4)},
        "qfi_anisotropy": qfi_spread,
        "pre_collapse": {"old_prob<0.6_at_epoch": prob_drop_ep,
                         "old_acc<0.9_at_epoch": acc_drop_ep,
                         "lead_epochs": (None if (prob_drop_ep is None or acc_drop_ep is None)
                                         else acc_drop_ep - prob_drop_ep)},
        "history": hist,
    }
    print(f"\n=== H1 === corr(R, dL_old): pearson={pear:.3f} spearman={spear:.3f} "
          f"lag1={pear_lag:.3f}")
    print(f"    CONTROL corr(||dtheta||^2, dL_old): pearson={pear_stepnorm:.3f}  "
          f"(R beats step-size by {pear - pear_stepnorm:+.3f})")
    print(f"    QFI anisotropy: min={qfi_spread['min']} max={qfi_spread['max']} "
          f"eff_rank={qfi_spread['effective_rank']}/{qfi_spread['n_weights']} "
          f"(max/min={qfi_spread['anisotropy_max_over_min']})")
    print(f"    pre-collapse: old_prob<0.6 @ep {prob_drop_ep}, old_acc<0.9 @ep {acc_drop_ep} "
          f"(lead {result['pre_collapse']['lead_epochs']} epochs)")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e007_h1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
