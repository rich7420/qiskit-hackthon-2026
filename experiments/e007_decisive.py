"""e007 (decisive) — does task-accessible directional geometry predict forgetting beyond step size?

Strip out step magnitude and test the directional Rayleigh quotient Q_F = v^T F v (v = update
direction) for three geometries: global pure-state QFI, readout-reduced mixed-state QFI, and
the classical (measurement) Fisher. The Go/No-Go question:

    After regressing forgetting on ||dtheta||^2, does Q_readout still explain the residual?

Evaluated at the CURRENT parameters (where the step and the loss change happen), over old-task
(MNIST) inputs. Run:
    python experiments/e007_decisive.py --epochs 40 --seed 42
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
from src.e005_softmax import _log_likelihood, accuracy, bce_loss, make_softmax_qnode  # noqa: E402
from src.e007_qmemguard import directional_qfi, make_state_qnode  # noqa: E402

N_QUBITS = 4
RESULTS = ROOT / "results"


def _ols_residual(y, X):
    """Return (residual, r2) for OLS y ~ [1, X...]."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return y - pred, 1 - ss_res / ss_tot, beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--pretrain-epochs", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--qfi-inputs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=RESULTS / "e007_decisive.json")
    args = ap.parse_args()

    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train,
                                  n_test=args.n_test, seed=args.seed)
    clf, wshape = make_softmax_qnode(N_QUBITS, args.layers)
    state_qnode = make_state_qnode(N_QUBITS, args.layers)

    w = pnp.array(0.01 * np.random.default_rng(args.seed).standard_normal(wshape),
                  requires_grad=True)
    X1 = pnp.array(task1.X_train, requires_grad=False)
    y1 = pnp.array(task1.y_train, requires_grad=False)
    opt = qml.AdamOptimizer(args.lr)
    print(f"Pretraining on {task1.name} ...")
    t0 = time.perf_counter()
    for _ in range(args.pretrain_epochs):
        w = opt.step(lambda W: bce_loss(clf, W, X1, y1), w)
    print(f"  MNIST test acc {accuracy(clf, w, task1.X_test, task1.y_test):.3f} "
          f"({time.perf_counter()-t0:.0f}s)")

    Xq = task1.X_train[:args.qfi_inputs]     # old-task inputs for QFI
    X1c = task1.X_train                       # old-task data for CFI + loss
    X2 = pnp.array(task2.X_train, requires_grad=False)
    y2 = pnp.array(task2.y_train, requires_grad=False)
    opt2 = qml.AdamOptimizer(args.lr)
    L_old_prev = float(bce_loss(clf, w, task1.X_test, task1.y_test))
    rows = []
    print(f"Training on {task2.name}; measuring directional predictors at each step ...")
    for epoch in range(1, args.epochs + 1):
        theta_before = np.asarray(w).copy()
        w = opt2.step(lambda W: bce_loss(clf, W, X2, y2), w)
        delta = np.asarray(w) - theta_before
        M = float(np.sum(delta**2))
        if M < 1e-14:
            continue
        # directional QFI at theta_before along the step (global + readout), over MNIST inputs
        Rg, Rr = directional_qfi(state_qnode, theta_before, delta, Xq, keep=(0, 1),
                                 n_qubits=N_QUBITS)
        # classical (measurement) Fisher, directional: mean_x (d_s log p(y|x))^2
        eps = 1e-3
        tp = pnp.array(theta_before + eps * delta, requires_grad=False)
        tm = pnp.array(theta_before - eps * delta, requires_grad=False)
        dlogp = (np.asarray(_log_likelihood(clf, tp, X1c, task1.y_train))
                 - np.asarray(_log_likelihood(clf, tm, X1c, task1.y_train))) / (2 * eps)
        Rc = float(np.mean(dlogp**2))
        L_old = float(bce_loss(clf, w, task1.X_test, task1.y_test))
        rows.append({"epoch": epoch, "step_norm2": M, "delta_old_loss": L_old - L_old_prev,
                     "R_global": Rg, "Q_global": Rg / M,
                     "R_readout": Rr, "Q_readout": Rr / M,
                     "R_cfi": Rc, "Q_cfi": Rc / M})
        L_old_prev = L_old
        if epoch % 5 == 0 or epoch == 1:
            r = rows[-1]
            print(f"  ep {epoch:3d} dL={r['delta_old_loss']:+.4f} "
                  f"Qg={r['Q_global']:.3f} Qs={r['Q_readout']:.3f} Qc={r['Q_cfi']:.2e}",
                  flush=True)

    dL = np.array([r["delta_old_loss"] for r in rows])
    M = np.array([r["step_norm2"] for r in rows])
    cols = {k: np.array([r[k] for r in rows]) for k in
            ("Q_global", "Q_readout", "Q_cfi", "R_global", "R_readout", "R_cfi")}

    def corr(a):
        return {"pearson": round(float(pearsonr(a, dL)[0]), 4),
                "spearman": round(float(spearmanr(a, dL)[0]), 4)}

    # Residualize forgetting on step size, then test directional signal on the residual.
    resid, r2_M, _ = _ols_residual(dL, [M])
    def partial(a):
        return round(float(pearsonr(a, resid)[0]), 4)

    incr = {}
    for name in ("Q_global", "Q_readout", "Q_cfi"):
        _, r2_both, beta = _ols_residual(dL, [M, cols[name]])
        incr[name] = {"delta_R2_over_stepsize": round(r2_both - r2_M, 4),
                      "beta_directional": round(float(beta[2]), 4)}

    result = {
        "experiment": "e007_decisive_directional",
        "tasks": [task1.name, task2.name], "layers": args.layers, "seed": args.seed,
        "evaluated_at": "current parameters (theta_before each step), old-task inputs",
        "corr_raw_R_vs_forgetting": {k: corr(cols[k]) for k in ("R_global", "R_readout", "R_cfi")},
        "corr_directional_Q_vs_forgetting": {k: corr(cols[k]) for k in ("Q_global", "Q_readout", "Q_cfi")},
        "corr_stepsize_vs_forgetting": corr(M),
        "R2_forgetting_on_stepsize": round(r2_M, 4),
        "partial_corr_directional_vs_residual": {k: partial(cols[k]) for k in ("Q_global", "Q_readout", "Q_cfi")},
        "incremental_over_stepsize": incr,
        "history": rows,
    }
    print("\n=== DECISIVE ===")
    print(f"  step-size alone: corr={result['corr_stepsize_vs_forgetting']['pearson']}  R^2={r2_M}")
    print("  directional Q (raw corr with forgetting):")
    for k in ("Q_global", "Q_readout", "Q_cfi"):
        print(f"    {k:10s} corr={corr(cols[k])['pearson']:+.3f}  "
              f"partial(resid)={partial(cols[k]):+.3f}  "
              f"dR^2={incr[k]['delta_R2_over_stepsize']:+.4f}")
    RESULTS.mkdir(exist_ok=True)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
