"""e007 (P2) — actual-Bures rescue: does finite state distance beat Euclidean step distance?

QFI is only a local quadratic approximation. This uses the ACTUAL Bures distance of each step
(global pure-state and readout-reduced mixed-state) and asks whether it predicts old-task
forgetting better than plain Euclidean parameter movement ||dtheta||. If even exact state
distance does not beat Euclidean, the quantum-direction line can be closed with confidence.

Run:
    python experiments/e007_bures_rescue.py --epochs 40 --seed 42
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
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_softmax import accuracy, bce_loss, make_softmax_qnode  # noqa: E402
from src.e007_qmemguard import make_state_qnode, mean_bures_distance, mean_readout_bures  # noqa: E402

N_QUBITS = 4
RESULTS = ROOT / "results"


def _incremental_r2(y, base, extra):
    def ols(cols):
        A = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in cols])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        p = A @ b
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), b
    r2_base, _ = ols([base])
    r2_both, beta = ols([base, extra])
    return r2_base, r2_both - r2_base, float(beta[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--pretrain-epochs", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bures-inputs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=RESULTS / "e007_bures_rescue.json")
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
    print("Pretraining on MNIST ...")
    t0 = time.perf_counter()
    for _ in range(args.pretrain_epochs):
        w = opt.step(lambda W: bce_loss(clf, W, X1, y1), w)
    print(f"  MNIST test acc {accuracy(clf, w, task1.X_test, task1.y_test):.3f} "
          f"({time.perf_counter()-t0:.0f}s)")

    Xb = task1.X_train[:args.bures_inputs]
    X2 = pnp.array(task2.X_train, requires_grad=False)
    y2 = pnp.array(task2.y_train, requires_grad=False)
    opt2 = qml.AdamOptimizer(args.lr)
    L_old_prev = float(bce_loss(clf, w, task1.X_test, task1.y_test))
    rows = []
    print("Training on Fashion; measuring actual Bures per step ...")
    for epoch in range(1, args.epochs + 1):
        wb = np.asarray(w).copy()
        w = opt2.step(lambda W: bce_loss(clf, W, X2, y2), w)
        euclid = float(np.linalg.norm(np.asarray(w) - wb))
        dbg = mean_bures_distance(state_qnode, wb, np.asarray(w), Xb)
        dbr = mean_readout_bures(state_qnode, wb, np.asarray(w), Xb)
        L_old = float(bce_loss(clf, w, task1.X_test, task1.y_test))
        rows.append({"epoch": epoch, "euclid_step": euclid, "bures_global": dbg,
                     "bures_readout": dbr, "delta_old_loss": L_old - L_old_prev})
        L_old_prev = L_old
        if epoch % 10 == 0 or epoch == 1:
            r = rows[-1]
            print(f"  ep {epoch:3d} dL={r['delta_old_loss']:+.4f} euclid={euclid:.3f} "
                  f"DBg={dbg:.3f} DBr={dbr:.3f}", flush=True)

    dL = np.array([r["delta_old_loss"] for r in rows])
    E = np.array([r["euclid_step"] for r in rows])
    DBg = np.array([r["bures_global"] for r in rows])
    DBr = np.array([r["bures_readout"] for r in rows])

    def pc(a):
        return round(float(pearsonr(a, dL)[0]), 4)
    r2_E, dR2_g, b_g = _incremental_r2(dL, E, DBg)
    _, dR2_r, b_r = _incremental_r2(dL, E, DBr)

    result = {
        "experiment": "e007_bures_rescue", "seed": args.seed, "layers": args.layers,
        "corr_with_forgetting": {"euclid_step": pc(E), "bures_global": pc(DBg),
                                 "bures_readout": pc(DBr)},
        "R2_forgetting_on_euclid": round(r2_E, 4),
        "incremental_R2_over_euclid": {"bures_global": round(dR2_g, 4),
                                       "bures_readout": round(dR2_r, 4)},
        "history": rows,
    }
    print("\n=== BURES RESCUE ===")
    print(f"  corr(euclid, dL)={pc(E)}   R^2={r2_E:.3f}")
    print(f"  corr(bures_global, dL)={pc(DBg)}   incremental over euclid dR^2={dR2_g:+.4f}")
    print(f"  corr(bures_readout, dL)={pc(DBr)}  incremental over euclid dR^2={dR2_r:+.4f}")
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
