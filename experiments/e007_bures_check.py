"""e007 (step 1) — validate R: QFI-predicted state drift vs actual Bures distance.

Trains the classifier on Task 1 (MNIST), computes the diagonal QFI at that operating point,
then perturbs the weights over a range of magnitudes/directions and compares the QFI
prediction (1/2)sqrt(R), R = delta^T F_Q delta, against the actual mean Bures distance
D_B(rho(theta), rho(theta+delta)). This is the physical grounding for Q-MemGuard's
certificate: R measures how far an update moves the old-task quantum state.

Run:
    python experiments/e007_bures_check.py --n-train 400 --epochs 20 --seed 42
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
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.e005_softmax import bce_loss, make_softmax_qnode  # noqa: E402
from src.e007_qmemguard import make_state_qnode, mean_bures_distance  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"
N_QUBITS = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--qfi-samples", type=int, default=48)
    ap.add_argument("--n-dirs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    task1, _ = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train,
                              n_test=args.n_test, seed=args.seed)
    clf_qnode, wshape = make_softmax_qnode(N_QUBITS, args.layers)
    qfi_qnode, _ = make_qnode(N_QUBITS, args.layers)
    state_qnode = make_state_qnode(N_QUBITS, args.layers)

    # Train on Task 1 (MNIST) to a realistic operating point theta*.
    w = pnp.array(0.01 * np.random.default_rng(args.seed).standard_normal(wshape),
                  requires_grad=True)
    Xtr = pnp.array(task1.X_train, requires_grad=False)
    ytr = pnp.array(task1.y_train, requires_grad=False)
    opt = qml.AdamOptimizer(args.lr)
    print(f"Training on {task1.name} for {args.epochs} epochs ...")
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        w = opt.step(lambda W: bce_loss(clf_qnode, W, Xtr, ytr), w)
    print(f"  trained ({time.perf_counter()-t0:.0f}s)")

    # Diagonal QFI at theta* over Task 1 data.
    fisher = quantum_fisher_diag(qfi_qnode, w, task1.X_train, n_samples=args.qfi_samples,
                                 seed=args.seed)
    theta = np.asarray(w)

    # Sweep update magnitudes and random directions.
    rng = np.random.default_rng(args.seed + 1)
    scales = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    Xb = task1.X_test[:16]
    rows = []
    print("Sweeping updates: predicted (1/2)sqrt(R) vs actual Bures D_B ...")
    for s in scales:
        for _ in range(args.n_dirs):
            u = rng.standard_normal(theta.size)
            u /= np.linalg.norm(u)
            delta_flat = s * u
            R = float(np.sum(fisher * delta_flat**2))
            pred = 0.5 * np.sqrt(R)
            actual = mean_bures_distance(state_qnode, theta, theta + delta_flat.reshape(wshape), Xb)
            rows.append({"scale": s, "R": R, "predicted_half_sqrt_R": pred, "actual_bures": actual})
        print(f"  scale={s:.3f}: pred~{np.mean([r['predicted_half_sqrt_R'] for r in rows if r['scale']==s]):.3f}"
              f"  actual~{np.mean([r['actual_bures'] for r in rows if r['scale']==s]):.3f}", flush=True)

    pred = np.array([r["predicted_half_sqrt_R"] for r in rows])
    act = np.array([r["actual_bures"] for r in rows])
    small = act < 0.15  # local regime
    pearson = float(np.corrcoef(pred, act)[0, 1])
    slope = float(np.polyfit(act, pred, 1)[0])
    slope_small = float(np.polyfit(act[small], pred[small], 1)[0]) if small.sum() > 2 else None

    result = {
        "experiment": "e007_bures_check",
        "operating_point": f"trained on {task1.name}, {args.epochs} epochs",
        "n_qubits": N_QUBITS, "layers": args.layers, "n_weights": int(np.prod(wshape)),
        "qfi": "diagonal, Fubini-Study metric", "seed": args.seed,
        "pearson_pred_vs_actual": round(pearson, 4),
        "slope_overall": round(slope, 3),
        "slope_small_updates": None if slope_small is None else round(slope_small, 3),
        "note": "diagonal QFI upper-bounds true drift -> (1/2)sqrt(R) is a conservative, "
                "proportional estimate of Bures D_B; calibration of eps absorbs the constant",
        "sweep": rows,
    }
    print(f"\n=== R validation === pearson={pearson:.3f}  slope={slope:.2f}  "
          f"slope(small)={slope_small}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "e007_bures.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
