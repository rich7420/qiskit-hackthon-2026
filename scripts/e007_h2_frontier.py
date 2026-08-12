"""e007 H2 frontier — retention-plasticity Pareto frontiers for the anchoring vs adaptive methods.

Sweeps regularization strength: lambda for QEWC and L2, and the trust-region budget scale for
the adaptive method. Produces (retention, plasticity, intervention) points per config so the
actual frontiers can be compared, not just single operating points.

Run:
    python scripts/e007_h2_frontier.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e007_h2 import N_QUBITS, train_method  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_softmax import make_softmax_qnode  # noqa: E402
from src.phase_data import load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train, n_test=args.n_test,
                            seed=args.seed)
    t3 = load_spt_atf(n_train=args.n_train, n_test=args.n_test, n_qubits=N_QUBITS, seed=args.seed)
    tasks = [t1, t2, t3]
    clf, wshape = make_softmax_qnode(N_QUBITS, args.layers)
    qfi_qnode, _ = make_qnode(N_QUBITS, args.layers)

    configs = (
        [("sequential", {})]
        + [("l2", {"lam_l2": v}) for v in (0.2, 0.5, 0.72, 1.5, 3.0)]
        + [("qewc", {"lam_qewc": v}) for v in (0.3, 0.8, 2.0)]
        + [("adaptive", {"eps_scale": v}) for v in (0.25, 0.5, 1.0, 2.0, 4.0)]
    )

    def run(method, **over):
        return train_method(method, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape,
                            lr=args.lr, epochs=args.epochs_per_task,
                            lam_qewc=over.get("lam_qewc", 0.8), lam_l2=over.get("lam_l2", 0.72),
                            eps_scale=over.get("eps_scale", 1.0), seed=args.seed, verbose=False)

    print(f"H2 frontier seed={args.seed}: {len(configs)} configs")
    t0 = time.perf_counter()
    points = []
    for method, over in configs:
        r = run(method, **over)
        label = method + (f" λ={list(over.values())[0]}" if method in ("qewc", "l2")
                          else (f" ε×{over['eps_scale']}" if method == "adaptive" else ""))
        pt = {"method": method, "label": label, "param": (list(over.values())[0] if over else None),
              "retention": r["retention_earlier_tasks"], "plasticity": r["plasticity_final_task"],
              "intervention": r["intervention_fraction"], "avg_forgetting": r["avg_forgetting"]}
        points.append(pt)
        print(f"  {label:18s} retention={pt['retention']:.3f} plasticity={pt['plasticity']:.3f} "
              f"interv={pt['intervention']:.2f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    out = {"experiment": "e007_h2_frontier", "seed": args.seed, "points": points,
           "train_time_sec": round(time.perf_counter() - t0, 1)}
    (RESULTS / "e007_h2_frontier.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {RESULTS / 'e007_h2_frontier.json'}")


if __name__ == "__main__":
    main()
