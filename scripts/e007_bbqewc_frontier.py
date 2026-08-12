"""e007 BB-QEWC frontier — soft anchoring vs QFI-state vs CFI-function trust regions.

Sweeps regularization strength (lambda for qewc/l2, budget for the trust regions) to compare
retention-plasticity frontiers on the 3-task sequence. Single seed for exploration.

Run:
    python scripts/e007_bbqewc_frontier.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e007_bbqewc import build, train_method  # noqa: E402

RESULTS = ROOT / "results"

SWEEP = (
    [("sequential", {})]
    + [("l2", {"lam": v}) for v in (0.2, 0.5, 0.72, 1.5)]
    + [("qewc", {"lam": v}) for v in (0.3, 0.8, 2.0)]
    + [("tr_qfi", {"budget": v}) for v in (0.005, 0.02, 0.05, 0.15, 0.5)]
    + [("tr_cfi", {"budget": v}) for v in (0.0002, 0.001, 0.005, 0.02, 0.1)]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks, clf, qfi_qnode, wshape = build(args.seed, args.layers, args.n_train, args.n_test)
    print(f"BB-QEWC frontier seed={args.seed}: {len(SWEEP)} configs")
    t0 = time.perf_counter()
    points = []
    for method, over in SWEEP:
        r = train_method(method, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape, lr=0.02,
                         epochs=args.epochs_per_task, lam=over.get("lam", 0.8),
                         budget=over.get("budget", 1e9), seed=args.seed)
        param = over.get("lam", over.get("budget"))
        label = (f"{method}" + (f" λ={param}" if method in ("qewc", "l2")
                                else (f" B={param}" if method.startswith("tr") else "")))
        pt = {"method": method, "label": label, "param": param,
              "retention": r["retention_earlier_tasks"], "plasticity": r["plasticity_final_task"],
              "intervention": r["intervention_fraction"], "avg_forgetting": r["avg_forgetting"]}
        points.append(pt)
        print(f"  {label:16s} retention={pt['retention']:.3f} plasticity={pt['plasticity']:.3f} "
              f"interv={pt['intervention']:.2f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    (RESULTS / "e007_bbqewc_frontier.json").write_text(
        json.dumps({"experiment": "e007_bbqewc_frontier", "seed": args.seed, "points": points},
                   indent=2) + "\n")
    print(f"Wrote {RESULTS / 'e007_bbqewc_frontier.json'}")


if __name__ == "__main__":
    main()
