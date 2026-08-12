"""e007 BB-QEWC multi-seed confirmation of representative operating points.

Confirms across seeds: soft anchoring (QEWC/L2) retains ~0.83 at full plasticity, while the
QFI-state and CFI-function trust regions only reach high retention by sacrificing new-task
learning (plasticity ~0.5). Reports mean +/- sample SD.

Run:
    python scripts/e007_bbqewc_multiseed.py --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e007_bbqewc import build, train_method  # noqa: E402

RESULTS = ROOT / "results"

CONFIGS = [
    ("sequential", "sequential", {}),
    ("qewc", "QEWC (λ=0.8)", {"lam": 0.8}),
    ("l2", "L2 (λ=0.2)", {"lam": 0.2}),
    ("tr_qfi", "QFI-TR (B=0.5)", {"budget": 0.5}),
    ("tr_cfi", "CFI-TR (B=0.005)", {"budget": 0.005}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    args = ap.parse_args()

    raw = {label: {"retention": [], "plasticity": [], "intervention": [], "avg_forgetting": []}
           for _, label, _ in CONFIGS}
    t0 = time.perf_counter()
    for seed in args.seeds:
        tasks, clf, qfi_qnode, wshape = build(seed, args.layers, args.n_train, args.n_test)
        for method, label, over in CONFIGS:
            r = train_method(method, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape, lr=0.02,
                             epochs=args.epochs_per_task, lam=over.get("lam", 0.8),
                             budget=over.get("budget", 1e9), seed=seed)
            for k in raw[label]:
                key = {"retention": "retention_earlier_tasks", "plasticity": "plasticity_final_task",
                       "intervention": "intervention_fraction", "avg_forgetting": "avg_forgetting"}[k]
                raw[label][k].append(r[key])
        print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    def ms(v):
        a = np.array(v)
        return {"mean": round(float(a.mean()), 4),
                "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}
    summary = {label: {k: ms(raw[label][k]) for k in raw[label]} for _, label, _ in CONFIGS}
    (RESULTS / "e007_bbqewc_multiseed.json").write_text(
        json.dumps({"experiment": "e007_bbqewc_multiseed", "seeds": args.seeds,
                    "summary": summary, "raw": raw}, indent=2) + "\n")

    print("\n=== BB-QEWC multi-seed (mean +/- sd) ===")
    for _, label, _ in CONFIGS:
        s = summary[label]
        print(f"  {label:16s} retention={s['retention']['mean']:.3f}+/-{s['retention']['sd']:.3f}"
              f"  plasticity={s['plasticity']['mean']:.3f}+/-{s['plasticity']['sd']:.3f}"
              f"  interv={s['intervention']['mean']:.2f}")
    print(f"Wrote {RESULTS / 'e007_bbqewc_multiseed.json'}")


if __name__ == "__main__":
    main()
