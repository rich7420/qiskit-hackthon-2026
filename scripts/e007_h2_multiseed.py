"""e007 H2 multi-seed confirmation of representative Pareto points.

Confirms the single-seed frontier finding across seeds: sequential (no protection), QEWC and
L2 at their best plasticity=1.0 operating points, and adaptive at its best point. Reports
mean +/- sample SD of retention, plasticity, and intervention fraction.

Run:
    python scripts/e007_h2_multiseed.py --seeds 42 43 44 45 46
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

from experiments.e007_h2 import N_QUBITS, train_method  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_softmax import make_softmax_qnode  # noqa: E402
from src.phase_data import load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"

# Representative operating points from the single-seed frontier.
CONFIGS = [
    ("sequential", "sequential", {}),
    ("qewc", "QEWC (λ=0.8)", {"lam_qewc": 0.8}),
    ("l2", "L2 (λ=0.2)", {"lam_l2": 0.2}),
    ("adaptive", "Adaptive (ε×2.0)", {"eps_scale": 2.0}),
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
        t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=args.n_train,
                                n_test=args.n_test, seed=seed)
        t3 = load_spt_atf(n_train=args.n_train, n_test=args.n_test, n_qubits=N_QUBITS, seed=seed)
        tasks = [t1, t2, t3]
        clf, wshape = make_softmax_qnode(N_QUBITS, args.layers)
        qfi_qnode, _ = make_qnode(N_QUBITS, args.layers)
        for method, label, over in CONFIGS:
            r = train_method(method, tasks, clf=clf, qfi_qnode=qfi_qnode, wshape=wshape,
                             lr=0.02, epochs=args.epochs_per_task,
                             lam_qewc=over.get("lam_qewc", 0.8), lam_l2=over.get("lam_l2", 0.72),
                             eps_scale=over.get("eps_scale", 1.0), seed=seed, verbose=False)
            raw[label]["retention"].append(r["retention_earlier_tasks"])
            raw[label]["plasticity"].append(r["plasticity_final_task"])
            raw[label]["intervention"].append(r["intervention_fraction"])
            raw[label]["avg_forgetting"].append(r["avg_forgetting"])
        print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    def ms(v):
        a = np.array(v)
        return {"mean": round(float(a.mean()), 4),
                "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}
    summary = {label: {k: ms(raw[label][k]) for k in raw[label]} for _, label, _ in CONFIGS}
    out = {"experiment": "e007_h2_multiseed", "seeds": args.seeds, "summary": summary, "raw": raw}
    (RESULTS / "e007_h2_multiseed.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== H2 multi-seed (mean +/- sd) ===")
    for _, label, _ in CONFIGS:
        s = summary[label]
        print(f"  {label:18s} retention={s['retention']['mean']:.3f}+/-{s['retention']['sd']:.3f}"
              f"  plasticity={s['plasticity']['mean']:.3f}+/-{s['plasticity']['sd']:.3f}"
              f"  interv={s['intervention']['mean']:.2f}")
    print(f"Wrote {RESULTS / 'e007_h2_multiseed.json'}")


if __name__ == "__main__":
    main()
