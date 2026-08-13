"""Aggregate E012 stability-plasticity frontiers across seeds (train-only selection)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
METHODS = ["output_cfi", "zz_cfi", "uniform_xyz", "mof_ewc", "readout_qewc", "qewc"]
L = {
    "output_cfi": "EWC(out-CFI)", "zz_cfi": "Joint-ZZ", "uniform_xyz": "Uniform-XYZ",
    "mof_ewc": "MOF-EWC", "readout_qewc": "Readout-QFI", "qewc": "QEWC(full-QFI)",
}
D = {s: json.loads((ROOT / f"results/e012_frontier_seed{s}.json").read_text()) for s in SEEDS}
split = sys.argv[1] if len(sys.argv) > 1 else "train"


def pts(s, m):
    return [
        (pt["lambda"], pt["metrics"][split]["task1_final_retention"],
         pt["metrics"][split]["task2_final_adaptation"])
        for pt in D[s]["frontier"][m]
    ]


def ret_at_plasticity(s, m, a0):
    cand = [ret for _, ret, ad in pts(s, m) if ad >= a0]
    return max(cand) if cand else np.nan


print(f"[{split}] frontier selection\n")
for a0 in (0.94, 0.92, 0.90):
    print(f"=== max retention s.t. adaptation >= {a0} (mean +/- SD over seeds) ===")
    rows = []
    for m in METHODS:
        vals = np.array([ret_at_plasticity(s, m, a0) for s in SEEDS])
        finite = vals[np.isfinite(vals)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        rows.append((mean, sd, m, vals))
    for mean, sd, m, vals in sorted(rows, key=lambda r: -r[0]):
        vv = "/".join(f"{v:.3f}" if np.isfinite(v) else "NA" for v in vals)
        print(f"   {L[m]:16s} {mean:.3f} +/- {sd:.3f}   [{vv}]")
    print()

print("=== best mean-final-acc over lambda grid (mean +/- SD over seeds) ===")
rows = []
for m in METHODS:
    per = np.array([max((r + a) / 2 for _, r, a in pts(s, m)) for s in SEEDS])
    rows.append((float(per.mean()), float(per.std(ddof=1)), m, per))
for mean, sd, m, per in sorted(rows, key=lambda r: -r[0]):
    vv = "/".join(f"{v:.3f}" for v in per)
    print(f"   {L[m]:16s} {mean:.4f} +/- {sd:.4f}   [{vv}]")

print("\n=== argmax lambda for retention@adapt>=0.92 per method/seed (operating point) ===")
for m in METHODS:
    locs = []
    for s in SEEDS:
        best = max(((ret, lam) for lam, ret, ad in pts(s, m) if ad >= 0.92), default=(np.nan, np.nan))
        locs.append(best[1])
    print(f"   {L[m]:16s} lambda*={locs}")
