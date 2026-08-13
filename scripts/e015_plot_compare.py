"""Aggregate e015 OI-QCL-forecast seeds and plot retention vs plasticity per method.

Reads results/e015_oiqcl_forecast_seed*.json, averages retention / plasticity / forgetting /
avg_final across seeds, prints a table, and writes figures/e015_oiqcl_forecast_compare.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ("sequential", "qewc", "frozen_head", "free_head", "anchor_head")
KEYS = ("retention_earlier_nmse", "plasticity_final_nmse", "avg_earlier_forgetting", "avg_final_nmse")
LABELS = {"sequential": "Sequential (naive)", "qewc": "QEWC", "frozen_head": "Frozen theta + heads (A)",
          "free_head": "Free theta + heads (B)", "anchor_head": "Anchor theta + heads (C)"}


def main() -> None:
    files = sorted(RESULTS.glob("e015_oiqcl_forecast_seed*.json"))
    if not files:
        sys.exit("no results/e015_oiqcl_forecast_seed*.json found")
    runs = [json.loads(f.read_text()) for f in files]
    seeds = [r["config"]["seed"] for r in runs]
    print(f"aggregating {len(runs)} seeds: {seeds}")

    agg = {m: {k: [] for k in KEYS} for m in METHODS}
    for r in runs:
        for m in METHODS:
            for k in KEYS:
                agg[m][k].append(r["methods"][m][k])
    mean = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in agg.items()}
    std = {m: {k: float(np.std(v)) for k, v in d.items()} for m, d in agg.items()}

    print(f"\n{'method':14s} {'retention':>12s} {'plasticity':>12s} {'forgetting':>12s} {'avg_final':>12s}")
    for m in METHODS:
        print(f"{m:14s} "
              f"{mean[m]['retention_earlier_nmse']:.3f}±{std[m]['retention_earlier_nmse']:.3f}  "
              f"{mean[m]['plasticity_final_nmse']:.3f}±{std[m]['plasticity_final_nmse']:.3f}  "
              f"{mean[m]['avg_earlier_forgetting']:+.3f}±{std[m]['avg_earlier_forgetting']:.3f}  "
              f"{mean[m]['avg_final_nmse']:.3f}±{std[m]['avg_final_nmse']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipped figure")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"sequential": "#888", "qewc": "#c44", "frozen_head": "#48a",
              "free_head": "#e90", "anchor_head": "#2a7"}
    for m in METHODS:
        x = mean[m]["plasticity_final_nmse"]
        y = mean[m]["retention_earlier_nmse"]
        ax.scatter(x, y, s=140, color=colors[m], zorder=3, edgecolor="k", linewidth=0.6)
        ax.annotate(LABELS[m], (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Plasticity: final-task test NMSE (lower = learns new task)")
    ax.set_ylabel("Retention: earlier-task test NMSE (lower = remembers)")
    ax.set_title("e015 OI-QCL on forecasting (Task-IL): retention vs plasticity\n"
                 "bottom-left is best; A/C isolate the measurement side")
    ax.grid(True, alpha=0.3)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "e015_oiqcl_forecast_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
