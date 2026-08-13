"""Accuracy (R^2 = 1 - NMSE) version of the advanced-method comparison (EWC / QEWC / QGR).

Same 3-panel per-task layout as e009_plot_advanced.py, but y = R^2 (higher = better, curves rise
during each task's phase like an accuracy plot). R^2 is clipped at 0 for display (R^2 < 0 means
"worse than predicting the mean", i.e. untrained). Reads results/e009_multiseed.json.

Run:
    python scripts/e009_plot_advanced_acc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_advanced_accuracy.png"

STYLE = {
    "naive": ("Baseline (naive seq.)", "0.5", "--", 1.6),
    "ewc": ("EWC (classical Fisher)", "#4477AA", "-.", 2.0),
    "qewc": ("QEWC (quantum Fisher)", "#228833", "-.", 2.0),
    "qgr": ("QGR (quantum generative replay)", "#CC3311", "-", 2.4),
}


def main() -> None:
    data = json.loads(RESULT.read_text())
    tasks = data["tasks"]
    curves = data["mean_curves"]
    ept = data["epochs_per_task"]
    total = curves["naive"][-1]["epoch"]
    boundaries = [ept * i for i in range(1, len(tasks))]
    seeds = data["seeds"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.4, 2.6 * len(tasks)), sharex=True)

    for i, (ax, task) in enumerate(zip(axes, tasks), start=1):
        for m, (label, color, ls, lw) in STYLE.items():
            rows = curves[m]
            ep = np.array([r["epoch"] for r in rows])
            nmse = np.array([r["nmse"][task]["mean"] for r in rows])
            sd = np.array([r["nmse"][task]["sd"] for r in rows])
            r2 = np.clip(1.0 - nmse, 0.0, 1.0)                     # accuracy-like R^2
            r2_lo = np.clip(1.0 - (nmse + sd), 0.0, 1.0)
            r2_hi = np.clip(1.0 - np.clip(nmse - sd, 0, None), 0.0, 1.0)
            ax.plot(ep, r2, ls, color=color, lw=lw, label=label)
            if m != "naive":
                ax.fill_between(ep, r2_lo, r2_hi, color=color, alpha=0.16)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.axvspan((i - 1) * ept, i * ept, color="green", alpha=0.05)
        ax.set_xlim(0, total)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"T{i}\nR² (=1−NMSE)")
        ax.set_title(f"Task {i}: {task}", loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower right", fontsize=9, ncol=2)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(f"Advanced methods on quantum forecasting — accuracy (R²) view: "
                 f"EWC vs QEWC vs QGR (seeds {seeds})", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
