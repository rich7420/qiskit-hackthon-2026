"""Plot E005 (Baseline / EWC / QEWC) in the style of figures/e004_continual_mean.png.

Three shared-x panels (one per task), test accuracy vs epoch, with a mean line and a +/-1
sample-SD band across seeds for each of the three methods, so Baseline can be compared with
EWC and QEWC. Reads results/e005_summary.json.

Run:
    python scripts/e005_plot_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e005_summary.json"
OUT = ROOT / "figures" / "e005_ewc_qewc.png"

TASK_KEYS = ("task1", "task2", "task3")
# method -> (label, color, linestyle) — Baseline gray dashed, EWC blue, QEWC green
STYLE = {
    "baseline": ("Baseline (no EWC)", "0.45", "--"),
    "ewc": ("EWC (CFI)", "#4477AA", "-"),
    "qewc": ("QEWC (QFI)", "#228833", "-"),
}


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    hist = data["aggregate_histories"]
    boundaries = data["config"]["task_boundaries"]
    # First epoch to draw for each task: skip everything before that task's training begins.
    intro = [1, boundaries[0], boundaries[1]]
    names = [data["aggregate_metrics"]["baseline"][k]["name"] for k in TASK_KEYS]
    total_epochs = hist["baseline"][-1]["epoch"]
    n_seeds = data["n_seeds"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, names), start=1):
        key = f"task{i}"
        for method, (label, color, ls) in STYLE.items():
            rows = [r for r in hist[method] if r["epoch"] >= intro[i - 1]]
            ep = np.array([r["epoch"] for r in rows])
            mean = np.array([r["test_accuracy"][key]["mean"] for r in rows])
            std = np.array([r["test_accuracy"][key]["sample_std"] for r in rows])
            ax.plot(ep, mean, ls, color=color, lw=2.0, label=label)
            ax.fill_between(ep, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.20)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.35, 1.03)
        ax.set_ylabel(f"T{i}\naccuracy")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=9, ncol=3)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"E005: Baseline vs EWC vs QEWC — seeds {data['seeds']} (test accuracy)",
                 fontsize=12)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}  (n_seeds={n_seeds})")


if __name__ == "__main__":
    main()
