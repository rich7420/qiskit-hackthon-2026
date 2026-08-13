"""Plot E011 (blocked vs interleaved) in the E005/E006 three-panel style.

Three shared-x panels (one per task), test accuracy vs epoch, mean line + /-1 sample-SD band
across seeds, overlaying the blocked (sequential) and interleaved schedules. The dotted
vertical lines mark the blocked-run task boundaries; the interleaved run has no blocks, so
its curves rise together from epoch 0. Reads results/e011_summary.json.

Run:
    python scripts/e011_plot_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e011_summary.json"
OUT = ROOT / "figures" / "e011_interleaved.png"

TASK_KEYS = ("task1", "task2", "task3")
# schedule -> (label, color, linestyle)
STYLE = {
    "blocked": ("Blocked / sequential (E005 baseline)", "0.45", "--"),
    "interleaved": ("Interleaved (T1->T2->T3 each round)", "#228833", "-"),
}


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    hist = data["aggregate_histories"]
    boundaries = data["config"]["task_boundaries"]
    names = [data["aggregate_metrics"]["blocked"][k]["name"] for k in TASK_KEYS]
    total_epochs = hist["blocked"][-1]["epoch"]
    n_seeds = data["n_seeds"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, names), start=1):
        key = f"task{i}"
        for schedule, (label, color, ls) in STYLE.items():
            rows = hist[schedule]
            ep = np.array([r["epoch"] for r in rows])
            mean = np.array([r["test_accuracy"][key]["mean"] for r in rows])
            std = np.array([r["test_accuracy"][key]["sample_std"] for r in rows])
            ax.plot(ep, mean, ls, color=color, lw=2.0, label=label)
            ax.fill_between(ep, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.20)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls=":", lw=1.0)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.35, 1.03)
        ax.set_ylabel(f"T{i}\naccuracy")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=8.5, ncol=1)
    axes[-1].set_xlabel("Epoch (gradient step; 20 per task either way)")
    fig.suptitle("E011: blocked vs interleaved schedule, fixed 20 epochs/task\n"
                 f"MNIST -> Fashion-MNIST -> SPT/ATF, seeds {data['seeds']} (test accuracy)",
                 fontsize=12)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}  (n_seeds={n_seeds})")


if __name__ == "__main__":
    main()
