"""Plot e009 continual forecasting TRAINING curves (per-task train NMSE over epochs).

Training-set counterpart of scripts/e009_plot.py (which shows test NMSE / forgetting). One panel
per task, method curves with mean +/- SD band across seeds, in the style of e005_ewc_qewc.png.
Reads results/e009_multiseed.json.

Run:
    python scripts/e009_plot_train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_training.png"

STYLE = {"naive": ("naive (no CL)", "0.45", "--"),
         "l2": ("L2 anchor", "#CCBB44", "-."),
         "ewc": ("EWC (classical Fisher)", "#4477AA", "-."),
         "qewc": ("QEWC (quantum Fisher)", "#228833", "-."),
         "replay": ("replay", "#EE6677", "-")}


def main() -> None:
    data = json.loads(RESULT.read_text())
    tasks = data["tasks"]
    curves = data["mean_curves"]
    epochs_per_task = data["epochs_per_task"]
    total_epochs = curves["naive"][-1]["epoch"]
    boundaries = [epochs_per_task * i for i in range(1, len(tasks))]
    seeds = data["seeds"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 2.5 * len(tasks)), sharex=True)

    for i, (ax, task) in enumerate(zip(axes, tasks), start=1):
        for m, (label, color, ls) in STYLE.items():
            rows = curves[m]
            ep = np.array([r["epoch"] for r in rows])
            mean = np.array([r["train_nmse"][task]["mean"] for r in rows])
            sd = np.array([r["train_nmse"][task]["sd"] for r in rows])
            ax.plot(ep, mean, ls, color=color, lw=2.0, label=label)
            ax.fill_between(ep, np.clip(mean - sd, 0, None), mean + sd, color=color, alpha=0.18)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.axvspan((i - 1) * epochs_per_task, i * epochs_per_task, color="green", alpha=0.05)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"T{i}\ntrain NMSE")
        ax.set_title(f"Task {i}: {task}", loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(f"E009: training-set NMSE (fit/plasticity) — naive vs L2 vs EWC vs QEWC vs "
                 f"replay (seeds {seeds})", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
