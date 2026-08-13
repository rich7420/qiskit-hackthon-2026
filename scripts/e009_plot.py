"""Plot e009 continual forecasting: per-task test NMSE over epochs for each method.

One panel per task (rows), each showing that task's test NMSE (lower=better) across the whole
sequential run for naive/l2/ewc/replay, with task-switch lines. Reads results/e009_multiseed.json
(mean curves across seeds) or a single-seed results/e009_continual_seed42.json.

Run:
    python scripts/e009_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "results" / "e009_multiseed.json"
SINGLE = ROOT / "results" / "e009_continual_seed42.json"
OUT = ROOT / "figures" / "e009_forgetting.png"

STYLE = {"naive": ("naive (no CL)", "0.45", "--"),
         "l2": ("L2 anchor", "#CCBB44", "-."),
         "ewc": ("EWC (Fisher)", "#4477AA", "-."),
         "replay": ("replay", "#EE6677", "-")}


def main() -> None:
    if MULTI.exists():
        data = json.loads(MULTI.read_text())
        tasks = data["tasks"]
        curves = {m: data["mean_curves"][m] for m in STYLE}
        title_suffix = f"(mean of {len(data['seeds'])} seeds)"
    else:
        data = json.loads(SINGLE.read_text())
        tasks = data["tasks"]
        curves = {m: data["methods"][m]["history"] for m in STYLE}
        title_suffix = f"(seed {data['seed']})"
    epochs_per_task = data["epochs_per_task"]
    boundaries = [epochs_per_task * i for i in range(1, len(tasks))]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 2.3 * len(tasks)), sharex=True)

    for i, (ax, task) in enumerate(zip(axes, tasks), start=1):
        for m, (label, color, ls) in STYLE.items():
            rows = curves[m]
            ep = [r["epoch"] for r in rows]
            v = [r["nmse"][task] for r in rows]
            ax.plot(ep, v, ls, color=color, lw=1.7, label=label)
        for b in boundaries:
            ax.axvline(b, ls="--", color="gray", lw=1.0, alpha=0.6)
        # shade the phase where this task is being trained
        ax.axvspan((i - 1) * epochs_per_task, i * epochs_per_task, color="green", alpha=0.05)
        ax.set_ylabel("test NMSE")
        ax.set_ylim(0, min(1.2, max(0.4, max(max(r["nmse"][task] for r in curves["l2"]) * 1.1,
                                             0.4))))
        ax.text(0.02, 0.86, f"Task {i}: {task}", transform=ax.transAxes, fontweight="bold")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[-1].set_xlabel("epoch (sequential training across tasks; shaded = task being trained)")
    fig.suptitle(f"e009: catastrophic forgetting in quantum time-series forecasting {title_suffix}",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
