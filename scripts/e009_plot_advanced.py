"""Compare the advanced methods EWC / QEWC / QGR (+ Baseline reference) per dataset.

Three stacked panels (one per task), test NMSE vs epoch (log-y so the high-start and drop show),
mean +/- SD band across seeds. Baseline is a gray dashed reference; EWC (classical Fisher), QEWC
(quantum Fisher) and QGR (quantum generative replay) are the compared methods. Reads
results/e009_multiseed.json.

Run:
    python scripts/e009_plot_advanced.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_advanced_compare.png"

# Baseline dashed-gray reference; EWC blue, QEWC green, QGR red (emphasized)
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
            mean = np.array([r["nmse"][task]["mean"] for r in rows])
            sd = np.array([r["nmse"][task]["sd"] for r in rows])
            ax.plot(ep, mean, ls, color=color, lw=lw, label=label)
            if m != "naive":
                ax.fill_between(ep, np.clip(mean - sd, 2e-2, None), mean + sd,
                                color=color, alpha=0.16)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.axvspan((i - 1) * ept, i * ept, color="green", alpha=0.05)
        ax.set_xlim(0, total)
        ax.set_yscale("log")
        ax.set_ylim(2e-2, 6)
        ax.set_ylabel(f"T{i}\ntest NMSE")
        ax.set_title(f"Task {i}: {task}", loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2, which="both")

    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(f"Advanced continual-learning methods on quantum forecasting — "
                 f"EWC vs QEWC vs QGR (seeds {seeds}, test NMSE)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
