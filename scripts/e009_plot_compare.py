"""e009 method comparison across all three datasets (grouped bars, final test NMSE).

For each of the 3 forecasting tasks, show the final test NMSE (lower = better) of every method,
with +/-1 sample-SD error bars across seeds. This makes per-dataset behaviour explicit (naive
forgets the earlier tasks; replay/QEWC retain them). Reads results/e009_multiseed.json.

Run:
    python scripts/e009_plot_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_compare.png"

STYLE = {"naive": ("naive (no CL)", "0.55"), "l2": ("L2 anchor", "#CCBB44"),
         "ewc": ("EWC (classical Fisher)", "#4477AA"), "qewc": ("QEWC (quantum Fisher)", "#228833"),
         "replay": ("replay", "#EE6677")}


def main() -> None:
    data = json.loads(RESULT.read_text())
    tasks = data["tasks"]
    methods = list(STYLE)
    n_seeds = len(data["seeds"])
    # final-epoch per-task test NMSE (mean + sd) for each method
    final = {m: data["mean_curves"][m][-1]["nmse"] for m in methods}

    x = np.arange(len(tasks))
    w = 0.16
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for j, m in enumerate(methods):
        means = [final[m][t]["mean"] for t in tasks]
        sds = [final[m][t]["sd"] for t in tasks]
        label, color = STYLE[m]
        ax.bar(x + (j - 2) * w, means, w, yerr=sds, capsize=3, color=color,
               edgecolor="k", linewidth=0.5, label=label,
               error_kw={"elinewidth": 1, "alpha": 0.7})

    # mark which task is trained in which phase (1st, 2nd, last)
    phase_lbl = ["trained 1st\n(most forgotten)", "trained 2nd", "trained last\n(newest)"]
    for i, t in enumerate(tasks):
        ax.text(i, -0.07, phase_lbl[i], ha="center", va="top", fontsize=8.5, color="dimgray",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {i+1}\n{t}" for i, t in enumerate(tasks)])
    ax.set_ylabel("final test NMSE  (lower = better)")
    ax.set_title(f"e009: per-dataset final forecasting error after sequential training "
                 f"({n_seeds} seeds)")
    ax.legend(loc="upper center", fontsize=9, ncol=5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, None)
    ax.margins(y=0.15)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
