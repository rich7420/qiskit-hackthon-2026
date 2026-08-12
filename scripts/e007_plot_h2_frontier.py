"""Plot the e007 H2 retention-plasticity Pareto frontiers (L2 / QEWC / adaptive).

Each method's regularization sweep becomes a frontier in the (plasticity, retention) plane;
upper-right dominates. Reads results/e007_h2_frontier.json.

Run:
    python scripts/e007_plot_h2_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_h2_frontier.json"
OUT = ROOT / "figures" / "e007_h2_frontier.png"

STYLE = {"l2": ("L2 anchoring", "#CCBB44", "o"),
         "qewc": ("QEWC (global QFI)", "#228833", "s"),
         "adaptive": ("Adaptive trust region", "#EE6677", "^")}


def main() -> None:
    pts = json.loads(RESULT.read_text())["points"]
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7, 5.4))

    for method, (label, color, marker) in STYLE.items():
        mp = sorted([p for p in pts if p["method"] == method], key=lambda p: p["plasticity"])
        if not mp:
            continue
        x = [p["plasticity"] for p in mp]
        y = [p["retention"] for p in mp]
        ax.plot(x, y, "-", color=color, lw=1.6, marker=marker, markersize=8,
                markeredgecolor="k", label=label, zorder=3)
        for p in mp:
            ax.annotate(f"{p['param']}", (p["plasticity"], p["retention"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)

    seq = next((p for p in pts if p["method"] == "sequential"), None)
    if seq:
        ax.scatter([seq["plasticity"]], [seq["retention"]], marker="x", s=110, color="k",
                   zorder=4, label="sequential (no protection)")

    ax.set_xlabel("plasticity — final-task accuracy")
    ax.set_ylabel("stability — mean earlier-task retention")
    ax.set_title("H2 Pareto frontiers: does adaptive control beat always-on anchoring?")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=10)
    ax.annotate("better", xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                fontsize=12, fontweight="bold", color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
