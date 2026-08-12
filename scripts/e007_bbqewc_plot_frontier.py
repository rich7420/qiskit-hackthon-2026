"""Plot the BB-QEWC retention-plasticity frontiers: soft anchoring vs QFI-state vs CFI-function.

Reads results/e007_bbqewc_frontier.json. Run:
    python scripts/e007_bbqewc_plot_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_bbqewc_frontier.json"
OUT = ROOT / "figures" / "e007_bbqewc_frontier.png"

STYLE = {"l2": ("L2 anchor (soft)", "#CCBB44", "o"),
         "qewc": ("QEWC / global-QFI (soft)", "#228833", "s"),
         "tr_qfi": ("QFI state trust region", "#4477AA", "^"),
         "tr_cfi": ("CFI function trust region", "#EE6677", "D")}


def main() -> None:
    pts = json.loads(RESULT.read_text())["points"]
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    for method, (label, color, marker) in STYLE.items():
        mp = sorted([p for p in pts if p["method"] == method], key=lambda p: p["plasticity"])
        if not mp:
            continue
        ax.plot([p["plasticity"] for p in mp], [p["retention"] for p in mp], "-",
                color=color, lw=1.6, marker=marker, markersize=8, markeredgecolor="k",
                label=label, zorder=3)
        for p in mp:
            ax.annotate(f"{p['param']}\n{p['intervention']:.0%}", (p["plasticity"], p["retention"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color=color)

    seq = next((p for p in pts if p["method"] == "sequential"), None)
    if seq:
        ax.scatter([seq["plasticity"]], [seq["retention"]], marker="x", s=120, color="k",
                   zorder=4, label="sequential")

    ax.set_xlabel("plasticity — final-task accuracy")
    ax.set_ylabel("stability — mean earlier-task retention")
    ax.set_title("Trust regions vs soft anchoring: does protecting state vs function matter?")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    ax.annotate("better", xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                fontsize=12, fontweight="bold", color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
