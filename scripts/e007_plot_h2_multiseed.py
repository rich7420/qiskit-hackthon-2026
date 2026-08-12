"""Plot the e007 H2 multi-seed retention-plasticity points with error bars.

Reads results/e007_h2_multiseed.json. Confirms across seeds that adaptive hard control is
Pareto-dominated by soft anchoring (QEWC/L2). Run:
    python scripts/e007_plot_h2_multiseed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_h2_multiseed.json"
OUT = ROOT / "figures" / "e007_h2_multiseed.png"

COLORS = {"sequential": "0.5", "QEWC (λ=0.8)": "#228833", "L2 (λ=0.2)": "#CCBB44",
          "Adaptive (ε×2.0)": "#EE6677"}


def main() -> None:
    summ = json.loads(RESULT.read_text())["summary"]
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    for label, s in summ.items():
        x, xe = s["plasticity"]["mean"], s["plasticity"]["sd"]
        y, ye = s["retention"]["mean"], s["retention"]["sd"]
        frac = s["intervention"]["mean"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=11, color=COLORS.get(label, "gray"),
                    ecolor=COLORS.get(label, "gray"), elinewidth=1.4, capsize=4,
                    markeredgecolor="k", zorder=3)
        tag = f"{label}" + (f"\n(interv {frac:.0%})" if frac > 0 else "")
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("plasticity — final-task accuracy")
    ax.set_ylabel("stability — mean earlier-task retention")
    ax.set_title("H2 multi-seed (5 seeds): adaptive hard control is Pareto-dominated")
    ax.grid(alpha=0.25)
    ax.annotate("better", xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                fontsize=12, fontweight="bold", color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
