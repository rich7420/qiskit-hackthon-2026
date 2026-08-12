"""Plot the BB-QEWC multi-seed retention-plasticity points with error bars.

Reads results/e007_bbqewc_multiseed.json. Run:
    python scripts/e007_bbqewc_plot_multiseed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_bbqewc_multiseed.json"
OUT = ROOT / "figures" / "e007_bbqewc_multiseed.png"

COLORS = {"sequential": "0.5", "QEWC (λ=0.8)": "#228833", "L2 (λ=0.2)": "#CCBB44",
          "QFI-TR (B=0.5)": "#4477AA", "CFI-TR (B=0.005)": "#EE6677"}


def main() -> None:
    summ = json.loads(RESULT.read_text())["summary"]
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    for label, s in summ.items():
        x, xe = s["plasticity"]["mean"], s["plasticity"]["sd"]
        y, ye = s["retention"]["mean"], s["retention"]["sd"]
        frac = s["intervention"]["mean"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=11, color=COLORS.get(label, "gray"),
                    ecolor=COLORS.get(label, "gray"), elinewidth=1.4, capsize=4,
                    markeredgecolor="k", zorder=3)
        ax.annotate(f"{label}" + (f"\n(interv {frac:.0%})" if frac > 0 else ""), (x, y),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("plasticity — final-task accuracy")
    ax.set_ylabel("stability — mean earlier-task retention")
    ax.set_title("BB-QEWC multi-seed: hard trust regions trade away new-task learning")
    ax.grid(alpha=0.25)
    ax.annotate("useful region →", xy=(0.98, 0.02), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=10, color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
