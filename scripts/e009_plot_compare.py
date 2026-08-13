"""e009 method comparison: retention vs plasticity (both NMSE, lower-left is best).

Reads results/e009_multiseed.json. x = mean earlier-task test NMSE (retention; lower = better
retained), y = final-task test NMSE (plasticity; lower = better learned), error bars = sample SD
across seeds. The bottom-left corner is the ideal (retain old + learn new).

Run:
    python scripts/e009_plot_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_compare.png"

STYLE = {"naive": ("naive (no CL)", "0.45"), "l2": ("L2 anchor", "#CCBB44"),
         "ewc": ("EWC (Fisher)", "#4477AA"), "replay": ("replay", "#EE6677")}


def main() -> None:
    data = json.loads(RESULT.read_text())
    summ = data["summary"]
    n_seeds = len(data["seeds"])

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7, 5.6))

    for m, (label, color) in STYLE.items():
        x = summ[m]["retention_earlier_nmse"]["mean"]
        xe = summ[m]["retention_earlier_nmse"]["sd"]
        y = summ[m]["plasticity_final_nmse"]["mean"]
        ye = summ[m]["plasticity_final_nmse"]["sd"]
        avg = summ[m]["avg_final_nmse"]["mean"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=13, color=color, ecolor=color,
                    elinewidth=1.5, capsize=4, markeredgecolor="k", zorder=3)
        ax.annotate(f"{label}\navg NMSE {avg:.3f}", (x, y), textcoords="offset points",
                    xytext=(10, 8), fontsize=10)

    ax.set_xlabel("retention — earlier-task test NMSE   (lower ← better retained)")
    ax.set_ylabel("plasticity — new-task test NMSE   (lower ↓ better learned)")
    ax.set_title(f"e009: balancing retention vs adaptation on quantum forecasting "
                 f"({n_seeds} seeds)")
    ax.grid(alpha=0.25)
    ax.annotate("BEST\n(retain old + learn new)", xy=(0.02, 0.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=11, fontweight="bold", color="tab:green")
    # a little arrow toward the ideal corner
    ax.annotate("", xy=(0.13, 0.06), xytext=(0.30, 0.22), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.5))
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
