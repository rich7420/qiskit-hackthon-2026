"""Plot the e007 H2 retention-plasticity Pareto for the five methods.

x = final-task accuracy (plasticity), y = earlier-task retention (stability); marker size
encodes intervention fraction. Upper-right dominates. Reads results/e007_h2_seed42.json
(or an aggregate if present).

Run:
    python scripts/e007_plot_h2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_h2_seed42.json"
OUT = ROOT / "figures" / "e007_h2_pareto.png"

COLORS = {"sequential": "0.5", "qewc": "#228833", "l2": "#CCBB44",
          "gradclip": "#66CCEE", "adaptive": "#EE6677"}


def main() -> None:
    data = json.loads(RESULT.read_text())
    methods = data["methods"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for name, r in methods.items():
        x = r["plasticity_final_task"]
        y = r["retention_earlier_tasks"]
        frac = r["intervention_fraction"]
        ax.scatter(x, y, s=120 + 900 * frac, color=COLORS.get(name, "gray"),
                   edgecolor="k", alpha=0.85, zorder=3)
        ax.annotate(f"{name}\n(interv {frac:.0%})", (x, y),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("plasticity — final-task (SPT/ATF) accuracy")
    ax.set_ylabel("stability — mean earlier-task retention")
    ax.set_title("H2: retention–plasticity Pareto (marker size = intervention fraction)")
    ax.grid(alpha=0.25)
    ax.annotate("better", xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=11, fontweight="bold", color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
