"""QGR vs QEWC, and what the single-axis architecture does — logged 5-seed comparison.

Left : retention-plasticity plane (lower-left = best). Colour = method, marker = architecture.
       An arrow shows what single-axis encoding does to QEWC (orig -> single-axis).
Right: average NMSE bar (sorted). The QGR + single-axis cell is NOT in any log, shown as a gap.

Reads results/e009_method_arch_compare.json (numbers cross-validated: QEWC-orig is identical in
both source logs). Run:
    python scripts/e009_method_arch_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "e009_method_arch_compare.json"
OUT = ROOT / "figures" / "e009_method_arch_compare.png"

MCOLOR = {"qgr": "#CC3311", "qewc": "#228833", "replay": "#4477AA", "naive": "0.6", "ewc": "#CCBB44"}
AMARK = {"orig": "o", "single-axis": "*"}


def main() -> None:
    data = json.loads(IN.read_text())
    combos = data["combos"]
    measured = [c for c in combos if c["avg"] is not None]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))

    # Panel A: retention-plasticity plane
    for c in measured:
        ms = 20 if c["arch"] == "single-axis" else 12
        axA.plot(c["ret"], c["plas"], AMARK[c["arch"]], ms=ms, color=MCOLOR[c["method"]],
                 mec="k", mew=1.1, zorder=4)
        axA.annotate(c["label"], (c["ret"], c["plas"]), textcoords="offset points",
                     xytext=(9, 5), fontsize=9,
                     fontweight="bold" if c["method"] in ("qgr", "qewc") else "normal")
    # arrow: what single-axis encoding does to QEWC
    o = next(c for c in measured if c["key"] == "qewc_orig")
    s = next(c for c in measured if c["key"] == "qewc_simpl")
    axA.annotate("", xy=(s["ret"], s["plas"]), xytext=(o["ret"], o["plas"]),
                 arrowprops=dict(arrowstyle="->", color="#228833", lw=2.0, ls="--"))
    axA.annotate("single-axis\narchitecture", (0.5 * (o["ret"] + s["ret"]), 0.5 * (o["plas"] + s["plas"])),
                 textcoords="offset points", xytext=(6, -22), fontsize=8.5, color="#228833",
                 fontweight="bold")
    axA.set_xlabel("retention — earlier-task NMSE  (lower = better)")
    axA.set_ylabel("plasticity — new-task NMSE  (lower = better)")
    axA.set_title("Retention-plasticity plane (logged, 5 seeds)")
    axA.grid(alpha=0.25)
    axA.annotate("BEST", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=11,
                 fontweight="bold", color="tab:green", ha="left", va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="0.7", mec="k", label="original arch (2-axis)"),
               plt.Line2D([], [], marker="*", ls="", ms=13, mfc="0.7", mec="k", label="single-axis arch")]
    axA.legend(handles=handles, fontsize=8.5, loc="upper left")

    # Panel B: average NMSE bar (sorted); QGR+single-axis is a "not measured" gap
    order = sorted(measured, key=lambda c: c["avg"])
    labels = [c["label"] for c in order] + ["QGR + single-axis"]
    vals = [c["avg"] for c in order] + [0.0]
    colors = [MCOLOR[c["method"]] for c in order] + ["none"]
    y = list(range(len(labels)))[::-1]
    bars = axB.barh(y, vals, color=colors, edgecolor="k", height=0.62)
    bars[-1].set_hatch("////")
    bars[-1].set_edgecolor("#CC3311")
    for c, yi in zip(order, y):
        axB.annotate(f"{c['avg']:.3f}", (c["avg"], yi), textcoords="offset points",
                     xytext=(4, -3), fontsize=9, fontweight="bold" if c["method"] in ("qgr", "qewc") else "normal")
    axB.annotate("?  not in any log", (0.005, y[-1]), textcoords="offset points", xytext=(4, -3),
                 fontsize=9, color="#CC3311", fontweight="bold")
    axB.set_yticks(y)
    axB.set_yticklabels(labels, fontsize=9)
    axB.set_xlabel("average final NMSE  (lower = better)")
    axB.set_title("Average NMSE — QEWC+single-axis leads what is measured")
    axB.grid(alpha=0.25, axis="x")
    axB.axvline(0.059, color="#4477AA", ls=":", lw=1.2)
    axB.annotate("replay = raw-data upper bound", (0.059, y[0]), textcoords="offset points",
                 xytext=(6, 8), fontsize=8, color="#4477AA")

    fig.suptitle("QGR vs QEWC across the single-axis architecture change — "
                 "method vs architecture are separate levers", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
