"""Plot the e009 gate-count ablation: forecasting NMSE vs two-qubit (CNOT) cost.

Reads results/e009_gate_ablation_naive.json (or --input) and draws a Pareto view — average
forecasting NMSE (and the retention/plasticity split) against CNOT count per ansatz variant.
Lower-left is best: a cheaper circuit AND lower error. This is the figure that backs a
"same NMSE, fewer two-qubit gates" claim (CLAUDE.md: compare depth + two-qubit gate count).

Run:
    python scripts/e009_gate_ablation_plot.py
    python scripts/e009_gate_ablation_plot.py --input results/e009_gate_ablation_qewc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "e009_gate_ablation_naive.json")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "e009_gate_ablation.png")
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    cfgs = data["configs"]
    method, seeds = data["method"], data["seeds"]

    names = sorted(cfgs, key=lambda n: cfgs[n]["cost"]["cnot"])

    def col(n):  # colour reveals the dominant lever: single-axis (ry) vs two-axis (ry_rz) encoding
        return "#CC3311" if cfgs[n]["config"]["encoding"] == "ry" else "#4477AA"

    def mark(n):  # marker reveals depth: 1 layer (square) vs 2 layers (circle)
        return "s" if cfgs[n]["config"]["n_layers"] == 1 else "o"

    def get(n, k, sub="mean"):
        return cfgs[n]["perf"][k][sub]

    def pareto(xs, ys):  # lower-left staircase (minimise both CNOT and NMSE)
        order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
        front, best = [], float("inf")
        for i in order:
            if ys[i] < best - 1e-12:
                front.append(i); best = ys[i]
        return front

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.3))

    # Panel A: average final NMSE vs CNOT, coloured by encoding, with the Pareto frontier
    cnot = [cfgs[n]["cost"]["cnot"] for n in names]
    avg = [get(n, "avg") for n in names]
    fr = pareto(cnot, avg)
    axA.plot([cnot[i] for i in fr], [avg[i] for i in fr], "--", color="0.55", lw=1.3,
             zorder=1, label="Pareto frontier")
    for n in names:
        x, y = cfgs[n]["cost"]["cnot"], get(n, "avg")
        axA.errorbar(x, y, yerr=get(n, "avg", "sd"), fmt=mark(n), ms=12, color=col(n),
                     ecolor="0.5", capsize=4, mec="k", mew=1.1, zorder=3)
        axA.annotate(f"{n}\n({x} CNOT, d{cfgs[n]['cost']['depth']})", (x, y),
                     textcoords="offset points", xytext=(9, 5), fontsize=8.3,
                     fontweight="bold" if n in ("baseline", "aggressive", "enc_ry") else "normal")
    axA.set_xlabel("two-qubit (CNOT) count per forward pass  (lower = cheaper / less noise)")
    axA.set_ylabel("average final NMSE  (lower = better)")
    axA.set_title(f"Quality vs 2-qubit cost  ({method}, {len(seeds)} seeds)")
    axA.grid(alpha=0.25)
    axA.annotate("cheaper + better", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=10,
                 fontweight="bold", color="tab:green", ha="left", va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="#CC3311", mec="k", label="encoding=ry (1-axis)"),
               plt.Line2D([], [], marker="o", ls="", mfc="#4477AA", mec="k", label="encoding=ry_rz (2-axis)"),
               plt.Line2D([], [], marker="s", ls="", mfc="0.7", mec="k", label="1 layer"),
               plt.Line2D([], [], marker="o", ls="", mfc="0.7", mec="k", label="2 layers")]
    axA.legend(handles=handles, fontsize=8.5, loc="upper center", ncol=2)

    # Panel B: retention (filled) & plasticity (open) vs CNOT, same encoding colours
    for n in names:
        x = cfgs[n]["cost"]["cnot"]
        axB.errorbar(x, get(n, "retention"), yerr=get(n, "retention", "sd"), fmt=mark(n), ms=11,
                     color=col(n), ecolor="0.6", capsize=3, mec="k", zorder=3)
        axB.plot(x, get(n, "plasticity"), mark(n), ms=9, mfc="none", mec=col(n), mew=1.6, zorder=3)
        axB.annotate(n, (x, get(n, "retention")), textcoords="offset points", xytext=(6, 6),
                     fontsize=8, fontweight="bold" if n in ("baseline", "aggressive", "enc_ry") else "normal")
    axB.set_xlabel("two-qubit (CNOT) count per forward pass")
    axB.set_ylabel("NMSE  (lower = better)")
    axB.set_title("Retention (filled) & plasticity (open) vs 2-qubit cost")
    axB.grid(alpha=0.25)
    hb = [plt.Line2D([], [], marker="o", ls="", mfc="0.4", mec="k", label="retention — earlier tasks"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", mew=1.6, label="plasticity — new task")]
    axB.legend(handles=hb, fontsize=8.5, loc="best")

    fig.suptitle("e009 gate-count ablation — how small can the circuit get without losing forecasting quality?",
                 fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
