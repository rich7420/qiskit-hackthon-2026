"""Plot the QEWC lam retune: does the 24-CNOT `aggressive` circuit match the 64-CNOT baseline?

Panel A: retention-plasticity plane, one curve per config as lam sweeps (lower-left = best) — if
aggressive's curve reaches baseline's region, the smaller circuit matches at 62% fewer CNOTs.
Panel B: average NMSE vs lam, marking each config's best operating point.

Run:
    python scripts/e009_gate_lam_sweep_plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"aggressive": "#CC3311", "baseline": "#4477AA", "enc_ry": "#EE7733", "chain": "#228833"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "e009_gate_lam_sweep.json")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "e009_gate_lam_sweep.png")
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    cfgs, seeds = data["configs"], data["seeds"]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.3))

    for name in cfgs:
        color = COLORS.get(name, "0.4")
        cnot = cfgs[name]["cost"]["cnot"]
        lam_keys = sorted(cfgs[name]["lams"], key=float)
        lams = [float(k) for k in lam_keys]
        ret = [cfgs[name]["lams"][k]["retention"]["mean"] for k in lam_keys]
        plas = [cfgs[name]["lams"][k]["plasticity"]["mean"] for k in lam_keys]
        avg = [cfgs[name]["lams"][k]["avg"]["mean"] for k in lam_keys]
        avg_sd = [cfgs[name]["lams"][k]["avg"]["sd"] for k in lam_keys]

        # Panel A: retention-plasticity tradeoff as lam sweeps
        axA.plot(ret, plas, "o-", color=color, ms=8, mec="k", lw=1.6,
                 label=f"{name} ({cnot} CNOT)")
        for L, x, y in zip(lams, ret, plas):
            axA.annotate(f"{L:g}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5,
                         color=color)

        # Panel B: avg NMSE vs lam, mark the best
        axB.errorbar(lams, avg, yerr=avg_sd, fmt="o-", color=color, ms=8, mec="k", capsize=3,
                     lw=1.6, label=f"{name} ({cnot} CNOT)")
        bi = min(range(len(avg)), key=lambda i: avg[i])
        axB.plot(lams[bi], avg[bi], "*", ms=20, color=color, mec="k", zorder=5)

    axA.set_xlabel("retention — earlier-task NMSE  (lower = better)")
    axA.set_ylabel("plasticity — new-task NMSE  (lower = better)")
    axA.set_title(f"Retention-plasticity tradeoff as lam sweeps ({len(seeds)} seeds)")
    axA.grid(alpha=0.25)
    axA.legend(fontsize=9, loc="best")
    axA.annotate("BEST", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=10,
                 fontweight="bold", color="tab:green", ha="left", va="bottom")

    axB.set_xlabel("QEWC anchor strength  lam  (log scale)")
    axB.set_ylabel("average final NMSE  (lower = better)")
    axB.set_title("Best operating point per config (star)")
    axB.set_xscale("log")
    axB.grid(alpha=0.25, which="both")
    axB.legend(fontsize=9, loc="best")

    fig.suptitle("e009 QEWC lam retune — can 24 CNOTs match 64 when each is tuned fairly?", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
