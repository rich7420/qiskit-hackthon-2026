"""e009 QGR comparison: retention vs plasticity for all six methods (both NMSE, lower-left best).

Highlights where Quantum Generative Replay (qgr) lands relative to classical replay and the
regularizers (Baseline/L2/EWC/QEWC). Reads results/e009_multiseed.json.

Run:
    python scripts/e009_qgr_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_multiseed.json"
OUT = ROOT / "figures" / "e009_qgr_compare.png"

# (label, color, marker) — QGR and replay emphasized
STYLE = {
    "naive": ("Baseline (naive seq.)", "0.55", "X"),
    "l2": ("L2 anchor", "#CCBB44", "o"),
    "ewc": ("EWC (classical Fisher)", "#4477AA", "o"),
    "qewc": ("QEWC (quantum Fisher)", "#228833", "s"),
    "replay": ("replay (stores raw data)", "#EE6677", "D"),
    "qgr": ("QGR (quantum generative)", "#AA3377", "*"),
}


def main() -> None:
    data = json.loads(RESULT.read_text())
    summ = data["summary"]
    n_seeds = len(data["seeds"])

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    for m, (label, color, marker) in STYLE.items():
        x = summ[m]["retention_earlier_nmse"]["mean"]
        xe = summ[m]["retention_earlier_nmse"]["sd"]
        y = summ[m]["plasticity_final_nmse"]["mean"]
        ye = summ[m]["plasticity_final_nmse"]["sd"]
        big = m in ("qgr", "replay")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=marker, ms=18 if m == "qgr" else 13,
                    color=color, ecolor=color, elinewidth=1.4, capsize=4,
                    markeredgecolor="k", markeredgewidth=1.2 if big else 0.6, zorder=4 if big else 3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(11, 7),
                    fontsize=10, fontweight="bold" if big else "normal")

    ax.set_xlabel("retention — earlier-task test NMSE   (lower ← better retained)")
    ax.set_ylabel("plasticity — new-task test NMSE   (lower ↓ better learned)")
    ax.set_title(f"Quantum Generative Replay vs baselines on quantum forecasting ({n_seeds} seeds)")
    ax.grid(alpha=0.25)
    ax.annotate("BEST\n(retain + learn)", xy=(0.02, 0.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=11, fontweight="bold", color="tab:green")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
