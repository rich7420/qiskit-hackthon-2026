"""Plot e007 R-validation: QFI-predicted (1/2)sqrt(R) vs actual Bures state drift.

Reads results/e007_bures.json. Left: predicted vs actual scatter (colored by update size)
with a linear fit and the y=x reference. Right: both quantities vs update magnitude.

Run:
    python scripts/e007_plot_bures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e007_bures.json"
OUT = ROOT / "figures" / "e007_bures.png"


def main() -> None:
    data = json.loads(RESULT.read_text())
    rows = data["sweep"]
    pred = np.array([r["predicted_half_sqrt_R"] for r in rows])
    act = np.array([r["actual_bures"] for r in rows])
    scale = np.array([r["scale"] for r in rows])
    pearson = data["pearson_pred_vs_actual"]
    slope = data["slope_overall"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: predicted vs actual
    sc = axl.scatter(act, pred, c=np.log10(scale), cmap="viridis", s=40, edgecolor="k",
                     linewidth=0.3)
    lim = max(act.max(), pred.max()) * 1.05
    axl.plot([0, lim], [0, lim], "--", color="gray", lw=1.2, label="y = x")
    xs = np.linspace(0, act.max(), 50)
    axl.plot(xs, slope * xs, "-", color="tab:red", lw=1.6,
             label=f"fit: slope={slope:.2f}")
    axl.set_xlabel(r"actual Bures drift  $D_B$")
    axl.set_ylabel(r"predicted  $\frac{1}{2}\sqrt{R}$  (QFI)")
    axl.set_title(f"R predicts state drift   (Pearson r = {pearson:.3f})")
    axl.legend(loc="upper left", fontsize=10)
    axl.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=axl); cb.set_label(r"update size  $\log_{10}\|\Delta\theta\|$")

    # Right: both vs update magnitude
    order = np.argsort(scale)
    axr.plot(scale[order], act[order], "o", color="tab:blue", ms=4, alpha=0.5,
             label=r"actual $D_B$")
    axr.plot(scale[order], pred[order], "^", color="tab:red", ms=4, alpha=0.5,
             label=r"predicted $\frac{1}{2}\sqrt{R}$")
    axr.set_xscale("log")
    axr.set_xlabel(r"update magnitude  $\|\Delta\theta\|$")
    axr.set_ylabel("state drift")
    axr.set_title("both grow together; tight in the local regime")
    axr.legend(loc="upper left", fontsize=10)
    axr.grid(alpha=0.25)

    fig.suptitle("Q-MemGuard step 1: the QFI displacement R measures old-task state drift",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
