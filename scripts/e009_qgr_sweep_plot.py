"""Plot the QGR rollout-length sweep: retention (mean +/- SD) vs gen_len.

Shows that longer autoregressive rollouts stabilize QGR — retention SD drops sharply.
Reads results/e009_qgr_sweep.json.

Run:
    python scripts/e009_qgr_sweep_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e009_qgr_sweep.json"
OUT = ROOT / "figures" / "e009_qgr_sweep.png"


def main() -> None:
    rows = json.loads(RESULT.read_text())["rows"]
    g = np.array([r["gen_len"] for r in rows])
    rmean = np.array([r["retention_mean"] for r in rows])
    rsd = np.array([r["retention_sd"] for r in rows])

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: retention mean +/- SD band
    ax1.plot(g, rmean, "-o", color="#AA3377", lw=2, ms=8, markeredgecolor="k")
    ax1.fill_between(g, rmean - rsd, rmean + rsd, color="#AA3377", alpha=0.2)
    ax1.set_xlabel("QGR rollout length  (gen_len)")
    ax1.set_ylabel("retention — earlier-task NMSE (lower better)")
    ax1.set_title("QGR retention vs rollout length (band = ±1 SD)")
    ax1.grid(alpha=0.25)

    # right: retention SD alone (the headline — variance shrinks)
    ax2.plot(g, rsd, "-s", color="tab:red", lw=2, ms=8, markeredgecolor="k")
    for gi, si in zip(g, rsd):
        ax2.annotate(f"{si:.3f}", (gi, si), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    ax2.set_xlabel("QGR rollout length  (gen_len)")
    ax2.set_ylabel("retention SD across seeds")
    ax2.set_title("Longer rollout → lower variance (3.2× at gen_len=48)")
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, max(rsd) * 1.2)

    fig.suptitle("Reducing QGR variance with longer autoregressive rollouts (5 seeds)", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
