"""Two big-font, low-text pitch figures for a professor audience.

  e009_arch.png        (b) the quantum forecaster: |0> -> U(x) encode -> V(theta) ansatz
                           (recurrent) -> <Z> -> head -> forecast
  e009_qgr_concept.png (d) our method QGR: freeze -> rollout -> rehearse (+ one loss line)

Deliberately minimal wording and large type so it reads from the back of a room.

Run:
    python scripts/e009_pitch_figs.py            # both
    python scripts/e009_pitch_figs.py --which arch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGURES = ROOT / "figures"

CREAM, CREAM_E = "#FBEFCF", "#B9974A"
LBLUE, LBLUE_E = "#D6E3F0", "#5E86BC"
HEAD, HEAD_E = "#ECE6F4", "#8877AA"
BLUE, RED, GREEN = "#4477AA", "#CC3311", "#228833"

# (code name, T-label, display name, one-line "what it is", color, generating formula)
DATASETS = [
    ("narma_5", "$T_1$", "NARMA-5", "nonlinear autoregressive · memory-5", GREEN,
     r"$y_{t+1}=0.3\,y_t+0.05\,y_t\!\sum_{i=0}^{4}y_{t-i}+1.5\,u_{t-4}u_t+0.1$"),
    ("damped_shm", "$T_2$", "Damped SHM", r"damped pendulum · series $=\dot{\theta}$", BLUE,
     r"$\ddot{\theta}+0.15\,\dot{\theta}+9.81\,\sin\theta=0$"),
    ("bessel_j2", "$T_3$", "Bessel $J_2$", r"Bessel function, order 2 · $x\in[2,100]$", "#CC9933",
     r"$x^2y''+x\,y'+(x^2-4)\,y=0,\;\; y=J_2(x)$"),
]


def _scaled(series):
    s = np.asarray(series, float)
    return 2 * (s - s.min()) / (s.max() - s.min() + 1e-12) - 1


def _canvas(w, h):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _badge(ax, x, y, num, color, r=26, fs=17):
    ax.plot([x], [y], marker="o", markersize=r, color=color, zorder=5)
    ax.text(x, y, num, ha="center", va="center", color="white", fontsize=fs, fontweight="bold", zorder=6)


def draw_arch(output: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = _canvas(12.2, 5.4)
    ax.text(0.5, 0.92, "Recurrent quantum forecaster", ha="center", va="center",
            fontsize=25, fontweight="bold")

    ys = [0.40, 0.50, 0.60, 0.70]
    for y in ys:
        ax.plot([0.175, 0.60], [y, y], color="0.25", lw=1.6, zorder=2)
        ax.text(0.155, y, r"$|0\rangle$", ha="right", va="center", fontsize=19)

    # recurrent bracket around U and V
    ax.add_patch(FancyBboxPatch((0.205, 0.315), 0.30, 0.44, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.6, edgecolor="0.5", facecolor="none",
                                linestyle=(0, (5, 3)), zorder=2))
    ax.add_patch(FancyArrowPatch((0.435, 0.775), (0.275, 0.775), connectionstyle="arc3,rad=0.4",
                                 arrowstyle="-|>", mutation_scale=16, lw=1.6, color="0.45"))
    ax.text(0.355, 0.86, r"recurrent $\times\, T$  (state persists)", ha="center", va="center",
            fontsize=15, color="0.4")

    ax.add_patch(FancyBboxPatch((0.225, 0.34, ), 0.115, 0.38, boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=2.0, edgecolor=CREAM_E, facecolor=CREAM, zorder=3))
    ax.text(0.2825, 0.58, r"$U(x_t)$", ha="center", va="center", fontsize=25, fontweight="bold")
    ax.text(0.2825, 0.47, "data\nencoding", ha="center", va="center", fontsize=15, color="0.25")

    ax.add_patch(FancyBboxPatch((0.365, 0.34), 0.115, 0.38, boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=2.0, edgecolor=LBLUE_E, facecolor=LBLUE, zorder=3))
    ax.text(0.4225, 0.58, r"$V(\theta)$", ha="center", va="center", fontsize=25, fontweight="bold")
    ax.text(0.4225, 0.47, "trainable\nansatz", ha="center", va="center", fontsize=15, color="0.25")

    # readout
    for y in ys:
        ax.add_patch(FancyBboxPatch((0.565, y - 0.028), 0.05, 0.056, boxstyle="round,pad=0.002,rounding_size=0.008",
                                    linewidth=1.3, edgecolor="0.4", facecolor="0.96", zorder=3))
        ax.text(0.59, y, r"$\langle Z\rangle$", ha="center", va="center", fontsize=13)
    ax.add_patch(FancyArrowPatch((0.625, 0.55), (0.70, 0.55), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.4, color="0.35"))
    ax.add_patch(FancyBboxPatch((0.705, 0.47), 0.115, 0.16, boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=2.0, edgecolor=HEAD_E, facecolor=HEAD, zorder=3))
    ax.text(0.7625, 0.55, "tanh head", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.82, 0.55), (0.885, 0.55), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.4, color="0.35"))
    ax.text(0.93, 0.57, r"$\hat{y}_{t+1}$", ha="center", va="center", fontsize=25, fontweight="bold",
            color=RED)
    ax.text(0.93, 0.47, "next-step\nforecast", ha="center", va="center", fontsize=14, color="0.3")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def draw_qgr(output: Path) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = _canvas(12.2, 4.7)
    ax.text(0.5, 0.90, "Quantum Generative Replay", ha="center", va="center",
            fontsize=25, fontweight="bold")

    chips = [(0.205, BLUE, "1", "FREEZE", "the forecaster"),
             (0.500, RED, "2", "ROLLOUT", "synthetic old data"),
             (0.795, GREEN, "3", "REHEARSE", "new + old jointly")]
    w, h, yc = 0.235, 0.34, 0.50
    for xc, col, num, head, tag in chips:
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.03",
                                    linewidth=2.4, edgecolor=col, facecolor=mcolors.to_rgba(col, 0.09),
                                    zorder=3))
        _badge(ax, xc - w / 2 + 0.03, yc + h / 2 - 0.02, num, col, r=22, fs=15)
        ax.text(xc, yc + 0.035, head, ha="center", va="center", fontsize=21, fontweight="bold", color=col)
        ax.text(xc, yc - 0.065, tag, ha="center", va="center", fontsize=15, color="0.25")
    for xa, xb in [(0.3225, 0.3825), (0.6175, 0.6775)]:
        ax.add_patch(FancyArrowPatch((xa, yc), (xb, yc), arrowstyle="-|>", mutation_scale=26,
                                     lw=3.0, color="0.4"))

    ax.text(0.5, 0.185, r"$L \;=\; \mathrm{MSE}_{\mathrm{new}} \;+\; \mathrm{MSE}_{\mathrm{replay}}$",
            ha="center", va="center", fontsize=21,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F4F4F4", ec="0.7"))
    ax.text(0.5, 0.07, "replay the learned function — memory lives in the circuit, not stored data",
            ha="center", va="center", fontsize=14, color=RED, fontweight="bold")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def draw_data(output: Path) -> None:
    """Introduce the three forecasting tasks — real series, one line each, big type."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from src.e009_data import _SERIES

    fig = plt.figure(figsize=(14.6, 5.6))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, 1)
    cv.set_ylim(0, 1)
    cv.axis("off")
    cv.text(0.5, 0.955, "Three forecasting tasks (learned in sequence)", ha="center", va="center",
            fontsize=24, fontweight="bold")

    xcs = [0.19, 0.51, 0.83]
    for xc, (code, tlab, name, blurb, col, formula) in zip(xcs, DATASETS):
        s = _scaled(_SERIES[code]())
        n = len(s)
        axs = fig.add_axes([xc - 0.145, 0.52, 0.29, 0.30])
        axs.plot(s, color=col, lw=1.8)
        axs.axvspan(int(0.8 * n), n, color=col, alpha=0.09)   # held-out test tail
        axs.axvline(int(0.8 * n), color="0.6", lw=0.9, ls=":")
        axs.set_xlim(0, n)
        axs.set_ylim(-1.18, 1.18)
        axs.set_xticks([])
        axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_edgecolor("0.75")
        # T-badge + big name above
        cv.add_patch(FancyBboxPatch((xc - 0.145, 0.85, ), 0.05, 0.065,
                                    boxstyle="round,pad=0.002,rounding_size=0.01",
                                    linewidth=0, facecolor=col, zorder=3))
        cv.text(xc - 0.12, 0.8825, tlab, ha="center", va="center", color="white",
                fontsize=14, fontweight="bold", zorder=4)
        cv.text(xc - 0.08, 0.8825, name, ha="left", va="center", color=col,
                fontsize=19, fontweight="bold")
        cv.text(xc, 0.455, blurb, ha="center", va="center", fontsize=12.5, color="0.25")
        cv.text(xc, 0.31, formula, ha="center", va="center", fontsize=12.5, color="0.1",
                bbox=dict(boxstyle="round,pad=0.5", fc=mcolors.to_rgba(col, 0.06),
                          ec=mcolors.to_rgba(col, 0.6), lw=1.3))
    # train/test labels on the first panel only
    cv.text(xcs[0] - 0.05, 0.565, "train", ha="center", va="center", fontsize=10, color="0.45")
    cv.text(xcs[0] + 0.105, 0.565, "test", ha="center", va="center", fontsize=10, color="0.45")

    cv.text(0.5, 0.08, "each scaled to [-1, 1] · 8-step window -> next-step forecast · "
            "trained T1 -> T2 -> T3, never revisiting old data",
            ha="center", va="center", fontsize=13, color="0.35")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def draw_metric(output: Path) -> None:
    """A clean definition card for the accuracy metric: R^2 = 1 - NMSE."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = _canvas(9.8, 4.4)
    ax.text(0.5, 0.91, "forecast accuracy  ·  higher is better ↑", ha="center", va="center",
            fontsize=14, color=GREEN, fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.16, 0.50), 0.68, 0.26, boxstyle="round,pad=0.01,rounding_size=0.04",
                                linewidth=2.0, edgecolor=GREEN, facecolor="#F1F7F1", zorder=2))
    ax.text(0.5, 0.63, r"$R^2 \;=\; 1 \;-\; \mathrm{NMSE}$", ha="center", va="center",
            fontsize=44, zorder=3)

    ax.text(0.5, 0.32, r"$\mathrm{NMSE} \;=\; \frac{\mathrm{MSE}}{\mathrm{Var}(y)} "
            r"\;=\; \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$",
            ha="center", va="center", fontsize=23, color="0.15")

    ax.text(0.5, 0.08, r"$R^2\!=\!1$ perfect   ·   $R^2\!=\!0$ predicts the mean   ·   "
            r"$R^2\!<\!0$ worse than the mean", ha="center", va="center", fontsize=14, color="0.4")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["arch", "qgr", "data", "metric", "both", "all"])
    args = ap.parse_args()
    if args.which in ("arch", "both", "all"):
        draw_arch(FIGURES / "e009_arch.png")
    if args.which in ("qgr", "both", "all"):
        draw_qgr(FIGURES / "e009_qgr_concept.png")
    if args.which in ("data", "all"):
        draw_data(FIGURES / "e009_datasets.png")
    if args.which in ("metric", "all"):
        draw_metric(FIGURES / "e009_r2_metric.png")


if __name__ == "__main__":
    main()
