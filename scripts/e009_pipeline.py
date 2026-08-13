"""Clean pipeline diagram of the e009 sequential continual-learning setup.

Three time-series tasks are learned in order by ONE shared quantum forecaster; earlier tasks are
forgotten unless a continual-learning method protects them. Slide-style boxes + arrows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "e009_pipeline.png"


def box(ax, x, y, w, h, text, face, edge, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=2, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    BLUE, PURPLE, GREY = "#4477AA", "#7733AA", "0.4"

    tasks = [("Task 1\nnarma_5", "#EAF2FB"), ("Task 2\ndamped_shm", "#EAF7EE"),
             ("Task 3\nbessel_j2", "#FBEEF3")]
    tx = [0.5, 5.0, 9.5]
    # task data boxes (top)
    for (name, face), x in zip(tasks, tx):
        box(ax, x, 4.9, 3.0, 1.1, name, face, "#556", 11)

    # shared quantum model (middle band)
    box(ax, 2.4, 2.5, 8.2, 1.2,
        "ONE shared quantum forecaster  $f_\\theta$  (4-qubit recurrent data re-uploading, 21 params)",
        "#F3EEFA", PURPLE, 11)

    # sequential training arrows: each task -> the shared model, in order
    for x, lab in zip(tx, ["train 1st", "train 2nd", "train last"]):
        ax.add_patch(FancyArrowPatch((x + 1.5, 4.9), (x + 1.5, 3.7), arrowstyle="-|>",
                                     mutation_scale=20, lw=2, color=GREY, zorder=2))
        ax.text(x + 1.5, 4.35, lab, ha="center", fontsize=8.5, color=GREY,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    # order arrow across tasks
    ax.annotate("", xy=(4.9, 5.45), xytext=(3.5, 5.45), arrowprops=dict(arrowstyle="-|>", color="#556", lw=2))
    ax.annotate("", xy=(9.5, 5.45), xytext=(8.0, 5.45), arrowprops=dict(arrowstyle="-|>", color="#556", lw=2))

    # evaluation + problem (bottom)
    box(ax, 0.5, 0.4, 5.7, 1.2,
        "Every epoch: measure NMSE on ALL tasks\n→ catastrophic forgetting (old-task error rises)",
        "#FDECEC", "#CC3311", 10)
    box(ax, 6.8, 0.4, 5.7, 1.2,
        "Continual-learning methods balance:\nBaseline·L2·EWC·QEWC·replay·QGR",
        "#EAF7EE", "#228833", 10)
    ax.add_patch(FancyArrowPatch((5.3, 2.5), (3.3, 1.6), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color="#CC3311", zorder=2))
    ax.add_patch(FancyArrowPatch((7.7, 2.5), (9.6, 1.6), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color="#228833", zorder=2))

    ax.text(6.5, 6.25, "Sequential continual learning on quantum time-series forecasting",
            ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
