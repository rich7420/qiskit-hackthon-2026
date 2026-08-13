"""Clean flowchart of the Quantum Generative Replay (QGR) method (boxes + arrows, no gates).

Separates the METHOD FLOW from the circuit diagram: after learning a task, freeze the quantum
forecaster as a generator; roll it out to synthesize old-task sequences; rehearse them alongside
the new task; the learning model updates while the generator stays frozen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "e009_qgr_flow.png"


def box(ax, x, y, w, h, text, face, edge, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=2, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, p0, p1, color="0.3", lw=2.2, ls="-", rad=0.0, label=None, lpos=None, lcolor=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22, lw=lw,
                                 color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        lx, ly = lpos if lpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=9.5,
                color=lcolor or color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    BLUE, GREEN, RED, YEL = "#4477AA", "#228833", "#CC3311", "#E8A33D"

    # --- generation lane (top) ---
    box(ax, 0.4, 4.7, 3.0, 1.2,
        "❄ Frozen generator\n$G_j = f_{\\theta_j^*}$\n(quantum circuit, 21 params)", "#EAF7EE", GREEN, 10)
    box(ax, 4.9, 4.7, 3.2, 1.2,
        "Autoregressive rollout\n$\\hat{s}_t = G_j(\\hat{s}_{t-8:t-1})$\n(feed output back in)", "#FBEEF3", RED, 10)
    box(ax, 9.6, 4.7, 3.0, 1.2,
        "Synthetic old-task\nsequences $\\tilde{\\mathcal{D}}_j$\n(no raw data stored)", "#FDECEC", RED, 10)
    arrow(ax, (3.4, 5.3), (4.9, 5.3), RED, label="rollout", lcolor=RED)
    arrow(ax, (8.1, 5.3), (9.6, 5.3), RED, label="window", lcolor=RED)

    # --- learning lane (bottom) ---
    box(ax, 0.4, 1.1, 3.0, 1.2, "New task data\n$\\mathcal{D}_k$", "#EAF2FB", BLUE, 11)
    box(ax, 4.9, 1.1, 3.2, 1.2,
        "Rehearse (loss)\n$\\mathcal{L}=\\mathrm{MSE}(\\mathcal{D}_k)+\\mathrm{MSE}(\\tilde{\\mathcal{D}}_j)$",
        "#FFF6E6", YEL, 10.5)
    box(ax, 9.6, 1.1, 3.0, 1.2, "Learning model $f_\\theta$\n(gets updated)", "#EAF2FB", BLUE, 10.5)
    arrow(ax, (3.4, 1.7), (4.9, 1.7), "0.35")
    arrow(ax, (8.1, 1.7), (9.6, 1.7), "0.35", label="∇ update", lcolor="0.3")

    # synthetic data feeds the rehearsal
    arrow(ax, (11.1, 4.7), (6.5, 2.3), RED, rad=-0.15, label="rehearse\ngenerated", lpos=(9.6, 3.5), lcolor=RED)
    # after learning a task, freeze the model as the next generator
    arrow(ax, (11.1, 2.3), (1.9, 4.7), GREEN, ls="--", rad=-0.28,
          label="after task $k$:  freeze  $f_\\theta \\to G_k$", lpos=(6.5, 3.75), lcolor=GREEN)

    ax.text(6.5, 6.55, "Quantum Generative Replay — memory lives in the quantum circuit, not a raw-data buffer",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(1.9, 4.35, "GENERATION (frozen)", ha="center", fontsize=8.5, color=GREEN, style="italic")
    ax.text(1.9, 0.75, "LEARNING (updates)", ha="center", fontsize=8.5, color=BLUE, style="italic")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
