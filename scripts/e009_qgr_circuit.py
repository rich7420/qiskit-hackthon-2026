"""QGR circuit schematic: the frozen quantum forecaster used as an autoregressive generator.

Shows one rollout step of the recurrent data-reuploading circuit (frozen at theta*), the
Pauli-Z readout + tanh head producing the next value, and the AUTOREGRESSIVE FEEDBACK LOOP that
appends the generated value to the window to produce the next input — this loop is what makes
QGR generate old-task sequences without stored data. The generated sequence is then rehearsed
into the (separately updated) learning model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "e009_qgr_circuit.png"

BLUE, GREEN, RED, GREY = "#4477AA", "#228833", "#CC3311", "0.45"


def gate(ax, x, y, label, color, w=0.62, h=0.42):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor="k", facecolor=color, alpha=0.85, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold", zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    qy = [6.4, 5.5, 4.6, 3.7]           # 4 qubit wires
    x0, x1 = 1.2, 9.6
    for i, y in enumerate(qy):
        ax.plot([x0, x1], [y, y], color="k", lw=1.1, zorder=1)
        ax.text(x0 - 0.35, y, f"$q_{i}$", ha="right", va="center", fontsize=11)

    # --- one frozen re-uploading step ---
    for y in qy:
        gate(ax, 2.3, y, "RY$(x_t)$", BLUE)
        gate(ax, 3.1, y, "RZ$(x_t)$", BLUE)
        gate(ax, 4.3, y, "RY$(\\theta^*)$", GREEN)
        gate(ax, 5.1, y, "RZ$(\\theta^*)$", GREEN)
    # CNOT ring
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        ax.plot([6.0, 6.0], [qy[a], qy[b]], color="k", lw=1.3, zorder=3)
        ax.add_patch(plt.Circle((6.0, qy[a]), 0.06, color="k", zorder=4))
        ax.add_patch(plt.Circle((6.0, qy[b]), 0.16, fill=False, ec="k", lw=1.3, zorder=4))
    ax.plot([6.0, 6.6], [qy[3], qy[3]], color="k", lw=1.1)
    ax.annotate("", xy=(6.9, (qy[0] + qy[3]) / 2), xytext=(6.6, (qy[0] + qy[3]) / 2),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(6.75, qy[3] - 0.5, "CNOT ring", ha="center", fontsize=8, color="0.3")

    # dashed box = one step, repeats x8, frozen
    ax.add_patch(Rectangle((1.9, qy[3] - 0.55), 4.5, qy[0] - qy[3] + 1.1, fill=False,
                           ls="--", ec=GREY, lw=1.6, zorder=2))
    ax.text(4.15, qy[0] + 0.85, "one time step  —  repeat $\\times 8$   (FROZEN $\\theta^*$: quantum memory)",
            ha="center", fontsize=10, color=GREY, fontweight="bold")

    # measurement
    for y in qy:
        gate(ax, 7.5, y, "$\\langle Z\\rangle$", "0.35", w=0.6)
    ax.plot([6.6, 7.2], [(qy[0] + qy[3]) / 2, (qy[0] + qy[3]) / 2], alpha=0)

    # tanh head -> next value
    ax.add_patch(FancyBboxPatch((8.4, 4.7), 1.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=1.5, edgecolor="k", facecolor="#EEDD88", zorder=4))
    ax.text(9.35, 5.15, r"$\tanh(\mathbf{w}\!\cdot\!\langle Z\rangle+b)$", ha="center", va="center",
            fontsize=9.5, zorder=5)
    ax.add_patch(FancyBboxPatch((10.9, 4.75), 1.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=1.6, edgecolor=RED, facecolor="#FBEEF3", zorder=4))
    ax.text(11.6, 5.15, r"$\hat{s}_{t}$", ha="center", va="center", fontsize=13,
            color=RED, fontweight="bold", zorder=5)
    for xa, xb in [(7.8, 8.4), (10.3, 10.9)]:
        ax.annotate("", xy=(xb, 5.15), xytext=(xa, 5.15), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))

    # --- AUTOREGRESSIVE FEEDBACK LOOP (the QGR mechanism) ---
    ax.add_patch(FancyArrowPatch((11.6, 4.75), (0.7, 2.4), connectionstyle="arc3,rad=0.32",
                                 arrowstyle="-|>", mutation_scale=24, lw=2.6, color=RED, zorder=2))
    ax.text(6.0, 2.05, "autoregressive rollout:  append $\\hat{s}_t$ to the window  →  next input  "
                       "$\\;\\hat{s}_{t}=f_{\\theta^*}(\\hat{s}_{t-8:t-1})$",
            ha="center", fontsize=10.5, color=RED, fontweight="bold")
    ax.text(0.7, 3.05, "seed →\nwindow", ha="center", va="center", fontsize=8.5, color=RED)

    # --- rehearse into the learning model ---
    ax.add_patch(FancyBboxPatch((3.2, 0.35), 6.6, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=1.6, edgecolor="#333", facecolor="#EAF2FB", zorder=3))
    ax.text(6.5, 0.85, r"generated old-task sequence  →  rehearse:  "
                       r"$\mathcal{L}=\mathrm{MSE}(\mathrm{new})+\mathrm{MSE}(\mathrm{generated})$  "
                       r"(updates the LEARNING model, not the frozen generator)",
            ha="center", va="center", fontsize=9.3)
    ax.annotate("", xy=(6.5, 1.35), xytext=(6.5, 1.9), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.6))

    fig.suptitle("QGR — the frozen quantum forecaster as an autoregressive generator of old tasks",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
