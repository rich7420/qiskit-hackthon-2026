"""Draw the MPI circuit: (1) an authentic PennyLane render of the VQC with a probs
readout, and (2) a concept schematic showing the shared quantum representation feeding
per-task classical observable heads selected by the task id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"


def draw_authentic(n_layers: int = 2) -> None:
    """Real PennyLane circuit for a small (readable) number of layers."""
    import pennylane as qml
    from pennylane import numpy as pnp

    from src.e014_oiqcl import make_probs_qnode

    qnode, shape = make_probs_qnode(n_qubits=4, n_layers=n_layers)
    x = np.random.default_rng(0).standard_normal(16)
    w = 0.5 * np.random.default_rng(1).standard_normal(shape)
    fig, ax = qml.draw_mpl(qnode, decimals=None, style="pennylane")(
        pnp.array(x, requires_grad=False), pnp.array(w))
    ax.set_title(f"MPI VQC (n=4, L={n_layers} shown; L=12 used) — probs readout",
                 fontsize=11)
    out = FIG / "e014_circuit_pennylane.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def _box(ax, x, y, w, h, text, fc, ec="#333", fs=10, lw=1.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=4, weight=weight)


def _arrow(ax, xy1, xy2, color="#333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, zorder=2))


def draw_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    wires_y = [5.0, 4.3, 3.6, 2.9]
    x0, x1 = 1.6, 7.4
    for i, y in enumerate(wires_y):
        ax.plot([x0, x1], [y, y], color="#222", lw=1.2, zorder=1)
        ax.text(x0 - 0.15, y, r"$|0\rangle$", ha="right", va="center", fontsize=11)
        ax.text(x1 + 0.12, y, f"$q_{i}$", ha="left", va="center", fontsize=9, color="#666")

    yb, yt = wires_y[-1] - 0.35, wires_y[0] + 0.35
    h = yt - yb

    # Encoding V(x)
    _box(ax, 1.75, yb, 0.95, h, "$V(x)$\namplitude\nembedding", "#dbeafe", fs=9)
    # Shared variational block U(theta)
    _box(ax, 3.0, yb, 2.7, h, "", "#e7f6e7", ec="#59A14F", lw=2.0)
    ax.text(4.35, yt + 0.12, r"$U(\theta)$  —  shared representation", ha="center",
            va="bottom", fontsize=11, weight="bold", color="#2f6b2f")
    ax.text(4.35, (yt + yb) / 2,
            r"per layer:  $R_Y(\theta)\,R_Z(\theta)$ on each qubit"
            "\n"
            r"+ CNOT ladder    $\times\,L{=}12$",
            ha="center", va="center", fontsize=9.5, color="#333")
    # Measurement (probs)
    for y in wires_y:
        _box(ax, 5.95, y - 0.16, 0.5, 0.32, "", "#fff3cd", ec="#b8860b", fs=7)
        ax.text(6.2, y, "▮", ha="center", va="center", fontsize=8, color="#b8860b", zorder=5)
    ax.text(6.2, yb - 0.12, "measure\n(comp. basis)", ha="center", va="top", fontsize=8.5)

    # probability vector
    _arrow(ax, (7.55, 3.95), (8.25, 3.95))
    _box(ax, 8.25, 3.45, 1.15, 1.0,
         r"$p_\theta(x)$" "\n" r"$\in\mathbb{R}^{2^n}$" "\n(probs)", "#fde2c8",
         ec="#E15759", fs=9.5, weight="bold")

    # per-task heads
    ax.text(11.1, 6.05, r"per-task heads $W^{(t)}$ (classical linear observables)",
            ha="center", va="bottom", fontsize=8.8, color="#444")
    head_specs = [("$W^{(1)}$  (Task 1)", 5.30, "#c7e9c0"),
                  ("$W^{(2)}$  (Task 2)", 4.15, "#c6dbef"),
                  ("$W^{(3)}$  (Task 3)", 3.00, "#dadaeb")]
    for label, y, fc in head_specs:
        _arrow(ax, (9.45, 3.95), (10.25, y + 0.24), color="#999", lw=1.2)
        _box(ax, 10.25, y, 1.7, 0.48, label + r"  $\cdot\,p_\theta$", fc, fs=9)

    # task-id selector -> prediction (example: head t selected)
    _box(ax, 8.15, 1.25, 1.35, 0.62, "task id $t$\n(Task-IL)", "#f2f2f2", ec="#888", fs=9)
    _arrow(ax, (9.55, 1.7), (10.30, 2.95), color="#E15759", lw=1.8)
    ax.text(9.6, 2.35, "select\nhead $t$", ha="left", va="center", fontsize=8.5,
            color="#E15759")
    _arrow(ax, (11.1, 2.98), (11.1, 1.95), color="#333", lw=1.6)
    ax.text(11.1, 1.55, r"$\hat y_t=\arg\max_c\,[W^{(t)}p_\theta(x)]_c$", ha="center",
            va="center", fontsize=9.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#333", lw=1.2))

    ax.text(6.25, 0.35,
            "Shared quantum circuit is never rewritten per task; each task adds one lightweight "
            "classical head over the measured probabilities.\n"
            r"$\langle H^{(t)}\rangle=\sum_k \lambda_k^{(t)} p_k(x;\theta)=W^{(t)}p_\theta(x)$   "
            "— a diagonal observable = linear head over probs.",
            ha="center", va="center", fontsize=9.2, color="#333")
    ax.set_title("MPI — Measurement-based Parameter Isolation",
                 fontsize=13, weight="bold")

    out = FIG / "e014_circuit.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    draw_authentic()
    draw_schematic()


if __name__ == "__main__":
    main()
