"""Draw the four-layer theoretical architecture of the e009 + QGR system (slide schematic)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "e009_architecture.png"

LAYERS = [
    ("① Problem  —  sequential empirical risk minimization", "#EAF2FB", "#4477AA",
     [r"Tasks arrive in order  $\mathcal{T}_1 \to \mathcal{T}_2 \to \mathcal{T}_3$;  when learning $\mathcal{T}_k$ only $\mathcal{D}_k$ is visible.",
      r"Forgetting  $F_j^{(k)} = \mathcal{R}_j(\theta^{(k)}) - \mathcal{R}_j(\theta^{(j)}) \geq 0$   (root cause: gradient conflict $\langle\nabla\mathcal{R}_k,\nabla\mathcal{R}_j\rangle<0$).",
      r"Goal = stability (retain old) vs plasticity (learn new)."]),
    ("② Model  —  quantum RNN (recurrent data re-uploading)", "#EAF7EE", "#228833",
     [r"$|\psi_t\rangle = U_{\mathrm{var}}(\vartheta)\,U_{\mathrm{enc}}(x_t)\,|\psi_{t-1}\rangle$   —  state persists across steps (a quantum hidden state).",
      r"4 qubits, shared block $\times$8 steps + CNOT ring, $\langle Z\rangle\!\to\!\tanh$ head.  21 params.",
      r"Re-uploading $\Rightarrow$ Fourier features  (well-matched to oscillatory series)."]),
    ("③ Method  —  every CL method = $\min_\theta\ \mathcal{R}_k(\theta) + \Omega(\theta)$", "#FFF6E6", "#CC8800",
     [r"$\Omega$ approximates the old-task constraint $\mathcal{R}_{<k}$:",
      r"param-space:  L2 / EWC (classical Fisher) / QEWC (quantum Fisher)",
      r"function-space:  replay (real data)  ·  QGR (generated data)"]),
    ("④ QGR  —  quantum generative replay", "#FBEEF3", "#CC3311",
     [r"Freeze the quantum forecaster as a generator $G_j\equiv f_{\theta^{(j)}}$;  roll out old sequences:",
      r"$\hat{s}_{t}=f_{\theta^{(j)}}(\hat{s}_{t-L:t-1})$   (forward orbit of a frozen quantum dynamical system).",
      r"$\mathcal{L}_{\mathrm{QGR}}=\hat{\mathcal{R}}_k(\theta;\mathcal{D}_k)+\sum_{j<k}\hat{\mathcal{R}}_j(\theta;\tilde{\mathcal{D}}_j)$   —  memory = quantum circuit, not raw data.",
      r"error = sampling variance $\sim 1/\sqrt{L_{\mathrm{gen}}}$  +  generation bias (fidelity ceiling)."]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    h, gap, top = 1.95, 0.55, 9.4
    centers = []
    for i, (title, face, edge, lines) in enumerate(LAYERS):
        y = top - i * (h + gap) - h
        box = FancyBboxPatch((0.3, y), 9.4, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=2, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(box)
        ax.text(0.55, y + h - 0.33, title, fontsize=13, fontweight="bold", color=edge, zorder=3)
        for j, ln in enumerate(lines):
            ax.text(0.65, y + h - 0.72 - j * 0.38, ln, fontsize=10.3, color="#222", zorder=3)
        centers.append((5.0, y, y + h))

    for i in range(len(LAYERS) - 1):
        _, ylow, _ = centers[i]
        _, _, yhigh_next = centers[i + 1]
        ax.add_patch(FancyArrowPatch((5.0, ylow), (5.0, yhigh_next), arrowstyle="-|>",
                                     mutation_scale=22, linewidth=2.2, color="0.4", zorder=1))

    fig.suptitle("e009 + QGR — four-layer theoretical architecture", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
