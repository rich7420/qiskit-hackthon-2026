"""Dataset intro figure for MPI: three continual tasks (images + a quantum state), not curves."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"


def _first_image(name, label):
    """Return a 28x28 sample of the given openml class, or None on failure."""
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name, version=1, as_frame=False, parser="auto")
        y = d.target.astype(int)
        idx = int(np.where(y == label)[0][0])
        return d.data[idx].reshape(28, 28)
    except Exception:
        return None


def main() -> None:
    from src.phase_data import _ground_state

    img0 = _first_image("mnist_784", 0)
    img1 = _first_image("mnist_784", 1)
    fashion = _first_image("Fashion-MNIST", 0)
    spt = _ground_state(0.2)  # deep-SPT 4-qubit ground state (16 real amplitudes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    green, blue, gold = "#2e8b57", "#3a6ea5", "#c8912a"

    # T1 MNIST 0/1
    ax = axes[0]
    if img0 is not None and img1 is not None:
        pair = np.hstack([img0, np.zeros((28, 5)), img1])  # white separator (0 -> white in gray_r)
        ax.imshow(pair, cmap="gray_r")
    ax.axis("off")
    ax.set_title("T₁ · MNIST 0/1", fontsize=21, weight="bold", color=green, pad=14)
    ax.text(0.5, -0.09, "handwritten digits  ·  classical image", transform=ax.transAxes,
            ha="center", fontsize=14, color="#333")

    # T2 Fashion 0/1
    ax = axes[1]
    if fashion is not None:
        ax.imshow(fashion, cmap="gray_r")
    ax.axis("off")
    ax.set_title("T₂ · Fashion 0/1", fontsize=21, weight="bold", color=blue, pad=14)
    ax.text(0.5, -0.09, "clothing images  ·  domain shift from T₁", transform=ax.transAxes,
            ha="center", fontsize=14, color="#333")

    # T3 SPT/ATF quantum phase — the 16 amplitudes of a 4-qubit ground state
    ax = axes[2]
    ax.bar(range(len(spt)), spt, color=gold, width=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("T₃ · SPT phases", fontsize=21, weight="bold", color=gold, pad=14)
    ax.text(0.5, -0.09, "cluster-Ising ground state  ·  quantum-native (4-qubit)",
            transform=ax.transAxes, ha="center", fontsize=14, color="#333")

    fig.suptitle("Three continual-learning tasks (learned in sequence)\n"
                 + r"$\it{all\ encoded\ into\ a\ 4\!-\!qubit\ quantum\ state\ (16\ amplitudes)}$",
                 fontsize=23, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_datasets.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}  (images ok: mnist={img0 is not None}, fashion={fashion is not None})")


if __name__ == "__main__":
    main()
