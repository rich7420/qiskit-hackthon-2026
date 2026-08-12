"""Plot E003 training-set accuracy for both tasks across sequential training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e003_continual_baseline_reference.json"
OUT = ROOT / "figures" / "e003_training_curve.png"


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    history = data["training"]["history"]
    epochs = [row["epoch"] for row in history]
    mnist = [row["mnist_train_accuracy"] for row in history]
    fashion = [row["fashion_train_accuracy"] for row in history]
    boundary = data["training"]["task_boundary_epoch"]

    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        epochs,
        mnist,
        "-o",
        color="tab:blue",
        markersize=4,
        markerfacecolor="none",
        markevery=3,
        label="MNIST train",
    )
    ax.plot(
        epochs,
        fashion,
        "-^",
        color="tab:orange",
        markersize=4,
        markerfacecolor="none",
        markevery=3,
        label="Fashion-MNIST train",
    )
    ax.axvline(boundary + 0.5, ls="--", color="black", lw=1.1, alpha=0.65)
    ax.text(boundary / 2, 0.17, "Train MNIST", ha="center", fontsize=10.5)
    ax.text(boundary * 1.5, 0.17, "Train Fashion-MNIST", ha="center", fontsize=10.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")
    ax.set_ylim(0.15, 1.03)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=9.5,
        frameon=True,
    )
    ax.set_title("Sequential training: MNIST then Fashion-MNIST")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
