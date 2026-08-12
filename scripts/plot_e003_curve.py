"""Plot E003 train/test accuracy by task to diagnose learning and forgetting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e003_continual_baseline_reference.json"
OUT = ROOT / "figures" / "e003_continual_baseline.png"


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    history = data["training"]["history"]
    epochs = [row["epoch"] for row in history]
    boundary = data["training"]["task_boundary_epoch"]

    series = (
        (
            "MNIST 0/1",
            [row["mnist_train_accuracy"] for row in history],
            [row["mnist_test_accuracy"] for row in history],
        ),
        (
            "Fashion-MNIST 0/1",
            [row["fashion_train_accuracy"] for row in history],
            [row["fashion_test_accuracy"] for row in history],
        ),
    )

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for axis, (title, train, test) in zip(axes, series, strict=True):
        axis.plot(epochs, train, color="tab:blue", lw=1.8, label="Train")
        axis.plot(epochs, test, "--", color="tab:orange", lw=1.6, label="Test")
        axis.axvline(boundary + 0.5, ls="--", color="black", lw=1.0, alpha=0.6)
        axis.set_ylim(0.1, 1.03)
        axis.set_ylabel("Accuracy")
        axis.grid(True, alpha=0.25)
        axis.text(0.02, 0.88, title, transform=axis.transAxes, fontweight="bold")

    axes[0].legend(loc="lower right", fontsize=10, frameon=True)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Sequential baseline: learning and forgetting", fontsize=12)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
