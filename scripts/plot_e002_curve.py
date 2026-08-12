"""Plot the E002 train/validation accuracy curves.

The test set is intentionally absent because it is evaluated once after training.

Run:
    python scripts/plot_e002_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e002_amplitude_qnn_reference.json"
OUT = ROOT / "figures" / "e002_training_curve.png"


def main() -> None:
    data = json.loads(RESULT.read_text())
    hist = data["training"]["history"]
    epochs = [h["epoch"] for h in hist]
    train = [h["train_accuracy"] for h in hist]
    validation = [h["validation_accuracy"] for h in hist]

    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, train, "-o", color="tab:blue", markersize=4,
            markerfacecolor="none", markevery=3, label="QNN training")
    ax.plot(epochs, validation, "-^", color="tab:orange", markersize=4,
            markerfacecolor="none", markevery=3, label="QNN validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(left=0)
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
