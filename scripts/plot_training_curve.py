"""Render the E001 optimizer-evaluation accuracy curve from a result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "results" / "e001_qnn_mnist_reference.json"
DEFAULT_OUTPUT = ROOT / "figures" / "e001_training_curve.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Keep CLI help usable even when an optional local plotting stack is unavailable.
    import matplotlib.pyplot as plt

    data = json.loads(args.result.read_text(encoding="utf-8"))
    history = data["training"]["history"]
    evaluations = [point["evaluation"] for point in history]
    train = [point["train_accuracy"] for point in history]
    test = [point["test_accuracy"] for point in history]

    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        evaluations,
        train,
        "-o",
        color="tab:blue",
        markersize=5,
        markerfacecolor="none",
        label="QNN training",
    )
    ax.plot(
        evaluations,
        test,
        "-^",
        color="tab:orange",
        markersize=5,
        markerfacecolor="none",
        label="QNN testing",
    )

    baseline = data["metrics"].get("logreg_test_accuracy")
    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            color="gray",
            linewidth=1.3,
            label=f"Classical baseline ({baseline:.2f})",
        )

    ax.set_xlabel("Objective evaluation (COBYLA)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(left=0)
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
