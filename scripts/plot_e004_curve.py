"""Plot three E004 seeded runs and their mean training accuracy with uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 43, 44)
TASK_KEYS = ("task1", "task2", "task3")
COLORS = ("#4477AA", "#EE6677", "#228833")


def _decorate(
    axes: np.ndarray,
    task_names: list[str],
    boundaries: list[int],
    total_epochs: int,
) -> None:
    for index, (ax, name) in enumerate(zip(axes, task_names, strict=True)):
        for boundary in boundaries:
            ax.axvline(boundary, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.0, 1.03)
        ax.set_ylabel(f"T{index + 1}\naccuracy")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Epoch")


def plot_seed(result: dict[str, Any], output: Path) -> None:
    history = result["training"]["history"]
    epochs = np.asarray([row["epoch"] for row in history])
    boundaries = result["training"]["task_boundaries"]
    task_starts = [1, *(boundary + 1 for boundary in boundaries)]
    task_metrics = result["metrics"]["tasks"]
    task_names = [task_metrics[key]["name"] for key in TASK_KEYS]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, key, color, start in zip(
        axes, TASK_KEYS, COLORS, task_starts, strict=True
    ):
        visible = epochs >= start
        accuracy = np.asarray([row["train_accuracy"][key] for row in history])
        ax.plot(epochs[visible], accuracy[visible], color=color, linewidth=2.0)
    _decorate(axes, task_names, boundaries, int(epochs[-1]))
    seed = result["training"]["seed"]
    fig.suptitle(f"E004 sequential baseline — seed {seed} (training accuracy)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean(summary: dict[str, Any], output: Path) -> None:
    history = summary["aggregate_history"]
    epochs = np.asarray([row["epoch"] for row in history])
    boundaries = summary["configuration"]["task_boundaries"]
    task_starts = [1, *(boundary + 1 for boundary in boundaries)]
    task_metrics = summary["aggregate_metrics"]
    task_names = [task_metrics[key]["name"] for key in TASK_KEYS]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, key, color, start in zip(
        axes, TASK_KEYS, COLORS, task_starts, strict=True
    ):
        mean = np.asarray([row["train_accuracy"][key]["mean"] for row in history])
        std = np.asarray(
            [row["train_accuracy"][key]["sample_std"] for row in history]
        )
        visible = epochs >= start
        ax.plot(
            epochs[visible],
            mean[visible],
            color=color,
            linewidth=2.0,
            label="3-seed mean",
        )
        ax.fill_between(
            epochs[visible],
            np.clip(mean[visible] - std[visible], 0.0, 1.0),
            np.clip(mean[visible] + std[visible], 0.0, 1.0),
            color=color,
            alpha=0.22,
            label="±1 sample SD",
        )
    _decorate(axes, task_names, boundaries, int(epochs[-1]))
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("E004 sequential baseline — seeds 42/43/44 (training accuracy)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "e004_continual_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    for seed in args.seeds:
        path = ROOT / "results" / f"e004_continual_seed{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        plot_seed(result, args.output_dir / f"e004_continual_seed{seed}.png")
        print(f"Wrote {args.output_dir / f'e004_continual_seed{seed}.png'}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot_mean(summary, args.output_dir / "e004_continual_mean.png")
    print(f"Wrote {args.output_dir / 'e004_continual_mean.png'}")


if __name__ == "__main__":
    main()
