"""Plot E006 learning curves and the paired stability-plasticity comparison."""

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
METHOD_STYLE = {
    "baseline": {"color": "#666666", "linestyle": "--", "label": "Baseline (E004 protocol)"},
    "replay": {"color": "#228833", "linestyle": "-", "label": "Advanced (balanced replay)"},
}


def plot_curves(summary: dict[str, Any], output: Path) -> None:
    boundaries = summary["configuration"]["task_boundaries"]
    starts = [1, *(boundary + 1 for boundary in boundaries)]
    task_names = summary["configuration"]["dataset"]["task_order"]
    histories = summary["aggregate_history"]
    total_epochs = histories["baseline"][-1]["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.8), sharex=True)
    for task_index, (axis, task_name, start) in enumerate(
        zip(axes, task_names, starts, strict=True), start=1
    ):
        key = f"task{task_index}"
        for method, style in METHOD_STYLE.items():
            history = histories[method]
            epochs = np.asarray([row["epoch"] for row in history])
            mean = np.asarray(
                [row["train_balanced_accuracy"][key]["mean"] for row in history]
            )
            std = np.asarray(
                [
                    row["train_balanced_accuracy"][key]["sample_std"]
                    for row in history
                ]
            )
            visible = epochs >= start
            axis.plot(
                epochs[visible],
                mean[visible],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                label=style["label"],
            )
            axis.fill_between(
                epochs[visible],
                np.clip(mean[visible] - std[visible], 0.0, 1.0),
                np.clip(mean[visible] + std[visible], 0.0, 1.0),
                color=style["color"],
                alpha=0.14,
            )
        for boundary in boundaries:
            axis.axvline(boundary, color="0.45", linestyle=":", linewidth=1.2)
        axis.set_xlim(0, total_epochs)
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel(f"T{task_index}\nbalanced acc.")
        axis.set_title(task_name, loc="left", fontsize=10)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[-1].set_xlabel("Epoch")
    seeds = "/".join(str(seed) for seed in summary["seeds"])
    fig.suptitle(
        "E006 Advanced temporal learning vs E004-style baseline\n"
        f"training balanced accuracy, mean ± sample SD across seeds {seeds}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: dict[str, Any], output: Path) -> None:
    paired = summary["paired_seed_metrics"]
    baseline_x = np.asarray(
        [row["baseline_old_task_balanced_retention"] for row in paired]
    )
    baseline_y = np.asarray(
        [row["baseline_new_task_balanced_adaptation"] for row in paired]
    )
    replay_x = np.asarray(
        [row["replay_old_task_balanced_retention"] for row in paired]
    )
    replay_y = np.asarray(
        [row["replay_new_task_balanced_adaptation"] for row in paired]
    )

    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for index, row in enumerate(paired):
        axis.annotate(
            "",
            xy=(replay_x[index], replay_y[index]),
            xytext=(baseline_x[index], baseline_y[index]),
            arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 1.3},
        )
        axis.text(
            replay_x[index] + 0.008,
            replay_y[index] + 0.008,
            f"seed {row['seed']}",
            fontsize=8,
            color="#176425",
        )
    axis.scatter(
        baseline_x,
        baseline_y,
        marker="o",
        s=55,
        color=METHOD_STYLE["baseline"]["color"],
        label=METHOD_STYLE["baseline"]["label"],
        zorder=3,
    )
    axis.scatter(
        replay_x,
        replay_y,
        marker="s",
        s=60,
        color=METHOD_STYLE["replay"]["color"],
        label=METHOD_STYLE["replay"]["label"],
        zorder=3,
    )
    for values_x, values_y, color, marker in (
        (baseline_x, baseline_y, METHOD_STYLE["baseline"]["color"], "o"),
        (replay_x, replay_y, METHOD_STYLE["replay"]["color"], "s"),
    ):
        axis.errorbar(
            np.mean(values_x),
            np.mean(values_y),
            xerr=np.std(values_x, ddof=1),
            yerr=np.std(values_y, ddof=1),
            fmt=marker,
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.3,
            color=color,
            capsize=4,
            zorder=4,
        )

    all_x = np.concatenate([baseline_x, replay_x])
    all_y = np.concatenate([baseline_y, replay_y])
    x_margin = max(0.05, 0.2 * np.ptp(all_x))
    y_margin = max(0.05, 0.2 * np.ptp(all_y))
    axis.set_xlim(max(0.0, all_x.min() - x_margin), min(1.02, all_x.max() + x_margin))
    axis.set_ylim(max(0.0, all_y.min() - y_margin), min(1.02, all_y.max() + y_margin))
    axis.set_xlabel("Final old-task retention (mean T1/T2 train balanced accuracy) →")
    axis.set_ylabel("New-task adaptation (T3 final train balanced accuracy) →")
    axis.set_title("Stability–plasticity trade-off (paired seeds)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    axis.text(
        0.02,
        0.02,
        "Large markers: 3-seed mean ± sample SD\nArrows: baseline → replay for the same seed",
        transform=axis.transAxes,
        fontsize=8,
        color="0.35",
        va="bottom",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "e006_advanced_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    curves = args.output_dir / "e006_advanced_vs_e004_baseline.png"
    tradeoff = args.output_dir / "e006_advanced_tradeoff.png"
    plot_curves(summary, curves)
    plot_tradeoff(summary, tradeoff)
    print(f"Wrote {curves}")
    print(f"Wrote {tradeoff}")


if __name__ == "__main__":
    main()
