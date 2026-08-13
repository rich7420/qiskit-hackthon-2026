"""Plot a scope-aware E004–E008 stability/plasticity overview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "naive": "#6b7280",
    "output_fisher": "#7c3aed",
    "measurement_fisher": "#2563eb",
    "measurement_optimized": "#ef4444",
    "quantum_fisher": "#15803d",
    "l2": "#f59e0b",
    "replay": "#0891b2",
    "adaptive": "#a16207",
    "trust_region": "#be123c",
}
FAMILY_LABELS = {
    "naive": "Naive / sequential",
    "output_fisher": "Output CFI",
    "measurement_fisher": "Measurement CFI",
    "measurement_optimized": "Measurement optimized",
    "quantum_fisher": "QFI consolidation",
    "l2": "L2 anchoring",
    "replay": "Replay",
    "adaptive": "Adaptive control",
    "trust_region": "Hard trust region",
}


def plot_comparison(data: dict, output: Path) -> None:
    points = data["points"]
    experiments = data["experiments"]
    y = np.arange(len(points))[::-1]
    figure, axes = plt.subplots(1, 3, figsize=(17, 10.8), sharey=True)
    metrics = (
        ("old_task_retention", "Old-task retention ↑", (0.38, 1.04)),
        ("new_task_plasticity", "Final-task plasticity ↑", (0.08, 1.04)),
        ("average_forgetting", "Average forgetting ↓", (-0.08, 0.55)),
    )

    group_ranges: dict[str, tuple[int, int]] = {}
    for experiment in experiments:
        indices = [index for index, point in enumerate(points) if point["experiment"] == experiment]
        group_ranges[experiment] = (min(indices), max(indices))

    for axis, (metric, title, limits) in zip(axes, metrics, strict=True):
        for group_index, (experiment, (start, end)) in enumerate(group_ranges.items()):
            if group_index % 2 == 0:
                upper = len(points) - start - 0.5
                lower = len(points) - end - 1.5
                axis.axhspan(lower, upper, color="#f3f4f6", zorder=0)
            if experiment == "E007":
                upper = len(points) - start - 0.5
                lower = len(points) - end - 1.5
                axis.axhspan(lower, upper, color="#fff7ed", alpha=0.65, zorder=0)

        for row, point in zip(y, points, strict=True):
            estimate = point[metric]
            family = point["family"]
            marker = "D" if family == "trust_region" else "o"
            verified = experiments[point["experiment"]].get("artifact_verified", True)
            axis.errorbar(
                estimate["mean"],
                row,
                xerr=estimate["sample_std"],
                fmt=marker,
                color=COLORS[family],
                markerfacecolor=COLORS[family] if verified else "white",
                markeredgewidth=1.8 if not verified else 1.0,
                markersize=7,
                capsize=3,
                linewidth=1.5,
                zorder=3,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(*limits)
        axis.grid(axis="x", alpha=0.22)
        axis.axvline(0.5 if metric != "average_forgetting" else 0.0, color="#9ca3af", lw=0.8)

    labels = []
    for point in points:
        experiment = experiments[point["experiment"]]
        suffix = "†" if experiment["test_tuned"] else ""
        if not experiment.get("artifact_verified", True):
            suffix += "‡"
        labels.append(f"{point['experiment']} · {point['method']}{suffix}")
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=9)

    for axis in axes:
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)

    used_families = dict.fromkeys(point["family"] for point in points)
    handles = [
        mlines.Line2D(
            [],
            [],
            color=COLORS[family],
            marker="D" if family == "trust_region" else "o",
            linestyle="None",
            markersize=7,
            label=FAMILY_LABELS[family],
        )
        for family in used_families
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    figure.suptitle(
        "E004–E008 continual-learning overview — compare within experiment blocks",
        fontsize=16,
        y=0.985,
    )
    figure.text(
        0.5,
        0.078,
        "Mean ± sample SD. E006 uses balanced accuracy; other blocks use accuracy. "
        "† E007 is exploratory/test-selected. ‡ E005 artifacts do not match final PR source. "
        "Task sequences and models differ across blocks.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    figure.tight_layout(rect=(0.10, 0.135, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "e004_e008_comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "e004_e008_comparison.png",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    plot_comparison(data, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
