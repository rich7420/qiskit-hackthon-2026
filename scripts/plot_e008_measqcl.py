"""Create the E008 MeasQCL continual-learning and measurement diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = (
    "naive",
    "output_cfi",
    "zz_cfi",
    "uniform_xyz",
    "mof_ewc",
    "readout_qewc",
    "qewc",
)
LABELS = {
    "naive": "Naive",
    "output_cfi": "Output CFI",
    "zz_cfi": "Joint ZZ CFI",
    "uniform_xyz": "Uniform XYZ",
    "mof_ewc": "MOF-EWC",
    "readout_qewc": "Readout QFI",
    "qewc": "Full QFI",
}
COLORS = {
    "naive": "#6b7280",
    "output_cfi": "#8b5cf6",
    "zz_cfi": "#2563eb",
    "uniform_xyz": "#06b6d4",
    "mof_ewc": "#ef4444",
    "readout_qewc": "#f59e0b",
    "qewc": "#15803d",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_phase2_curves(summary: dict, output: Path) -> None:
    """Plot only Task-2 training, where methods diverge and forgetting is observable."""
    _style()
    epochs_per_task = summary["configuration"]["training"]["epochs_per_task"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    panels = (("task1", "MNIST retention"), ("task2", "Fashion-MNIST adaptation"))

    for axis, (task, title) in zip(axes, panels, strict=True):
        for method in METHODS:
            rows = summary["aggregate_history"][method]
            visible = [row for row in rows if row["epoch"] >= epochs_per_task]
            task2_epoch = np.asarray(
                [row["epoch"] - epochs_per_task for row in visible], dtype=float
            )
            mean = np.asarray(
                [row["test_accuracy"][task]["mean"] for row in visible], dtype=float
            )
            std = np.asarray(
                [row["test_accuracy"][task]["sample_std"] for row in visible],
                dtype=float,
            )
            axis.plot(
                task2_epoch,
                mean,
                color=COLORS[method],
                linewidth=2.0,
                label=LABELS[method],
            )
            axis.fill_between(
                task2_epoch,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=COLORS[method],
                alpha=0.09,
                linewidth=0,
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Fashion-MNIST training epoch")
        axis.set_ylim(0.35, 1.01)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Held-out test accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )
    figure.suptitle(
        "E008 MeasQCL — memory and adaptation during Task 2 (mean ± sample SD, n=3)"
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_stability_plasticity(summary: dict, output: Path) -> None:
    """Show paired seeds and mean uncertainty on the final test frontier."""
    _style()
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    paired = summary["paired_seed_metrics"]
    for method in METHODS:
        seed_rows = [
            row
            for row in paired
            if row["split"] == "test" and row["method"] == method
        ]
        x_seed = np.asarray([row["adaptation"] for row in seed_rows])
        y_seed = np.asarray([row["retention"] for row in seed_rows])
        aggregate = summary["aggregate_metrics"][method]["test"]
        x = aggregate["task2_final_adaptation"]["mean"]
        y = aggregate["task1_final_retention"]["mean"]
        xerr = aggregate["task2_final_adaptation"]["sample_std"]
        yerr = aggregate["task1_final_retention"]["sample_std"]
        axis.scatter(x_seed, y_seed, color=COLORS[method], s=18, alpha=0.24)
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=8,
            capsize=3,
            color=COLORS[method],
            label=LABELS[method],
            zorder=3,
        )
    axis.set_xlabel("Fashion-MNIST final test accuracy (plasticity)")
    axis.set_ylabel("MNIST final test accuracy (retention)")
    axis.set_xlim(0.89, 1.002)
    axis.set_ylim(0.43, 0.94)
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left", ncol=2, frameon=True)
    axis.set_title(
        "Stability–plasticity frontier\nlarge markers: mean ± sample SD; faint markers: seeds 42/43/44",
        loc="left",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _normalized(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array * array.size / array.sum()


def plot_measurement_geometry(summary: dict, seed_run: dict, output: Path) -> None:
    """Connect measurement choice, Fisher geometry, retention, and exact cost."""
    _style()
    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.42, wspace=0.30)

    axis_heat = figure.add_subplot(grid[0, :])
    raw_basis = seed_run["fisher_profiles"]["raw_basis_fisher"]
    raw_method = seed_run["fisher_profiles"]["raw_method_fisher"]
    fisher_rows = [
        _normalized(raw_method["output_cfi"]),
        _normalized(raw_basis["ZZ"]),
        _normalized(raw_basis["XX"]),
        _normalized(raw_basis["YY"]),
        _normalized(raw_method["mof_ewc"]),
        _normalized(raw_method["readout_qewc"]),
        _normalized(raw_method["qewc"]),
    ]
    heat = np.log10(np.asarray(fisher_rows) + 1e-3)
    image = axis_heat.imshow(heat, aspect="auto", cmap="magma", vmin=-3, vmax=1.2)
    axis_heat.set_yticks(range(len(fisher_rows)))
    axis_heat.set_yticklabels(
        ["Output CFI", "ZZ", "XX", "YY", "MOF mixture", "Readout QFI", "Full QFI"]
    )
    axis_heat.set_xlabel("Trainable parameter (10 layers × 6 qubits × RY/RZ)")
    axis_heat.set_title(
        "A. Different measurements protect different parameter directions (seed 42)",
        loc="left",
    )
    for boundary in range(12, heat.shape[1], 12):
        axis_heat.axvline(boundary - 0.5, color="white", alpha=0.18, linewidth=0.6)
    colorbar = figure.colorbar(image, ax=axis_heat, fraction=0.018, pad=0.015)
    colorbar.set_label("log10(mean-normalized importance + 1e-3)")

    axis_allocation = figure.add_subplot(grid[1, 0])
    run_paths = [ROOT / path for path in summary["result_files"]]
    runs = [_load_json(path) for path in run_paths]
    bases = ("ZZ", "XX", "YY")
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    basis_colors = {"ZZ": "#2563eb", "XX": "#ef4444", "YY": "#8b5cf6"}
    for basis in bases:
        values = np.asarray(
            [run["fisher_profiles"]["allocations"]["mof_ewc"][basis] for run in runs]
        )
        axis_allocation.bar(
            x,
            values,
            bottom=bottom,
            color=basis_colors[basis],
            label=basis,
        )
        bottom += values
    axis_allocation.set_xticks(x, [f"seed {seed}" for seed in summary["seeds"]])
    axis_allocation.set_ylim(0, 1)
    axis_allocation.set_ylabel("Optimized measurement probability / shot fraction")
    axis_allocation.legend(ncol=3, frameon=False)
    axis_allocation.set_title("B. Exact diagonal D-optimal-like allocation", loc="left")
    axis_allocation.grid(axis="y", alpha=0.2)

    axis_geometry = figure.add_subplot(grid[1, 1])
    methods = ("zz_cfi", "uniform_xyz", "mof_ewc")
    x = np.arange(len(methods))
    width = 0.25
    fields = (
        ("cosine_to_full_qfi", "cosine to full QFI", "#15803d"),
        ("cosine_to_output_cfi", "cosine to output CFI", "#8b5cf6"),
        ("readout_qfi_trace_coverage_proxy", "readout-QFI trace proxy", "#f59e0b"),
    )
    for offset, (field, label, color) in enumerate(fields):
        means = [summary["geometry"][method][field]["mean"] for method in methods]
        stds = [summary["geometry"][method][field]["sample_std"] for method in methods]
        axis_geometry.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=color,
            alpha=0.88,
            label=label,
        )
    axis_geometry.set_xticks(x, [LABELS[method] for method in methods])
    axis_geometry.set_ylim(0, 1.02)
    axis_geometry.set_ylabel("Geometry diagnostic")
    axis_geometry.legend(fontsize=8, frameon=False)
    axis_geometry.set_title("C. More accessible geometry is not necessarily task memory", loc="left")
    axis_geometry.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "E008 measurement physics — exact probabilities, task-boundary measurements only",
        y=0.985,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "e008_measqcl_summary.json",
    )
    parser.add_argument(
        "--representative-seed",
        type=Path,
        default=ROOT / "results" / "e008_measqcl_seed42.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    summary = _load_json(args.summary)
    seed_run = _load_json(args.representative_seed)
    outputs = (
        args.output_dir / "e008_measqcl_curves.png",
        args.output_dir / "e008_measqcl_stability_plasticity.png",
        args.output_dir / "e008_measqcl_measurement_geometry.png",
    )
    plot_phase2_curves(summary, outputs[0])
    plot_stability_plasticity(summary, outputs[1])
    plot_measurement_geometry(summary, seed_run, outputs[2])
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
