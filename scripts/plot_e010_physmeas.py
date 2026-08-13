"""Generate the checked E010 task-relevance, shot-cost, and locality figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "naive": "#9CA3AF",
    "output_cfi": "#D97706",
    "zz_cfi": "#2563EB",
    "mof_ewc": "#7C3AED",
    "ewc_dr": "#EA580C",
    "task_relevant_mof": "#059669",
    "qewc": "#111827",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def plot_main() -> Path:
    exact = _load("e010_physmeas_summary.json")
    phase = _load("e010_phase_locality_summary.json")
    parent = _load("e008_measqcl_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    methods = (
        "naive",
        "output_cfi",
        "zz_cfi",
        "mof_ewc",
        "ewc_dr",
        "task_relevant_mof",
        "qewc",
    )
    labels = {
        "naive": "Naive",
        "output_cfi": "Output CFI",
        "zz_cfi": "Joint ZZ",
        "mof_ewc": "Task-agnostic MOF",
        "ewc_dr": "EWC-DR",
        "task_relevant_mof": "Task-relevant MOF",
        "qewc": "Full QFI",
    }
    ax = axes[0]
    for method in methods:
        metric = exact["aggregate_metrics"][method]["test"]
        x = metric["task2_final_adaptation"]
        y = metric["task1_final_retention"]
        ax.errorbar(
            x["mean"],
            y["mean"],
            xerr=x["sample_std"],
            yerr=y["sample_std"],
            fmt="o",
            ms=8 if method == "task_relevant_mof" else 6,
            color=COLORS[method],
            capsize=3,
            label=labels[method],
            zorder=3,
        )
    ax.set_title("A  Stability–plasticity (MNIST → Fashion)")
    ax.set_xlabel("New-task test adaptation")
    ax.set_ylabel("Old-task test retention")
    ax.set_xlim(0.90, 0.985)
    ax.set_ylim(0.55, 0.92)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    bases = ("ZZ", "XX", "YY")
    task_agnostic = parent["mof_allocation"]
    task_relevant = exact["task_relevant_allocation"]
    x = np.arange(len(bases))
    width = 0.36
    ax.bar(
        x - width / 2,
        [task_agnostic[basis]["mean"] for basis in bases],
        width,
        yerr=[task_agnostic[basis]["sample_std"] for basis in bases],
        capsize=3,
        color="#7C3AED",
        label="Task-agnostic coverage",
    )
    ax.bar(
        x + width / 2,
        [task_relevant[basis]["mean"] for basis in bases],
        width,
        yerr=[task_relevant[basis]["sample_std"] for basis in bases],
        capsize=3,
        color="#059669",
        label="Task-relevant coverage",
    )
    ax.set_title("B  Relevance changes what is measured")
    ax.set_xticks(x, bases)
    ax.set_ylabel("Measurement allocation")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper center")

    phase_methods = (
        "naive",
        "output_cfi",
        "readout_pauli",
        "one_local",
        "two_local",
        "hamiltonian",
        "nonlocal",
        "task_relevant_all",
        "qewc",
    )
    phase_labels = (
        "Naive",
        "Output\nCFI",
        "Readout\nPauli",
        "1-local",
        "2-local",
        "Hamiltonian",
        "XYYX\nweight-4",
        "Task-rel.\nall",
        "QFI",
    )
    ax = axes[2]
    values = [
        phase["aggregate_metrics"][method]["test"]["phase_final_retention"]
        for method in phase_methods
    ]
    bar_colors = [
        "#9CA3AF",
        "#D97706",
        "#3B82F6",
        "#60A5FA",
        "#2563EB",
        "#0EA5E9",
        "#7C3AED",
        "#059669",
        "#111827",
    ]
    ax.bar(
        np.arange(len(values)),
        [value["mean"] for value in values],
        yerr=[value["sample_std"] for value in values],
        capsize=3,
        color=bar_colors,
    )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(np.arange(len(values)), phase_labels, rotation=35, ha="right")
    ax.set_ylim(0.35, 1.08)
    ax.set_ylabel("Final phase-task test retention")
    ax.set_title("C  Output locality is not monotonic")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.02,
        "mean ± sample SD, 3 paired seeds",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    fig.suptitle(
        "E010 PhysMeas-QCL — task relevance repairs allocation; QFI has highest mean phase retention",
        fontsize=15,
    )
    FIGURES.mkdir(exist_ok=True)
    output = FIGURES / "e010_physmeas_main.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_finite_shot() -> Path:
    result = _load("e010_finite_shot_summary.json")
    budgets = sorted(result["aggregate"], key=int)
    total_shots = np.asarray(
        [result["aggregate"][budget]["pilot_shots_per_seed_repetition"] for budget in budgets]
    )
    cosine = [
        result["aggregate"][budget]["selected_profile_cosine_to_exact"]["seed_mean"]
        for budget in budgets
    ]
    allocation_error = [
        result["aggregate"][budget]["allocation_l1_error_to_exact"]["seed_mean"]
        for budget in budgets
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(
        total_shots,
        [value["mean"] for value in cosine],
        yerr=[value["sample_std"] for value in cosine],
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#059669",
    )
    for x, budget, value in zip(total_shots, budgets, cosine, strict=True):
        ax.annotate(
            f"{budget} shots/circuit\n{value['mean']:.3f}",
            (x, value["mean"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_ylim(0.94, 1.005)
    ax.set_xlabel("Total pilot shots per seed/repetition (all shifts, anchors, bases)")
    ax.set_ylabel("Selected Fisher profile cosine to exact")
    ax.set_title("A  Selected Fisher profile")
    ax.grid(alpha=0.25, which="both")
    ax.text(
        0.01,
        0.03,
        "conditional on exact task relevance",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    ax = axes[1]
    ax.errorbar(
        total_shots,
        [value["mean"] for value in allocation_error],
        yerr=[value["sample_std"] for value in allocation_error],
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#7C3AED",
    )
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Total pilot shots per seed/repetition")
    ax.set_ylabel("Allocation L1 error to exact")
    ax.set_title("B  Allocation is less stable than its profile")
    ax.grid(alpha=0.25, which="both")
    fig.suptitle(
        "Conditional finite-shot basis-Fisher audit — all anchors, shifts, and bases counted",
        fontsize=14,
    )
    output = FIGURES / "e010_finite_shot.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    for path in (plot_main(), plot_finite_shot()):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
