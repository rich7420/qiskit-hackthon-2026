"""Plot E013 learnable full-output measurements on the phase-first sequence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/e013_phase_learnable_summary.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(summary: dict, method: str, metric: str) -> tuple[float, float]:
    value = summary["aggregate_metrics"][method]["test"][metric]
    return value["mean"], value["sample_std"]


def _future_adaptation(summary: dict, method: str) -> tuple[float, float]:
    values = np.asarray(
        [
            0.5
            * (
                seed_metrics[method]["mnist_final_adaptation"]
                + seed_metrics[method]["fashion_final_adaptation"]
            )
            for seed_metrics in summary["paired_seed_metrics"].values()
        ]
    )
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    colors = {
        "joint_zzzz": "#4C78A8",
        "uniform_xyz_joint": "#72B7B2",
        "info_learn_basis_alloc": "#F28E2B",
        "task_learn_basis_uniform": "#E15759",
        "task_learn_basis_alloc": "#B07AA1",
        "qewc": "#333333",
    }
    labels = {
        "joint_zzzz": "Joint ZZZZ",
        "uniform_xyz_joint": "Uniform joint XYZ",
        "info_learn_basis_alloc": "LearnBasis info+alloc",
        "task_learn_basis_uniform": "LearnBasis task",
        "task_learn_basis_alloc": "LearnBasis task+alloc",
        "qewc": "Full-state QEWC",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    ax = axes[0]
    seeds = summary["seeds"]
    x = np.arange(len(seeds))
    for method, marker in (
        ("joint_zzzz", "o"),
        ("uniform_xyz_joint", "s"),
        ("info_learn_basis_alloc", "^"),
        ("task_learn_basis_alloc", "P"),
        ("qewc", "*"),
    ):
        values = [
            summary["paired_seed_metrics"][str(seed)][method]["phase_final_retention"]
            for seed in seeds
        ]
        ax.plot(
            x,
            values,
            marker=marker,
            color=colors[method],
            linewidth=2,
            markersize=8,
            label=labels[method],
        )
    ax.set_xticks(x, [f"seed {seed}" for seed in seeds])
    ax.set_ylim(0.68, 1.02)
    ax.set_ylabel("Final phase-task retention")
    ax.set_title("A  Paired phase retention by seed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    comparison = (
        ("one_local", "1-local E010", "#9ECAE9", "o"),
        ("two_local", "2-local E010", "#59A14F", "s"),
        ("nonlocal", "Nonlocal E010", "#EDC948", "D"),
        ("joint_zzzz", labels["joint_zzzz"], colors["joint_zzzz"], "o"),
        ("uniform_xyz_joint", labels["uniform_xyz_joint"], colors["uniform_xyz_joint"], "s"),
        (
            "info_learn_basis_alloc",
            labels["info_learn_basis_alloc"],
            colors["info_learn_basis_alloc"],
            "^",
        ),
        (
            "task_learn_basis_alloc",
            labels["task_learn_basis_alloc"],
            colors["task_learn_basis_alloc"],
            "P",
        ),
        ("qewc", labels["qewc"], colors["qewc"], "*"),
    )
    for method, label, color, marker in comparison:
        retention, retention_sd = _metric(summary, method, "phase_final_retention")
        adaptation, adaptation_sd = _future_adaptation(summary, method)
        ax.errorbar(
            adaptation,
            retention,
            xerr=adaptation_sd,
            yerr=retention_sd,
            fmt=marker,
            color=color,
            markersize=9,
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Mean final MNIST/Fashion adaptation")
    ax.set_ylabel("Final phase-task retention")
    ax.set_title("B  Stability–plasticity comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[2]
    learned = (
        ("Info + alloc", "info_learn_basis_alloc"),
        ("Task", "task_learn_basis_uniform"),
        ("Task + alloc", "task_learn_basis_alloc"),
    )
    x = np.arange(len(learned))
    bottom = np.zeros(len(learned))
    pauli_colors = {"X": "#4C78A8", "Y": "#F28E2B", "Z": "#59A14F"}
    for pauli in ("X", "Y", "Z"):
        values = np.asarray(
            [
                summary["measurement_geometry"][method]["orientation_power"][pauli][
                    "mean"
                ]
                for _, method in learned
            ]
        )
        ax.bar(x, values, bottom=bottom, color=pauli_colors[pauli], label=rf"${pauli}^2$")
        bottom += values
    ax.set_xticks(x, [label for label, _ in learned])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Allocation-weighted axis power")
    ax.set_title("C  Learned product measurements are non-Z")
    ax.legend(ncols=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "E013 phase memory — full-output learnable product measurements, exact statevector, seeds 42/43/44",
        fontsize=14,
    )
    output = ROOT / "figures/e013_phase_learnable.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_file": str(SUMMARY_PATH.relative_to(ROOT)),
        "summary_file_sha256": _digest(SUMMARY_PATH),
        "plot_source_file": str(Path(__file__).relative_to(ROOT)),
        "plot_source_sha256": _digest(Path(__file__)),
        "figure_file": str(output.relative_to(ROOT)),
        "figure_file_sha256": _digest(output),
    }
    provenance_path = ROOT / "results/e013_phase_figure_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
