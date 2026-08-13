"""Plot the E013 learnable-measurement result against fixed-basis E008/E010."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(summary: dict, method: str, metric: str) -> tuple[float, float]:
    value = summary["aggregate_metrics"][method]["test"][metric]
    return value["mean"], value["sample_std"]


def main() -> None:
    e008 = _load("e008_measqcl_summary.json")
    e010 = _load("e010_physmeas_summary.json")
    e013 = _load("e013_learnable_measqcl_summary.json")

    methods = [
        ("Joint ZZ", e008, "zz_cfi", "#4C78A8", "o"),
        ("Alloc XYZ\n(info)", e008, "mof_ewc", "#9ECAE9", "s"),
        ("Alloc XYZ\n(task)", e010, "task_relevant_mof", "#59A14F", "D"),
        ("LearnBasis\n(info+alloc)", e013, "info_learn_basis_alloc", "#F28E2B", "^"),
        ("LearnBasis\n(task)", e013, "task_learn_basis_uniform", "#E15759", "v"),
        ("LearnBasis\n(task+alloc)", e013, "task_learn_basis_alloc", "#B07AA1", "P"),
        ("Readout QFI", e008, "readout_qewc", "#79706E", "*"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    ax = axes[0]
    for label, summary, method, color, marker in methods:
        retention, retention_sd = _metric(summary, method, "task1_final_retention")
        adaptation, adaptation_sd = _metric(summary, method, "task2_final_adaptation")
        ax.errorbar(
            adaptation,
            retention,
            xerr=adaptation_sd,
            yerr=retention_sd,
            fmt=marker,
            color=color,
            markersize=9,
            capsize=3,
            label=label.replace("\n", " "),
        )
    ax.set_xlabel("Fashion-MNIST adaptation")
    ax.set_ylabel("MNIST retention")
    ax.set_title("A  Stability–plasticity frontier")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    learned_methods = [
        ("Info + alloc", "info_learn_basis_alloc"),
        ("Task", "task_learn_basis_uniform"),
        ("Task + alloc", "task_learn_basis_alloc"),
    ]
    colors = {"X": "#4C78A8", "Y": "#F28E2B", "Z": "#59A14F"}
    x = np.arange(len(learned_methods))
    bottom = np.zeros(len(learned_methods))
    for pauli in ("X", "Y", "Z"):
        values = np.asarray(
            [
                e013["measurement_geometry"][method]["orientation_power"][pauli]["mean"]
                for _, method in learned_methods
            ]
        )
        ax.bar(x, values, bottom=bottom, color=colors[pauli], label=rf"${pauli}^2$")
        bottom += values
    ax.set_xticks(x, [label for label, _ in learned_methods])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Allocation-weighted axis power")
    ax.set_title("B  What the learned ensemble measures")
    ax.legend(ncols=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    for label, method, color, marker in (
        ("Info + alloc", "info_learn_basis_alloc", "#F28E2B", "^"),
        ("Task", "task_learn_basis_uniform", "#E15759", "v"),
        ("Task + alloc", "task_learn_basis_alloc", "#B07AA1", "P"),
    ):
        similarity = e013["measurement_geometry"][method]["cosine_to_readout_qfi"]
        retention = e013["aggregate_metrics"][method]["test"]["task1_final_retention"]
        ax.errorbar(
            similarity["mean"],
            retention["mean"],
            xerr=similarity["sample_std"],
            yerr=retention["sample_std"],
            fmt=marker,
            color=color,
            markersize=10,
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Cosine similarity to readout-subsystem QFI")
    ax.set_ylabel("MNIST retention")
    ax.set_title("C  More QFI-like is not more memorable")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.suptitle(
        "E013 learnable measurement Fisher consolidation — exact statevector, seeds 42/43/44",
        fontsize=14,
    )
    output = ROOT / "figures/e013_learnable_measqcl.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    summary_path = ROOT / "results/e013_learnable_measqcl_summary.json"
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_file": str(summary_path.relative_to(ROOT)),
        "summary_file_sha256": _digest(summary_path),
        "plot_source_file": str(Path(__file__).relative_to(ROOT)),
        "plot_source_sha256": _digest(Path(__file__)),
        "figure_file": str(output.relative_to(ROOT)),
        "figure_file_sha256": _digest(output),
    }
    provenance_path = ROOT / "results/e013_figure_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
