"""Plot E013 phase-first training curves in the E009 three-panel style."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
TASKS = ("phase", "mnist", "fashion")
E013_TEMPLATE = "results/e013_phase_learnable_seed{seed}.json"
E010_TEMPLATE = "results/e010_phase_locality_seed{seed}.json"
OUTPUT = ROOT / "figures/e013_phase_training.png"
PROVENANCE = ROOT / "results/e013_phase_training_figure_provenance.json"

METHODS = {
    "naive": ("Baseline (naive seq.)", "#777777", "--", "e010"),
    "joint_zzzz": ("Joint ZZZZ", "#4C78A8", "-.", "e013"),
    "uniform_xyz_joint": ("Uniform joint XYZ", "#72B7B2", "-.", "e013"),
    "info_learn_basis_alloc": ("LearnBasis info + allocation", "#F28E2B", "-", "e013"),
    "qewc": ("Full-state QEWC", "#333333", ":", "e010"),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runs() -> tuple[dict[int, dict], dict[int, dict], list[Path]]:
    e013_runs: dict[int, dict] = {}
    e010_runs: dict[int, dict] = {}
    paths: list[Path] = []
    for seed in SEEDS:
        e013_path = ROOT / E013_TEMPLATE.format(seed=seed)
        e010_path = ROOT / E010_TEMPLATE.format(seed=seed)
        e013_runs[seed] = json.loads(e013_path.read_text(encoding="utf-8"))
        e010_runs[seed] = json.loads(e010_path.read_text(encoding="utf-8"))
        paths.extend((e013_path, e010_path))
    return e013_runs, e010_runs, paths


def _history(
    method: str,
    source: str,
    seed: int,
    e013_runs: dict[int, dict],
    e010_runs: dict[int, dict],
) -> list[dict]:
    run = e013_runs[seed] if source == "e013" else e010_runs[seed]
    return run["histories"][method]


def _validate_histories(e013_runs: dict[int, dict], e010_runs: dict[int, dict]) -> np.ndarray:
    reference_epochs: np.ndarray | None = None
    for method, (_, _, _, source) in METHODS.items():
        for seed in SEEDS:
            rows = _history(method, source, seed, e013_runs, e010_runs)
            epochs = np.asarray([row["epoch"] for row in rows], dtype=int)
            if reference_epochs is None:
                reference_epochs = epochs
            elif not np.array_equal(epochs, reference_epochs):
                raise ValueError(f"unaligned history for {method}, seed {seed}")
            if any(set(row["train_accuracy"]) != set(TASKS) for row in rows):
                raise ValueError(f"incomplete task history for {method}, seed {seed}")
    if reference_epochs is None:
        raise ValueError("no histories available")
    return reference_epochs


def main() -> None:
    e013_runs, e010_runs, input_paths = _load_runs()
    epochs = _validate_histories(e013_runs, e010_runs)
    epochs_per_task_values = {
        int(run["training"]["epochs_per_task"])
        for run in (*e013_runs.values(), *e010_runs.values())
    }
    if len(epochs_per_task_values) != 1:
        raise ValueError("inconsistent epochs_per_task across input artifacts")
    epochs_per_task = epochs_per_task_values.pop()
    total_epochs = int(epochs[-1])
    if total_epochs != epochs_per_task * len(TASKS):
        raise ValueError("history length does not match the declared task schedule")
    boundaries = tuple(epochs_per_task * index for index in range(1, len(TASKS)))

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    for task_index, (ax, task) in enumerate(zip(axes, TASKS, strict=True), start=1):
        for method, (label, color, linestyle, source) in METHODS.items():
            values = np.asarray(
                [
                    [
                        row["train_accuracy"][task]
                        for row in _history(method, source, seed, e013_runs, e010_runs)
                    ]
                    for seed in SEEDS
                ],
                dtype=float,
            )
            mean = np.mean(values, axis=0)
            sample_sd = np.std(values, axis=0, ddof=1)
            ax.plot(
                epochs,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
            ax.fill_between(
                epochs,
                np.clip(mean - sample_sd, 0.0, 1.0),
                np.clip(mean + sample_sd, 0.0, 1.0),
                color=color,
                alpha=0.14,
            )
        for boundary in boundaries:
            ax.axvline(boundary, color="0.35", linestyle="--", linewidth=1.0)
        phase_start = (task_index - 1) * epochs_per_task
        ax.axvspan(
            phase_start,
            task_index * epochs_per_task,
            color="#59A14F",
            alpha=0.06,
        )
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.0, 1.03)
        ax.set_ylabel(f"T{task_index}\ntrain accuracy")
        title = "SPT/ATF phase" if task == "phase" else task.upper()
        ax.set_title(f"Task {task_index}: {title}", loc="left", fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower right", fontsize=8.5, ncols=2)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(
        "E013: phase-first learnable measurements — training accuracy "
        "(seeds 42/43/44, mean ± sample SD)",
        fontsize=13,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "uncertainty": "sample standard deviation",
        "input_artifacts": [
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": _digest(path),
            }
            for path in input_paths
        ],
        "plot_source_file": str(Path(__file__).relative_to(ROOT)),
        "plot_source_sha256": _digest(Path(__file__)),
        "figure_file": str(OUTPUT.relative_to(ROOT)),
        "figure_file_sha256": _digest(OUTPUT),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {PROVENANCE}")


if __name__ == "__main__":
    main()
