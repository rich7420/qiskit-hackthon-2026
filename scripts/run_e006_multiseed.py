"""Run E006 for seeds 42/43/44 and aggregate paired baseline/replay results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e006_advanced_temporal import (  # noqa: E402
    METHODS,
    run_experiment,
    write_result,
)

DEFAULT_SEEDS = (42, 43, 44)
RESULTS = ROOT / "results"


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6) if len(array) > 1 else 0.0,
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _numeric_keys(mapping: dict[str, Any]) -> tuple[str, ...]:
    return tuple(key for key, value in mapping.items() if isinstance(value, (int, float)))


def build_summary(
    results: list[dict[str, Any]], result_paths: list[Path]
) -> dict[str, Any]:
    """Aggregate compatible runs and preserve paired method deltas by seed."""
    if not results or len(results) != len(result_paths):
        raise ValueError("results and paths must have the same nonzero length")
    seeds = [result["training"]["seed"] for result in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("results contain duplicate seeds")

    reference = results[0]
    epochs = [row["epoch"] for row in reference["methods"]["baseline"]["history"]]
    compatibility_fields = ("dataset", "model")
    training_fields = (
        "comparison",
        "optimizer",
        "loss",
        "optimizer_state_reset_at_boundaries",
        "learning_rate",
        "epochs_per_task",
        "task_boundaries",
        "memory_per_previous_task",
        "memory_selection",
        "replay_weight",
        "record_test_during_training",
        "test_used_for_selection",
    )
    for result in results:
        if result["source_code_sha256"] != reference["source_code_sha256"]:
            raise ValueError("runs were produced by different source revisions")
        for field in compatibility_fields:
            if result[field] != reference[field]:
                raise ValueError(f"runs have incompatible {field}")
        if any(result["training"][field] != reference["training"][field] for field in training_fields):
            raise ValueError("runs have incompatible training configurations")
        for method in METHODS:
            candidate_epochs = [row["epoch"] for row in result["methods"][method]["history"]]
            if candidate_epochs != epochs:
                raise ValueError("runs have incompatible histories")

    aggregate_history = {}
    for method in METHODS:
        aggregate_history[method] = []
        for row_index, epoch in enumerate(epochs):
            template = reference["methods"][method]["history"][row_index]
            aggregate_row = {"epoch": epoch, "phase": template["phase"]}
            for metric in ("train_accuracy", "train_balanced_accuracy"):
                aggregate_row[metric] = {
                    key: _mean_std(
                        [
                            result["methods"][method]["history"][row_index][metric][
                                key
                            ]
                            for result in results
                        ]
                    )
                    for key in template[metric]
                }
            aggregate_history[method].append(aggregate_row)

    aggregate_metrics = {}
    for method in METHODS:
        aggregate_metrics[method] = {}
        for section in ("task1", "task2", "task3", "summary"):
            template = reference["methods"][method]["metrics"][section]
            aggregate_metrics[method][section] = {
                key: _mean_std(
                    [result["methods"][method]["metrics"][section][key] for result in results]
                )
                for key in _numeric_keys(template)
                if key != "phase_end_epoch"
            }
            if "phase_end_epoch" in template:
                aggregate_metrics[method][section]["phase_end_epoch"] = template[
                    "phase_end_epoch"
                ]

    paired_seed_metrics = []
    for result in results:
        baseline = result["methods"]["baseline"]["metrics"]["summary"]
        replay = result["methods"]["replay"]["metrics"]["summary"]
        paired_seed_metrics.append(
            {
                "seed": result["training"]["seed"],
                "baseline_old_task_balanced_retention": baseline[
                    "old_task_balanced_retention_final"
                ],
                "replay_old_task_balanced_retention": replay[
                    "old_task_balanced_retention_final"
                ],
                "baseline_new_task_balanced_adaptation": baseline[
                    "new_task_balanced_adaptation"
                ],
                "replay_new_task_balanced_adaptation": replay[
                    "new_task_balanced_adaptation"
                ],
                "old_task_balanced_retention_gain": round(
                    replay["old_task_balanced_retention_final"]
                    - baseline["old_task_balanced_retention_final"],
                    4,
                ),
                "new_task_balanced_adaptation_change": round(
                    replay["new_task_balanced_adaptation"]
                    - baseline["new_task_balanced_adaptation"],
                    4,
                ),
                "average_final_train_balanced_accuracy_gain": round(
                    replay["average_final_train_balanced_accuracy"]
                    - baseline["average_final_train_balanced_accuracy"],
                    4,
                ),
                "old_task_balanced_forgetting_reduction": round(
                    baseline["average_old_task_balanced_forgetting"]
                    - replay["average_old_task_balanced_forgetting"],
                    4,
                ),
            }
        )
    paired_aggregate = {
        key: _mean_std([row[key] for row in paired_seed_metrics])
        for key in (
            "old_task_balanced_retention_gain",
            "new_task_balanced_adaptation_change",
            "average_final_train_balanced_accuracy_gain",
            "old_task_balanced_forgetting_reduction",
        )
    }
    compute = {
        method: {
            "train_time_sec": _mean_std(
                [result["methods"][method]["train_time_sec"] for result in results]
            ),
            "objective_sample_exposures": _mean_std(
                [
                    result["methods"][method]["objective_sample_exposures"]
                    for result in results
                ]
            ),
            "relative_objective_sample_exposure": _mean_std(
                [
                    result["methods"][method]["relative_objective_sample_exposure"]
                    for result in results
                ]
            ),
        }
        for method in METHODS
    }
    compute["paired_replay_minus_baseline"] = {
        "train_time_sec": _mean_std(
            [
                result["methods"]["replay"]["train_time_sec"]
                - result["methods"]["baseline"]["train_time_sec"]
                for result in results
            ]
        ),
        "train_time_ratio": _mean_std(
            [
                result["methods"]["replay"]["train_time_sec"]
                / result["methods"]["baseline"]["train_time_sec"]
                for result in results
            ]
        ),
    }

    return {
        "schema_version": 1,
        "experiment": "e006_advanced_temporal_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": reference["source_code_sha256"],
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": seeds,
        "n_seeds": len(results),
        "uncertainty": "sample standard deviation across seeds (ddof=1)",
        "comparison_scope": "same E006 tasks, data, initialization, optimizer, and epoch budget",
        "baseline_definition": reference["training"]["comparison"]["baseline"],
        "advanced_definition": reference["training"]["comparison"]["replay"],
        "result_files": [path.relative_to(ROOT).as_posix() for path in result_paths],
        "data_sha256_by_seed": {
            str(result["training"]["seed"]): result["data_sha256"] for result in results
        },
        "configuration": {
            "dataset": reference["dataset"],
            "model": reference["model"],
            **{field: reference["training"][field] for field in training_fields},
        },
        "aggregate_history": aggregate_history,
        "aggregate_metrics": aggregate_metrics,
        "paired_seed_metrics": paired_seed_metrics,
        "paired_aggregate": paired_aggregate,
        "compute": compute,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.03, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--memory-per-task", type=int, default=16)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--offline", action="store_false", dest="allow_download")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS / "e006_advanced_summary.json",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    results = []
    paths = []
    for seed in args.seeds:
        result = run_experiment(
            n_qubits=args.n_qubits,
            layers=args.layers,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            memory_per_task=args.memory_per_task,
            replay_weight=args.replay_weight,
            seed=seed,
            allow_download=args.allow_download,
        )
        path = write_result(result, RESULTS / f"e006_advanced_seed{seed}.json")
        results.append(result)
        paths.append(path)
        print(f"Wrote {path}")

    summary = build_summary(results, paths)
    output = args.summary_output if args.summary_output.is_absolute() else ROOT / args.summary_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
