"""Run the E004 baseline for three fixed seeds and write an aggregate artifact."""

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

from experiments.e004_continual_three import run_experiment, write_result  # noqa: E402

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


def build_summary(
    results: list[dict[str, Any]], result_paths: list[Path]
) -> dict[str, Any]:
    """Aggregate compatible seeded results with sample standard deviations."""
    if not results or len(results) != len(result_paths):
        raise ValueError("results and result paths must have the same nonzero length")

    epochs = [row["epoch"] for row in results[0]["training"]["history"]]
    task_keys = tuple(results[0]["metrics"]["tasks"])
    source_hashes = {result["source_code_sha256"] for result in results}
    reference_configuration = {
        "dataset": results[0]["dataset"],
        "model": results[0]["model"],
        "task_order": results[0]["training"]["task_order"],
        "optimizer": results[0]["training"]["optimizer"],
        "learning_rate": results[0]["training"]["learning_rate"],
        "epochs_per_task": results[0]["training"]["epochs_per_task"],
        "task_boundaries": results[0]["training"]["task_boundaries"],
    }
    for result in results[1:]:
        candidate_epochs = [row["epoch"] for row in result["training"]["history"]]
        if candidate_epochs != epochs or tuple(result["metrics"]["tasks"]) != task_keys:
            raise ValueError("seeded results have incompatible histories or tasks")
        candidate_configuration = {
            "dataset": result["dataset"],
            "model": result["model"],
            "task_order": result["training"]["task_order"],
            "optimizer": result["training"]["optimizer"],
            "learning_rate": result["training"]["learning_rate"],
            "epochs_per_task": result["training"]["epochs_per_task"],
            "task_boundaries": result["training"]["task_boundaries"],
        }
        if candidate_configuration != reference_configuration:
            raise ValueError("seeded results have incompatible configurations")
    if len(source_hashes) != 1:
        raise ValueError("seeded results were produced by different source revisions")
    seeds = [result["training"]["seed"] for result in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeded results contain duplicate seeds")

    aggregate_history = []
    for row_index, epoch in enumerate(epochs):
        template = results[0]["training"]["history"][row_index]
        aggregate_history.append(
            {
                "epoch": epoch,
                "phase": template["phase"],
                "train_accuracy": {
                    key: _mean_std(
                        [
                            result["training"]["history"][row_index]["train_accuracy"][key]
                            for result in results
                        ]
                    )
                    for key in task_keys
                },
            }
        )

    aggregate_metrics = {}
    numeric_fields = (
        "train_accuracy_at_phase_end",
        "train_accuracy_final",
        "train_accuracy_drop_since_phase_end",
        "test_accuracy_at_phase_end",
        "test_accuracy_final",
        "test_accuracy_drop_since_phase_end",
        "logreg_test_accuracy",
    )
    for key in task_keys:
        aggregate_metrics[key] = {
            "name": results[0]["metrics"]["tasks"][key]["name"],
            **{
                field: _mean_std(
                    [result["metrics"]["tasks"][key][field] for result in results]
                )
                for field in numeric_fields
            },
        }

    return {
        "schema_version": 1,
        "experiment": "e004_three_task_sequential_baseline_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": next(iter(source_hashes)),
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": seeds,
        "n_seeds": len(results),
        "uncertainty": "sample standard deviation across seeds (ddof=1)",
        "result_files": [path.relative_to(ROOT).as_posix() for path in result_paths],
        "data_split_sha256_by_seed": {
            str(result["training"]["seed"]): result["data_split_sha256"]
            for result in results
        },
        "configuration": {
            "task_order": results[0]["training"]["task_order"],
            "epochs_per_task": results[0]["training"]["epochs_per_task"],
            "task_boundaries": results[0]["training"]["task_boundaries"],
            "n_train_per_task": results[0]["dataset"]["n_train_per_task"],
            "n_test_per_task": results[0]["dataset"]["n_test_per_task"],
            "model": results[0]["model"],
            "optimizer": results[0]["training"]["optimizer"],
            "learning_rate": results[0]["training"]["learning_rate"],
        },
        "aggregate_history": aggregate_history,
        "aggregate_metrics": aggregate_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS / "e004_continual_summary.json",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    results = []
    result_paths = []
    for seed in args.seeds:
        result = run_experiment(
            layers=args.layers,
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=seed,
        )
        path = write_result(result, RESULTS / f"e004_continual_seed{seed}.json")
        results.append(result)
        result_paths.append(path)
        print(f"Wrote {path}")

    summary = build_summary(results, result_paths)
    summary_path = args.summary_output
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
