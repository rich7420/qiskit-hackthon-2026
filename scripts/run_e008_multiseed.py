"""Run and aggregate the exact-statevector MeasQCL reference over paired seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e008_measqcl import (  # noqa: E402
    METHODS,
    _source_digest,
    run_experiment,
    write_result,
)

RESULTS = ROOT / "results"
DEFAULT_SEEDS = (42, 43, 44)
MEASUREMENT_METHODS = ("zz_cfi", "uniform_xyz", "mof_ewc")


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6)
        if len(array) > 1
        else 0.0,
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _configuration_signature(result: dict[str, Any]) -> dict[str, Any]:
    training = result["training"]
    return {
        "data": {
            key: value
            for key, value in result["data"].items()
            if key != "test_used_for_selection"
        },
        "model": result["model"],
        "training": {
            key: training[key]
            for key in (
                "optimizer",
                "learning_rate",
                "epochs_per_task",
                "ewc_lambda_shared_by_all_consolidation_methods",
                "fisher_normalization",
                "fisher_samples",
                "reference_shots_for_allocation_only",
                "record_test_history",
            )
        },
    }


def _correlation_record(
    x: list[float],
    y: list[float],
) -> dict[str, float | int | bool | None]:
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("correlation inputs must have the same length of at least three")
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        return {
            "n_seed_method_points": len(x),
            "defined": False,
            "pearson": None,
            "spearman": None,
        }
    pearson = float(np.corrcoef(x_array, y_array)[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    return {
        "n_seed_method_points": len(x),
        "defined": True,
        "pearson": round(pearson, 6),
        "spearman": round(spearman, 6),
    }


def build_summary(
    runs: list[dict[str, Any]],
    paths: list[Path],
) -> dict[str, Any]:
    """Aggregate compatible paired runs and preserve geometry/retention diagnostics."""
    if not runs or len(runs) != len(paths):
        raise ValueError("runs and paths must have the same nonzero length")
    seeds = [run["training"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("runs contain duplicate seeds")
    reference = runs[0]
    signature = _configuration_signature(reference)
    epochs = [row["epoch"] for row in reference["histories"]["naive"]]
    for run in runs:
        if run["source_code_sha256"] != reference["source_code_sha256"]:
            raise ValueError("runs were produced by different source revisions")
        if _configuration_signature(run) != signature:
            raise ValueError("runs have incompatible configurations")
        if run["data"]["test_used_for_selection"] is not False:
            raise ValueError("test data must not be used for configuration selection")
        for method in METHODS:
            if [row["epoch"] for row in run["histories"][method]] != epochs:
                raise ValueError("runs have incompatible histories")

    aggregate_history: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        aggregate_history[method] = []
        for row_index, epoch in enumerate(epochs):
            template = reference["histories"][method][row_index]
            row: dict[str, Any] = {"epoch": epoch, "phase": template["phase"]}
            for field in ("train_accuracy", "test_accuracy"):
                if template[field] is None:
                    row[field] = None
                else:
                    row[field] = {
                        task: _mean_std(
                            [run["histories"][method][row_index][field][task] for run in runs]
                        )
                        for task in template[field]
                    }
            aggregate_history[method].append(row)

    aggregate_metrics: dict[str, Any] = {}
    for method in METHODS:
        aggregate_metrics[method] = {}
        for split in reference["metrics"][method]:
            aggregate_metrics[method][split] = {
                metric: _mean_std(
                    [run["metrics"][method][split][metric] for run in runs]
                )
                for metric in reference["metrics"][method][split]
            }

    allocations = {
        basis: _mean_std(
            [run["fisher_profiles"]["allocations"]["mof_ewc"][basis] for run in runs]
        )
        for basis in reference["fisher_profiles"]["allocations"]["mof_ewc"]
    }
    geometry = {
        method: {
            "cosine_to_full_qfi": _mean_std(
                [
                    run["fisher_profiles"]["cosine_similarity_to_qfi"][method]
                    for run in runs
                ]
            ),
            "cosine_to_readout_qfi": _mean_std(
                [
                    run["fisher_profiles"]["cosine_similarity_to_readout_qfi"][method]
                    for run in runs
                ]
            ),
            "cosine_to_output_cfi": _mean_std(
                [
                    (
                        1.0
                        if method == "output_cfi"
                        else run["fisher_profiles"]["cosine_similarity_to_output_cfi"][method]
                    )
                    for run in runs
                ]
            ),
        }
        for method in ("output_cfi", *MEASUREMENT_METHODS)
    }
    for method in MEASUREMENT_METHODS:
        geometry[method]["full_qfi_trace_coverage_proxy"] = _mean_std(
            [
                run["fisher_profiles"]["global_qfi_trace_coverage_proxy"][method]
                for run in runs
            ]
        )
        geometry[method]["readout_qfi_trace_coverage_proxy"] = _mean_std(
            [
                run["fisher_profiles"]["readout_qfi_trace_coverage_proxy"][method]
                for run in runs
            ]
        )

    paired_seed_metrics = []
    for run in runs:
        seed = run["training"]["seed"]
        for method in METHODS:
            for split in ("train", "test"):
                metric = run["metrics"][method][split]
                zz_metric = run["metrics"]["zz_cfi"][split]
                paired_seed_metrics.append(
                    {
                        "seed": seed,
                        "method": method,
                        "split": split,
                        "retention": metric["task1_final_retention"],
                        "adaptation": metric["task2_final_adaptation"],
                        "forgetting": metric["task1_forgetting"],
                        "average_final_accuracy": metric["average_final_accuracy"],
                        "retention_minus_zz": round(
                            metric["task1_final_retention"]
                            - zz_metric["task1_final_retention"],
                            4,
                        ),
                        "average_final_accuracy_minus_zz": round(
                            metric["average_final_accuracy"]
                            - zz_metric["average_final_accuracy"],
                            4,
                        ),
                    }
                )

    qfi_similarity: list[float] = []
    output_alignment: list[float] = []
    readout_coverage: list[float] = []
    test_forgetting: list[float] = []
    for run in runs:
        for method in MEASUREMENT_METHODS:
            qfi_similarity.append(
                run["fisher_profiles"]["cosine_similarity_to_qfi"][method]
            )
            output_alignment.append(
                run["fisher_profiles"]["cosine_similarity_to_output_cfi"][method]
            )
            readout_coverage.append(
                run["fisher_profiles"]["readout_qfi_trace_coverage_proxy"][method]
            )
            test_forgetting.append(run["metrics"][method]["test"]["task1_forgetting"])

    runtime = {
        method: {
            "boundary_fisher_estimation_sec": _mean_std(
                [run["runtime_sec"]["boundary_fisher_estimation"][method] for run in runs]
            ),
            "phase2_training_sec": _mean_std(
                [run["runtime_sec"]["phase2_training"][method] for run in runs]
            ),
        }
        for method in METHODS
    }
    return {
        "schema_version": 1,
        "experiment": "e008_measqcl_exact_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": reference["source_code_sha256"],
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "uncertainty": "sample standard deviation across seeds (ddof=1)",
        "result_files": [path.relative_to(ROOT).as_posix() for path in paths],
        "data_sha256_by_seed": {
            str(run["training"]["seed"]): run["data_sha256"] for run in runs
        },
        "configuration": signature,
        "aggregate_history": aggregate_history,
        "aggregate_metrics": aggregate_metrics,
        "mof_allocation": allocations,
        "geometry": geometry,
        "paired_seed_metrics": paired_seed_metrics,
        "descriptive_correlations": {
            "warning": (
                "Exploratory only: seed-method points are not independent and do not "
                "support population-level inference"
            ),
            "full_qfi_cosine_vs_test_forgetting": _correlation_record(
                qfi_similarity, test_forgetting
            ),
            "output_cfi_cosine_vs_test_forgetting": _correlation_record(
                output_alignment, test_forgetting
            ),
            "readout_qfi_trace_coverage_vs_test_forgetting": _correlation_record(
                readout_coverage, test_forgetting
            ),
        },
        "runtime_sec": runtime,
        "claim_boundaries": reference["claim_boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=RESULTS / "e008_measqcl_summary.json")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    runs = []
    paths = []
    for seed in args.seeds:
        path = RESULTS / f"e008_measqcl_seed{seed}.json"
        if args.reuse_existing and path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["source_code_sha256"] != _source_digest():
                raise ValueError(f"existing seed {seed} artifact has a stale source hash")
            if result["training"]["seed"] != seed:
                raise ValueError(f"existing seed {seed} artifact records another seed")
            print(f"Reusing {path}", flush=True)
        else:
            result = run_experiment(seed=seed)
            write_result(result, path)
            print(f"Wrote {path}", flush=True)
        runs.append(result)
        paths.append(path)

    summary = build_summary(runs, paths)
    output = args.summary_output if args.summary_output.is_absolute() else ROOT / args.summary_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
