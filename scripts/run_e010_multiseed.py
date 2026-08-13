"""Validate and aggregate the three paired E010 PhysMeas-QCL runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e008_measqcl import METHODS as PARENT_METHODS  # noqa: E402
from experiments.e010_physmeas_qcl import (  # noqa: E402
    NEW_METHODS,
    _file_digest,
    _source_digest,
)

SEEDS = (42, 43, 44)
METHODS = PARENT_METHODS + NEW_METHODS
METRICS = (
    "task1_at_boundary",
    "task1_final_retention",
    "task1_forgetting",
    "backward_transfer",
    "task2_final_adaptation",
    "average_final_accuracy",
)


def _sample_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6),
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_summary(runs: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    if len(runs) != len(SEEDS) or len(paths) != len(runs):
        raise ValueError("E010 formal summary requires exactly seeds 42/43/44")
    if [run["training"]["seed"] for run in runs] != list(SEEDS):
        raise ValueError("E010 runs must be ordered seeds 42/43/44")
    for run, path in zip(runs, paths, strict=True):
        if run["source_code_sha256"] != _source_digest():
            raise ValueError(f"{path} does not match current E010 sources")
        parent_path = ROOT / run["parent"]["result_file"]
        if _file_digest(parent_path) != run["parent"]["result_file_sha256"]:
            raise ValueError(f"{path} references a changed E008 artifact")
        parent_result = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent_result["source_code_sha256"] != run["parent"]["source_code_sha256"]:
            raise ValueError(f"{path} parent source digest is inconsistent")
        if parent_result["data_sha256"] != run["data_sha256"]:
            raise ValueError(f"{path} parent and extension use different data")
        if set(run["metrics"]) != set(NEW_METHODS):
            raise ValueError(f"{path} has incomplete E010 method metrics")
        if set(run["parent_metrics"]) != set(PARENT_METHODS):
            raise ValueError(f"{path} has incomplete inherited method metrics")

    aggregate_metrics: dict[str, Any] = {}
    for method in METHODS:
        source = "parent_metrics" if method in PARENT_METHODS else "metrics"
        aggregate_metrics[method] = {
            split: {
                metric: _sample_summary(
                    [run[source][method][split][metric] for run in runs]
                )
                for metric in METRICS
            }
            for split in ("train", "test")
        }

    allocation = {
        basis: _sample_summary(
            [run["measurement_design"]["allocation"][basis] for run in runs]
        )
        for basis in runs[0]["measurement_design"]["allocation"]
    }
    diagnostics = {
        key: _sample_summary(
            [run["measurement_design"]["diagnostics"][key] for run in runs]
        )
        for key in runs[0]["measurement_design"]["diagnostics"]
    }
    paired_differences = {}
    for comparator in ("mof_ewc", "zz_cfi", "output_cfi"):
        paired_differences[f"task_relevant_mof_minus_{comparator}"] = {
            metric: _sample_summary(
                [
                    run["metrics"]["task_relevant_mof"]["test"][metric]
                    - run["parent_metrics"][comparator]["test"][metric]
                    for run in runs
                ]
            )
            for metric in (
                "task1_final_retention",
                "task2_final_adaptation",
                "average_final_accuracy",
            )
        }
    return {
        "schema_version": 1,
        "experiment": "e010_task_relevant_measqcl_exact_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "uncertainty": "sample standard deviation across paired seeds (ddof=1)",
        "result_files": [str(path.relative_to(ROOT)) for path in paths],
        "configuration": {
            "data": runs[0]["data"],
            "model": runs[0]["model"],
            "training": {
                key: runs[0]["training"][key]
                for key in (
                    "optimizer",
                    "learning_rate",
                    "epochs_per_task",
                    "ewc_lambda_shared",
                    "selection_policy",
                )
            },
            "measurement_design": {
                key: runs[0]["measurement_design"][key]
                for key in (
                    "task_relevance_estimator",
                    "relevance_role",
                    "relevance_floor_fraction",
                    "minimum_allocation_per_basis",
                    "objective",
                )
            },
        },
        "aggregate_metrics": aggregate_metrics,
        "task_relevant_allocation": allocation,
        "geometry_diagnostics": diagnostics,
        "paired_differences": paired_differences,
        "paired_seed_metrics": {
            str(run["training"]["seed"]): {
                method: (
                    run["parent_metrics"][method]["test"]
                    if method in PARENT_METHODS
                    else run["metrics"][method]["test"]
                )
                for method in METHODS
            }
            for run in runs
        },
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "locality_resolved_result": False,
            "statistical_significance_claimed": False,
            "quantum_advantage": False,
            "interpretation": (
                "paired three-seed exact-statevector diagnostic; it tests whether "
                "task relevance repairs task-agnostic measurement selection"
            ),
        },
    }


def main() -> None:
    paths = [ROOT / f"results/e010_physmeas_seed{seed}.json" for seed in SEEDS]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = build_summary(runs, paths)
    output = ROOT / "results/e010_physmeas_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for method, values in summary["aggregate_metrics"].items():
        test = values["test"]
        print(
            f"  {method:18s} retention={test['task1_final_retention']['mean']:.3f} "
            f"adaptation={test['task2_final_adaptation']['mean']:.3f} "
            f"average={test['average_final_accuracy']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
