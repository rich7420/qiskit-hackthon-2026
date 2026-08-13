"""Validate and aggregate paired E013 learnable-measurement runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e013_learnable_measqcl import (  # noqa: E402
    METHODS,
    _file_digest,
    _source_digest,
)

SEEDS = (42, 43, 44)
METRICS = (
    "task1_at_boundary",
    "task1_final_retention",
    "task1_forgetting",
    "backward_transfer",
    "task2_final_adaptation",
    "average_final_accuracy",
)


def _sample_summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6),
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _orientation_power(optimization: dict[str, Any]) -> np.ndarray:
    """Gauge/permutation-invariant weighted X^2/Y^2/Z^2 measurement content."""
    axes = np.asarray(optimization["axes"], dtype=float)
    allocation = np.asarray(optimization["allocation"], dtype=float)
    power = np.sum(allocation[:, np.newaxis, np.newaxis] * axes**2, axis=(0, 1))
    power /= axes.shape[1]
    return power / np.sum(power)


def _validate_parent_reference(run: dict[str, Any], key: str, path: Path) -> None:
    reference = run["parents"][key]
    parent_path = ROOT / reference["result_file"]
    if _file_digest(parent_path) != reference["result_file_sha256"]:
        raise ValueError(f"{path} references a changed {key.upper()} artifact")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["source_code_sha256"] != reference["source_code_sha256"]:
        raise ValueError(f"{path} has an inconsistent {key.upper()} source digest")
    if parent["data_sha256"] != run["data_sha256"]:
        raise ValueError(f"{path} and its {key.upper()} parent use different data")


def _configuration_signature(run: dict[str, Any]) -> str:
    invariant = {
        "environment": run["environment"],
        "data": {
            key: run["data"][key]
            for key in ("task_order", "n_train_per_task", "n_test_per_task")
        },
        "model": {
            key: run["model"][key]
            for key in (
                "n_qubits",
                "layers",
                "n_parameters",
                "consolidation_readout_wires",
                "learned_measurement_class",
                "n_learned_settings",
                "prediction_measurement_unchanged",
            )
        },
        "training": {
            key: run["training"][key]
            for key in (
                "optimizer",
                "learning_rate",
                "epochs_per_task",
                "ewc_lambda_shared",
                "record_test_history",
                "selection_policy",
            )
        },
        "measurement_design": {
            key: run["measurement_design"][key]
            for key in (
                "fisher_parameter",
                "task_relevance_source",
                "objective_variants",
                "initialization",
                "initialization_noise",
                "minimum_allocation",
                "relevance_floor",
                "diversity_coefficient",
                "outer_iterations",
                "axis_max_iterations",
                "optimizer",
            )
        },
    }
    return json.dumps(invariant, sort_keys=True, separators=(",", ":"))


def build_summary(runs: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    if len(runs) != len(SEEDS) or len(paths) != len(runs):
        raise ValueError("E013 formal summary requires exactly seeds 42/43/44")
    if [run["training"]["seed"] for run in runs] != list(SEEDS):
        raise ValueError("E013 runs must be ordered seeds 42/43/44")
    expected_configuration = _configuration_signature(runs[0])
    for run, path in zip(runs, paths, strict=True):
        if run["source_code_sha256"] != _source_digest():
            raise ValueError(f"{path} does not match current E013 sources")
        if run["experiment"] != "e013_learnable_measurement_fisher_exact":
            raise ValueError(f"{path} is not an E013 exact run")
        if set(run["metrics"]) != set(METHODS):
            raise ValueError(f"{path} has incomplete E013 method metrics")
        if _configuration_signature(run) != expected_configuration:
            raise ValueError(f"{path} uses a different formal E013 configuration")
        for method, optimization in run["measurement_design"]["optimization"].items():
            if not optimization["axis_solver_messages"][-1].startswith("CONVERGENCE"):
                raise ValueError(f"{path} has a non-converged final axis solve for {method}")
            if optimization["axis_physical_gradient_norms"][-1] > optimization[
                "axis_stationarity_tolerance"
            ]:
                raise ValueError(f"{path} has a non-stationary final axis solve for {method}")
        _validate_parent_reference(run, "e008", path)
        _validate_parent_reference(run, "e010", path)

    aggregate_metrics = {
        method: {
            split: {
                metric: _sample_summary(
                    [run["metrics"][method][split][metric] for run in runs]
                )
                for metric in METRICS
            }
            for split in ("train", "test")
        }
        for method in METHODS
    }
    geometry = {}
    for method in METHODS:
        powers = np.stack(
            [
                _orientation_power(run["measurement_design"]["optimization"][method])
                for run in runs
            ]
        )
        geometry[method] = {
            "orientation_power": {
                pauli: _sample_summary(powers[:, index])
                for index, pauli in enumerate(("X", "Y", "Z"))
            },
            "diversity_penalty": _sample_summary(
                [
                    run["measurement_design"]["optimization"][method][
                        "diversity_penalty"
                    ]
                    for run in runs
                ]
            ),
            "cosine_to_output_cfi": _sample_summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "cosine_to_output_cfi"
                    ]
                    for run in runs
                ]
            ),
            "cosine_to_readout_qfi": _sample_summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "cosine_to_readout_qfi"
                    ]
                    for run in runs
                ]
            ),
            "cosine_to_full_qfi": _sample_summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "cosine_to_full_qfi"
                    ]
                    for run in runs
                ]
            ),
            "minimum_readout_qfi_minus_basis_cfi": _sample_summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "information_hierarchy"
                    ]["minimum_readout_qfi_minus_basis_cfi"]
                    for run in runs
                ]
            ),
            "minimum_readout_qfi_minus_accessible_cfi": _sample_summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "information_hierarchy"
                    ]["minimum_readout_qfi_minus_accessible_cfi"]
                    for run in runs
                ]
            ),
        }

    paired_differences = {}
    comparators = {
        "task_relevant_mof": ("e010", "task_relevant_mof"),
        "joint_zz": ("e008", "zz_cfi"),
        "task_agnostic_mof": ("e008", "mof_ewc"),
        "readout_qfi": ("e008", "readout_qewc"),
        "full_qfi": ("e008", "qewc"),
    }
    for method in METHODS:
        paired_differences[method] = {}
        for label, (parent_key, parent_method) in comparators.items():
            paired_differences[method][f"minus_{label}"] = {
                metric: _sample_summary(
                    [
                        run["metrics"][method]["test"][metric]
                        - run["parent_metrics"][parent_key][parent_method]["test"][metric]
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
        "experiment": "e013_learnable_measurement_fisher_exact_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "uncertainty": "sample standard deviation across paired seeds (ddof=1)",
        "result_artifacts": [
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": _file_digest(path),
            }
            for path in paths
        ],
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
                    "fisher_parameter",
                    "task_relevance_source",
                    "objective_variants",
                    "initialization",
                    "initialization_noise",
                    "minimum_allocation",
                    "relevance_floor",
                    "diversity_coefficient",
                    "outer_iterations",
                    "axis_max_iterations",
                    "optimizer",
                )
            },
        },
        "aggregate_metrics": aggregate_metrics,
        "measurement_geometry": geometry,
        "paired_differences": paired_differences,
        "paired_seed_metrics": {
            str(run["training"]["seed"]): {
                method: run["metrics"][method]["test"] for method in METHODS
            }
            for run in runs
        },
        "resource_accounting": {
            key: runs[0]["resource_accounting"][key]
            for key in (
                "cached_quantum_circuit_configurations",
                "formula",
                "n_anchors",
                "n_classifier_parameters",
                "basis_independent_cache_reused_by_every_objective_evaluation",
                "axis_optimization_quantum_circuit_configurations",
                "finite_shots",
                "hardware_execution",
            )
        },
        "claim_boundaries": {
            **runs[0]["claim_boundaries"],
            "statistical_significance_claimed": False,
            "interpretation": (
                "paired three-seed exact-statevector diagnostic; continuous local basis "
                "learning does not establish superiority over fixed Joint ZZ"
            ),
        },
    }


def main() -> None:
    force = "--force" in sys.argv
    paths = [ROOT / f"results/e013_learnable_measqcl_seed{seed}.json" for seed in SEEDS]
    for seed, path in zip(SEEDS, paths, strict=True):
        if force or not path.exists():
            subprocess.run(
                [sys.executable, str(ROOT / "experiments/e013_learnable_measqcl.py"), "--seed", str(seed)],
                cwd=ROOT,
                check=True,
            )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = build_summary(runs, paths)
    output = ROOT / "results/e013_learnable_measqcl_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for method, values in summary["aggregate_metrics"].items():
        test = values["test"]
        print(
            f"  {method:26s} "
            f"retention={test['task1_final_retention']['mean']:.3f} +/- "
            f"{test['task1_final_retention']['sample_std']:.3f}; "
            f"adaptation={test['task2_final_adaptation']['mean']:.3f} +/- "
            f"{test['task2_final_adaptation']['sample_std']:.3f}"
        )


if __name__ == "__main__":
    main()
