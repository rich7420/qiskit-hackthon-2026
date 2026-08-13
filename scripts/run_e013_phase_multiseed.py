"""Validate and aggregate paired E013 phase-memory measurement runs."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e013_phase_learnable import (  # noqa: E402
    METHODS,
    _file_digest,
    _source_digest,
)

SEEDS = (42, 43, 44)
PARENT_COMPARATORS = (
    "naive",
    "output_cfi",
    "ewc_dr",
    "one_local",
    "two_local",
    "hamiltonian",
    "nonlocal",
    "task_relevant_all",
    "qewc",
)
METRICS = (
    "phase_at_boundary",
    "phase_final_retention",
    "phase_forgetting",
    "mnist_final_adaptation",
    "fashion_final_adaptation",
    "average_final_accuracy",
)


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6),
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _orientation_power(optimization: dict[str, Any]) -> np.ndarray:
    axes = np.asarray(optimization["axes"], dtype=float)
    allocation = np.asarray(optimization["allocation"], dtype=float)
    power = np.sum(allocation[:, None, None] * axes**2, axis=(0, 1))
    return power / np.sum(power)


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
                "learned_measurement_domain",
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
                "ewc_lambda",
                "record_test_history",
                "selection_policy",
            )
        },
        "measurement": {
            key: run["measurement_design"][key]
            for key in (
                "fisher_parameter",
                "outcomes_per_setting",
                "fixed_axes_order",
                "task_relevance_source",
                "minimum_allocation",
                "relevance_floor",
                "diversity_coefficient",
                "outer_iterations",
                "axis_max_iterations",
                "initialization_noise",
                "optimizer",
            )
        },
    }
    return json.dumps(invariant, sort_keys=True, separators=(",", ":"))


def build_summary(runs: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    if [run["training"]["seed"] for run in runs] != list(SEEDS):
        raise ValueError("formal E013 phase runs must be ordered seeds 42/43/44")
    expected_config = _configuration_signature(runs[0])
    for run, path in zip(runs, paths, strict=True):
        if run["experiment"] != "e013_phase_first_learnable_full_output_exact":
            raise ValueError(f"{path} is not an E013 phase run")
        if run["source_code_sha256"] != _source_digest():
            raise ValueError(f"{path} does not match current E013 phase sources")
        if run["data"]["test_used_for_selection"]:
            raise ValueError(f"{path} reports test-based selection")
        if set(run["metrics"]) != set(METHODS):
            raise ValueError(f"{path} has incomplete learned-measurement methods")
        if _configuration_signature(run) != expected_config:
            raise ValueError(f"{path} uses a different formal configuration")
        parent_path = ROOT / run["parent"]["result_file"]
        if _file_digest(parent_path) != run["parent"]["result_file_sha256"]:
            raise ValueError(f"{path} references a changed E010 parent")
        for method, optimization in run["measurement_design"]["optimization"].items():
            if optimization["allocation_optimality_gap"] > optimization[
                "allocation_optimality_tolerance"
            ]:
                raise ValueError(f"{path} has a non-optimal allocation for {method}")
            if not optimization["axis_solver_messages"][-1].startswith("CONVERGENCE"):
                raise ValueError(f"{path} has a non-converged final axis solve for {method}")
            if optimization["axis_physical_gradient_norms"][-1] > optimization[
                "axis_stationarity_tolerance"
            ]:
                raise ValueError(f"{path} has a non-stationary final axis solve for {method}")
        if not all(
            diagnostic["information_hierarchy"]["passed"]
            for diagnostic in run["measurement_design"]["diagnostics"].values()
        ):
            raise ValueError(f"{path} violates the CFI <= full-QFI hierarchy")

    all_methods = (*METHODS, *PARENT_COMPARATORS)

    def metric_record(run: dict[str, Any], method: str) -> dict[str, Any]:
        if method in METHODS:
            return run["metrics"][method]
        return run["parent_metrics"][method]

    aggregate = {
        method: {
            split: {
                metric: _summary(
                    [metric_record(run, method)[split][metric] for run in runs]
                )
                for metric in METRICS
            }
            for split in ("train", "test")
        }
        for method in all_methods
    }
    geometry = {}
    for method in (
        "info_learn_basis_alloc",
        "task_learn_basis_uniform",
        "task_learn_basis_alloc",
    ):
        powers = np.stack(
            [
                _orientation_power(run["measurement_design"]["optimization"][method])
                for run in runs
            ]
        )
        geometry[method] = {
            "orientation_power": {
                pauli: _summary(powers[:, index])
                for index, pauli in enumerate(("X", "Y", "Z"))
            },
            "allocation": {
                f"setting_{setting}": _summary(
                    [
                        run["measurement_design"]["optimization"][method]["allocation"][
                            setting
                        ]
                        for run in runs
                    ]
                )
                for setting in range(runs[0]["model"]["n_learned_settings"])
            },
            "cosine_to_full_qfi": _summary(
                [
                    run["measurement_design"]["diagnostics"][method][
                        "cosine_to_full_qfi"
                    ]
                    for run in runs
                ]
            ),
        }

    paired = {}
    for method in METHODS:
        paired[method] = {}
        for comparator in ("joint_zzzz", "uniform_xyz_joint", "qewc"):
            if method == comparator:
                continue
            paired[method][f"minus_{comparator}"] = {
                metric: _summary(
                    [
                        metric_record(run, method)["test"][metric]
                        - metric_record(run, comparator)["test"][metric]
                        for run in runs
                    ]
                )
                for metric in (
                    "phase_final_retention",
                    "mnist_final_adaptation",
                    "fashion_final_adaptation",
                    "average_final_accuracy",
                )
            }

    return {
        "schema_version": 1,
        "experiment": "e013_phase_learnable_joint_measurement_exact_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "uncertainty": "sample standard deviation across paired seeds (ddof=1)",
        "result_artifacts": [
            {"file": str(path.relative_to(ROOT)), "sha256": _file_digest(path)}
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
                    "ewc_lambda",
                    "selection_policy",
                )
            },
            "measurement_design": {
                key: runs[0]["measurement_design"][key]
                for key in (
                    "fisher_parameter",
                    "outcomes_per_setting",
                    "fixed_axes_order",
                    "task_relevance_source",
                    "minimum_allocation",
                    "relevance_floor",
                    "diversity_coefficient",
                    "outer_iterations",
                    "axis_max_iterations",
                    "initialization_noise",
                    "optimizer",
                )
            },
        },
        "aggregate_metrics": aggregate,
        "measurement_geometry": geometry,
        "paired_differences": paired,
        "paired_seed_metrics": {
            str(run["training"]["seed"]): {
                method: metric_record(run, method)["test"] for method in all_methods
            }
            for run in runs
        },
        "ceiling_effect_diagnostic": {
            "definition": "Joint ZZZZ phase forgetting <= 0.01 at final evaluation",
            "ceiling_seeds": [
                run["training"]["seed"]
                for run in runs
                if run["metrics"]["joint_zzzz"]["test"]["phase_forgetting"] <= 0.01
            ],
            "interpretation": (
                "post-hoc diagnostic only; the formal headline remains the paired "
                "three-seed mean and sample SD"
            ),
        },
        "claim_boundaries": {
            **runs[0]["claim_boundaries"],
            "statistical_significance_claimed": False,
            "superiority_claimed": False,
            "interpretation": (
                "exact-statevector three-seed diagnostic: learned full-output product "
                "measurements improve the one seed with substantial phase forgetting, "
                "while two seeds have a retention ceiling"
            ),
        },
    }


def main() -> None:
    paths = [ROOT / f"results/e013_phase_learnable_seed{seed}.json" for seed in SEEDS]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = build_summary(runs, paths)
    output = ROOT / "results/e013_phase_learnable_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for method in (*METHODS, "qewc"):
        test = summary["aggregate_metrics"][method]["test"]
        print(
            f"  {method:26s} phase={test['phase_final_retention']['mean']:.3f} +/- "
            f"{test['phase_final_retention']['sample_std']:.3f}; "
            f"average={test['average_final_accuracy']['mean']:.3f} +/- "
            f"{test['average_final_accuracy']['sample_std']:.3f}"
        )


if __name__ == "__main__":
    main()
