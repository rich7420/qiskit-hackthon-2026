"""E013: learn task-boundary product measurements for Fisher consolidation.

The E008 classifier and Task-1 trajectory remain fixed.  At the Task-1 boundary this
experiment caches the two-readout-qubit reduced density matrices at the optimum and at
every classifier parameter-shift point.  It then learns local Bloch measurement axes
classically, optionally alternating with a certified shot-allocation solve.  Prediction
continues to use the original two-Z softmax readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import e008_measqcl as parent  # noqa: E402
from experiments import e010_physmeas_qcl as task_parent  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.measqcl_fisher import (  # noqa: E402
    fisher_cosine_similarity,
    normalize_fisher_mass,
)
from src.measqcl_learnable import (  # noqa: E402
    cache_parameter_shift_density_matrices,
    canonical_product_axes,
    optimize_learnable_measurements,
)
from src.measqcl_model import (  # noqa: E402
    make_classifier_qnode,
    make_reduced_state_qnode,
)

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
METHODS = (
    "info_learn_basis_alloc",
    "task_learn_basis_uniform",
    "task_learn_basis_alloc",
)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "src/measqcl_task_relevance.py",
        ROOT / "src/measqcl_learnable.py",
        ROOT / "experiments/e008_measqcl.py",
        ROOT / "experiments/e010_physmeas_qcl.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(name: str, seed: int) -> tuple[dict[str, Any], Path]:
    path = RESULTS / f"{name}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing paired parent artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _validate_task_parent(
    extension: dict[str, Any],
    *,
    seed: int,
    data_sha256: str,
    boundary_weights,
) -> None:
    if extension.get("experiment") != "e010_task_relevant_measqcl_exact":
        raise ValueError("task-relevance parent is not an E010 exact run")
    if extension.get("source_code_sha256") != task_parent._source_digest():
        raise ValueError("task-relevance parent does not match current E010 sources")
    if extension["training"]["seed"] != seed or extension["data_sha256"] != data_sha256:
        raise ValueError("task-relevance parent seed/data split does not match E013")
    np.testing.assert_allclose(
        np.asarray(extension["training"]["boundary_weights"]),
        np.asarray(boundary_weights),
        atol=1e-12,
        rtol=1e-12,
    )


def _optimization_record(result) -> dict[str, Any]:
    return {
        "axes": result.axes.tolist(),
        "allocation": result.allocation.tolist(),
        "raw_basis_fisher": result.basis_fishers.tolist(),
        "raw_accessible_fisher": result.accessible_fisher.tolist(),
        "information_objective": result.information_objective,
        "diversity_penalty": result.diversity_penalty,
        "total_objective": result.objective,
        "outer_iterations": result.outer_iterations,
        "axis_iterations": result.axis_iterations,
        "objective_evaluations": result.objective_evaluations,
        "allocation_solver": result.allocation_solver,
        "allocation_optimality_gap": result.allocation_optimality_gap,
        "allocation_optimality_tolerance": result.allocation_optimality_tolerance,
        "axis_solver_messages": list(result.axis_solver_messages),
        "axis_physical_gradient_norms": list(result.axis_physical_gradient_norms),
        "axis_stationarity_tolerance": result.axis_stationarity_tolerance,
        "history": list(result.history),
    }


def run_experiment(
    *,
    seed: int = 42,
    n_settings: int = 3,
    minimum_allocation: float = 0.01,
    relevance_floor: float = 1e-3,
    diversity_coefficient: float = 1e-3,
    outer_iterations: int = 100,
    axis_max_iterations: int = 3_000,
    initialization_noise: float = 1e-2,
    verbose: bool = True,
    parent_result: dict[str, Any] | None = None,
    parent_path: Path | None = None,
    task_result: dict[str, Any] | None = None,
    task_path: Path | None = None,
) -> dict[str, Any]:
    """Run a paired exact-statevector learnable-measurement comparison."""
    if parent_result is None:
        parent_result, parent_path = _load_result("e008_measqcl", seed)
    if task_result is None:
        task_result, task_path = _load_result("e010_physmeas", seed)
    data_config = parent_result["data"]
    model_config = parent_result["model"]
    training_config = parent_result["training"]
    n_qubits = int(model_config["n_qubits"])
    layers = int(model_config["layers"])
    readout_wires = tuple(int(wire) for wire in model_config["consolidation_readout_wires"])
    n_train = int(data_config["n_train_per_task"])
    n_test = int(data_config["n_test_per_task"])
    epochs = int(training_config["epochs_per_task"])
    learning_rate = float(training_config["learning_rate"])
    ewc_lambda = float(training_config["ewc_lambda_shared_by_all_consolidation_methods"])
    record_test = bool(training_config["record_test_history"])

    tasks = load_two_tasks(
        n_features=2**n_qubits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    data_sha256 = parent._digest_arrays(*(array for task in tasks for array in task[1:]))
    classifier_qnode, weight_shape = make_classifier_qnode(
        n_qubits,
        layers,
        readout_wires,
    )
    if verbose:
        print(
            f"E013 seed={seed}: replaying paired Task 1 and learning {n_settings} "
            "product-measurement settings",
            flush=True,
        )
    (
        initial_weights,
        boundary_weights,
        boundary_optimizer,
        common_history,
        phase1_time,
    ) = parent._train_first_task(
        qnode=classifier_qnode,
        weight_shape=weight_shape,
        tasks=tasks,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        record_test=record_test,
        verbose=verbose,
    )
    task_parent._validate_parent(
        parent_result,
        seed=seed,
        data_sha256=data_sha256,
        initial_weights=initial_weights,
        boundary_weights=boundary_weights,
        common_history=common_history,
    )
    _validate_task_parent(
        task_result,
        seed=seed,
        data_sha256=data_sha256,
        boundary_weights=boundary_weights,
    )

    anchor_indices = np.asarray(
        parent_result["fisher_profiles"]["anchor_indices"], dtype=int
    )
    task_relevance = np.asarray(
        task_result["measurement_design"]["raw_task_relevance"], dtype=float
    )
    density_qnode = make_reduced_state_qnode(
        n_qubits=n_qubits,
        n_layers=layers,
        readout_wires=readout_wires,
    )
    started = time.perf_counter()
    cache = cache_parameter_shift_density_matrices(
        density_qnode,
        boundary_weights,
        tasks[0].X_train,
        anchor_indices,
    )
    cache_time = time.perf_counter() - started
    initial_axes = canonical_product_axes(
        n_settings,
        len(readout_wires),
        seed=seed + 11_000,
        initialization_noise=initialization_noise,
    )

    optimization_started = time.perf_counter()
    info_result = optimize_learnable_measurements(
        cache,
        np.ones(cache.n_parameters),
        initial_axes=initial_axes,
        n_settings=n_settings,
        learn_allocation=True,
        minimum_allocation=minimum_allocation,
        relevance_floor=0.0,
        diversity_coefficient=diversity_coefficient,
        outer_iterations=outer_iterations,
        axis_max_iterations=axis_max_iterations,
        seed=seed,
    )
    task_uniform_result = optimize_learnable_measurements(
        cache,
        task_relevance,
        initial_axes=initial_axes,
        n_settings=n_settings,
        learn_allocation=False,
        minimum_allocation=minimum_allocation,
        relevance_floor=relevance_floor,
        diversity_coefficient=diversity_coefficient,
        outer_iterations=1,
        axis_max_iterations=axis_max_iterations,
        seed=seed,
    )
    task_alloc_result = optimize_learnable_measurements(
        cache,
        task_relevance,
        initial_axes=initial_axes,
        n_settings=n_settings,
        learn_allocation=True,
        minimum_allocation=minimum_allocation,
        relevance_floor=relevance_floor,
        diversity_coefficient=diversity_coefficient,
        outer_iterations=outer_iterations,
        axis_max_iterations=axis_max_iterations,
        seed=seed,
    )
    optimization_time = time.perf_counter() - optimization_started
    optimized = {
        "info_learn_basis_alloc": info_result,
        "task_learn_basis_uniform": task_uniform_result,
        "task_learn_basis_alloc": task_alloc_result,
    }
    importance = {
        method: normalize_fisher_mass(result.accessible_fisher)
        for method, result in optimized.items()
    }

    histories: dict[str, list[dict[str, Any]]] = {}
    final_weights: dict[str, list[float]] = {}
    phase2_times: dict[str, float] = {}
    for method in METHODS:
        history, weights, phase2_time = parent._train_second_task(
            method=method,
            qnode=classifier_qnode,
            boundary_weights=boundary_weights,
            boundary_optimizer=boundary_optimizer,
            common_history=common_history,
            importance=importance[method],
            tasks=tasks,
            ewc_lambda=ewc_lambda,
            epochs=epochs,
            record_test=record_test,
            verbose=verbose,
        )
        histories[method] = history
        final_weights[method] = weights.tolist()
        phase2_times[method] = phase2_time
    metrics = {
        method: parent._method_metrics(histories[method], epochs) for method in METHODS
    }
    full_qfi = np.asarray(
        parent_result["fisher_profiles"]["raw_method_fisher"]["qewc"], dtype=float
    )
    readout_qfi = np.asarray(
        parent_result["fisher_profiles"]["raw_method_fisher"]["readout_qewc"],
        dtype=float,
    )
    output_cfi = np.asarray(
        parent_result["fisher_profiles"]["raw_method_fisher"]["output_cfi"], dtype=float
    )

    def parent_reference(path: Path | None) -> dict[str, Any]:
        return {
            "result_file": (
                str(path.relative_to(ROOT))
                if path is not None and path.is_relative_to(ROOT)
                else None
            ),
            "result_file_sha256": (
                _file_digest(path) if path is not None and path.exists() else None
            ),
        }

    hierarchy = {}
    for method, result in optimized.items():
        minimum_basis_margin = float(np.min(readout_qfi[np.newaxis, :] - result.basis_fishers))
        minimum_accessible_margin = float(np.min(readout_qfi - result.accessible_fisher))
        if min(minimum_basis_margin, minimum_accessible_margin) < -1e-6:
            raise ValueError(
                f"{method} measurement CFI exceeds its readout-subsystem QFI bound"
            )
        hierarchy[method] = {
            "minimum_readout_qfi_minus_basis_cfi": minimum_basis_margin,
            "minimum_readout_qfi_minus_accessible_cfi": minimum_accessible_margin,
            "tolerance": 1e-6,
            "passed": True,
        }

    cache_circuit_configurations = cache.n_anchors * (1 + 2 * cache.n_parameters)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e013_learnable_measurement_fisher_exact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": data_sha256,
        "parents": {
            "e008": {
                **parent_reference(parent_path),
                "source_code_sha256": parent_result["source_code_sha256"],
            },
            "e010": {
                **parent_reference(task_path),
                "source_code_sha256": task_result["source_code_sha256"],
            },
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in ("autograd", "numpy", "pennylane", "scipy")
            },
        },
        "data": {
            "task_order": [task.name for task in tasks],
            "n_train_per_task": n_train,
            "n_test_per_task": n_test,
            "test_used_for_selection": False,
        },
        "model": {
            **model_config,
            "learned_measurement_class": (
                "local fixed-spectrum projective axes on each readout qubit, with "
                "joint bitstring outcomes"
            ),
            "n_learned_settings": n_settings,
            "prediction_measurement_unchanged": True,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs_per_task": epochs,
            "ewc_lambda_shared": ewc_lambda,
            "seed": seed,
            "phase1_replayed_and_verified_against_parents": True,
            "boundary_adam_state_sha256": parent._optimizer_state_digest(
                boundary_optimizer
            ),
            "boundary_weights": np.asarray(boundary_weights).tolist(),
            "record_test_history": record_test,
            "selection_policy": (
                "inherits E008 train-only classifier/lambda and E010 EWC-DR relevance; "
                "measurement hyperparameters are prespecified for the seed-42 development run"
            ),
        },
        "measurement_design": {
            "fisher_parameter": "classifier weights theta, not measurement axes phi",
            "task_relevance_source": "paired E010 EWC-DR boundary importance",
            "objective_variants": {
                "info_learn_basis_alloc": "uniform relevance; learned axes and allocation",
                "task_learn_basis_uniform": "EWC-DR relevance; learned axes; uniform allocation",
                "task_learn_basis_alloc": "EWC-DR relevance; learned axes and allocation",
            },
            "initialization": "Z/X/Y product axes plus deterministic small noise",
            "initialization_noise": initialization_noise,
            "initial_axes": initial_axes.tolist(),
            "minimum_allocation": minimum_allocation,
            "relevance_floor": relevance_floor,
            "diversity_coefficient": diversity_coefficient,
            "outer_iterations": outer_iterations,
            "axis_max_iterations": axis_max_iterations,
            "optimizer": (
                "analytic-gradient local tangent trust regions with sphere retraction and "
                "physical-gradient stopping; certified convex q solve alternated afterward"
            ),
            "optimization": {
                method: _optimization_record(result) for method, result in optimized.items()
            },
            "normalized_method_fisher": {
                method: values.tolist() for method, values in importance.items()
            },
            "diagnostics": {
                method: {
                    "cosine_to_output_cfi": fisher_cosine_similarity(
                        result.accessible_fisher, output_cfi
                    ),
                    "cosine_to_readout_qfi": fisher_cosine_similarity(
                        result.accessible_fisher, readout_qfi
                    ),
                    "cosine_to_full_qfi": fisher_cosine_similarity(
                        result.accessible_fisher, full_qfi
                    ),
                    "information_hierarchy": hierarchy[method],
                }
                for method, result in optimized.items()
            },
        },
        "resource_accounting": {
            "cached_quantum_circuit_configurations": cache_circuit_configurations,
            "formula": "n_anchors * (1 + 2 * n_classifier_parameters)",
            "n_anchors": cache.n_anchors,
            "n_classifier_parameters": cache.n_parameters,
            "basis_independent_cache_reused_by_every_objective_evaluation": True,
            "axis_optimization_quantum_circuit_configurations": 0,
            "finite_shots": False,
            "hardware_execution": False,
        },
        "histories": histories,
        "metrics": metrics,
        "parent_metrics": {
            "e008": parent_result["metrics"],
            "e010": task_result["metrics"],
        },
        "final_weights": final_weights,
        "runtime_sec": {
            "replayed_phase1_training": round(phase1_time, 4),
            "density_cache": round(cache_time, 4),
            "classical_measurement_optimization": round(optimization_time, 4),
            "phase2_training": {
                method: round(value, 4) for method, value in phase2_times.items()
            },
        },
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "global_or_entangled_povm": False,
            "qfi_attainability": False,
            "quantum_advantage": False,
            "scope": (
                "exact-statevector two-readout-qubit product-measurement basis learning"
            ),
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e013_learnable_measqcl_seed{result['training']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-settings", type=int, default=3)
    parser.add_argument("--minimum-allocation", type=float, default=0.01)
    parser.add_argument("--relevance-floor", type=float, default=1e-3)
    parser.add_argument("--diversity-coefficient", type=float, default=1e-3)
    parser.add_argument("--outer-iterations", type=int, default=100)
    parser.add_argument("--axis-max-iterations", type=int, default=3_000)
    parser.add_argument("--initialization-noise", type=float, default=1e-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        seed=args.seed,
        n_settings=args.n_settings,
        minimum_allocation=args.minimum_allocation,
        relevance_floor=args.relevance_floor,
        diversity_coefficient=args.diversity_coefficient,
        outer_iterations=args.outer_iterations,
        axis_max_iterations=args.axis_max_iterations,
        initialization_noise=args.initialization_noise,
    )
    print("\n=== E013 learnable-measurement comparison ===")
    for method in METHODS:
        metric = result["metrics"][method]["test"]
        print(
            f"  {method:26s} retention={metric['task1_final_retention']:.3f} "
            f"adaptation={metric['task2_final_adaptation']:.3f} "
            f"forgetting={metric['task1_forgetting']:.3f}"
        )
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
