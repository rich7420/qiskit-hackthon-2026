"""E013 phase-first learnable full-output product measurements.

The four-qubit SPT/ATF task is learned once and verified against E010.  At that
boundary, three product-measurement settings learn one local Bloch axis per output
qubit while retaining all 16 joint outcomes.  The learned Fisher profile protects the
phase boundary while the classifier subsequently learns MNIST and Fashion-MNIST.
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

from experiments import e010_phase_locality as parent  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.measqcl_fisher import (  # noqa: E402
    fisher_cosine_similarity,
    normalize_fisher_mass,
)
from src.measqcl_learnable import (  # noqa: E402
    basis_fisher_diags_from_cache,
    cache_parameter_shift_density_matrices,
    canonical_product_axes,
    optimize_learnable_measurements,
)
from src.measqcl_model import (  # noqa: E402
    make_classifier_qnode,
    make_reduced_state_qnode,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
METHODS = (
    "joint_zzzz",
    "uniform_xyz_joint",
    "info_learn_basis_alloc",
    "task_learn_basis_uniform",
    "task_learn_basis_alloc",
)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "src/measqcl_task_relevance.py",
        ROOT / "src/measqcl_learnable.py",
        ROOT / "src/physmeas_observables.py",
        ROOT / "experiments/e010_phase_locality.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_parent(seed: int) -> tuple[dict[str, Any], Path]:
    path = RESULTS / f"e010_phase_locality_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing paired E010 phase artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _validate_parent(
    result: dict[str, Any],
    *,
    seed: int,
    data_sha256: str,
    initial_weights: np.ndarray,
    boundary_weights,
    common_history: list[dict[str, Any]],
) -> None:
    if result.get("experiment") != "e010_phase_first_locality_resolved_exact":
        raise ValueError("parent artifact is not an E010 phase-first exact run")
    if result.get("source_code_sha256") != parent._source_digest():
        raise ValueError("parent artifact does not match current E010 phase sources")
    if result["training"]["seed"] != seed or result["data_sha256"] != data_sha256:
        raise ValueError("parent seed/data split does not match E013 phase extension")
    np.testing.assert_allclose(
        np.asarray(result["training"]["initial_weights"]),
        initial_weights,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result["training"]["phase_boundary_weights"]),
        np.asarray(boundary_weights),
        atol=1e-12,
        rtol=1e-12,
    )
    phase_length = int(result["training"]["epochs_per_task"]) + 1
    reference = result["histories"][result["training"]["methods"][0]][:phase_length]
    if common_history != reference:
        raise ValueError("replayed phase history differs from paired E010 trajectory")
    if any(
        result["histories"][method][:phase_length] != reference
        for method in result["training"]["methods"][1:]
    ):
        raise ValueError("E010 phase methods do not share one verified boundary")


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
        "axis_solver_messages": list(result.axis_solver_messages),
        "axis_physical_gradient_norms": list(result.axis_physical_gradient_norms),
        "axis_stationarity_tolerance": result.axis_stationarity_tolerance,
        "allocation_solver": result.allocation_solver,
        "allocation_optimality_gap": result.allocation_optimality_gap,
        "allocation_optimality_tolerance": result.allocation_optimality_tolerance,
        "history": list(result.history),
    }


def run_experiment(
    *,
    n_settings: int = 3,
    minimum_allocation: float = 0.01,
    relevance_floor: float = 1e-3,
    diversity_coefficient: float = 1e-3,
    outer_iterations: int = 100,
    axis_max_iterations: int = 3_000,
    initialization_noise: float = 1e-2,
    seed: int = 42,
    verbose: bool = True,
    parent_result: dict[str, Any] | None = None,
    parent_path: Path | None = None,
) -> dict[str, Any]:
    """Run one paired phase-first learned-product-measurement comparison."""
    if parent_result is None:
        parent_result, parent_path = _load_parent(seed)
    model_config = parent_result["model"]
    training_config = parent_result["training"]
    data_config = parent_result["data"]
    layers = int(model_config["layers"])
    learning_rate = float(training_config["learning_rate"])
    epochs = int(training_config["epochs_per_task"])
    ewc_lambda = float(training_config["ewc_lambda"])
    n_train = int(data_config["n_train_per_task"])
    n_test = int(data_config["n_test_per_task"])
    record_test = bool(training_config["record_test_history"])

    phase_task = load_spt_atf(n_train=n_train, n_test=n_test, seed=seed)
    image_tasks = load_two_tasks(
        n_features=2**N_QUBITS,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    tasks = (phase_task, *image_tasks)
    data_sha256 = parent._digest_arrays(*(array for task in tasks for array in task[1:]))
    classifier_qnode, weight_shape = make_classifier_qnode(N_QUBITS, layers)
    if verbose:
        print(
            f"E013 phase seed={seed}: replaying one verified phase boundary and "
            f"learning {n_settings} full-output product settings",
            flush=True,
        )
    initial, boundary, boundary_optimizer, common_history = parent._train_phase(
        qnode=classifier_qnode,
        weight_shape=weight_shape,
        tasks=tasks,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        record_test=record_test,
        verbose=verbose,
    )
    _validate_parent(
        parent_result,
        seed=seed,
        data_sha256=data_sha256,
        initial_weights=initial,
        boundary_weights=boundary,
        common_history=common_history,
    )

    analysis = parent_result["locality_analysis"]
    anchor_indices = np.asarray(analysis["anchor_indices"], dtype=int)
    task_relevance = np.asarray(
        analysis["raw_method_fisher"]["ewc_dr"], dtype=float
    )
    full_qfi = np.asarray(analysis["raw_method_fisher"]["qewc"], dtype=float)
    density_qnode = make_reduced_state_qnode(
        n_qubits=N_QUBITS,
        n_layers=layers,
        readout_wires=tuple(range(N_QUBITS)),
    )
    cache_started = time.perf_counter()
    cache = cache_parameter_shift_density_matrices(
        density_qnode,
        boundary,
        phase_task.X_train,
        anchor_indices,
    )
    cache_time = time.perf_counter() - cache_started

    fixed_axes = canonical_product_axes(
        3,
        N_QUBITS,
        seed=0,
        initialization_noise=0.0,
    )
    fixed_fishers = basis_fisher_diags_from_cache(cache, fixed_axes)
    raw: dict[str, np.ndarray] = {
        "joint_zzzz": fixed_fishers[0],
        "uniform_xyz_joint": np.mean(fixed_fishers, axis=0),
    }
    initial_axes = canonical_product_axes(
        n_settings,
        N_QUBITS,
        seed=seed + 13_000,
        initialization_noise=initialization_noise,
    )
    optimization_started = time.perf_counter()
    optimized = {
        "info_learn_basis_alloc": optimize_learnable_measurements(
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
        ),
        "task_learn_basis_uniform": optimize_learnable_measurements(
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
        ),
        "task_learn_basis_alloc": optimize_learnable_measurements(
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
        ),
    }
    optimization_time = time.perf_counter() - optimization_started
    for method, result in optimized.items():
        raw[method] = result.accessible_fisher

    hierarchy: dict[str, Any] = {}
    candidate_profiles = {
        "joint_zzzz": fixed_fishers[0:1],
        "fixed_xxxx": fixed_fishers[1:2],
        "fixed_yyyy": fixed_fishers[2:3],
        **{method: result.basis_fishers for method, result in optimized.items()},
    }
    for name, profiles in candidate_profiles.items():
        margin = float(np.min(full_qfi[np.newaxis, :] - profiles))
        if margin < -2e-6:
            raise ValueError(f"{name} product CFI exceeds full-state QFI")
        hierarchy[name] = {
            "minimum_full_qfi_minus_basis_cfi": margin,
            "tolerance": 2e-6,
            "passed": True,
        }
    for method, fisher in raw.items():
        margin = float(np.min(full_qfi - fisher))
        if margin < -2e-6:
            raise ValueError(f"{method} accessible CFI exceeds full-state QFI")
        hierarchy.setdefault(method, {"tolerance": 2e-6, "passed": True})[
            "minimum_full_qfi_minus_accessible_cfi"
        ] = margin

    importance = {
        method: normalize_fisher_mass(fisher) for method, fisher in raw.items()
    }
    histories: dict[str, list[dict[str, Any]]] = {}
    final_weights: dict[str, list[float]] = {}
    future_times: dict[str, float] = {}
    for method in METHODS:
        started = time.perf_counter()
        history, weights = parent._train_future_tasks(
            method=method,
            qnode=classifier_qnode,
            boundary_weights=boundary,
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
        future_times[method] = time.perf_counter() - started

    parent_reference = (
        str(parent_path.relative_to(ROOT))
        if parent_path is not None and parent_path.is_relative_to(ROOT)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e013_phase_first_learnable_full_output_exact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": data_sha256,
        "parent": {
            "experiment": parent_result["experiment"],
            "result_file": parent_reference,
            "result_file_sha256": (
                _file_digest(parent_path)
                if parent_path is not None and parent_path.exists()
                else None
            ),
            "source_code_sha256": parent_result["source_code_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in ("autograd", "numpy", "pennylane", "scipy")
            },
        },
        "data": {
            **data_config,
            "test_used_for_selection": False,
        },
        "model": {
            **model_config,
            "learned_measurement_domain": "all four output qubits",
            "learned_measurement_class": (
                "local fixed-spectrum projective axes with all 16 joint outcomes"
            ),
            "n_learned_settings": n_settings,
            "prediction_measurement_unchanged": True,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs_per_task": epochs,
            "ewc_lambda": ewc_lambda,
            "seed": seed,
            "record_test_history": record_test,
            "phase_boundary_weights": np.asarray(boundary).tolist(),
            "phase_replayed_and_verified_against_parent": True,
            "phase_anchor_strength_constant_across_both_future_tasks": True,
            "selection_policy": (
                "inherits E010 train-only phase capacity and shared lambda; inherits "
                "prespecified E013 measurement optimizer without phase-test tuning"
            ),
        },
        "measurement_design": {
            "fisher_parameter": "classifier weights theta, not measurement axes phi",
            "outcomes_per_setting": 2**N_QUBITS,
            "fixed_axes_order": ["ZZZZ", "XXXX", "YYYY"],
            "raw_fixed_basis_fisher": fixed_fishers.tolist(),
            "task_relevance_source": "paired E010 phase-boundary EWC-DR importance",
            "initial_axes": initial_axes.tolist(),
            "minimum_allocation": minimum_allocation,
            "relevance_floor": relevance_floor,
            "diversity_coefficient": diversity_coefficient,
            "outer_iterations": outer_iterations,
            "axis_max_iterations": axis_max_iterations,
            "initialization_noise": initialization_noise,
            "optimizer": (
                "analytic-gradient local tangent trust regions with sphere retraction and "
                "physical-gradient stopping; certified convex q solve alternated afterward"
            ),
            "optimization": {
                method: _optimization_record(result)
                for method, result in optimized.items()
            },
            "raw_method_fisher": {
                method: fisher.tolist() for method, fisher in raw.items()
            },
            "normalized_method_fisher": {
                method: fisher.tolist() for method, fisher in importance.items()
            },
            "diagnostics": {
                method: {
                    "cosine_to_full_qfi": fisher_cosine_similarity(fisher, full_qfi),
                    "information_hierarchy": hierarchy[method],
                }
                for method, fisher in raw.items()
            },
            "all_basis_hierarchy": hierarchy,
            "anchor_indices": anchor_indices.tolist(),
        },
        "resource_accounting": {
            "cached_quantum_circuit_configurations": (
                cache.n_anchors * (1 + 2 * cache.n_parameters)
            ),
            "formula": "n_anchors * (1 + 2 * n_classifier_parameters)",
            "n_anchors": cache.n_anchors,
            "n_classifier_parameters": cache.n_parameters,
            "basis_independent_cache_reused": True,
            "axis_optimization_quantum_circuit_configurations": 0,
            "finite_shots": False,
            "hardware_execution": False,
        },
        "histories": histories,
        "metrics": {
            method: parent._metrics(history, epochs)
            for method, history in histories.items()
        },
        "parent_metrics": parent_result["metrics"],
        "final_weights": final_weights,
        "runtime_sec": {
            "density_cache": round(cache_time, 4),
            "classical_measurement_optimization": round(optimization_time, 4),
            "future_task_training": {
                method: round(value, 4) for method, value in future_times.items()
            },
        },
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "entangled_or_global_povm": False,
            "thermodynamic_phase_transition": False,
            "input_locality_equivalence": False,
            "qfi_attainability": False,
            "quantum_advantage": False,
            "scope": "exact-statevector four-qubit joint product-measurement diagnostic",
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e013_phase_learnable_seed{result['training']['seed']}.json"
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
    print("\n=== E013 phase-memory comparison ===")
    for method in METHODS:
        metric = result["metrics"][method]["test"]
        print(
            f"  {method:26s} phase={metric['phase_final_retention']:.3f} "
            f"MNIST={metric['mnist_final_adaptation']:.3f} "
            f"Fashion={metric['fashion_final_adaptation']:.3f}"
        )
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
