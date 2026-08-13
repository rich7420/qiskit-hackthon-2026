"""E008 MeasQCL: measurement-optimized Fisher consolidation.

This exact-statevector MVP trains MNIST 0/1 followed by Fashion-MNIST 0/1. All methods
share the same Task-1 trajectory, boundary checkpoint, Adam state, Task-2 data, and EWC
strength. They differ only in the parameter-importance diagonal consolidated at the task
boundary.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import Task, load_two_tasks  # noqa: E402
from src.e005_consolidation import EWC, quantum_fisher_diag  # noqa: E402
from src.e005_softmax import (  # noqa: E402
    accuracy as classifier_accuracy,
    bce_loss,
    classical_fisher_diag,
)
from src.measqcl_fisher import (  # noqa: E402
    accessible_fisher_diag,
    allocate_integer_shots,
    fisher_cosine_similarity,
    measurement_fisher_diag,
    normalize_fisher_mass,
    optimize_measurement_allocation,
    parameter_shift_evaluation_budget,
    reduced_state_qfi_diag,
    select_anchor_indices,
)
from src.measqcl_model import (  # noqa: E402
    DEFAULT_BASES,
    make_classifier_qnode,
    make_measurement_qnode,
    make_qfi_qnode,
    make_reduced_state_qnode,
)

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2")
METHODS = (
    "naive",
    "output_cfi",
    "zz_cfi",
    "uniform_xyz",
    "mof_ewc",
    "readout_qewc",
    "qewc",
)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "src/measqcl_fisher.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("pennylane", "numpy", "scipy", "scikit-learn")
    }


def _optimizer_state_digest(optimizer: qml.AdamOptimizer) -> str:
    digest = hashlib.sha256()
    accumulation = optimizer.accumulation
    if accumulation is None:
        return digest.hexdigest()
    digest.update(str(accumulation["t"]).encode())
    for key in ("fm", "sm"):
        for array in accumulation[key]:
            contiguous = np.ascontiguousarray(np.asarray(array, dtype=float))
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _snapshot(
    history: list[dict[str, Any]],
    *,
    epoch: int,
    phase: int,
    qnode,
    weights,
    tasks: tuple[Task, Task],
    task_loss: float | None,
    penalty: float | None,
    objective: float | None,
    record_test: bool,
) -> None:
    train_accuracy = {
        key: classifier_accuracy(qnode, weights, task.X_train, task.y_train)
        for key, task in zip(TASK_KEYS, tasks, strict=True)
    }
    test_accuracy = (
        {
            key: classifier_accuracy(qnode, weights, task.X_test, task.y_test)
            for key, task in zip(TASK_KEYS, tasks, strict=True)
        }
        if record_test
        else None
    )
    history.append(
        {
            "epoch": epoch,
            "phase": phase,
            "task_loss": task_loss,
            "consolidation_penalty": penalty,
            "objective": objective,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    )


def _train_first_task(
    *,
    qnode,
    weight_shape: tuple[int, ...],
    tasks: tuple[Task, Task],
    learning_rate: float,
    epochs: int,
    seed: int,
    record_test: bool,
    verbose: bool,
) -> tuple[np.ndarray, pnp.ndarray, qml.AdamOptimizer, list[dict[str, Any]], float]:
    initial_weights = 0.05 * np.random.default_rng(seed).standard_normal(weight_shape)
    weights = pnp.array(
        initial_weights,
        requires_grad=True,
    )
    optimizer = qml.AdamOptimizer(learning_rate)
    task = tasks[0]
    X_train = pnp.array(task.X_train, requires_grad=False)
    y_train = pnp.array(task.y_train, requires_grad=False)
    history: list[dict[str, Any]] = []
    _snapshot(
        history,
        epoch=0,
        phase=0,
        qnode=qnode,
        weights=weights,
        tasks=tasks,
        task_loss=None,
        penalty=None,
        objective=None,
        record_test=record_test,
    )

    def cost(candidate):
        return bce_loss(qnode, candidate, X_train, y_train)

    started = time.perf_counter()
    for local_epoch in range(1, epochs + 1):
        weights = optimizer.step(cost, weights)
        loss = float(cost(weights))
        _snapshot(
            history,
            epoch=local_epoch,
            phase=1,
            qnode=qnode,
            weights=weights,
            tasks=tasks,
            task_loss=loss,
            penalty=0.0,
            objective=loss,
            record_test=record_test,
        )
        if verbose and (local_epoch == 1 or local_epoch % 5 == 0):
            print(
                f"  shared T1 epoch {local_epoch:2d}: "
                f"train={history[-1]['train_accuracy']['task1']:.3f}",
                flush=True,
            )
    return initial_weights, weights, optimizer, history, time.perf_counter() - started


def _estimate_importance(
    *,
    classifier_qnode,
    qfi_qnode,
    reduced_state_qnode,
    measurement_qnodes: dict[str, Any],
    weights,
    task: Task,
    anchor_indices: np.ndarray,
    reference_shots: int,
    verbose: bool,
) -> tuple[dict[str, np.ndarray | None], dict[str, Any], dict[str, float]]:
    raw: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    started = time.perf_counter()
    raw["output_cfi"] = classical_fisher_diag(
        classifier_qnode,
        weights,
        task.X_train[anchor_indices],
        task.y_train[anchor_indices],
    )
    timings["output_cfi"] = time.perf_counter() - started

    basis_fishers: dict[str, np.ndarray] = {}
    basis_timings: dict[str, float] = {}
    for basis, qnode in measurement_qnodes.items():
        started = time.perf_counter()
        basis_fishers[basis] = measurement_fisher_diag(
            qnode,
            weights,
            task.X_train,
            anchor_indices,
        )
        basis_timings[basis] = time.perf_counter() - started
        if verbose:
            print(
                f"  {basis} CFI raw mass={np.sum(basis_fishers[basis]):.4f}",
                flush=True,
            )

    uniform = {basis: 1.0 / len(basis_fishers) for basis in basis_fishers}
    fixed_zz = {basis: float(basis == "ZZ") for basis in basis_fishers}
    started = time.perf_counter()
    optimized_result = optimize_measurement_allocation(basis_fishers)
    optimization_time = time.perf_counter() - started
    optimized = dict(
        zip(optimized_result.bases, optimized_result.weights.tolist(), strict=True)
    )

    raw["zz_cfi"] = accessible_fisher_diag(basis_fishers, fixed_zz)
    raw["uniform_xyz"] = accessible_fisher_diag(basis_fishers, uniform)
    raw["mof_ewc"] = accessible_fisher_diag(basis_fishers, optimized)

    started = time.perf_counter()
    raw["readout_qewc"] = reduced_state_qfi_diag(
        reduced_state_qnode,
        weights,
        task.X_train,
        anchor_indices,
    )
    timings["readout_qewc"] = time.perf_counter() - started

    started = time.perf_counter()
    raw["qewc"] = quantum_fisher_diag(
        qfi_qnode,
        weights,
        task.X_train[anchor_indices],
        n_samples=len(anchor_indices),
        seed=0,
    )
    timings["qewc"] = time.perf_counter() - started
    timings["zz_cfi"] = basis_timings["ZZ"]
    timings["uniform_xyz"] = sum(basis_timings.values())
    timings["mof_ewc"] = sum(basis_timings.values()) + optimization_time
    timings["naive"] = 0.0

    normalized: dict[str, np.ndarray | None] = {"naive": None}
    for method in METHODS[1:]:
        normalized[method] = normalize_fisher_mass(raw[method])

    qfi = raw["qewc"]
    readout_qfi = raw["readout_qewc"]
    hierarchy_tolerance = 1e-6
    for basis, fisher in basis_fishers.items():
        if np.any(fisher > readout_qfi + hierarchy_tolerance):
            raise ValueError(f"{basis} CFI exceeds the readout-subsystem QFI")
    if np.any(readout_qfi > qfi + hierarchy_tolerance):
        raise ValueError("readout-subsystem QFI exceeds the full-state QFI")
    n_parameters = int(np.asarray(weights).size)
    per_setting_budget = parameter_shift_evaluation_budget(
        n_parameters,
        len(anchor_indices),
        1,
    )
    profiles = {
        "anchor_indices": anchor_indices.tolist(),
        "n_anchor_samples": len(anchor_indices),
        "raw_basis_fisher": {
            basis: values.tolist() for basis, values in basis_fishers.items()
        },
        "raw_basis_mass": {
            basis: float(np.sum(values)) for basis, values in basis_fishers.items()
        },
        "allocations": {
            "zz_cfi": fixed_zz,
            "uniform_xyz": uniform,
            "mof_ewc": optimized,
        },
        "reference_shots": reference_shots,
        "integer_shot_plans": {
            name: allocate_integer_shots(allocation, reference_shots)
            for name, allocation in (
                ("zz_cfi", fixed_zz),
                ("uniform_xyz", uniform),
                ("mof_ewc", optimized),
            )
        },
        "allocation_objective": optimized_result.objective,
        "allocation_iterations": optimized_result.iterations,
        "raw_method_fisher": {
            method: raw[method].tolist() for method in METHODS if method != "naive"
        },
        "normalized_method_fisher": {
            method: normalized[method].tolist()
            for method in METHODS
            if method != "naive"
        },
        "raw_method_mass": {
            method: float(np.sum(raw[method]))
            for method in METHODS
            if method != "naive"
        },
        "cosine_similarity_to_qfi": {
            method: fisher_cosine_similarity(raw[method], qfi)
            for method in METHODS[1:-1]
        },
        "cosine_similarity_to_readout_qfi": {
            method: fisher_cosine_similarity(raw[method], readout_qfi)
            for method in ("output_cfi", "zz_cfi", "uniform_xyz", "mof_ewc")
        },
        "cosine_similarity_to_output_cfi": {
            method: fisher_cosine_similarity(raw[method], raw["output_cfi"])
            for method in (
                "zz_cfi",
                "uniform_xyz",
                "mof_ewc",
                "readout_qewc",
                "qewc",
            )
        },
        "global_qfi_trace_coverage_proxy": {
            method: float(np.sum(raw[method]) / np.sum(qfi))
            for method in ("zz_cfi", "uniform_xyz", "mof_ewc")
        },
        "readout_qfi_trace_coverage_proxy": {
            method: float(np.sum(raw[method]) / np.sum(readout_qfi))
            for method in ("zz_cfi", "uniform_xyz", "mof_ewc")
        },
        "geometry_hierarchy_min_margin": {
            "readout_qfi_minus_each_basis_cfi": {
                basis: float(np.min(readout_qfi - fisher))
                for basis, fisher in basis_fishers.items()
            },
            "full_qfi_minus_readout_qfi": float(np.min(qfi - readout_qfi)),
        },
        "optimization_target": (
            "task-agnostic sum of log accessible diagonal CFI; higher coverage is not "
            "assumed to imply higher task retention"
        ),
        "logical_parameter_shift_probability_evaluations": {
            "zz_cfi": per_setting_budget,
            "uniform_xyz": 3 * per_setting_budget,
            "mof_ewc": 3 * per_setting_budget,
            "readout_qewc_density_matrix_reference": per_setting_budget,
        },
        "exact_estimation_only": True,
        "finite_shot_estimation_performed": False,
    }
    return normalized, profiles, timings


def _train_second_task(
    *,
    method: str,
    qnode,
    boundary_weights,
    boundary_optimizer: qml.AdamOptimizer,
    common_history: list[dict[str, Any]],
    importance: np.ndarray | None,
    tasks: tuple[Task, Task],
    ewc_lambda: float,
    epochs: int,
    record_test: bool,
    verbose: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    weights = pnp.array(np.asarray(boundary_weights).copy(), requires_grad=True)
    optimizer = copy.deepcopy(boundary_optimizer)
    history = copy.deepcopy(common_history)
    regularizer = EWC(0.0 if method == "naive" else ewc_lambda)
    if importance is not None:
        regularizer.consolidate(np.asarray(boundary_weights).reshape(-1), importance)
    task = tasks[1]
    X_train = pnp.array(task.X_train, requires_grad=False)
    y_train = pnp.array(task.y_train, requires_grad=False)

    def components(candidate):
        task_loss = bce_loss(qnode, candidate, X_train, y_train)
        penalty = regularizer.penalty(candidate.flatten(), task_index=2)
        return task_loss, penalty, task_loss + penalty

    def cost(candidate):
        return components(candidate)[2]

    started = time.perf_counter()
    for local_epoch in range(1, epochs + 1):
        weights = optimizer.step(cost, weights)
        task_loss, penalty, objective = components(weights)
        epoch = common_history[-1]["epoch"] + local_epoch
        _snapshot(
            history,
            epoch=epoch,
            phase=2,
            qnode=qnode,
            weights=weights,
            tasks=tasks,
            task_loss=float(task_loss),
            penalty=float(penalty),
            objective=float(objective),
            record_test=record_test,
        )
        if verbose and (local_epoch == 1 or local_epoch % 5 == 0):
            accuracy = history[-1]["train_accuracy"]
            print(
                f"    {method:11s} epoch {epoch:2d}: "
                f"T1={accuracy['task1']:.3f} T2={accuracy['task2']:.3f}",
                flush=True,
            )
    return history, np.asarray(weights), time.perf_counter() - started


def _method_metrics(
    history: list[dict[str, Any]],
    boundary_epoch: int,
) -> dict[str, Any]:
    boundary = history[boundary_epoch]
    final = history[-1]
    metrics: dict[str, Any] = {}
    for split in ("train", "test"):
        field = f"{split}_accuracy"
        if final[field] is None:
            continue
        old_boundary = boundary[field]["task1"]
        old_final = final[field]["task1"]
        new_final = final[field]["task2"]
        metrics[split] = {
            "task1_at_boundary": round(old_boundary, 4),
            "task1_final_retention": round(old_final, 4),
            "task1_forgetting": round(old_boundary - old_final, 4),
            "backward_transfer": round(old_final - old_boundary, 4),
            "task2_final_adaptation": round(new_final, 4),
            "average_final_accuracy": round((old_final + new_final) / 2.0, 4),
        }
    return metrics


def run_experiment(
    *,
    n_qubits: int = 6,
    layers: int = 10,
    learning_rate: float = 0.02,
    epochs_per_task: int = 40,
    ewc_lambda: float = 0.1,
    fisher_samples: int = 32,
    reference_shots: int = 1024,
    n_train: int = 400,
    n_test: int = 200,
    seed: int = 42,
    record_test: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the paired exact-statevector MeasQCL comparison for one seed."""
    if min(
        n_qubits,
        layers,
        epochs_per_task,
        fisher_samples,
        reference_shots,
        n_train,
        n_test,
    ) <= 0:
        raise ValueError("model, data, epoch, Fisher, and shot sizes must be positive")
    if learning_rate <= 0.0 or ewc_lambda < 0.0:
        raise ValueError("learning_rate must be positive and ewc_lambda non-negative")

    tasks = load_two_tasks(
        n_features=2**n_qubits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    classifier_qnode, weight_shape = make_classifier_qnode(n_qubits, layers)
    measurement_qnodes = {
        basis: make_measurement_qnode(basis, n_qubits, layers)
        for basis in DEFAULT_BASES
    }
    qfi_qnode = make_qfi_qnode(n_qubits, layers)
    reduced_state_qnode = make_reduced_state_qnode(n_qubits, layers)

    if verbose:
        print(
            f"MeasQCL seed={seed}: {tasks[0].name} -> {tasks[1].name}; "
            f"{n_qubits} qubits, {layers} layers, lambda={ewc_lambda}",
            flush=True,
        )
    (
        initial_weights,
        boundary_weights,
        boundary_optimizer,
        common_history,
        phase1_time,
    ) = _train_first_task(
        qnode=classifier_qnode,
        weight_shape=weight_shape,
        tasks=tasks,
        learning_rate=learning_rate,
        epochs=epochs_per_task,
        seed=seed,
        record_test=record_test,
        verbose=verbose,
    )
    anchor_indices = select_anchor_indices(
        len(tasks[0].X_train),
        fisher_samples,
        seed + 10_000,
    )
    importance, profiles, fisher_timings = _estimate_importance(
        classifier_qnode=classifier_qnode,
        qfi_qnode=qfi_qnode,
        reduced_state_qnode=reduced_state_qnode,
        measurement_qnodes=measurement_qnodes,
        weights=boundary_weights,
        task=tasks[0],
        anchor_indices=anchor_indices,
        reference_shots=reference_shots,
        verbose=verbose,
    )

    histories: dict[str, list[dict[str, Any]]] = {}
    final_weights: dict[str, list[float]] = {}
    phase2_times: dict[str, float] = {}
    for method in METHODS:
        if verbose:
            print(f"  training Task 2 with {method}", flush=True)
        history, method_weights, phase2_time = _train_second_task(
            method=method,
            qnode=classifier_qnode,
            boundary_weights=boundary_weights,
            boundary_optimizer=boundary_optimizer,
            common_history=common_history,
            importance=importance[method],
            tasks=tasks,
            ewc_lambda=ewc_lambda,
            epochs=epochs_per_task,
            record_test=record_test,
            verbose=verbose,
        )
        histories[method] = history
        final_weights[method] = method_weights.tolist()
        phase2_times[method] = phase2_time

    metrics = {
        method: _method_metrics(history, epochs_per_task)
        for method, history in histories.items()
    }
    classical_references = {}
    if record_test:
        for key, task in zip(TASK_KEYS, tasks, strict=True):
            model = LogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight="balanced",
            ).fit(task.X_train, task.y_train)
            classical_references[key] = round(
                float(np.mean(model.predict(task.X_test) == task.y_test)),
                4,
            )

    resources = qml.specs(classifier_qnode)(
        tasks[0].X_train[0],
        pnp.array(boundary_weights),
    )["resources"]
    gate_types = dict(resources.gate_types)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e008_measqcl_exact_mvp",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": _digest_arrays(
            *(array for task in tasks for array in task[1:])
        ),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "data": {
            "task_order": [task.name for task in tasks],
            "representation": (
                "one PCA transform fit on MNIST training data; 2^n features; "
                "per-sample L2 normalization for amplitude encoding"
            ),
            "n_train_per_task": n_train,
            "n_test_per_task": n_test,
            "test_used_for_selection": False,
        },
        "model": {
            "n_qubits": n_qubits,
            "layers": layers,
            "n_parameters": int(np.prod(weight_shape)),
            "ansatz": "independent RY/RZ plus nearest-neighbor CNOT chain",
            "classifier_readout": "Pauli-Z expectations on qubits 0 and 1 plus softmax",
            "consolidation_measurements": list(DEFAULT_BASES),
            "consolidation_readout_wires": [0, 1],
            "candidate_measurement_scope": (
                "same-axis Pauli products on the two readout qubits; ZZ/XX/YY is "
                "not tomographically complete"
            ),
            "readout_qfi_reference": (
                "SLD QFI of the two-readout-qubit reduced density matrix; an upper "
                "bound for that subsystem, not a jointly attainable multiparameter CFI"
            ),
            "prediction_measurement_unchanged": True,
            "device": "default.qubit",
            "shots": None,
            "logical_depth": int(resources.depth),
            "two_qubit_gates": int(gate_types.get("CNOT", 0)),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task,
            "ewc_lambda_shared_by_all_consolidation_methods": ewc_lambda,
            "fisher_normalization": "mean parameter importance = 1",
            "fisher_samples": len(anchor_indices),
            "reference_shots_for_allocation_only": reference_shots,
            "seed": seed,
            "phase1_trained_once_then_branched": True,
            "phase1_weights_and_adam_state_identical": True,
            "boundary_adam_state_sha256": _optimizer_state_digest(
                boundary_optimizer
            ),
            "initial_weights": initial_weights.tolist(),
            "boundary_weights": np.asarray(boundary_weights).tolist(),
            "record_test_history": record_test,
            "configuration_selection": (
                "capacity and shared lambda selected using seed-42 training metrics only; "
                "lambda calibrated on the output-CFI baseline; test was not evaluated"
            ),
        },
        "fisher_profiles": profiles,
        "histories": histories,
        "metrics": metrics,
        "final_weights": final_weights,
        "runtime_sec": {
            "shared_phase1_training": round(phase1_time, 4),
            "boundary_fisher_estimation": {
                method: round(fisher_timings[method], 4) for method in METHODS
            },
            "phase2_training": {
                method: round(phase2_times[method], 4) for method in METHODS
            },
        },
        "classical_logistic_regression_test_accuracy": classical_references,
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "qfi_attained": False,
            "quantum_advantage": False,
            "scope": "exact-statevector two-task engineering MVP",
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e008_measqcl_seed{result['training']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=40)
    parser.add_argument("--ewc-lambda", type=float, default=0.1)
    parser.add_argument("--fisher-samples", type=int, default=32)
    parser.add_argument("--reference-shots", type=int, default=1024)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-only", action="store_false", dest="record_test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_experiment(
        n_qubits=args.n_qubits,
        layers=args.layers,
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        reference_shots=args.reference_shots,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        record_test=args.record_test,
    )
    print("\n=== MeasQCL final training stability/plasticity ===")
    for method in METHODS:
        metric = result["metrics"][method]["train"]
        print(
            f"  {method:11s} retention={metric['task1_final_retention']:.3f} "
            f"adaptation={metric['task2_final_adaptation']:.3f} "
            f"forgetting={metric['task1_forgetting']:.3f}"
        )
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
