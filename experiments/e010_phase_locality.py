"""Phase-first, locality-resolved PhysMeas-QCL experiment.

The learner first fits the four-qubit SPT/ATF task, then learns MNIST 0/1 and
Fashion-MNIST 0/1.  A single verified phase boundary is branched into consolidation
methods whose importance profiles come from output observables of increasing Pauli
weight.  The experiment measures phase-memory retention; it does not claim that output
Pauli support is identical to locality in the input ground state after a deep unitary.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import Task, load_two_tasks  # noqa: E402
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.e005_softmax import (  # noqa: E402
    accuracy,
    bce_loss,
    classical_fisher_diag,
)
from src.measqcl_fisher import (  # noqa: E402
    accessible_fisher_diag,
    fisher_cosine_similarity,
    normalize_fisher_mass,
    select_anchor_indices,
)
from src.measqcl_model import make_classifier_qnode, make_qfi_qnode  # noqa: E402
from src.measqcl_task_relevance import (  # noqa: E402
    normalize_task_relevance,
    optimize_task_relevant_allocation,
    reversed_logits_fisher_diag,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402
from src.physmeas_observables import (  # noqa: E402
    family_resource_summary,
    make_pauli_expectation_qnode,
    pauli_fisher_diag,
    phase_measurement_families,
    uniform_family_fisher,
)

RESULTS = ROOT / "results"
TASK_KEYS = ("phase", "mnist", "fashion")
METHODS = (
    "naive",
    "output_cfi",
    "ewc_dr",
    "readout_pauli",
    "one_local",
    "two_local",
    "hamiltonian",
    "nonlocal",
    "task_relevant_all",
    "qewc",
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
        ROOT / "src/physmeas_observables.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _snapshot(
    history: list[dict[str, Any]],
    *,
    epoch: int,
    phase: int,
    qnode,
    weights,
    tasks: tuple[Task, ...],
    record_test: bool,
) -> None:
    history.append(
        {
            "epoch": epoch,
            "phase": phase,
            "train_accuracy": {
                key: accuracy(qnode, weights, task.X_train, task.y_train)
                for key, task in zip(TASK_KEYS, tasks, strict=True)
            },
            "test_accuracy": (
                {
                    key: accuracy(qnode, weights, task.X_test, task.y_test)
                    for key, task in zip(TASK_KEYS, tasks, strict=True)
                }
                if record_test
                else None
            ),
        }
    )


def _train_phase(
    *,
    qnode,
    weight_shape: tuple[int, ...],
    tasks: tuple[Task, ...],
    learning_rate: float,
    epochs: int,
    seed: int,
    record_test: bool,
    verbose: bool,
):
    initial = 0.05 * np.random.default_rng(seed).standard_normal(weight_shape)
    weights = pnp.array(initial, requires_grad=True)
    optimizer = qml.AdamOptimizer(learning_rate)
    history: list[dict[str, Any]] = []
    _snapshot(
        history,
        epoch=0,
        phase=0,
        qnode=qnode,
        weights=weights,
        tasks=tasks,
        record_test=record_test,
    )
    X = pnp.array(tasks[0].X_train, requires_grad=False)
    y = pnp.array(tasks[0].y_train, requires_grad=False)

    def cost(candidate):
        return bce_loss(qnode, candidate, X, y)

    for epoch in range(1, epochs + 1):
        weights = optimizer.step(cost, weights)
        _snapshot(
            history,
            epoch=epoch,
            phase=1,
            qnode=qnode,
            weights=weights,
            tasks=tasks,
            record_test=record_test,
        )
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"  shared phase epoch {epoch:2d}: "
                f"train={history[-1]['train_accuracy']['phase']:.3f}",
                flush=True,
            )
    return initial, weights, optimizer, history


def _train_future_tasks(
    *,
    method: str,
    qnode,
    boundary_weights,
    boundary_optimizer,
    common_history: list[dict[str, Any]],
    importance: np.ndarray | None,
    tasks: tuple[Task, ...],
    ewc_lambda: float,
    epochs: int,
    record_test: bool,
    verbose: bool,
):
    weights = pnp.array(np.asarray(boundary_weights).copy(), requires_grad=True)
    optimizer = copy.deepcopy(boundary_optimizer)
    history = copy.deepcopy(common_history)
    anchor = np.asarray(boundary_weights).reshape(-1)
    profile = None if importance is None else pnp.array(importance, requires_grad=False)
    for task_index, task in enumerate(tasks[1:], start=2):
        X = pnp.array(task.X_train, requires_grad=False)
        y = pnp.array(task.y_train, requires_grad=False)

        def cost(candidate):
            task_loss = bce_loss(qnode, candidate, X, y)
            if profile is None:
                return task_loss
            penalty = 0.5 * ewc_lambda * pnp.sum(
                profile * (candidate.flatten() - anchor) ** 2
            )
            return task_loss + penalty

        for local_epoch in range(1, epochs + 1):
            weights = optimizer.step(cost, weights)
            epoch = history[-1]["epoch"] + 1
            _snapshot(
                history,
                epoch=epoch,
                phase=task_index,
                qnode=qnode,
                weights=weights,
                tasks=tasks,
                record_test=record_test,
            )
            if verbose and (local_epoch == 1 or local_epoch % 10 == 0):
                values = history[-1]["train_accuracy"]
                print(
                    f"    {method:18s} epoch {epoch:3d}: "
                    f"phase={values['phase']:.3f} "
                    f"active={values[TASK_KEYS[task_index - 1]]:.3f}",
                    flush=True,
                )
    return history, np.asarray(weights)


def _metrics(history: list[dict[str, Any]], phase_boundary: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    boundary = history[phase_boundary]
    final = history[-1]
    for split in ("train", "test"):
        field = f"{split}_accuracy"
        if final[field] is None:
            continue
        phase_boundary_accuracy = boundary[field]["phase"]
        phase_final = final[field]["phase"]
        result[split] = {
            "phase_at_boundary": round(phase_boundary_accuracy, 4),
            "phase_final_retention": round(phase_final, 4),
            "phase_forgetting": round(phase_boundary_accuracy - phase_final, 4),
            "mnist_final_adaptation": round(final[field]["mnist"], 4),
            "fashion_final_adaptation": round(final[field]["fashion"], 4),
            "average_final_accuracy": round(
                float(np.mean(list(final[field].values()))), 4
            ),
        }
    return result


def run_experiment(
    *,
    layers: int = 3,
    learning_rate: float = 0.02,
    epochs_per_task: int = 40,
    ewc_lambda: float = 0.1,
    fisher_samples: int = 32,
    minimum_allocation: float = 1e-6,
    n_train: int = 400,
    n_test: int = 200,
    seed: int = 42,
    record_test: bool = True,
    methods: tuple[str, ...] = METHODS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one paired phase-first locality comparison."""
    if any(method not in METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("methods must be a unique subset of the formal method ladder")
    if min(layers, epochs_per_task, fisher_samples, n_train, n_test) <= 0:
        raise ValueError("capacity, epochs, Fisher sample size, and data sizes must be positive")
    if learning_rate <= 0.0 or ewc_lambda < 0.0 or minimum_allocation < 0.0:
        raise ValueError("learning rate must be positive; lambda/floor must be non-negative")
    phase_task = load_spt_atf(n_train=n_train, n_test=n_test, seed=seed)
    image_tasks = load_two_tasks(
        n_features=2**N_QUBITS,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    tasks = (phase_task, *image_tasks)
    classifier_qnode, weight_shape = make_classifier_qnode(N_QUBITS, layers)
    qfi_qnode = make_qfi_qnode(N_QUBITS, layers)
    initial, boundary, boundary_optimizer, common_history = _train_phase(
        qnode=classifier_qnode,
        weight_shape=weight_shape,
        tasks=tasks,
        learning_rate=learning_rate,
        epochs=epochs_per_task,
        seed=seed,
        record_test=record_test,
        verbose=verbose,
    )
    indices = select_anchor_indices(n_train, fisher_samples, seed + 20_000)
    anchor_X = phase_task.X_train[indices]
    anchor_y = phase_task.y_train[indices]
    raw: dict[str, np.ndarray | None] = {"naive": None}
    raw["output_cfi"] = classical_fisher_diag(
        classifier_qnode, boundary, anchor_X, anchor_y
    )
    raw["ewc_dr"] = reversed_logits_fisher_diag(
        classifier_qnode, boundary, anchor_X, anchor_y
    )
    relevance = normalize_task_relevance(raw["ewc_dr"])
    families = phase_measurement_families(N_QUBITS)
    observable_fishers: dict[str, dict[str, np.ndarray]] = {}
    observable_metadata: dict[str, Any] = {}
    unique_observables = {}
    for family, measurements in families.items():
        observable_fishers[family] = {}
        for measurement in measurements:
            fisher = pauli_fisher_diag(
                make_pauli_expectation_qnode(
                    measurement.pauli, N_QUBITS, layers
                ),
                boundary,
                phase_task.X_train,
                indices,
            )
            observable_fishers[family][measurement.name] = fisher
            unique_observables.setdefault(measurement.pauli, fisher)
            observable_metadata[measurement.name] = {
                "pauli": measurement.pauli,
                "setting": measurement.setting,
                "family": family,
                "weight": measurement.weight,
                "diameter": measurement.diameter,
                "raw_fisher_mass": float(np.sum(fisher)),
            }
        raw_name = "readout_pauli" if family == "readout" else family
        raw[raw_name] = uniform_family_fisher(observable_fishers[family])
    optimization = optimize_task_relevant_allocation(
        unique_observables,
        raw["ewc_dr"],
        minimum_allocation=minimum_allocation,
        relevance_floor=1e-3,
    )
    optimized_allocation = dict(
        zip(optimization.bases, optimization.weights.tolist(), strict=True)
    )
    raw["task_relevant_all"] = accessible_fisher_diag(
        unique_observables, optimized_allocation
    )
    raw["qewc"] = quantum_fisher_diag(
        qfi_qnode,
        boundary,
        anchor_X,
        n_samples=len(indices),
        seed=seed,
    )
    for pauli, fisher in unique_observables.items():
        if np.any(fisher > raw["qewc"] + 2e-6):
            raise ValueError(f"observable CFI {pauli} exceeds full-state QFI")
    importance = {
        method: None if raw[method] is None else normalize_fisher_mass(raw[method])
        for method in METHODS
    }
    histories = {}
    final_weights = {}
    started = time.perf_counter()
    for method in methods:
        if verbose:
            print(f"  training future tasks with {method}", flush=True)
        history, weights = _train_future_tasks(
            method=method,
            qnode=classifier_qnode,
            boundary_weights=boundary,
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
        final_weights[method] = weights.tolist()
    future_training_time = time.perf_counter() - started
    return {
        "schema_version": 1,
        "experiment": "e010_phase_first_locality_resolved_exact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": _digest_arrays(
            *(array for task in tasks for array in task[1:])
        ),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in ("pennylane", "numpy", "scipy", "scikit-learn")
            },
        },
        "data": {
            "task_order": [task.name for task in tasks],
            "n_train_per_task": n_train,
            "n_test_per_task": n_test,
            "test_used_for_selection": False,
            "phase_ranges": {"SPT": [0.0, 0.5], "ATF": [2.5, 3.0]},
        },
        "model": {
            "n_qubits": N_QUBITS,
            "layers": layers,
            "n_parameters": int(np.prod(weight_shape)),
            "ansatz": "independent RY/RZ plus directed nearest-neighbor CNOT chain",
            "classifier_readout": "Z0/Z1 expectations plus softmax",
            "shots": None,
        },
        "training": {
            "methods": list(methods),
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task,
            "ewc_lambda": ewc_lambda,
            "fisher_normalization": "mean parameter importance = 1",
            "fisher_samples": len(indices),
            "seed": seed,
            "initial_weights": initial.tolist(),
            "phase_boundary_weights": np.asarray(boundary).tolist(),
            "phase_trained_once_then_branched": True,
            "optimizer_state_identical_at_branch": True,
            "phase_anchor_strength_constant_across_both_future_tasks": True,
            "record_test_history": record_test,
        },
        "locality_analysis": {
            "domain": (
                "Pauli support on the learned output state; because the VQC unitary "
                "can enlarge Heisenberg support, this is not asserted to equal input "
                "ground-state locality"
            ),
            "binary_pauli_cfi": (
                "exact (d<P>/dtheta)^2/(1-<P>^2), averaged over phase anchors"
            ),
            "families": {
                family: {
                    "resources": family_resource_summary(measurements),
                    "measurements": [measurement.name for measurement in measurements],
                }
                for family, measurements in families.items()
            },
            "observable_metadata": observable_metadata,
            "raw_observable_fisher": {
                family: {
                    name: values.tolist() for name, values in fishers.items()
                }
                for family, fishers in observable_fishers.items()
            },
            "raw_method_fisher": {
                method: raw[method].tolist() for method in METHODS if method != "naive"
            },
            "normalized_method_fisher": {
                method: importance[method].tolist()
                for method in METHODS
                if method != "naive"
            },
            "cosine_to_qfi": {
                method: fisher_cosine_similarity(raw[method], raw["qewc"])
                for method in METHODS
                if method not in ("naive", "qewc")
            },
            "task_relevant_allocation": optimized_allocation,
            "task_relevant_minimum_allocation": minimum_allocation,
            "task_relevant_objective": optimization.objective,
            "task_relevant_optimizer": optimization.solver,
            "task_relevant_optimality_gap": optimization.optimality_gap,
            "task_relevant_optimality_tolerance": optimization.tolerance,
            "task_relevant_allocation_unit": (
                "probability of independently measuring one binary Pauli observable; "
                "compatible-bitstring reuse is reported but not credited"
            ),
            "anchor_indices": indices.tolist(),
        },
        "histories": histories,
        "metrics": {
            method: _metrics(history, epochs_per_task)
            for method, history in histories.items()
        },
        "final_weights": final_weights,
        "runtime_sec": {"future_task_training": round(future_training_time, 4)},
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "thermodynamic_phase_transition": False,
            "input_locality_equivalence": False,
            "quantum_advantage": False,
            "scope": "exact-statevector four-qubit phase-memory diagnostic",
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e010_phase_locality_seed{result['training']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=40)
    parser.add_argument("--ewc-lambda", type=float, default=0.1)
    parser.add_argument("--fisher-samples", type=int, default=32)
    parser.add_argument("--minimum-allocation", type=float, default=1e-6)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-only", action="store_false", dest="record_test")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        layers=args.layers,
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        minimum_allocation=args.minimum_allocation,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        record_test=args.record_test,
        methods=tuple(args.methods),
    )
    split = "test" if args.record_test else "train"
    print(f"\n=== E010 phase memory ({split}) ===")
    for method in args.methods:
        metric = result["metrics"][method][split]
        print(
            f"  {method:18s} phase={metric['phase_final_retention']:.3f} "
            f"MNIST={metric['mnist_final_adaptation']:.3f} "
            f"Fashion={metric['fashion_final_adaptation']:.3f}"
        )
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
