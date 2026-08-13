"""E010: task-relevant measurement design for quantum continual learning.

This experiment extends the paired E008 run without changing its classifier, data,
Task-1 trajectory, optimizer state, EWC strength, anchor examples, or candidate
measurements.  E008 maximized task-agnostic diagonal Fisher coverage.  E010 instead
uses EWC-DR importance only to choose the measurement mixture, then consolidates with
the resulting physically accessible CFI.  This isolates measurement selection from
the EWC penalty itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import e008_measqcl as parent  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.measqcl_fisher import (  # noqa: E402
    accessible_fisher_diag,
    fisher_cosine_similarity,
    normalize_fisher_mass,
)
from src.measqcl_model import make_classifier_qnode  # noqa: E402
from src.measqcl_task_relevance import (  # noqa: E402
    normalize_task_relevance,
    optimize_task_relevant_allocation,
    reversed_logits_fisher_diag,
)

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
NEW_METHODS = ("ewc_dr", "task_relevant_mof")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "experiments/e008_measqcl.py",
        ROOT / "src/measqcl_task_relevance.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_parent(seed: int) -> tuple[dict[str, Any], Path]:
    path = RESULTS / f"e008_measqcl_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing paired E008 artifact: {path}")
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
    """Reject extensions that are not paired to the checked E008 trajectory."""
    if result.get("experiment") != "e008_measqcl_exact_mvp":
        raise ValueError("parent artifact is not an E008 exact-statevector run")
    if result.get("source_code_sha256") != parent._source_digest():
        raise ValueError("parent artifact does not match the current E008 sources")
    if result["training"]["seed"] != seed or result["data_sha256"] != data_sha256:
        raise ValueError("parent seed/data split does not match this extension")
    np.testing.assert_allclose(
        np.asarray(result["training"]["initial_weights"]),
        initial_weights,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result["training"]["boundary_weights"]),
        np.asarray(boundary_weights),
        atol=1e-12,
        rtol=1e-12,
    )
    first_parent_history = result["histories"][parent.METHODS[0]][: len(common_history)]
    if first_parent_history != common_history:
        raise ValueError("replayed Task-1 history differs from the E008 paired trajectory")
    if any(
        result["histories"][method][: len(common_history)] != first_parent_history
        for method in parent.METHODS[1:]
    ):
        raise ValueError("E008 methods do not share one Task-1 trajectory")


def run_experiment(
    *,
    seed: int = 42,
    minimum_allocation: float = 0.01,
    relevance_floor: float = 1e-3,
    verbose: bool = True,
    parent_result: dict[str, Any] | None = None,
    parent_path: Path | None = None,
) -> dict[str, Any]:
    """Extend one formal E008 seed with EWC-DR and task-relevant MeasQCL."""
    if minimum_allocation < 0.0 or relevance_floor < 0.0:
        raise ValueError("allocation and relevance floors must be non-negative")
    if parent_result is None:
        parent_result, parent_path = _load_parent(seed)
    config = parent_result["configuration"] if "configuration" in parent_result else None
    # Per-seed E008 artifacts keep configuration under these three top-level sections.
    data_config = parent_result["data"] if config is None else config["data"]
    model_config = parent_result["model"] if config is None else config["model"]
    training_config = parent_result["training"] if config is None else config["training"]
    n_qubits = int(model_config["n_qubits"])
    layers = int(model_config["layers"])
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
    data_sha256 = parent._digest_arrays(
        *(array for task in tasks for array in task[1:])
    )
    classifier_qnode, weight_shape = make_classifier_qnode(n_qubits, layers)
    if verbose:
        print(
            f"PhysMeas-QCL seed={seed}: replaying paired Task 1, then branching "
            f"{', '.join(NEW_METHODS)}",
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
    _validate_parent(
        parent_result,
        seed=seed,
        data_sha256=data_sha256,
        initial_weights=initial_weights,
        boundary_weights=boundary_weights,
        common_history=common_history,
    )

    profiles = parent_result["fisher_profiles"]
    anchor_indices = np.asarray(profiles["anchor_indices"], dtype=int)
    basis_fishers = {
        basis: np.asarray(values, dtype=float)
        for basis, values in profiles["raw_basis_fisher"].items()
    }
    anchor_features = tasks[0].X_train[anchor_indices]
    anchor_labels = tasks[0].y_train[anchor_indices]
    started = time.perf_counter()
    ewc_dr_raw = reversed_logits_fisher_diag(
        classifier_qnode,
        pnp.array(boundary_weights, requires_grad=True),
        anchor_features,
        anchor_labels,
    )
    relevance = normalize_task_relevance(ewc_dr_raw, floor=relevance_floor)
    allocation_result = optimize_task_relevant_allocation(
        basis_fishers,
        ewc_dr_raw,
        minimum_allocation=minimum_allocation,
        relevance_floor=relevance_floor,
    )
    allocation = dict(
        zip(
            allocation_result.bases,
            allocation_result.weights.tolist(),
            strict=True,
        )
    )
    task_relevant_raw = accessible_fisher_diag(basis_fishers, allocation)
    fisher_time = time.perf_counter() - started
    importance = {
        "ewc_dr": normalize_fisher_mass(ewc_dr_raw),
        "task_relevant_mof": normalize_fisher_mass(task_relevant_raw),
    }

    histories: dict[str, list[dict[str, Any]]] = {}
    final_weights: dict[str, list[float]] = {}
    phase2_times: dict[str, float] = {}
    for method in NEW_METHODS:
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
        method: parent._method_metrics(history, epochs)
        for method, history in histories.items()
    }

    original_mof = np.asarray(profiles["raw_method_fisher"]["mof_ewc"])
    output_cfi = np.asarray(profiles["raw_method_fisher"]["output_cfi"])
    parent_reference = (
        str(parent_path.relative_to(ROOT))
        if parent_path is not None and parent_path.is_relative_to(ROOT)
        else None
    )
    parent_file_sha256 = (
        _file_digest(parent_path)
        if parent_path is not None and parent_path.exists()
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e010_task_relevant_measqcl_exact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": data_sha256,
        "parent": {
            "experiment": parent_result["experiment"],
            "result_file": parent_reference,
            "result_file_sha256": parent_file_sha256,
            "source_code_sha256": parent_result["source_code_sha256"],
            "inherited_methods": list(parent.METHODS),
        },
        "data": {
            "task_order": [task.name for task in tasks],
            "n_train_per_task": n_train,
            "n_test_per_task": n_test,
            "test_used_for_selection": False,
        },
        "model": {
            **model_config,
            "candidate_measurements": list(basis_fishers),
            "prediction_measurement_unchanged": True,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs_per_task": epochs,
            "ewc_lambda_shared": ewc_lambda,
            "seed": seed,
            "phase1_replayed_and_verified_against_parent": True,
            "boundary_adam_state_sha256": parent._optimizer_state_digest(
                boundary_optimizer
            ),
            "boundary_weights": np.asarray(boundary_weights).tolist(),
            "record_test_history": record_test,
            "selection_policy": (
                "inherits E008 train-only capacity/lambda; EWC-DR relevance floor and "
                "one-percent basis floor are prespecified estimator safeguards"
            ),
        },
        "measurement_design": {
            "task_relevance_estimator": (
                "EWC-DR empirical diagonal: squared gradients of reversed-logit "
                "old-task log likelihood"
            ),
            "relevance_role": (
                "used only to optimize measurement allocation; the final task-relevant "
                "MOF penalty uses accessible measurement CFI"
            ),
            "relevance_floor_fraction": relevance_floor,
            "minimum_allocation_per_basis": minimum_allocation,
            "objective": (
                "sum_i task_relevance_i * log(epsilon + sum_m q_m F_mi)"
            ),
            "allocation": allocation,
            "allocation_objective": allocation_result.objective,
            "allocation_iterations": allocation_result.iterations,
            "allocation_solver": allocation_result.solver,
            "allocation_optimality_gap": allocation_result.optimality_gap,
            "allocation_optimality_tolerance": allocation_result.tolerance,
            "raw_task_relevance": ewc_dr_raw.tolist(),
            "normalized_task_relevance": relevance.tolist(),
            "raw_task_relevant_accessible_fisher": task_relevant_raw.tolist(),
            "normalized_method_fisher": {
                method: values.tolist() for method, values in importance.items()
            },
            "diagnostics": {
                "ewc_dr_to_output_cfi_cosine": fisher_cosine_similarity(
                    ewc_dr_raw, output_cfi
                ),
                "task_relevant_to_output_cfi_cosine": fisher_cosine_similarity(
                    task_relevant_raw, output_cfi
                ),
                "task_relevant_to_original_mof_cosine": fisher_cosine_similarity(
                    task_relevant_raw, original_mof
                ),
            },
            "exact_estimation_only": True,
            "finite_shot_estimation_performed": False,
        },
        "histories": histories,
        "metrics": metrics,
        "parent_metrics": parent_result["metrics"],
        "final_weights": final_weights,
        "runtime_sec": {
            "replayed_phase1_training": round(phase1_time, 4),
            "new_relevance_and_allocation": round(fisher_time, 4),
            "phase2_training": {
                method: round(value, 4) for method, value in phase2_times.items()
            },
        },
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "locality_resolved_result": False,
            "quantum_advantage": False,
            "scope": (
                "exact-statevector paired two-task test of task-relevant measurement "
                "selection"
            ),
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e010_physmeas_seed{result['training']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-allocation", type=float, default=0.01)
    parser.add_argument("--relevance-floor", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        seed=args.seed,
        minimum_allocation=args.minimum_allocation,
        relevance_floor=args.relevance_floor,
    )
    print("\n=== E010 exact task-relevant comparison ===")
    for method in NEW_METHODS:
        metric = result["metrics"][method]["test"]
        print(
            f"  {method:18s} retention={metric['task1_final_retention']:.3f} "
            f"adaptation={metric['task2_final_adaptation']:.3f} "
            f"forgetting={metric['task1_forgetting']:.3f}"
        )
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
