"""E012 axis C: stability-plasticity frontier of EWC vs QEWC vs readout-QFI.

Decisive test for the E008-vs-2607.16030 discrepancy. A single lambda point (as in E008)
conflates the Fisher *direction* with the operating point on each method's frontier.
Normalizing to mean-one and sweeping a shared lambda is a reparametrization of "raw Fisher
with a per-method lambda" (lambda_raw = lambda_shared / mean(F_raw)), so the frontier itself
is protocol-invariant. If QEWC's (retention, adaptation) frontier dominates EWC's, the quantum
Fisher is genuinely the better importance signal and E008 merely sampled a bad lambda. If the
frontiers overlap or cross, E005's QEWC>EWC ordering was purely an operating-point (lambda)
choice and the paper's advantage does not reproduce under matched geometry.

Reuses the exact E008 harness (6q/10L, MNIST->Fashion, shared Task-1 trajectory/Adam state).
lambda=0.1 reproduces the E008 single-point numbers as a sanity check. Test labels are recorded
for diagnosis only; frontier selection must use train metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.e005_softmax import classical_fisher_diag  # noqa: E402
from src.measqcl_fisher import (  # noqa: E402
    accessible_fisher_diag,
    measurement_fisher_diag,
    normalize_fisher_mass,
    optimize_measurement_allocation,
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
from experiments.e008_measqcl import (  # noqa: E402
    _method_metrics,
    _source_digest,
    _train_first_task,
    _train_second_task,
)

RESULTS = ROOT / "results"
# Shared effective-lambda grid on mean-one Fisher. lambda=0.1 == the E008 operating point.
DEFAULT_LAMBDAS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
FRONTIER_METHODS = (
    "output_cfi",
    "zz_cfi",
    "uniform_xyz",
    "mof_ewc",
    "readout_qewc",
    "qewc",
)
LABELS = {
    "output_cfi": "EWC (output-CFI)",
    "zz_cfi": "Joint-ZZ CFI",
    "uniform_xyz": "Uniform XYZ",
    "mof_ewc": "MOF-EWC",
    "readout_qewc": "Readout-QFI",
    "qewc": "QEWC (full-QFI)",
}


def run_frontier(
    *,
    n_qubits: int = 6,
    layers: int = 10,
    learning_rate: float = 0.02,
    epochs_per_task: int = 40,
    fisher_samples: int = 32,
    n_train: int = 400,
    n_test: int = 200,
    seed: int = 42,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    verbose: bool = True,
) -> dict[str, Any]:
    tasks = load_two_tasks(
        n_features=2**n_qubits, n_train=n_train, n_test=n_test, seed=seed
    )
    classifier_qnode, weight_shape = make_classifier_qnode(n_qubits, layers)
    measurement_qnodes = {
        basis: make_measurement_qnode(basis, n_qubits, layers) for basis in DEFAULT_BASES
    }
    qfi_qnode = make_qfi_qnode(n_qubits, layers)
    reduced_state_qnode = make_reduced_state_qnode(n_qubits, layers)

    if verbose:
        print(
            f"E012 frontier seed={seed}: {tasks[0].name} -> {tasks[1].name}; "
            f"{n_qubits}q/{layers}L; lambdas={lambdas}",
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
        record_test=True,
        verbose=verbose,
    )
    anchor_indices = select_anchor_indices(
        len(tasks[0].X_train), fisher_samples, seed + 10_000
    )

    # Raw, then mean-one normalized importance diagonals (same estimators as E008).
    started = time.perf_counter()
    basis_fishers = {
        basis: measurement_fisher_diag(
            qnode, boundary_weights, tasks[0].X_train, anchor_indices
        )
        for basis, qnode in measurement_qnodes.items()
    }
    uniform = {basis: 1.0 / len(basis_fishers) for basis in basis_fishers}
    fixed_zz = {basis: float(basis == "ZZ") for basis in basis_fishers}
    allocation = optimize_measurement_allocation(basis_fishers)
    optimized = dict(zip(allocation.bases, allocation.weights.tolist(), strict=True))
    raw = {
        "output_cfi": classical_fisher_diag(
            classifier_qnode,
            boundary_weights,
            tasks[0].X_train[anchor_indices],
            tasks[0].y_train[anchor_indices],
        ),
        "zz_cfi": accessible_fisher_diag(basis_fishers, fixed_zz),
        "uniform_xyz": accessible_fisher_diag(basis_fishers, uniform),
        "mof_ewc": accessible_fisher_diag(basis_fishers, optimized),
        "readout_qewc": reduced_state_qfi_diag(
            reduced_state_qnode,
            boundary_weights,
            tasks[0].X_train,
            anchor_indices,
        ),
        "qewc": quantum_fisher_diag(
            qfi_qnode,
            boundary_weights,
            tasks[0].X_train[anchor_indices],
            n_samples=len(anchor_indices),
            seed=0,
        ),
    }
    fisher_time = time.perf_counter() - started
    normalized = {m: normalize_fisher_mass(raw[m]) for m in FRONTIER_METHODS}
    raw_mass = {m: float(np.sum(raw[m])) for m in FRONTIER_METHODS}

    frontier: dict[str, list[dict[str, Any]]] = {m: [] for m in ("naive", *FRONTIER_METHODS)}
    # Naive baseline: single point (lambda has no effect).
    history, _, _ = _train_second_task(
        method="naive",
        qnode=classifier_qnode,
        boundary_weights=boundary_weights,
        boundary_optimizer=boundary_optimizer,
        common_history=common_history,
        importance=None,
        tasks=tasks,
        ewc_lambda=0.0,
        epochs=epochs_per_task,
        record_test=True,
        verbose=False,
    )
    metrics = _method_metrics(history, epochs_per_task)
    frontier["naive"].append({"lambda": 0.0, "raw_lambda": 0.0, "metrics": metrics})

    for method in FRONTIER_METHODS:
        for lam in lambdas:
            history, _, _ = _train_second_task(
                method=method,
                qnode=classifier_qnode,
                boundary_weights=boundary_weights,
                boundary_optimizer=boundary_optimizer,
                common_history=common_history,
                importance=normalized[method],
                tasks=tasks,
                ewc_lambda=lam,
                epochs=epochs_per_task,
                record_test=True,
                verbose=False,
            )
            metrics = _method_metrics(history, epochs_per_task)
            # raw_lambda: the per-method lambda that gives the same penalty on RAW Fisher.
            raw_lambda = lam * len(normalized[method]) / raw_mass[method]
            frontier[method].append(
                {"lambda": lam, "raw_lambda": raw_lambda, "metrics": metrics}
            )
            if verbose:
                tr = metrics["train"]
                print(
                    f"  {LABELS[method]:18s} lam={lam:5.3f} "
                    f"ret={tr['task1_final_retention']:.3f} "
                    f"adapt={tr['task2_final_adaptation']:.3f}",
                    flush=True,
                )

    return {
        "experiment": "e012_lambda_frontier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "axis": "C: stability-plasticity frontier, raw-vs-normalized lambda invariance",
        "model": {"n_qubits": n_qubits, "layers": layers,
                  "n_parameters": int(np.prod(weight_shape))},
        "config": {
            "seed": seed, "epochs_per_task": epochs_per_task,
            "learning_rate": learning_rate, "fisher_samples": len(anchor_indices),
            "lambdas": list(lambdas), "n_train": n_train, "n_test": n_test,
            "test_used_for_selection": False,
        },
        "raw_fisher_mass": raw_mass,
        "mof_allocation": optimized,
        "runtime_sec": {"phase1": round(phase1_time, 2), "fisher": round(fisher_time, 2)},
        "frontier": frontier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_frontier(seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    output = args.output or RESULTS / f"e012_frontier_seed{args.seed}.json"
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
