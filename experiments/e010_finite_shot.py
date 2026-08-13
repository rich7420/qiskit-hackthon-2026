"""Finite-shot stability audit for E010 task-relevant measurement selection.

This is deliberately a measurement-estimation diagnostic, not a claim of finite-shot
training or retention.  For every formal seed it evaluates the exact parameter-shift
probability tables once, draws independent multinomial pilot experiments, and asks how
stable the selected measurement allocation is at an explicitly counted shot cost.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.measqcl_fisher import fisher_cosine_similarity  # noqa: E402
from src.measqcl_model import make_measurement_qnode  # noqa: E402
from src.measqcl_task_relevance import (  # noqa: E402
    exact_fisher_from_probability_table,
    finite_shot_fisher_from_probability_table,
    optimize_task_relevant_allocation,
    parameter_shift_probability_table,
)

RESULTS = ROOT / "results"


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "src/measqcl_task_relevance.py",
        ROOT / "experiments/e010_physmeas_qcl.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "sample_std": np.std(values, axis=0, ddof=1).tolist(),
    }


def run_audit(
    *,
    seed: int,
    shots_per_probability_circuit: tuple[int, ...] = (64, 256, 1024),
    repetitions: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    if repetitions < 2 or any(shots <= 0 for shots in shots_per_probability_circuit):
        raise ValueError("finite-shot audit requires positive shots and >=2 repetitions")
    extension_path = RESULTS / f"e010_physmeas_seed{seed}.json"
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    parent_path = ROOT / extension["parent"]["result_file"]
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != extension["parent"][
        "result_file_sha256"
    ]:
        raise ValueError("E010 references a changed E008 parent artifact")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    n_qubits = int(extension["model"]["n_qubits"])
    layers = int(extension["model"]["layers"])
    tasks = load_two_tasks(
        n_features=2**n_qubits,
        n_train=int(extension["data"]["n_train_per_task"]),
        n_test=int(extension["data"]["n_test_per_task"]),
        seed=seed,
    )
    weights = np.asarray(extension["training"]["boundary_weights"], dtype=float)
    indices = np.asarray(parent["fisher_profiles"]["anchor_indices"], dtype=int)
    relevance = np.asarray(
        extension["measurement_design"]["raw_task_relevance"], dtype=float
    )
    exact_basis = {
        basis: np.asarray(values, dtype=float)
        for basis, values in parent["fisher_profiles"]["raw_basis_fisher"].items()
    }
    tables: dict[str, dict[str, np.ndarray]] = {}
    table_times: dict[str, float] = {}
    for basis in exact_basis:
        if verbose:
            print(f"  seed {seed}: evaluating {basis} parameter-shift table", flush=True)
        started = time.perf_counter()
        tables[basis] = parameter_shift_probability_table(
            make_measurement_qnode(basis, n_qubits, layers),
            weights,
            tasks[0].X_train,
            indices,
        )
        table_times[basis] = time.perf_counter() - started
        recovered = exact_fisher_from_probability_table(tables[basis])
        np.testing.assert_allclose(recovered, exact_basis[basis], atol=2e-7, rtol=2e-6)

    exact_allocation = extension["measurement_design"]["allocation"]
    exact_vector = np.asarray([exact_allocation[basis] for basis in exact_basis])
    by_budget: dict[str, Any] = {}
    circuits_per_basis = len(indices) * (1 + 2 * weights.size)
    for shots in shots_per_probability_circuit:
        estimates = {}
        for offset, (basis, table) in enumerate(tables.items()):
            estimates[basis] = finite_shot_fisher_from_probability_table(
                table,
                shots_per_circuit=shots,
                repetitions=repetitions,
                seed=seed * 1_000_003 + shots * 101 + offset,
            ).estimates
        allocations = []
        cosine_to_exact = []
        l1_error = []
        failures = 0
        for repetition in range(repetitions):
            noisy_basis = {
                basis: values[repetition] for basis, values in estimates.items()
            }
            try:
                allocation_result = optimize_task_relevant_allocation(
                    noisy_basis,
                    relevance,
                    minimum_allocation=float(
                        extension["measurement_design"]["minimum_allocation_per_basis"]
                    ),
                    relevance_floor=float(
                        extension["measurement_design"]["relevance_floor_fraction"]
                    ),
                )
            except RuntimeError:
                failures += 1
                continue
            allocation = np.asarray(allocation_result.weights)
            allocations.append(allocation)
            l1_error.append(float(np.sum(np.abs(allocation - exact_vector))))
            noisy_accessible = sum(
                allocation[index] * noisy_basis[basis]
                for index, basis in enumerate(exact_basis)
            )
            exact_accessible = np.asarray(
                extension["measurement_design"]["raw_task_relevant_accessible_fisher"]
            )
            cosine_to_exact.append(
                fisher_cosine_similarity(noisy_accessible, exact_accessible)
            )
        if not allocations:
            raise RuntimeError(f"all finite-shot allocations failed at shots={shots}")
        allocation_values = np.stack(allocations)
        by_budget[str(shots)] = {
            "shots_per_probability_circuit": shots,
            "pilot_circuits_per_basis": circuits_per_basis,
            "pilot_measurement_settings": len(exact_basis),
            "pilot_shots_per_repetition": (
                len(exact_basis) * circuits_per_basis * shots
            ),
            "repetitions": repetitions,
            "successful_repetitions": len(allocations),
            "optimizer_failures": failures,
            "allocation": {
                basis: {
                    "mean": float(np.mean(allocation_values[:, index])),
                    "sample_std": float(np.std(allocation_values[:, index], ddof=1)),
                }
                for index, basis in enumerate(exact_basis)
            },
            "allocation_l1_error_to_exact": _summary(np.asarray(l1_error)),
            "selected_profile_cosine_to_exact": _summary(
                np.asarray(cosine_to_exact)
            ),
            "basis_fisher": {
                basis: _summary(values) for basis, values in estimates.items()
            },
        }
    return {
        "schema_version": 1,
        "experiment": "e010_conditional_finite_shot_basis_fisher_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "seed": seed,
        "extension_result_file": str(extension_path.relative_to(ROOT)),
        "extension_result_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "probability_table_sha256": _array_digest(
            *(
                table[key]
                for table in tables.values()
                for key in ("base", "plus", "minus")
            )
        ),
        "n_anchor_samples": len(indices),
        "n_parameters": int(weights.size),
        "bases": list(exact_basis),
        "exact_allocation": exact_allocation,
        "estimator": {
            "sampling": "independent multinomial draws per exact probability circuit",
            "derivatives": "parameter shift +/- pi/2",
            "pseudocount": "Jeffreys 0.5 per outcome",
            "bias_correction": (
                "first-order non-negative subtraction of squared-derivative sampling noise"
            ),
            "pilot_and_production_split": (
                "this audit covers pilot selection only; production allocation and "
                "finite-shot continual retraining are not performed"
            ),
            "exact_conditioning": (
                "task-relevance weights are loaded from the exact statevector E010 "
                "artifact; their sampling noise and hardware measurement cost are excluded"
            ),
        },
        "budgets": by_budget,
        "runtime_sec": {
            "exact_probability_tables": {
                basis: round(value, 4) for basis, value in table_times.items()
            }
        },
        "claim_boundaries": {
            "conditional_finite_shot_basis_fisher_selection": True,
            "complete_finite_shot_measurement_selection": False,
            "finite_shot_training_or_retention": False,
            "hardware_result": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(
        seed=args.seed,
        shots_per_probability_circuit=tuple(args.shots),
        repetitions=args.repetitions,
    )
    output = args.output or RESULTS / f"e010_finite_shot_seed{args.seed}.json"
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
