"""Reproduce E008 capacity and shared-lambda selection without test evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e008_measqcl import (  # noqa: E402
    _digest_arrays,
    _train_first_task,
    _train_second_task,
)
from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_softmax import classical_fisher_diag  # noqa: E402
from src.measqcl_fisher import normalize_fisher_mass, select_anchor_indices  # noqa: E402
from src.measqcl_model import make_classifier_qnode  # noqa: E402

CAPACITY_CANDIDATES = (
    {"layers": 2, "learning_rate": 0.02},
    {"layers": 3, "learning_rate": 0.02},
    {"layers": 3, "learning_rate": 0.05},
    {"layers": 6, "learning_rate": 0.02},
    {"layers": 10, "learning_rate": 0.02},
    {"layers": 12, "learning_rate": 0.02},
)
LAMBDA_CANDIDATES = (0.003, 0.01, 0.03, 0.1, 0.3)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "experiments/e008_measqcl.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _final_train_metrics(history: list[dict], boundary_epoch: int) -> dict[str, float]:
    boundary = history[boundary_epoch]["train_accuracy"]["task1"]
    final = history[-1]["train_accuracy"]
    return {
        "task1_boundary": round(boundary, 4),
        "task1_retention": round(final["task1"], 4),
        "task2_adaptation": round(final["task2"], 4),
        "task1_forgetting": round(boundary - final["task1"], 4),
        "average_final_accuracy": round((final["task1"] + final["task2"]) / 2, 4),
    }


def run_tuning(
    *,
    seed: int = 42,
    n_qubits: int = 6,
    epochs_per_task: int = 40,
    n_train: int = 400,
    n_test_loaded_but_never_evaluated: int = 200,
    fisher_samples: int = 32,
    verbose: bool = True,
) -> dict:
    """Run the documented train-only scan and return its auditable record."""
    tasks = load_two_tasks(
        n_features=2**n_qubits,
        n_train=n_train,
        n_test=n_test_loaded_but_never_evaluated,
        seed=seed,
    )
    capacity_results = []
    for candidate in CAPACITY_CANDIDATES:
        qnode, shape = make_classifier_qnode(n_qubits, candidate["layers"])
        started = time.perf_counter()
        _, _, _, history, _ = _train_first_task(
            qnode=qnode,
            weight_shape=shape,
            tasks=tasks,
            learning_rate=candidate["learning_rate"],
            epochs=epochs_per_task,
            seed=seed,
            record_test=False,
            verbose=False,
        )
        row = {
            **candidate,
            "n_parameters": int(np.prod(shape)),
            "task1_final_train_accuracy": round(
                history[-1]["train_accuracy"]["task1"], 4
            ),
            "runtime_sec": round(time.perf_counter() - started, 4),
            "test_evaluations": 0,
        }
        capacity_results.append(row)
        if verbose:
            print(
                f"capacity layers={row['layers']:2d} lr={row['learning_rate']:.3f}: "
                f"T1 train={row['task1_final_train_accuracy']:.4f}",
                flush=True,
            )

    selected_capacity = {"layers": 10, "learning_rate": 0.02}
    qnode, shape = make_classifier_qnode(n_qubits, selected_capacity["layers"])
    initial, boundary_weights, boundary_optimizer, common_history, _ = _train_first_task(
        qnode=qnode,
        weight_shape=shape,
        tasks=tasks,
        learning_rate=selected_capacity["learning_rate"],
        epochs=epochs_per_task,
        seed=seed,
        record_test=False,
        verbose=False,
    )
    anchors = select_anchor_indices(len(tasks[0].X_train), fisher_samples, seed + 10_000)
    importance = normalize_fisher_mass(
        classical_fisher_diag(
            qnode,
            boundary_weights,
            tasks[0].X_train[anchors],
            tasks[0].y_train[anchors],
        )
    )
    lambda_results = []
    for ewc_lambda in LAMBDA_CANDIDATES:
        started = time.perf_counter()
        history, _, _ = _train_second_task(
            method="output_cfi",
            qnode=qnode,
            boundary_weights=boundary_weights,
            boundary_optimizer=boundary_optimizer,
            common_history=common_history,
            importance=importance,
            tasks=tasks,
            ewc_lambda=ewc_lambda,
            epochs=epochs_per_task,
            record_test=False,
            verbose=False,
        )
        row = {
            "ewc_lambda": ewc_lambda,
            **_final_train_metrics(history, epochs_per_task),
            "runtime_sec": round(time.perf_counter() - started, 4),
            "test_evaluations": 0,
        }
        lambda_results.append(row)
        if verbose:
            print(
                f"lambda={ewc_lambda:.3f}: retention={row['task1_retention']:.4f} "
                f"adaptation={row['task2_adaptation']:.4f} "
                f"average={row['average_final_accuracy']:.4f}",
                flush=True,
            )

    selected_lambda = max(
        lambda_results,
        key=lambda row: row["average_final_accuracy"],
    )["ewc_lambda"]
    return {
        "schema_version": 1,
        "experiment": "e008_train_only_configuration_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": _digest_arrays(
            *(array for task in tasks for array in task[1:])
        ),
        "seed": seed,
        "test_evaluations": 0,
        "selection_data": "training metrics only",
        "capacity_scan": {
            "epochs": epochs_per_task,
            "results": capacity_results,
            "selected": selected_capacity,
            "selection_rule": (
                "smallest tested model reaching at least 0.88 Task-1 train accuracy; "
                "12 layers had only a marginal gain at higher parameter cost"
            ),
        },
        "lambda_scan": {
            "importance_estimator": "normalized output CFI",
            "fisher_anchor_indices": anchors.tolist(),
            "results": lambda_results,
            "selected": selected_lambda,
            "selection_rule": "highest average final Task-1/Task-2 training accuracy",
        },
        "initial_weights_sha256": _digest_arrays(initial),
        "claim_boundaries": {
            "test_used_for_selection": False,
            "mof_used_for_selection": False,
            "retrospective_test_tuning": False,
        },
    }


def main() -> None:
    output = ROOT / "results" / "e008_train_only_tuning.json"
    result = run_tuning()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
