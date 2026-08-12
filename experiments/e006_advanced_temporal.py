"""E006 Advanced - temporal quantum continual learning with balanced replay.

The same compact recurrent/data-reuploading QNN is trained on ECG200, GunPoint, and Coffee.
An E004-style naive sequential baseline is compared with balanced episodic replay using the same
data, initialization, optimizer settings, and task order.
"""

from __future__ import annotations

import argparse
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from src.qnn_temporal import make_temporal_qnode, predict  # noqa: E402
from src.temporal_data import TemporalTask, load_temporal_tasks  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
METHODS = ("baseline", "replay")


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/temporal_data.py",
        ROOT / "src/qnn_temporal.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("pennylane", "numpy", "scipy", "scikit-learn")
    }


def _score_pair(qnode, circuit_weights, head_weights, X, y) -> tuple[float, float]:
    output = np.asarray(predict(qnode, circuit_weights, head_weights, X))
    labels = np.where(output >= 0.0, 1, -1)
    accuracy = float(np.mean(labels == y))
    recalls = [float(np.mean(labels[y == label] == label)) for label in (-1, 1)]
    return accuracy, float(np.mean(recalls))


def _class_weights(labels: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights with mean one across both classes."""
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {-1, 1}:
        raise ValueError("class-balanced loss requires both {-1, +1} labels")
    counts = {label: int(np.sum(labels == label)) for label in (-1, 1)}
    return np.asarray(
        [len(labels) / (2.0 * counts[int(label)]) for label in labels],
        dtype=float,
    )


def _balanced_memory_indices(
    task: TemporalTask,
    memory_per_task: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if memory_per_task <= 0 or memory_per_task % 2:
        raise ValueError("memory_per_task must be a positive even number")
    per_class = memory_per_task // 2
    selected = []
    for label in (-1, 1):
        candidates = np.flatnonzero(task.y_train == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"{task.name} has only {len(candidates)} examples for label {label}; "
                f"cannot select {per_class}"
            )
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return rng.permutation(np.asarray(selected, dtype=int))


def _task_metrics(history: list[dict[str, Any]], epochs_per_task: int) -> dict[str, Any]:
    final = history[-1]
    boundary_rows = [
        history[index * epochs_per_task] for index in range(1, 4)
    ]
    metrics = {}
    for task_index in range(3):
        key = f"task{task_index + 1}"
        phase_end_epoch = (task_index + 1) * epochs_per_task
        phase_end = history[phase_end_epoch]
        boundary_balanced = [
            row["train_balanced_accuracy"][key]
            for row in boundary_rows[task_index:]
        ]
        task_metrics = {
            "phase_end_epoch": phase_end_epoch,
            "train_accuracy_at_phase_end": round(phase_end["train_accuracy"][key], 4),
            "train_accuracy_final": round(final["train_accuracy"][key], 4),
            "train_balanced_accuracy_at_phase_end": round(
                phase_end["train_balanced_accuracy"][key], 4
            ),
            "train_balanced_accuracy_final": round(
                final["train_balanced_accuracy"][key], 4
            ),
            "train_balanced_phase_end_drop": round(
                phase_end["train_balanced_accuracy"][key]
                - final["train_balanced_accuracy"][key],
                4,
            ),
            "train_balanced_forgetting": round(
                max(boundary_balanced) - boundary_balanced[-1], 4
            ),
        }
        if phase_end["test_accuracy"] is not None:
            boundary_test_balanced = [
                row["test_balanced_accuracy"][key]
                for row in boundary_rows[task_index:]
            ]
            task_metrics.update(
                {
                    "test_accuracy_at_phase_end": round(
                        phase_end["test_accuracy"][key], 4
                    ),
                    "test_accuracy_final": round(final["test_accuracy"][key], 4),
                    "test_balanced_accuracy_at_phase_end": round(
                        phase_end["test_balanced_accuracy"][key], 4
                    ),
                    "test_balanced_accuracy_final": round(
                        final["test_balanced_accuracy"][key], 4
                    ),
                    "test_balanced_phase_end_drop": round(
                        phase_end["test_balanced_accuracy"][key]
                        - final["test_balanced_accuracy"][key],
                        4,
                    ),
                    "test_balanced_forgetting": round(
                        max(boundary_test_balanced) - boundary_test_balanced[-1],
                        4,
                    ),
                }
            )
        metrics[key] = task_metrics

    metrics["summary"] = {
        "average_final_train_accuracy": round(
            float(np.mean(list(final["train_accuracy"].values()))), 4
        ),
        "average_final_train_balanced_accuracy": round(
            float(np.mean(list(final["train_balanced_accuracy"].values()))), 4
        ),
        "old_task_balanced_retention_final": round(
            float(
                np.mean(
                    [
                        final["train_balanced_accuracy"]["task1"],
                        final["train_balanced_accuracy"]["task2"],
                    ]
                )
            ),
            4,
        ),
        "new_task_balanced_adaptation": round(
            final["train_balanced_accuracy"]["task3"], 4
        ),
        "average_old_task_balanced_forgetting": round(
            float(
                np.mean(
                    [
                        metrics["task1"]["train_balanced_forgetting"],
                        metrics["task2"]["train_balanced_forgetting"],
                    ]
                )
            ),
            4,
        ),
    }
    if final["test_accuracy"] is not None:
        metrics["summary"]["average_final_test_accuracy"] = round(
            float(np.mean(list(final["test_accuracy"].values()))), 4
        )
        metrics["summary"]["average_final_test_balanced_accuracy"] = round(
            float(np.mean(list(final["test_balanced_accuracy"].values()))), 4
        )
    return metrics


def _run_method(
    *,
    method: str,
    tasks: tuple[TemporalTask, ...],
    qnode,
    initial_circuit: np.ndarray,
    initial_head: np.ndarray,
    memories: tuple[np.ndarray, ...],
    learning_rate: float,
    epochs_per_task: int,
    replay_weight: float,
    record_test: bool,
    verbose: bool,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    circuit_weights = pnp.array(initial_circuit, requires_grad=True)
    head_weights = pnp.array(initial_head, requires_grad=True)
    optimizer = qml.AdamOptimizer(learning_rate)
    task_keys = tuple(f"task{index + 1}" for index in range(len(tasks)))
    history: list[dict[str, Any]] = []
    objective_sample_exposures = 0

    def snapshot(
        *,
        epoch: int,
        phase: int,
        current_loss: float | None,
        replay_loss: float | None,
        objective_loss: float | None,
    ) -> None:
        train_scores = {
            key: _score_pair(
                qnode,
                circuit_weights,
                head_weights,
                task.X_train,
                task.y_train,
            )
            for key, task in zip(task_keys, tasks, strict=True)
        }
        test_scores = (
            {
                key: _score_pair(
                    qnode,
                    circuit_weights,
                    head_weights,
                    task.X_test,
                    task.y_test,
                )
                for key, task in zip(task_keys, tasks, strict=True)
            }
            if record_test
            else None
        )
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "current_task_loss": current_loss,
                "replay_loss": replay_loss,
                "objective_loss": objective_loss,
                "train_accuracy": {
                    key: scores[0] for key, scores in train_scores.items()
                },
                "train_balanced_accuracy": {
                    key: scores[1] for key, scores in train_scores.items()
                },
                "test_accuracy": (
                    {key: scores[0] for key, scores in test_scores.items()}
                    if test_scores is not None
                    else None
                ),
                "test_balanced_accuracy": (
                    {key: scores[1] for key, scores in test_scores.items()}
                    if test_scores is not None
                    else None
                ),
            }
        )

    snapshot(epoch=0, phase=0, current_loss=None, replay_loss=None, objective_loss=None)
    started = time.perf_counter()
    for phase, task in enumerate(tasks, start=1):
        X_current = pnp.array(task.X_train, requires_grad=False)
        y_current = pnp.array(task.y_train, requires_grad=False)
        current_weights = pnp.array(_class_weights(task.y_train), requires_grad=False)
        if method == "replay" and phase > 1:
            replay_features = np.concatenate(
                [tasks[index].X_train[memories[index]] for index in range(phase - 1)]
            )
            replay_labels = np.concatenate(
                [tasks[index].y_train[memories[index]] for index in range(phase - 1)]
            )
            X_replay = pnp.array(replay_features, requires_grad=False)
            y_replay = pnp.array(replay_labels, requires_grad=False)
            replay_weights = pnp.array(
                _class_weights(replay_labels), requires_grad=False
            )
        else:
            X_replay = None
            y_replay = None
            replay_weights = None

        if verbose:
            replay_count = 0 if X_replay is None else len(X_replay)
            print(
                f"  {method}: phase {phase} {task.name}, "
                f"current={len(X_current)}, replay={replay_count}",
                flush=True,
            )

        def losses(candidate_circuit, candidate_head):
            current = pnp.mean(
                current_weights
                * (
                    predict(qnode, candidate_circuit, candidate_head, X_current)
                    - y_current
                )
                ** 2
            )
            if X_replay is None:
                return current, None, current
            replay = pnp.mean(
                replay_weights
                * (
                    predict(qnode, candidate_circuit, candidate_head, X_replay)
                    - y_replay
                )
                ** 2
            )
            objective = (1.0 - replay_weight) * current + replay_weight * replay
            return current, replay, objective

        def cost(candidate_circuit, candidate_head):
            return losses(candidate_circuit, candidate_head)[2]

        for local_epoch in range(1, epochs_per_task + 1):
            circuit_weights, head_weights = optimizer.step(
                cost, circuit_weights, head_weights
            )
            objective_sample_exposures += len(X_current) + (
                0 if X_replay is None else len(X_replay)
            )
            current, replay, objective = losses(circuit_weights, head_weights)
            epoch = history[-1]["epoch"] + 1
            snapshot(
                epoch=epoch,
                phase=phase,
                current_loss=float(current),
                replay_loss=None if replay is None else float(replay),
                objective_loss=float(objective),
            )
            if verbose and (local_epoch == 1 or local_epoch % 5 == 0):
                values = history[-1]["train_balanced_accuracy"]
                print(
                    f"    epoch {epoch:3d}: "
                    + "  ".join(
                        f"T{index + 1} balanced={values[key]:.3f}"
                        for index, key in enumerate(task_keys)
                    ),
                    flush=True,
                )

    return {
        "history": history,
        "metrics": _task_metrics(history, epochs_per_task),
        "final_circuit_weights": np.asarray(circuit_weights).tolist(),
        "final_head_weights": np.asarray(head_weights).tolist(),
        "train_time_sec": round(time.perf_counter() - started, 3),
        "objective_sample_exposures": objective_sample_exposures,
    }


def run_experiment(
    *,
    n_qubits: int = 4,
    layers: int = 1,
    n_steps: int = 12,
    learning_rate: float = 0.03,
    epochs_per_task: int = 20,
    memory_per_task: int = 16,
    replay_weight: float = 0.5,
    seed: int = 42,
    record_test: bool = True,
    allow_download: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one fair seeded comparison of baseline and balanced replay."""
    if min(n_qubits, layers, n_steps, epochs_per_task, memory_per_task) <= 0:
        raise ValueError("model sizes, epochs, and memory size must be positive")
    if not 0.0 < learning_rate or not 0.0 < replay_weight < 1.0:
        raise ValueError("learning_rate must be positive and replay_weight must be in (0, 1)")

    tasks = load_temporal_tasks(
        n_steps=n_steps,
        allow_download=allow_download,
    )
    if len(tasks) != 3:
        raise ValueError("E006 requires exactly three temporal tasks")
    qnode, circuit_shape, head_shape = make_temporal_qnode(
        n_qubits=n_qubits,
        n_layers=layers,
        n_steps=n_steps,
    )
    rng = np.random.default_rng(seed)
    initial_circuit = 0.05 * rng.standard_normal(circuit_shape)
    initial_head = 0.10 * rng.standard_normal(head_shape)
    memory_rng = np.random.default_rng(seed + 10_000)
    memories = tuple(
        _balanced_memory_indices(task, memory_per_task, memory_rng) for task in tasks
    )

    if verbose:
        print("E006 task order: " + " -> ".join(task.name for task in tasks))
    methods = {
        method: _run_method(
            method=method,
            tasks=tasks,
            qnode=qnode,
            initial_circuit=initial_circuit,
            initial_head=initial_head,
            memories=memories,
            learning_rate=learning_rate,
            epochs_per_task=epochs_per_task,
            replay_weight=replay_weight,
            record_test=record_test,
            verbose=verbose,
        )
        for method in METHODS
    }
    baseline_exposures = methods["baseline"]["objective_sample_exposures"]
    for method in METHODS:
        methods[method]["relative_objective_sample_exposure"] = round(
            methods[method]["objective_sample_exposures"] / baseline_exposures,
            4,
        )

    classical_test_metrics = {}
    if record_test:
        for index, task in enumerate(tasks, start=1):
            model = LogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight="balanced",
            ).fit(task.X_train, task.y_train)
            predictions = model.predict(task.X_test)
            classical_test_metrics[f"task{index}"] = {
                "accuracy": round(float(np.mean(predictions == task.y_test)), 4),
                "balanced_accuracy": round(
                    float(balanced_accuracy_score(task.y_test, predictions)), 4
                ),
            }
    majority_baselines = {
        f"task{index}": {
            "train_accuracy": round(
                max(float(np.mean(task.y_train == label)) for label in (-1, 1)), 4
            ),
            "train_balanced_accuracy": 0.5,
        }
        for index, task in enumerate(tasks, start=1)
    }

    resources = qml.specs(qnode)(tasks[0].X_train[0], pnp.array(initial_circuit))[
        "resources"
    ]
    gate_types = dict(resources.gate_types)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e006_advanced_temporal_continual_learning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_sha256": _digest_arrays(
            *(
                array
                for task in tasks
                for array in (task.X_train, task.X_test, task.y_train, task.y_test)
            )
        ),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "dataset": {
            "task_order": [task.name for task in tasks],
            "n_steps": n_steps,
            "preprocessing": "per-series z-normalization; equal-width PAA; clip at 3 sigma; scale to [-pi, pi]",
            "official_train_test_splits": True,
            "tasks": {
                f"task{index}": {
                    "archive": task.spec.name,
                    "domain": task.spec.domain,
                    "source_url": task.spec.url,
                    "archive_sha256": task.spec.archive_sha256,
                    "original_length": task.spec.original_length,
                    "train_size": len(task.X_train),
                    "test_size": len(task.X_test),
                    "class_names": list(task.spec.class_names),
                    "train_class_counts": {
                        str(label): int(np.sum(task.y_train == label))
                        for label in (-1, 1)
                    },
                }
                for index, task in enumerate(tasks, start=1)
            },
        },
        "model": {
            "type": "hybrid recurrent/data-reuploading quantum temporal classifier",
            "device": "default.qubit",
            "shots": None,
            "n_qubits": n_qubits,
            "n_steps": n_steps,
            "shared_variational_layers_per_step": layers,
            "n_quantum_weights": int(np.prod(circuit_shape)),
            "n_classical_head_weights": int(np.prod(head_shape)),
            "readouts": [f"PauliZ({qubit})" for qubit in range(n_qubits)],
            "classical_head": "linear + tanh",
            "logical_depth": int(resources.depth),
            "two_qubit_gates": int(gate_types.get("CNOT", 0)),
        },
        "training": {
            "comparison": {
                "baseline": "E004 protocol: naive sequential fine-tuning",
                "replay": "balanced episodic replay",
            },
            "optimizer": "Adam",
            "loss": "inverse-class-frequency weighted MSE",
            "optimizer_state_reset_at_boundaries": False,
            "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task,
            "task_boundaries": [epochs_per_task, 2 * epochs_per_task],
            "memory_per_previous_task": memory_per_task,
            "memory_selection": "fixed, seeded, class-balanced, without replacement",
            "replay_weight": replay_weight,
            "seed": seed,
            "record_test_during_training": record_test,
            "test_used_for_selection": False,
            "initial_circuit_weights": initial_circuit.tolist(),
            "initial_head_weights": initial_head.tolist(),
            "memory_indices": {
                f"task{index + 1}": memory.tolist()
                for index, memory in enumerate(memories)
            },
        },
        "methods": methods,
        "majority_class_train_baselines": majority_baselines,
        "classical_balanced_logreg_test_metrics": classical_test_metrics,
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = RESULTS / f"e006_seed{result['training']['seed']}_{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.03, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--memory-per-task", type=int, default=16)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-test", action="store_false", dest="record_test")
    parser.add_argument("--offline", action="store_false", dest="allow_download")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_args = {key: value for key, value in vars(args).items() if key != "output"}
    result = run_experiment(**run_args)
    print(json.dumps({key: value["metrics"] for key, value in result["methods"].items()}, indent=2))
    print(f"Wrote {write_result(result, args.output)}")


if __name__ == "__main__":
    main()
