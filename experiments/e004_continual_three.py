"""E004 - three-task sequential baseline: MNIST -> Fashion-MNIST -> SPT/ATF.

This is a paper-inspired engineering baseline, not an exact reproduction. One QNN and one
Adam optimizer are continued across all three tasks. Train and test accuracy on every task
are recorded at epoch zero and after each update.
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

from src.continual_data import Task, load_two_tasks  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1


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
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        ROOT / "src/qnn_pennylane.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    packages = ("pennylane", "numpy", "scipy", "scikit-learn")
    return {name: version(name) for name in packages}


def _accuracy(qnode, weights, features: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.where(np.asarray(qnode(features, weights)) >= 0.0, 1, -1)
    return float(np.mean(predictions == labels))


def run_experiment(
    *,
    n_qubits: int = N_QUBITS,
    layers: int = 20,
    learning_rate: float = 0.02,
    epochs_per_task: int = 20,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one seeded three-task sequence and return a JSON-safe result."""
    if n_qubits != N_QUBITS:
        raise ValueError(f"E004 requires n_qubits={N_QUBITS}")
    if min(layers, epochs_per_task, n_train, n_test) <= 0 or learning_rate <= 0:
        raise ValueError("layers, epochs, split sizes, and learning rate must be positive")

    def report(message: str) -> None:
        if verbose:
            print(message)

    report("Loading MNIST -> Fashion-MNIST -> SPT/ATF ...")
    task1, task2 = load_two_tasks(
        n_features=2**n_qubits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    task3 = load_spt_atf(
        n_train=n_train,
        n_test=n_test,
        n_qubits=n_qubits,
        seed=seed,
    )
    tasks: list[Task] = [task1, task2, task3]
    task_keys = ["task1", "task2", "task3"]
    report("  " + "  ".join(f"T{i + 1}={task.name}" for i, task in enumerate(tasks)))

    qnode, weight_shape = make_qnode(n_qubits=n_qubits, n_layers=layers)
    initial_weights = 0.01 * np.random.default_rng(seed).standard_normal(weight_shape)
    weights = pnp.array(initial_weights, requires_grad=True)
    optimizer = qml.AdamOptimizer(learning_rate)
    report(
        f"  qubits={n_qubits} layers={layers} weights={int(np.prod(weight_shape))} "
        f"(one Adam optimizer, lr={learning_rate}, no consolidation)"
    )

    boundaries = [epochs_per_task, 2 * epochs_per_task]
    history: list[dict[str, Any]] = []

    def snapshot(*, epoch: int, phase: int, trained_task: str | None, active_loss=None) -> None:
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "trained_task": trained_task,
                "active_training_loss": None if active_loss is None else float(active_loss),
                "train_accuracy": {
                    key: _accuracy(qnode, weights, task.X_train, task.y_train)
                    for key, task in zip(task_keys, tasks, strict=True)
                },
                "test_accuracy": {
                    key: _accuracy(qnode, weights, task.X_test, task.y_test)
                    for key, task in zip(task_keys, tasks, strict=True)
                },
            }
        )

    snapshot(epoch=0, phase=0, trained_task=None)
    started = time.perf_counter()
    for phase, task in enumerate(tasks, start=1):
        report(f"Phase {phase}: train on {task.name} for {epochs_per_task} epochs ...")
        X_train = pnp.array(task.X_train, requires_grad=False)
        y_train = pnp.array(task.y_train, requires_grad=False)

        def cost(candidate, X_train=X_train, y_train=y_train):
            return pnp.mean((qnode(X_train, candidate) - y_train) ** 2)

        for _ in range(epochs_per_task):
            weights = optimizer.step(cost, weights)
            epoch = history[-1]["epoch"] + 1
            loss = cost(weights)
            snapshot(epoch=epoch, phase=phase, trained_task=task.name, active_loss=loss)
            if verbose and (epoch == 1 or epoch % 5 == 0):
                values = history[-1]["train_accuracy"]
                print(
                    f"  epoch {epoch:3d} (T{phase})  "
                    + "  ".join(
                        f"T{i + 1} train {values[key]:.3f}"
                        for i, key in enumerate(task_keys)
                    ),
                    flush=True,
                )
    train_time = time.perf_counter() - started

    logistic_test_accuracy = {}
    for key, task in zip(task_keys, tasks, strict=True):
        baseline = LogisticRegression(max_iter=2000, random_state=seed).fit(
            task.X_train, task.y_train
        )
        logistic_test_accuracy[key] = round(
            float(baseline.score(task.X_test, task.y_test)), 4
        )

    final_row = history[-1]
    task_metrics = {}
    for index, (key, task) in enumerate(zip(task_keys, tasks, strict=True), start=1):
        phase_end_epoch = index * epochs_per_task
        phase_end_row = history[phase_end_epoch]
        task_metrics[key] = {
            "name": task.name,
            "phase_end_epoch": phase_end_epoch,
            "train_accuracy_at_phase_end": round(
                phase_end_row["train_accuracy"][key], 4
            ),
            "train_accuracy_final": round(final_row["train_accuracy"][key], 4),
            "train_accuracy_drop_since_phase_end": round(
                phase_end_row["train_accuracy"][key]
                - final_row["train_accuracy"][key],
                4,
            ),
            "test_accuracy_at_phase_end": round(
                phase_end_row["test_accuracy"][key], 4
            ),
            "test_accuracy_final": round(final_row["test_accuracy"][key], 4),
            "test_accuracy_drop_since_phase_end": round(
                phase_end_row["test_accuracy"][key]
                - final_row["test_accuracy"][key],
                4,
            ),
            "logreg_test_accuracy": logistic_test_accuracy[key],
        }

    resources = qml.specs(qnode)(task1.X_train[0], weights)["resources"]
    gate_types = dict(resources.gate_types)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e004_three_task_sequential_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_split_sha256": _digest_arrays(
            *(
                array
                for task in tasks
                for array in (task.X_train, task.X_test, task.y_train, task.y_test)
            )
        ),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "dataset": {
            "tasks": [task.name for task in tasks],
            "classes": {
                "task1": ["MNIST 0", "MNIST 1"],
                "task2": ["Fashion-MNIST 0", "Fashion-MNIST 1"],
                "task3": ["SPT (-1)", "ATF (+1)"],
            },
            "n_train_per_task": n_train,
            "n_test_per_task": n_test,
            "n_features": 2**n_qubits,
            "image_preprocessing": "PCA fit on MNIST train only, shared with Fashion-MNIST; L2 normalized",
            "phase_preparation": "open-boundary cluster-Ising exact ground states; even-parity representative",
            "phase_parameter_ranges": {"SPT": [0.0, 0.5], "ATF": [2.5, 3.0]},
        },
        "model": {
            "type": "PennyLane QNode",
            "device": "default.qubit",
            "shots": None,
            "n_qubits": n_qubits,
            "ansatz": f"independent RY/RZ + nearest-neighbor CNOT ({layers} layers)",
            "observable": "PauliZ(0)",
            "n_weights": int(np.prod(weight_shape)),
            "logical_depth": int(resources.depth),
            "two_qubit_gates": int(gate_types.get("CNOT", 0)),
        },
        "training": {
            "method": "naive sequential fine-tuning (no consolidation)",
            "task_order": [task.name for task in tasks],
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "optimizer_state_reset_at_boundaries": False,
            "epochs_per_task": epochs_per_task,
            "task_boundaries": boundaries,
            "seed": seed,
            "initial_weights": initial_weights.tolist(),
            "final_weights": np.asarray(weights).tolist(),
            "history": history,
            "train_time_sec": round(train_time, 3),
        },
        "metrics": {"tasks": task_metrics},
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    """Write a result without replacing an earlier run unless a path is explicit."""
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = RESULTS / f"e004_seed{result['training']['seed']}_{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_args = {key: value for key, value in vars(args).items() if key != "output"}
    result = run_experiment(**run_args)
    print("\n=== E004 result ===")
    print(json.dumps(result["metrics"], indent=2))
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
