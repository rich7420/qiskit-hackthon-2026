"""E003 - sequential-training baseline: MNIST first, then Fashion-MNIST.

One QNN and one Adam optimizer are continued across the task boundary. Training and test
accuracy on both tasks are recorded after every epoch so learning and forgetting can be
distinguished explicitly.
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
    for path in (ROOT / "src/continual_data.py", ROOT / "src/qnn_pennylane.py", Path(__file__)):
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
    n_qubits: int = 4,
    layers: int = 20,
    learning_rate: float = 0.02,
    epochs_per_task: int = 40,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model sequentially and return a self-contained JSON-safe result."""
    if min(n_qubits, layers, epochs_per_task, n_train, n_test) <= 0 or learning_rate <= 0:
        raise ValueError("qubits, layers, epochs, split sizes, and learning rate must be positive")

    def report(message: str) -> None:
        if verbose:
            print(message)

    n_features = 2**n_qubits
    report("Loading MNIST -> Fashion-MNIST tasks in one MNIST-fitted PCA space ...")
    task1, task2 = load_two_tasks(
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    report(
        f"  T1={task1.name}  T2={task2.name}  "
        f"(train={len(task1.X_train)} test={len(task1.X_test)})"
    )

    qnode, weight_shape = make_qnode(n_qubits=n_qubits, n_layers=layers)
    initial_weights = 0.01 * np.random.default_rng(seed).standard_normal(weight_shape)
    weights = pnp.array(initial_weights, requires_grad=True)
    optimizer = qml.AdamOptimizer(learning_rate)
    report(
        f"  qubits={n_qubits} layers={layers} weights={int(np.prod(weight_shape))} "
        f"(one Adam optimizer, lr={learning_rate}, no consolidation)"
    )

    history: list[dict[str, Any]] = []

    def snapshot(*, epoch: int, phase: int, trained_task: str | None, active_loss=None) -> None:
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "trained_task": trained_task,
                "active_training_loss": None if active_loss is None else float(active_loss),
                "mnist_train_accuracy": _accuracy(
                    qnode, weights, task1.X_train, task1.y_train
                ),
                "mnist_test_accuracy": _accuracy(qnode, weights, task1.X_test, task1.y_test),
                "fashion_train_accuracy": _accuracy(
                    qnode, weights, task2.X_train, task2.y_train
                ),
                "fashion_test_accuracy": _accuracy(
                    qnode, weights, task2.X_test, task2.y_test
                ),
            }
        )

    # Epoch zero makes the before-training and task-boundary comparisons explicit.
    snapshot(epoch=0, phase=0, trained_task=None)
    started = time.perf_counter()

    def train_on(task: Task, phase: int) -> None:
        nonlocal weights
        X_train = pnp.array(task.X_train, requires_grad=False)
        y_train = pnp.array(task.y_train, requires_grad=False)

        def cost(candidate):
            return pnp.mean((qnode(X_train, candidate) - y_train) ** 2)

        for _ in range(epochs_per_task):
            weights = optimizer.step(cost, weights)
            epoch = history[-1]["epoch"] + 1
            loss = cost(weights)  # post-update, aligned with the accuracy snapshot
            snapshot(epoch=epoch, phase=phase, trained_task=task.name, active_loss=loss)
            if verbose and (epoch == 1 or epoch % 5 == 0):
                row = history[-1]
                print(
                    f"  epoch {epoch:3d} (T{phase})  "
                    f"MNIST train {row['mnist_train_accuracy']:.3f}  "
                    f"Fashion train {row['fashion_train_accuracy']:.3f}",
                    flush=True,
                )

    report(f"Phase 1: train on {task1.name} for {epochs_per_task} epochs ...")
    train_on(task1, 1)
    boundary_row = history[-1]
    report(
        f"Phase 2: continue the same model and Adam state on {task2.name} "
        f"for {epochs_per_task} epochs ..."
    )
    train_on(task2, 2)
    train_time = time.perf_counter() - started
    final_row = history[-1]

    mnist_baseline = LogisticRegression(max_iter=2000, random_state=seed).fit(
        task1.X_train, task1.y_train
    )
    fashion_baseline = LogisticRegression(max_iter=2000, random_state=seed).fit(
        task2.X_train, task2.y_train
    )
    resources = qml.specs(qnode)(task1.X_train[0], weights)["resources"]
    gate_types = dict(resources.gate_types)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e003_sequential_mnist_fashion_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_split_sha256": _digest_arrays(
            task1.X_train,
            task1.X_test,
            task1.y_train,
            task1.y_test,
            task2.X_train,
            task2.X_test,
            task2.y_train,
            task2.y_test,
        ),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "dataset": {
            "tasks": [task1.name, task2.name],
            "classes": [0, 1],
            "n_train_per_task": len(task1.X_train),
            "n_test_per_task": len(task1.X_test),
            "n_features": n_features,
            "preprocessing": "PCA fit on MNIST train only, shared across both tasks; L2 normalized",
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
            "task_order": [task1.name, task2.name],
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "optimizer_state_reset_at_boundary": False,
            "epochs_per_task": epochs_per_task,
            "task_boundary_epoch": epochs_per_task,
            "seed": seed,
            "initial_weights": initial_weights.tolist(),
            "final_weights": np.asarray(weights).tolist(),
            "history": history,
            "train_time_sec": round(train_time, 3),
        },
        "metrics": {
            "mnist_train_accuracy_at_boundary": round(
                boundary_row["mnist_train_accuracy"], 4
            ),
            "mnist_train_accuracy_final": round(final_row["mnist_train_accuracy"], 4),
            "mnist_train_forgetting": round(
                boundary_row["mnist_train_accuracy"] - final_row["mnist_train_accuracy"], 4
            ),
            "fashion_train_accuracy_before_phase2": round(
                boundary_row["fashion_train_accuracy"], 4
            ),
            "fashion_train_accuracy_final": round(
                final_row["fashion_train_accuracy"], 4
            ),
            "mnist_test_accuracy_at_boundary": round(
                boundary_row["mnist_test_accuracy"], 4
            ),
            "mnist_test_accuracy_final": round(final_row["mnist_test_accuracy"], 4),
            "fashion_test_accuracy_final": round(final_row["fashion_test_accuracy"], 4),
            "mnist_logreg_test_accuracy": round(
                float(mnist_baseline.score(task1.X_test, task1.y_test)), 4
            ),
            "fashion_logreg_test_accuracy": round(
                float(fashion_baseline.score(task2.X_test, task2.y_test)), 4
            ),
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    """Write a result without replacing an earlier run unless a path is explicit."""
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = RESULTS / f"e003_sequential_seed{result['training']['seed']}_{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    parser.add_argument("--epochs-per-task", type=int, default=40)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_args = {key: value for key, value in vars(args).items() if key != "output"}
    result = run_experiment(**run_args)
    print("\n=== E003 sequential-training result ===")
    print(json.dumps(result["metrics"], indent=2))
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
