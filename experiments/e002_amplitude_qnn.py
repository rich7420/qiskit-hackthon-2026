"""E002 - single-seed, paper-inspired amplitude QNN on binary MNIST.

Run the frozen reference configuration with::

    python experiments/e002_amplitude_qnn.py \
      --output results/e002_amplitude_qnn_reference.json
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

from src.data_amplitude import load_mnist_amplitude_pca  # noqa: E402
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
    for path in (ROOT / "src/data_amplitude.py", ROOT / "src/qnn_pennylane.py", Path(__file__)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    packages = ("pennylane", "numpy", "scipy", "scikit-learn")
    return {name: version(name) for name in packages}


def _accuracy(qnode, weights, features: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(qnode(features, weights))
    predictions = np.where(values >= 0.0, 1, -1)
    return float(np.mean(predictions == labels))


def run_experiment(
    *,
    n_qubits: int = 4,
    layers: int = 30,
    epochs: int = 75,
    lr: float = 0.01,
    n_train: int = 600,
    n_validation: int = 200,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the frozen configuration and evaluate its held-out test split once."""
    if min(n_qubits, layers, epochs, n_train, n_validation, n_test) <= 0 or lr <= 0:
        raise ValueError("qubits, layers, epochs, split sizes, and learning rate must be positive")

    def report(message: str) -> None:
        if verbose:
            print(message)

    n_features = 2**n_qubits
    report(f"Loading MNIST 0-vs-1, PCA -> {n_features} amplitude features ...")
    splits = load_mnist_amplitude_pca(
        n_features=n_features,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        seed=seed,
    )
    X_train, X_validation, X_test = (
        splits.X_train,
        splits.X_validation,
        splits.X_test,
    )
    y_train, y_validation, y_test = (
        splits.y_train,
        splits.y_validation,
        splits.y_test,
    )
    report(
        f"  source={splits.source}  train={len(X_train)} "
        f"validation={len(X_validation)} test={len(X_test)}"
    )

    qnode, weight_shape = make_qnode(n_qubits=n_qubits, n_layers=layers)
    initial_weights = 0.01 * np.random.default_rng(seed).standard_normal(weight_shape)
    weights = pnp.array(initial_weights, requires_grad=True)
    X_train_p = pnp.array(X_train, requires_grad=False)
    y_train_p = pnp.array(y_train, requires_grad=False)

    def cost(candidate):
        return pnp.mean((qnode(X_train_p, candidate) - y_train_p) ** 2)

    optimizer = qml.AdamOptimizer(lr)
    history: list[dict[str, float | int]] = []
    report(f"Training Adam(lr={lr}) for {epochs} epochs ...")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        weights = optimizer.step(cost, weights)
        loss = float(cost(weights))
        train_accuracy = _accuracy(qnode, weights, X_train, y_train)
        validation_accuracy = _accuracy(qnode, weights, X_validation, y_validation)
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"  epoch {epoch:3d}  loss {loss:.4f}  train {train_accuracy:.3f} "
                f"validation {validation_accuracy:.3f}",
                flush=True,
            )
    train_time = time.perf_counter() - started

    # The test split is first touched here, after architecture and epoch count are frozen.
    qnn_test_accuracy = _accuracy(qnode, weights, X_test, y_test)
    baseline = LogisticRegression(max_iter=2000, random_state=seed).fit(X_train, y_train)
    baseline_test_accuracy = float(baseline.score(X_test, y_test))

    specification = qml.specs(qnode)(X_train[0], weights)
    resources = specification["resources"]
    gate_types = dict(resources.gate_types)
    final_weights = np.asarray(weights)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e002_amplitude_qnn",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_split_sha256": _digest_arrays(
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        ),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "dataset": {
            "actual": splits.source,
            "classes": [0, 1],
            "n_train": len(X_train),
            "n_validation": len(X_validation),
            "n_test": len(X_test),
            "n_features": n_features,
            "preprocessing": "train-only PCA followed by per-sample L2 normalization",
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
            "optimizer": "Adam",
            "learning_rate": lr,
            "epochs": epochs,
            "seed": seed,
            "selection": "layers/lr/epochs fixed using validation-only tuning",
            "initial_weights": initial_weights.tolist(),
            "final_weights": final_weights.tolist(),
            "history": history,
            "train_time_sec": round(train_time, 3),
        },
        "metrics": {
            "qnn_train_accuracy": round(history[-1]["train_accuracy"], 4),
            "qnn_validation_accuracy": round(history[-1]["validation_accuracy"], 4),
            "qnn_test_accuracy": round(qnn_test_accuracy, 4),
            "logreg_test_accuracy": round(baseline_test_accuracy, 4),
            "final_loss": round(history[-1]["loss"], 8),
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    """Write a result without replacing an earlier run unless a path is explicit."""
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = RESULTS / f"e002_amplitude_qnn_seed{result['training']['seed']}_{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-validation", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_args = {key: value for key, value in vars(args).items() if key != "output"}
    result = run_experiment(**run_args)
    print("\n=== E002 result ===")
    print(json.dumps(result["metrics"], indent=2))
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
