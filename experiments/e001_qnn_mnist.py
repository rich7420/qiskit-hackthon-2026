"""E001 - reproducible QNN binary-digit classification smoke test.

Run from an activated project environment:
    python experiments/e001_qnn_mnist.py --dataset mnist --seed 42

Use ``--dataset digits`` for a fully offline, explicitly different dataset.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier  # noqa: E402
from qiskit_machine_learning.optimizers import COBYLA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.data import DatasetName, load_mnist_binary  # noqa: E402
from src.qnn import build_estimator_qnn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
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
    for path in (ROOT / "src/data.py", ROOT / "src/qnn.py", Path(__file__)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    packages = ("qiskit", "qiskit-machine-learning", "numpy", "scipy", "scikit-learn")
    return {name: version(name) for name in packages}


def run_experiment(
    *,
    dataset: DatasetName = "mnist",
    n_qubits: int = 4,
    n_train: int = 100,
    n_test: int = 100,
    ansatz_reps: int = 1,
    maxiter: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run E001 in memory and return a self-contained, JSON-safe result."""
    if min(maxiter, ansatz_reps) <= 0:
        raise ValueError("maxiter and ansatz_reps must be positive")

    def report(message: str) -> None:
        if verbose:
            print(message)

    report(f"Loading {dataset} 0-vs-1, PCA -> {n_qubits} features ...")
    X_tr, X_te, y_tr, y_te, source = load_mnist_binary(
        n_features=n_qubits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        dataset=dataset,
    )
    report(f"  source={source}  train={len(X_tr)}  test={len(X_te)}")

    qnn = build_estimator_qnn(n_qubits=n_qubits, ansatz_reps=ansatz_reps, seed=seed)
    history: list[dict[str, float | int]] = []

    def accuracy_at(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        values = np.asarray(qnn.forward(features, weights)).reshape(-1)
        predictions = np.where(values >= 0.0, 1, -1)
        return float(np.mean(predictions == labels))

    def callback(weights, obj_value) -> None:
        history.append(
            {
                "evaluation": len(history) + 1,
                "objective": float(obj_value),
                "train_accuracy": accuracy_at(weights, X_tr, y_tr),
                "test_accuracy": accuracy_at(weights, X_te, y_te),
            }
        )
        if verbose and len(history) % 5 == 0:
            print(f"  evaluation {len(history):3d}  objective {obj_value:.4f}")

    initial_weights = np.random.default_rng(seed).uniform(-0.1, 0.1, qnn.num_weights)
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=maxiter),
        loss="squared_error",
        initial_point=initial_weights,
        callback=callback,
    )

    report(f"Training (COBYLA maxiter={maxiter}) on an exact statevector estimator ...")
    started = time.perf_counter()
    classifier.fit(X_tr, y_tr)
    train_time = time.perf_counter() - started

    baseline = LogisticRegression(max_iter=1000, random_state=seed).fit(X_tr, y_tr)
    decomposed = qnn.circuit.decompose(reps=20)
    fit_result = classifier.fit_result

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e001_qnn_binary_digits",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_split_sha256": _digest_arrays(X_tr, X_te, y_tr, y_te),
        "environment": {"python": platform.python_version(), "packages": _versions()},
        "dataset": {
            "requested": dataset,
            "actual": source,
            "classes": [0, 1],
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_features": n_qubits,
            "feature_range": [0.0, round(float(np.pi), 15)],
        },
        "model": {
            "type": "EstimatorQNN",
            "feature_map": "zz_feature_map(reps=1)",
            "ansatz": f"real_amplitudes(reps={ansatz_reps})",
            "observable": "Z" * n_qubits,
            "estimator": type(qnn.estimator).__name__,
            "shots": None,
            "n_qubits": n_qubits,
            "n_weights": qnn.num_weights,
            "decomposed_depth": decomposed.depth(),
            "two_qubit_gates": int(decomposed.count_ops().get("cx", 0)),
        },
        "training": {
            "optimizer": "COBYLA",
            "maxiter": maxiter,
            "seed": seed,
            "initial_weights": initial_weights.tolist(),
            "final_weights": np.asarray(classifier.weights).tolist(),
            "history": history,
            "final_objective": float(fit_result.fun),
            "function_evaluations": int(fit_result.nfev),
            "train_time_sec": round(train_time, 3),
        },
        "metrics": {
            "qnn_train_accuracy": round(float(classifier.score(X_tr, y_tr)), 4),
            "qnn_test_accuracy": round(float(classifier.score(X_te, y_te)), 4),
            "logreg_test_accuracy": round(float(baseline.score(X_te, y_te)), 4),
        },
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    """Write without replacing an earlier run unless an explicit path is supplied."""
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = RESULTS / (
            f"e001_qnn_{result['dataset']['actual']}_seed"
            f"{result['training']['seed']}_{stamp}.json"
        )
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "digits", "auto"), default="mnist")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--ansatz-reps", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_args = {key: value for key, value in vars(args).items() if key != "output"}
    result = run_experiment(**run_args)
    print("\n=== E001 result ===")
    print(json.dumps(result["metrics"], indent=2))
    output = write_result(result, args.output)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
