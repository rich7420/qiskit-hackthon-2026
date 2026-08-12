"""Validation-only, single-seed tuning for E002.

The held-out test split is never evaluated here. After choosing a configuration, run the
E002 entrypoint once to create the reference result and its test metric.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_amplitude import load_mnist_amplitude_pca  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402


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


def accuracy(qnode, weights, features, labels) -> float:
    predictions = np.where(np.asarray(qnode(features, weights)) >= 0.0, 1, -1)
    return float(np.mean(predictions == labels))


def train(
    X_train,
    y_train,
    X_validation,
    y_validation,
    *,
    n_qubits: int,
    layers: int,
    lr: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Train one candidate and score it only on train/validation data."""
    qnode, weight_shape = make_qnode(n_qubits=n_qubits, n_layers=layers)
    weights = pnp.array(
        0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
        requires_grad=True,
    )
    X_train_p = pnp.array(X_train, requires_grad=False)
    y_train_p = pnp.array(y_train, requires_grad=False)

    def cost(candidate):
        return pnp.mean((qnode(X_train_p, candidate) - y_train_p) ** 2)

    optimizer = qml.AdamOptimizer(lr)
    best_validation = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        weights = optimizer.step(cost, weights)
        validation = accuracy(qnode, weights, X_validation, y_validation)
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch
    return {
        "final_validation_accuracy": accuracy(
            qnode, weights, X_validation, y_validation
        ),
        "best_validation_accuracy": best_validation,
        "best_epoch": best_epoch,
        "train_accuracy": accuracy(qnode, weights, X_train, y_train),
        "n_weights": int(np.prod(weight_shape)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-validation", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = load_mnist_amplitude_pca(
        n_features=16,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(
        f"data: train={len(splits.X_train)} validation={len(splits.X_validation)} "
        f"(test={len(splits.X_test)} held out), epochs={args.epochs}\n"
    )

    configs = [
        (6, 0.05),
        (10, 0.05),
        (10, 0.02),
        (15, 0.03),
        (20, 0.02),
        (30, 0.02),
        (30, 0.01),
    ]
    rows = []
    for layers, lr in configs:
        started = time.perf_counter()
        row = train(
            splits.X_train,
            splits.y_train,
            splits.X_validation,
            splits.y_validation,
            n_qubits=4,
            layers=layers,
            lr=lr,
            epochs=args.epochs,
            seed=args.seed,
        )
        row.update(
            {
                "layers": layers,
                "learning_rate": lr,
                "train_time_sec": round(time.perf_counter() - started, 3),
            }
        )
        rows.append(row)
        print(
            f"  layers={layers:2d} lr={lr:<4} weights={row['n_weights']:3d}  "
            f"final_val={row['final_validation_accuracy']:.3f} "
            f"best_val={row['best_validation_accuracy']:.3f}@{row['best_epoch']:02d} "
            f"train={row['train_accuracy']:.3f}",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            row["best_validation_accuracy"],
            row["final_validation_accuracy"],
            -row["n_weights"],
        ),
        reverse=True,
    )
    print("\n=== ranked by validation accuracy (test remains unobserved) ===")
    for row in rows:
        print(
            f"  best {row['best_validation_accuracy']:.3f}@{row['best_epoch']:02d} "
            f"final {row['final_validation_accuracy']:.3f}  "
            f"layers={row['layers']} lr={row['learning_rate']}"
        )

    result = {
        "schema_version": 1,
        "experiment": "e002_validation_tuning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "data_split_sha256": _digest_arrays(
            splits.X_train,
            splits.X_validation,
            splits.X_test,
            splits.y_train,
            splits.y_validation,
            splits.y_test,
        ),
        "environment": {
            name: version(name)
            for name in ("pennylane", "numpy", "scipy", "scikit-learn")
        },
        "dataset": {
            "source": splits.source,
            "n_train": len(splits.X_train),
            "n_validation": len(splits.X_validation),
            "n_test_held_out": len(splits.X_test),
        },
        "epochs": args.epochs,
        "seed": args.seed,
        "selection_metric": "best_validation_accuracy",
        "test_evaluations": 0,
        "results": rows,
    }
    output = ROOT / "results" / "tune_e002.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
