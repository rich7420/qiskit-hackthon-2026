"""Unit and offline regression tests for E003 sequential training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _raw_binary(seed: int, count: int = 60, width: int = 8):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], count // 2)
    features = rng.normal(size=(count, width))
    features[labels == 1, :2] += 1.5
    return features, labels


def test_shared_pca_is_deterministic_and_mnist_fitted_only():
    from src.continual_data import _build_tasks

    mnist_X, mnist_y = _raw_binary(1)
    fashion_X, fashion_y = _raw_binary(2)
    first = _build_tasks(
        mnist_X,
        mnist_y,
        fashion_X,
        fashion_y,
        n_features=4,
        n_train=40,
        n_test=20,
        seed=7,
    )
    second = _build_tasks(
        mnist_X,
        mnist_y,
        fashion_X + 1000.0,
        fashion_y,
        n_features=4,
        n_train=40,
        n_test=20,
        seed=7,
    )

    # Changing Fashion-MNIST cannot change the MNIST-fitted PCA or Task 1 arrays.
    for left, right in zip(first[0][1:], second[0][1:], strict=True):
        np.testing.assert_array_equal(left, right)
    for task in first:
        np.testing.assert_allclose(np.linalg.norm(task.X_train, axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(task.X_test, axis=1), 1.0)
        assert set(task.y_train) == {-1, 1}
        assert set(task.y_test) == {-1, 1}


def test_zero_vector_is_rejected_as_continual_amplitude_state():
    from src.continual_data import _normalize_amplitudes

    with pytest.raises(ValueError, match="all-zero"):
        _normalize_amplitudes(np.zeros((1, 4)))


def test_e003_runs_sequentially_offline(monkeypatch):
    import experiments.e003_continual_baseline as experiment
    from src.continual_data import Task

    rng = np.random.default_rng(10)

    def normalized(count: int) -> np.ndarray:
        values = rng.normal(size=(count, 4))
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    task1 = Task(
        "MNIST 0/1",
        normalized(16),
        np.tile([-1, 1], 8),
        normalized(8),
        np.tile([-1, 1], 4),
    )
    task2 = Task(
        "Fashion-MNIST 0/1",
        normalized(16),
        np.tile([-1, 1], 8),
        normalized(8),
        np.tile([-1, 1], 4),
    )
    monkeypatch.setattr(experiment, "load_two_tasks", lambda **_kwargs: (task1, task2))

    result = experiment.run_experiment(
        n_qubits=2,
        layers=1,
        learning_rate=0.02,
        epochs_per_task=2,
        n_train=16,
        n_test=8,
        seed=3,
        verbose=False,
    )

    history = result["training"]["history"]
    assert result["schema_version"] == 1
    assert result["training"]["task_order"] == ["MNIST 0/1", "Fashion-MNIST 0/1"]
    assert result["training"]["optimizer_state_reset_at_boundary"] is False
    assert [row["phase"] for row in history] == [0, 1, 1, 2, 2]
    assert [row["trained_task"] for row in history] == [
        None,
        "MNIST 0/1",
        "MNIST 0/1",
        "Fashion-MNIST 0/1",
        "Fashion-MNIST 0/1",
    ]
    for row in history:
        for key in (
            "mnist_train_accuracy",
            "mnist_test_accuracy",
            "fashion_train_accuracy",
            "fashion_test_accuracy",
        ):
            assert 0.0 <= row[key] <= 1.0
    assert result["model"]["shots"] is None
    assert len(result["source_code_sha256"]) == 64
    assert len(result["data_split_sha256"]) == 64
    assert result["training"]["initial_weights"] != result["training"]["final_weights"]


def test_checked_in_e003_reference_matches_current_sources():
    from experiments.e003_continual_baseline import _source_digest

    root = Path(__file__).resolve().parents[1]
    reference = json.loads(
        (root / "results/e003_continual_baseline_reference.json").read_text(
            encoding="utf-8"
        )
    )

    assert reference["source_code_sha256"] == _source_digest()
    assert reference["training"]["seed"] == 42
    assert reference["training"]["task_order"] == ["MNIST 0/1", "Fashion-MNIST 0/1"]
    assert reference["training"]["optimizer_state_reset_at_boundary"] is False
    assert len(reference["training"]["history"]) == 81
    assert reference["training"]["history"][40]["phase"] == 1
    assert reference["training"]["history"][41]["phase"] == 2
