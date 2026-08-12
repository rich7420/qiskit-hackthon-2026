"""Unit and offline regression tests for E002."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pennylane as qml
import pytest


def _mock_mnist(seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array(["0", "1"]), 80)
    features = rng.normal(size=(len(labels), 20))
    features[labels == "1", :4] += 2.0
    return SimpleNamespace(data=features, target=labels)


def test_amplitude_loader_is_reproducible_and_normalized(monkeypatch):
    import sklearn.datasets

    from src.data_amplitude import load_mnist_amplitude_pca

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", lambda *_args, **_kwargs: _mock_mnist())
    first = load_mnist_amplitude_pca(
        n_features=4, n_train=60, n_validation=20, n_test=20, seed=7
    )
    second = load_mnist_amplitude_pca(
        n_features=4, n_train=60, n_validation=20, n_test=20, seed=7
    )

    assert first.source == "mnist_784"
    for left, right in zip(first[:-1], second[:-1], strict=True):
        np.testing.assert_array_equal(left, right)
    for features in first[:3]:
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)
    for labels in first[3:6]:
        assert set(labels) == {-1, 1}


def test_pca_is_fit_on_training_split_only():
    from src.data_amplitude import _fit_transform_pca

    rng = np.random.default_rng(4)
    train = rng.normal(size=(20, 6))
    validation = rng.normal(size=(5, 6))
    test = rng.normal(size=(5, 6))

    transformed, *_ = _fit_transform_pca(
        train, validation, test, n_features=3, seed=9
    )
    transformed_after_holdout_change, *_ = _fit_transform_pca(
        train,
        validation + 10_000,
        test - 10_000,
        n_features=3,
        seed=9,
    )
    np.testing.assert_allclose(transformed, transformed_after_holdout_change)


def test_zero_vector_is_rejected_as_amplitude_state():
    from src.data_amplitude import _normalize_amplitudes

    with pytest.raises(ValueError, match="all-zero"):
        _normalize_amplitudes(np.zeros((1, 4)))


def test_amplitude_qnode_batch_range_and_gradient():
    from pennylane import numpy as pnp

    from src.qnn_pennylane import make_qnode

    qnode, weight_shape = make_qnode(n_qubits=2, n_layers=1)
    features = pnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    weights = pnp.array(np.full(weight_shape, 0.1), requires_grad=True)
    output = qnode(features, weights)
    gradient = qml.grad(lambda candidate: pnp.mean(qnode(features, candidate) ** 2))(weights)

    assert output.shape == (2,)
    assert np.all(np.asarray(output) >= -1.0) and np.all(np.asarray(output) <= 1.0)
    assert gradient.shape == weight_shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_e002_runs_end_to_end_offline(monkeypatch):
    import experiments.e002_amplitude_qnn as experiment
    from src.data_amplitude import AmplitudeSplits

    rng = np.random.default_rng(12)

    def normalized(count: int) -> np.ndarray:
        values = rng.normal(size=(count, 4))
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    splits = AmplitudeSplits(
        normalized(16),
        normalized(8),
        normalized(8),
        np.tile([-1, 1], 8),
        np.tile([-1, 1], 4),
        np.tile([-1, 1], 4),
        "offline_fixture",
    )
    monkeypatch.setattr(experiment, "load_mnist_amplitude_pca", lambda **_kwargs: splits)

    result = experiment.run_experiment(
        n_qubits=2,
        layers=1,
        epochs=2,
        n_train=16,
        n_validation=8,
        n_test=8,
        seed=3,
        verbose=False,
    )

    assert result["schema_version"] == 1
    assert result["dataset"]["actual"] == "offline_fixture"
    assert result["model"]["shots"] is None
    assert result["model"]["n_weights"] == 4
    assert len(result["training"]["history"]) == 2
    assert "test_accuracy" not in result["training"]["history"][0]
    assert len(result["source_code_sha256"]) == 64
    assert len(result["data_split_sha256"]) == 64
    assert 0.0 <= result["metrics"]["qnn_test_accuracy"] <= 1.0


def test_checked_in_e002_reference_matches_current_sources():
    from experiments.e002_amplitude_qnn import _source_digest

    root = Path(__file__).resolve().parents[1]
    reference = json.loads(
        (root / "results/e002_amplitude_qnn_reference.json").read_text(encoding="utf-8")
    )

    assert reference["source_code_sha256"] == _source_digest()
    assert reference["training"]["seed"] == 42
    assert reference["training"]["epochs"] == 75
    assert len(reference["training"]["history"]) == 75
    assert "test_accuracy" not in reference["training"]["history"][0]
