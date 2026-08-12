"""Unit and offline end-to-end regression tests for E001."""

import numpy as np
import pytest


def test_estimator_qnn_shapes():
    from src.qnn import build_estimator_qnn

    qnn = build_estimator_qnn(n_qubits=4, ansatz_reps=1)

    # 4 input features; RealAmplitudes(4, reps=1) has (reps+1)*4 = 8 weights.
    assert qnn.num_inputs == 4
    assert qnn.num_weights == 8


def test_estimator_qnn_rejects_single_qubit_zz_map():
    from src.qnn import build_estimator_qnn

    with pytest.raises(ValueError, match="at least 2"):
        build_estimator_qnn(n_qubits=1)


def test_estimator_qnn_forward_in_range():
    from src.qnn import build_estimator_qnn

    qnn = build_estimator_qnn(n_qubits=4)
    x = np.random.default_rng(0).random((5, 4))
    w = np.random.default_rng(1).random(qnn.num_weights)

    out = qnn.forward(x, w)

    # Single Z-parity expectation per sample, bounded in [-1, 1].
    assert out.shape == (5, 1)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_offline_data_is_bounded_and_reproducible():
    from src.data import load_mnist_binary

    first = load_mnist_binary(dataset="digits", n_train=40, n_test=20, seed=9)
    second = load_mnist_binary(dataset="digits", n_train=40, n_test=20, seed=9)

    assert first[-1] == "sklearn_digits_8x8"
    for left, right in zip(first[:-1], second[:-1], strict=True):
        np.testing.assert_array_equal(left, right)
    for features in first[:2]:
        assert np.all(features >= 0.0)
        assert np.all(features <= np.pi)
    assert set(first[2]) == {-1, 1}
    assert set(first[3]) == {-1, 1}


def test_mnist_failure_is_explicit(monkeypatch):
    import sklearn.datasets

    from src.data import load_mnist_binary

    def unavailable(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", unavailable)
    with pytest.raises(RuntimeError, match="dataset='digits'"):
        load_mnist_binary(dataset="mnist", n_train=20, n_test=10)

    with pytest.warns(RuntimeWarning, match="different dataset"):
        *_, source = load_mnist_binary(dataset="auto", n_train=20, n_test=10)
    assert source == "sklearn_digits_8x8"


def test_e001_runs_end_to_end_offline():
    from experiments.e001_qnn_mnist import run_experiment

    result = run_experiment(
        dataset="digits",
        n_qubits=2,
        n_train=20,
        n_test=10,
        maxiter=6,
        seed=7,
        verbose=False,
    )

    assert result["schema_version"] == 1
    assert result["dataset"]["actual"] == "sklearn_digits_8x8"
    assert result["model"]["estimator"] == "StatevectorEstimator"
    assert result["model"]["shots"] is None
    assert len(result["training"]["initial_weights"]) == result["model"]["n_weights"]
    assert len(result["training"]["final_weights"]) == result["model"]["n_weights"]
    assert result["training"]["history"]
    assert len(result["source_code_sha256"]) == 64
    assert len(result["data_split_sha256"]) == 64
    assert 0.0 <= result["metrics"]["qnn_test_accuracy"] <= 1.0


def test_result_writer_uses_requested_path(tmp_path):
    from experiments.e001_qnn_mnist import write_result

    output = tmp_path / "nested" / "result.json"
    result = {"dataset": {"actual": "test"}, "training": {"seed": 1}}

    assert write_result(result, output) == output
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_checked_in_reference_matches_current_sources():
    import json
    from pathlib import Path

    from experiments.e001_qnn_mnist import _source_digest

    root = Path(__file__).resolve().parents[1]
    reference = json.loads(
        (root / "results/e001_qnn_mnist_reference.json").read_text(encoding="utf-8")
    )

    assert reference["source_code_sha256"] == _source_digest()
    assert reference["dataset"]["actual"] == "mnist_784"
    assert reference["model"]["ansatz"] == "real_amplitudes(reps=2)"
    assert len(reference["training"]["history"]) == 60
    assert reference["metrics"]["qnn_test_accuracy"] == 0.78
