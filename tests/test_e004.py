"""Unit, offline, and checked-artifact tests for E004."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_phase_hamiltonian_uses_open_boundaries():
    from src.phase_data import _X, _Y, _Z, _hamiltonian, _op

    h = 0.37
    expected = np.zeros((16, 16), dtype=complex)
    for center in (1, 2):
        expected -= _op({center - 1: _X, center: _Z, center + 1: _X}, 4)
    for left in (0, 1, 2):
        expected += h * _op({left: _Y, left + 1: _Y}, 4)

    actual = _hamiltonian(h)
    np.testing.assert_allclose(actual, expected.real, atol=1e-12)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)


@pytest.mark.parametrize("h", [0.0, 0.25, 0.5, 2.5, 2.75, 3.0])
def test_phase_ground_state_is_deterministic_normalized_and_even(h):
    from src.phase_data import _Z, _ground_state, _hamiltonian, _op

    first = _ground_state(h)
    second = _ground_state(h)
    ground_energy = np.linalg.eigvalsh(_hamiltonian(h))[0]
    parity = _op({qubit: _Z for qubit in range(4)}, 4).real

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0, atol=1e-12)
    np.testing.assert_allclose(_hamiltonian(h) @ first, ground_energy * first, atol=1e-10)
    np.testing.assert_allclose(first @ parity @ first, 1.0, atol=1e-10)
    assert first[np.argmax(np.abs(first))] > 0.0


def test_phase_loader_is_balanced_normalized_and_reproducible():
    from src.phase_data import load_spt_atf

    first = load_spt_atf(n_train=12, n_test=8, seed=9)
    second = load_spt_atf(n_train=12, n_test=8, seed=9)
    for left, right in zip(first[1:], second[1:], strict=True):
        np.testing.assert_array_equal(left, right)
    for features, labels in (
        (first.X_train, first.y_train),
        (first.X_test, first.y_test),
    ):
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)
        assert dict(zip(*np.unique(labels, return_counts=True), strict=True)) == {
            -1: len(labels) // 2,
            1: len(labels) // 2,
        }
    with pytest.raises(ValueError, match="positive even"):
        load_spt_atf(n_train=11, n_test=8)
    with pytest.raises(ValueError, match="n_qubits=4"):
        load_spt_atf(n_qubits=3)


def test_e004_runs_three_tasks_continuously_offline(monkeypatch):
    import experiments.e004_continual_three as experiment
    from src.continual_data import Task

    rng = np.random.default_rng(10)

    def make_task(name: str) -> Task:
        def normalized(count: int) -> np.ndarray:
            values = rng.normal(size=(count, 16))
            return values / np.linalg.norm(values, axis=1, keepdims=True)

        return Task(
            name,
            normalized(8),
            np.tile([-1, 1], 4),
            normalized(4),
            np.tile([-1, 1], 2),
        )

    task1 = make_task("MNIST 0/1")
    task2 = make_task("Fashion-MNIST 0/1")
    task3 = make_task("SPT/ATF phases")
    monkeypatch.setattr(experiment, "load_two_tasks", lambda **_kwargs: (task1, task2))
    monkeypatch.setattr(experiment, "load_spt_atf", lambda **_kwargs: task3)

    result = experiment.run_experiment(
        layers=1,
        learning_rate=0.02,
        epochs_per_task=1,
        n_train=8,
        n_test=4,
        seed=3,
        verbose=False,
    )

    history = result["training"]["history"]
    assert [row["phase"] for row in history] == [0, 1, 2, 3]
    assert result["training"]["optimizer_state_reset_at_boundaries"] is False
    assert result["training"]["task_boundaries"] == [1, 2]
    assert result["model"]["shots"] is None
    assert result["model"]["n_weights"] == 8
    assert len(result["source_code_sha256"]) == 64
    assert len(result["data_split_sha256"]) == 64
    assert result["training"]["initial_weights"] != result["training"]["final_weights"]
    for row in history:
        assert set(row["train_accuracy"]) == {"task1", "task2", "task3"}
        assert set(row["test_accuracy"]) == {"task1", "task2", "task3"}
        assert all(0.0 <= value <= 1.0 for value in row["train_accuracy"].values())


def test_multiseed_summary_uses_sample_standard_deviation():
    from scripts.run_e004_multiseed import build_summary

    def seeded(seed: int, accuracy: float):
        return {
            "source_code_sha256": "a" * 64,
            "data_split_sha256": str(seed) * 64,
            "dataset": {"n_train_per_task": 8, "n_test_per_task": 4},
            "model": {"n_qubits": 4},
            "training": {
                "seed": seed,
                "task_order": ["one", "two", "three"],
                "epochs_per_task": 1,
                "task_boundaries": [1, 2],
                "optimizer": "Adam",
                "learning_rate": 0.02,
                "history": [
                    {
                        "epoch": 0,
                        "phase": 0,
                        "train_accuracy": {
                            "task1": accuracy,
                            "task2": accuracy,
                            "task3": accuracy,
                        },
                    }
                ],
            },
            "metrics": {
                "tasks": {
                    key: {
                        "name": key,
                        "train_accuracy_at_phase_end": accuracy,
                        "train_accuracy_final": accuracy,
                        "train_accuracy_drop_since_phase_end": accuracy,
                        "test_accuracy_at_phase_end": accuracy,
                        "test_accuracy_final": accuracy,
                        "test_accuracy_drop_since_phase_end": accuracy,
                        "logreg_test_accuracy": accuracy,
                    }
                    for key in ("task1", "task2", "task3")
                }
            },
        }

    values = [seeded(1, 0.2), seeded(2, 0.4), seeded(3, 0.6)]
    paths = [Path(f"results/seed{seed}.json").resolve() for seed in (1, 2, 3)]
    summary = build_summary(values, paths)

    aggregate = summary["aggregate_history"][0]["train_accuracy"]["task1"]
    assert aggregate["mean"] == pytest.approx(0.4)
    assert aggregate["sample_std"] == pytest.approx(0.2)
    assert summary["uncertainty"] == "sample standard deviation across seeds (ddof=1)"


def test_checked_in_e004_artifacts_match_current_sources():
    from experiments.e004_continual_three import _source_digest

    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "results/e004_continual_summary.json").read_text(encoding="utf-8")
    )
    assert summary["source_code_sha256"] == _source_digest()
    assert len(summary["aggregation_code_sha256"]) == 64
    assert summary["seeds"] == [42, 43, 44]
    assert summary["n_seeds"] == 3
    assert len(summary["aggregate_history"]) == 61
    for seed in summary["seeds"]:
        result = json.loads(
            (root / f"results/e004_continual_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        assert result["source_code_sha256"] == _source_digest()
        assert result["training"]["seed"] == seed
        assert result["training"]["optimizer_state_reset_at_boundaries"] is False
        assert len(result["training"]["history"]) == 61
