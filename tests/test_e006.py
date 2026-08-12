"""Unit, offline integration, and checked-artifact tests for E006 Advanced."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import pytest


def _spec(name: str = "Fixture", length: int = 8):
    from src.temporal_data import UCRSpec

    return UCRSpec(
        name=name,
        display_name=name,
        domain="fixture",
        url="https://example.invalid/fixture.zip",
        archive_sha256="a" * 64,
        train_size=8,
        test_size=4,
        original_length=length,
        class_labels=("negative", "positive"),
        class_names=("negative", "positive"),
    )


def test_ts_parser_validates_shape_labels_and_values():
    from src.temporal_data import _parse_ts

    spec = _spec(length=4)
    text = """@problemName Fixture
@univariate true
@data
0,1,2,3:negative
3,2,1,0:positive
"""
    features, labels = _parse_ts(text, spec, expected_size=2)
    np.testing.assert_array_equal(features, [[0, 1, 2, 3], [3, 2, 1, 0]])
    np.testing.assert_array_equal(labels, [-1, 1])
    with pytest.raises(ValueError, match="expected 3 rows"):
        _parse_ts(text, spec, expected_size=3)
    with pytest.raises(ValueError, match="unexpected labels"):
        _parse_ts(text.replace("positive", "unknown"), spec, expected_size=2)


def test_temporal_reduction_is_deterministic_finite_and_bounded():
    from src.temporal_data import _reduce_series

    features = np.asarray(
        [[0, 1, 2, 3, 4, 5, 6, 7], [2, 0, 2, 0, 2, 0, 2, 0]], dtype=float
    )
    first = _reduce_series(features, n_steps=4)
    second = _reduce_series(features, n_steps=4)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 4)
    assert np.all(np.isfinite(first))
    assert np.max(np.abs(first)) <= np.pi
    with pytest.raises(ValueError, match="constant"):
        _reduce_series(np.ones((2, 8)), n_steps=4)


def test_temporal_qnode_batch_range_and_two_parameter_gradients():
    from src.qnn_temporal import make_temporal_qnode, predict

    qnode, circuit_shape, head_shape = make_temporal_qnode(
        n_qubits=2, n_layers=1, n_steps=4
    )
    X = pnp.array(
        [[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]],
        requires_grad=False,
    )
    circuit = pnp.array(np.full(circuit_shape, 0.05), requires_grad=True)
    head = pnp.array(np.full(head_shape, 0.1), requires_grad=True)

    def cost(candidate_circuit, candidate_head):
        return pnp.mean(predict(qnode, candidate_circuit, candidate_head, X) ** 2)

    output = predict(qnode, circuit, head, X)
    circuit_gradient, head_gradient = qml.grad(cost, argnums=[0, 1])(circuit, head)
    assert output.shape == (2,)
    assert np.all(np.abs(np.asarray(output)) <= 1.0)
    assert circuit_gradient.shape == circuit_shape
    assert head_gradient.shape == head_shape
    assert np.all(np.isfinite(np.asarray(circuit_gradient)))
    assert np.all(np.isfinite(np.asarray(head_gradient)))


def test_balanced_memory_is_seeded_and_class_balanced():
    from experiments.e006_advanced_temporal import _balanced_memory_indices
    from src.temporal_data import TemporalTask

    task = TemporalTask(
        "fixture",
        np.arange(40).reshape(10, 4),
        np.asarray([-1] * 6 + [1] * 4),
        np.zeros((4, 4)),
        np.asarray([-1, -1, 1, 1]),
        _spec(length=4),
    )
    first = _balanced_memory_indices(task, 6, np.random.default_rng(7))
    second = _balanced_memory_indices(task, 6, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert dict(zip(*np.unique(task.y_train[first], return_counts=True), strict=True)) == {
        -1: 3,
        1: 3,
    }
    with pytest.raises(ValueError, match="positive even"):
        _balanced_memory_indices(task, 5, np.random.default_rng(7))


def test_e006_runs_paired_methods_offline(monkeypatch):
    import experiments.e006_advanced_temporal as experiment
    from src.temporal_data import TemporalTask

    rng = np.random.default_rng(9)

    def task(index: int) -> TemporalTask:
        train = rng.normal(size=(8, 4))
        test = rng.normal(size=(4, 4))
        return TemporalTask(
            f"task {index}",
            train,
            np.tile([-1, 1], 4),
            test,
            np.tile([-1, 1], 2),
            _spec(name=f"Fixture{index}", length=4),
        )

    tasks = tuple(task(index) for index in range(1, 4))
    monkeypatch.setattr(experiment, "load_temporal_tasks", lambda **_kwargs: tasks)
    result = experiment.run_experiment(
        n_qubits=2,
        layers=1,
        n_steps=4,
        learning_rate=0.02,
        epochs_per_task=1,
        memory_per_task=2,
        replay_weight=0.5,
        seed=3,
        verbose=False,
    )

    assert result["schema_version"] == 1
    assert result["model"]["shots"] is None
    assert result["training"]["test_used_for_selection"] is False
    assert set(result["methods"]) == {"baseline", "replay"}
    baseline = result["methods"]["baseline"]["history"]
    replay = result["methods"]["replay"]["history"]
    assert [row["phase"] for row in baseline] == [0, 1, 2, 3]
    assert baseline[:2] == replay[:2]
    assert replay[2]["replay_loss"] is not None
    assert baseline[2]["replay_loss"] is None
    assert set(baseline[0]["train_balanced_accuracy"]) == {
        "task1",
        "task2",
        "task3",
    }
    assert result["methods"]["replay"]["relative_objective_sample_exposure"] > 1.0
    assert result["methods"]["baseline"]["final_circuit_weights"] != result[
        "training"
    ]["initial_circuit_weights"]


def test_multiseed_summary_computes_paired_deltas(tmp_path):
    from scripts.run_e006_multiseed import build_summary

    root = Path(__file__).resolve().parents[1]

    def result(seed: int, baseline_retention: float, replay_retention: float):
        def method(retention: float, adaptation: float):
            task_metrics = {
                "phase_end_epoch": 1,
                "train_accuracy_at_phase_end": retention,
                "train_accuracy_final": retention,
                "train_balanced_accuracy_at_phase_end": retention,
                "train_balanced_accuracy_final": retention,
                "train_balanced_phase_end_drop": 0.0,
                "train_balanced_forgetting": 0.0,
            }
            return {
                "history": [
                    {
                        "epoch": 0,
                        "phase": 0,
                        "train_accuracy": {
                            "task1": retention,
                            "task2": retention,
                            "task3": adaptation,
                        },
                        "train_balanced_accuracy": {
                            "task1": retention,
                            "task2": retention,
                            "task3": adaptation,
                        },
                    }
                ],
                "metrics": {
                    "task1": task_metrics,
                    "task2": task_metrics,
                    "task3": task_metrics,
                    "summary": {
                        "average_final_train_accuracy": (2 * retention + adaptation) / 3,
                        "average_final_train_balanced_accuracy": (
                            2 * retention + adaptation
                        )
                        / 3,
                        "old_task_balanced_retention_final": retention,
                        "new_task_balanced_adaptation": adaptation,
                        "average_old_task_balanced_forgetting": 0.0,
                    },
                },
                "train_time_sec": 1.0 + retention,
                "objective_sample_exposures": 10 if retention < 0.8 else 12,
                "relative_objective_sample_exposure": 1.0 if retention < 0.8 else 1.2,
            }

        training = {
            "comparison": {"baseline": "baseline", "replay": "replay"},
            "optimizer": "Adam",
            "loss": "balanced MSE",
            "optimizer_state_reset_at_boundaries": False,
            "learning_rate": 0.02,
            "epochs_per_task": 1,
            "task_boundaries": [1, 2],
            "memory_per_previous_task": 2,
            "memory_selection": "balanced",
            "replay_weight": 0.5,
            "record_test_during_training": True,
            "test_used_for_selection": False,
            "seed": seed,
        }
        return {
            "source_code_sha256": "a" * 64,
            "data_sha256": "b" * 64,
            "dataset": {"task_order": ["one", "two", "three"]},
            "model": {"n_qubits": 2},
            "training": training,
            "methods": {
                "baseline": method(baseline_retention, 0.9),
                "replay": method(replay_retention, 0.8),
            },
        }

    results = [result(1, 0.5, 0.7), result(2, 0.6, 0.7), result(3, 0.7, 0.7)]
    paths = [root / f"results/fixture{seed}.json" for seed in (1, 2, 3)]
    summary = build_summary(results, paths)
    assert summary["paired_aggregate"]["old_task_balanced_retention_gain"] == {
        "mean": pytest.approx(0.1),
        "sample_std": pytest.approx(0.1),
    }
    assert summary["paired_aggregate"]["new_task_balanced_adaptation_change"][
        "mean"
    ] == pytest.approx(-0.1)
    assert summary["compute"]["paired_replay_minus_baseline"]["train_time_ratio"][
        "mean"
    ] > 1.0


def test_checked_in_e006_artifacts_match_current_sources():
    from experiments.e006_advanced_temporal import _source_digest
    from scripts.run_e006_multiseed import _aggregation_digest

    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "results/e006_advanced_summary.json").read_text(encoding="utf-8")
    )
    assert summary["source_code_sha256"] == _source_digest()
    assert summary["seeds"] == [42, 43, 44]
    assert summary["n_seeds"] == 3
    assert summary["aggregation_code_sha256"] == _aggregation_digest()
    assert len(set(summary["data_sha256_by_seed"].values())) == 1
    for seed in summary["seeds"]:
        result = json.loads(
            (root / f"results/e006_advanced_seed{seed}.json").read_text(encoding="utf-8")
        )
        assert result["source_code_sha256"] == _source_digest()
        assert result["training"]["seed"] == seed
        assert result["training"]["test_used_for_selection"] is False
        assert result["training"]["optimizer_state_reset_at_boundaries"] is False
        assert result["data_sha256"] == summary["data_sha256_by_seed"][str(seed)]
        baseline = result["methods"]["baseline"]
        replay = result["methods"]["replay"]
        assert len(baseline["history"]) == 61
        assert len(replay["history"]) == 61
        assert baseline["history"][:21] == replay["history"][:21]
        assert baseline["objective_sample_exposures"] == 3560
        assert replay["objective_sample_exposures"] == 4520
