"""Formulation, fairness, and offline integration tests for E008 MeasQCL."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from pennylane import numpy as pnp
import pytest


def test_zz_joint_probabilities_recover_classifier_z_scores():
    from src.measqcl_model import make_classifier_qnode, make_measurement_qnode

    classifier, shape = make_classifier_qnode(n_qubits=3, n_layers=1)
    zz_qnode = make_measurement_qnode("ZZ", n_qubits=3, n_layers=1)
    rng = np.random.default_rng(2)
    features = rng.normal(size=8)
    features /= np.linalg.norm(features)
    weights = pnp.array(rng.normal(scale=0.2, size=shape), requires_grad=True)

    score0, score1 = (float(value) for value in classifier(features, weights))
    probabilities = np.asarray(zz_qnode(features, weights))
    expected0 = probabilities[0] + probabilities[1] - probabilities[2] - probabilities[3]
    expected1 = probabilities[0] - probabilities[1] + probabilities[2] - probabilities[3]
    assert np.isclose(score0, expected0, atol=1e-9)
    assert np.isclose(score1, expected1, atol=1e-9)


def test_y_basis_rotation_identifies_plus_y_state():
    from src.measqcl_model import make_measurement_qnode

    qnode = make_measurement_qnode(
        "Y",
        n_qubits=1,
        n_layers=1,
        readout_wires=(0,),
    )
    plus_y = np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    zero_weights = pnp.array(np.zeros((1, 1, 2)), requires_grad=True)
    probabilities = np.asarray(qnode(plus_y, zero_weights))
    np.testing.assert_allclose(probabilities, [1.0, 0.0], atol=1e-9)


def test_measurement_fisher_matches_one_qubit_analytic_result():
    from src.measqcl_fisher import measurement_fisher_diag
    from src.measqcl_model import make_measurement_qnode

    qnode = make_measurement_qnode(
        "Z",
        n_qubits=1,
        n_layers=1,
        readout_wires=(0,),
    )
    features = np.asarray([[1.0, 0.0]])
    weights = pnp.array(np.asarray([[[0.61, 0.27]]]), requires_grad=True)
    fisher = measurement_fisher_diag(qnode, weights, features, [0])
    assert fisher.shape == (2,)
    assert np.isclose(fisher[0], 1.0, atol=1e-7)
    assert np.isclose(fisher[1], 0.0, atol=1e-9)


def test_accessible_fisher_normalization_and_allocation_optimizer():
    from src.measqcl_fisher import (
        accessible_fisher_diag,
        allocate_integer_shots,
        normalize_fisher_mass,
        optimize_measurement_allocation,
    )

    fishers = {
        "ZZ": np.asarray([10.0, 0.1]),
        "XX": np.asarray([0.1, 10.0]),
        "YY": np.asarray([0.0, 0.0]),
    }
    result = optimize_measurement_allocation(fishers)
    allocation = dict(zip(result.bases, result.weights, strict=True))
    assert allocation["ZZ"] == pytest.approx(0.5, abs=1e-6)
    assert allocation["XX"] == pytest.approx(0.5, abs=1e-6)
    assert allocation["YY"] == pytest.approx(0.0, abs=1e-8)
    accessible = accessible_fisher_diag(fishers, allocation)
    assert np.mean(normalize_fisher_mass(accessible)) == pytest.approx(1.0)
    shots = allocate_integer_shots(allocation, total_shots=11)
    assert sum(shots.values()) == 11
    assert abs(shots["ZZ"] - shots["XX"]) <= 1
    assert shots["YY"] == 0


def test_zero_mass_fisher_is_rejected():
    from src.measqcl_fisher import normalize_fisher_mass, optimize_measurement_allocation

    with pytest.raises(ValueError, match="zero-mass"):
        normalize_fisher_mass(np.zeros(3))
    with pytest.raises(ValueError, match="zero Fisher"):
        optimize_measurement_allocation({"ZZ": np.zeros(3)})


def test_reduced_state_qfi_matches_one_qubit_pure_state_and_bounds_cfi():
    from src.measqcl_fisher import measurement_fisher_diag, reduced_state_qfi_diag
    from src.measqcl_model import make_measurement_qnode, make_reduced_state_qnode

    features = np.asarray([[1.0, 0.0]])
    weights = pnp.array(np.asarray([[[0.61, 0.27]]]), requires_grad=True)
    reduced_qnode = make_reduced_state_qnode(
        n_qubits=1,
        n_layers=1,
        readout_wires=(0,),
    )
    qfi = reduced_state_qfi_diag(reduced_qnode, weights, features, [0])
    assert qfi[0] == pytest.approx(1.0, abs=1e-7)
    assert qfi[1] == pytest.approx(np.sin(0.61) ** 2, abs=1e-7)
    for basis in "XYZ":
        measurement_qnode = make_measurement_qnode(
            basis,
            n_qubits=1,
            n_layers=1,
            readout_wires=(0,),
        )
        cfi = measurement_fisher_diag(
            measurement_qnode,
            weights,
            features,
            [0],
        )
        assert np.all(cfi <= qfi + 1e-7)


def test_reduced_state_qfi_matches_rank_deficient_mixed_two_qubit_state():
    from src.measqcl_fisher import reduced_state_qfi_diag

    # This 4x4 state is the two-qubit reduction of
    # cos(theta/2)|000> + sin(theta/2)|011>. It is mixed for 0 < theta < pi,
    # rank deficient, and has the independent analytic SLD-QFI F_theta=1.
    def reduced_density(_features, weights):
        theta = float(np.asarray(weights).reshape(-1)[0])
        probabilities = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2, 0.0, 0.0]
        return np.diag(probabilities).astype(complex)

    weights = np.asarray([[[0.73]]])
    qfi = reduced_state_qfi_diag(
        reduced_density,
        weights,
        np.asarray([[1.0]]),
        [0],
    )
    assert qfi.shape == (1,)
    assert qfi[0] == pytest.approx(1.0, abs=1e-9)


def test_e008_runs_all_paired_methods_offline(monkeypatch):
    import experiments.e008_measqcl as experiment
    from src.continual_data import Task

    rng = np.random.default_rng(8)

    def make_task(name: str) -> Task:
        train = rng.normal(size=(12, 4))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test = rng.normal(size=(6, 4))
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        return Task(
            name,
            train,
            np.tile([-1, 1], 6),
            test,
            np.tile([-1, 1], 3),
        )

    tasks = (make_task("task A"), make_task("task B"))
    monkeypatch.setattr(experiment, "load_two_tasks", lambda **_kwargs: tasks)
    result = experiment.run_experiment(
        n_qubits=2,
        layers=1,
        learning_rate=0.02,
        epochs_per_task=1,
        ewc_lambda=0.5,
        fisher_samples=2,
        reference_shots=10,
        n_train=12,
        n_test=6,
        seed=5,
        verbose=False,
    )

    assert set(result["histories"]) == set(experiment.METHODS)
    common_histories = [
        result["histories"][method][:2] for method in experiment.METHODS
    ]
    assert all(history == common_histories[0] for history in common_histories[1:])
    assert result["training"]["phase1_trained_once_then_branched"] is True
    assert result["data"]["test_used_for_selection"] is False
    assert result["model"]["prediction_measurement_unchanged"] is True
    assert result["claim_boundaries"]["finite_shot_result"] is False
    assert "readout_qewc" in result["histories"]
    assert len(result["fisher_profiles"]["anchor_indices"]) == 2
    assert sum(
        result["fisher_profiles"]["integer_shot_plans"]["mof_ewc"].values()
    ) == 10
    for method in experiment.METHODS[1:]:
        normalized = np.asarray(
            result["fisher_profiles"]["normalized_method_fisher"][method]
        )
        assert np.mean(normalized) == pytest.approx(1.0)
        assert np.all(np.isfinite(normalized))


def test_multiseed_summary_rejects_incompatible_runs(monkeypatch):
    import experiments.e008_measqcl as experiment
    from scripts.run_e008_multiseed import build_summary
    from src.continual_data import Task

    rng = np.random.default_rng(18)

    def make_task(name: str) -> Task:
        train = rng.normal(size=(8, 4))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test = rng.normal(size=(4, 4))
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        return Task(name, train, np.tile([-1, 1], 4), test, np.tile([-1, 1], 2))

    monkeypatch.setattr(
        experiment,
        "load_two_tasks",
        lambda **_kwargs: (make_task("task A"), make_task("task B")),
    )
    run = experiment.run_experiment(
        n_qubits=2,
        layers=1,
        epochs_per_task=1,
        fisher_samples=2,
        reference_shots=10,
        n_train=8,
        n_test=4,
        seed=1,
        verbose=False,
    )
    second = copy.deepcopy(run)
    second["training"]["seed"] = 2
    second["data_sha256"] = "different-split"
    root = Path(__file__).resolve().parents[1]
    paths = [root / "results/fake_seed1.json", root / "results/fake_seed2.json"]
    with pytest.raises(ValueError, match="incompatible configurations"):
        second["training"]["learning_rate"] = 0.03
        build_summary([run, second], paths)
    second["training"]["learning_rate"] = run["training"]["learning_rate"]
    summary = build_summary([run, second], paths)
    assert summary["seeds"] == [1, 2]
    assert summary["n_seeds"] == 2
    assert summary["configuration"]["model"]["prediction_measurement_unchanged"] is True


def test_checked_in_e008_artifacts_match_current_sources_and_invariants():
    from experiments.e008_measqcl import METHODS, _source_digest
    from scripts.run_e008_multiseed import _aggregation_digest, build_summary
    from scripts.tune_e008_train_only import _source_digest as tuning_source_digest

    root = Path(__file__).resolve().parents[1]
    paths = [root / f"results/e008_measqcl_seed{seed}.json" for seed in (42, 43, 44)]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    stored = json.loads(
        (root / "results/e008_measqcl_summary.json").read_text(encoding="utf-8")
    )
    tuning = json.loads(
        (root / "results/e008_train_only_tuning.json").read_text(encoding="utf-8")
    )
    source_digest = _source_digest()
    assert all(run["source_code_sha256"] == source_digest for run in runs)
    assert stored["source_code_sha256"] == source_digest
    assert stored["aggregation_code_sha256"] == _aggregation_digest()
    assert tuning["source_code_sha256"] == tuning_source_digest()
    assert tuning["data_sha256"] == runs[0]["data_sha256"]
    assert tuning["test_evaluations"] == 0
    assert tuning["capacity_scan"]["selected"] == {
        "layers": 10,
        "learning_rate": 0.02,
    }
    assert tuning["lambda_scan"]["selected"] == 0.1
    assert all(
        row["test_evaluations"] == 0
        for scan in ("capacity_scan", "lambda_scan")
        for row in tuning[scan]["results"]
    )

    for run in runs:
        assert run["data"]["test_used_for_selection"] is False
        assert run["training"]["phase1_trained_once_then_branched"] is True
        common = [run["histories"][method][:41] for method in METHODS]
        assert all(history == common[0] for history in common[1:])
        profiles = run["fisher_profiles"]
        assert profiles["finite_shot_estimation_performed"] is False
        for allocation in profiles["allocations"].values():
            assert sum(allocation.values()) == pytest.approx(1.0)
        for plan in profiles["integer_shot_plans"].values():
            assert sum(plan.values()) == 1024
        for fisher in profiles["normalized_method_fisher"].values():
            values = np.asarray(fisher)
            assert np.all(np.isfinite(values))
            assert np.mean(values) == pytest.approx(1.0)
        hierarchy = profiles["geometry_hierarchy_min_margin"]
        assert min(hierarchy["readout_qfi_minus_each_basis_cfi"].values()) >= -1e-7
        assert hierarchy["full_qfi_minus_readout_qfi"] >= -1e-7

    rebuilt = build_summary(runs, paths)
    rebuilt["created_at_utc"] = stored["created_at_utc"]
    assert rebuilt == stored
