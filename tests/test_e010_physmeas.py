"""Theory and estimator contracts for task-relevant PhysMeas-QCL."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from pennylane import numpy as pnp
import pytest

from src.e005_softmax import classical_fisher_diag
from src.measqcl_task_relevance import (
    exact_fisher_from_probability_table,
    finite_shot_fisher_from_probability_table,
    normalize_task_relevance,
    optimize_task_relevant_allocation,
    reversed_logits_fisher_diag,
)


def test_reversed_logits_prevents_high_confidence_importance_collapse():
    def qnode(features, weights):
        batch = pnp.ones(len(features))
        return weights[0] * batch, -weights[0] * batch

    features = np.ones((4, 1))
    labels = np.full(4, -1)
    weights = pnp.array([1.0], requires_grad=True)
    vanilla = classical_fisher_diag(qnode, weights, features, labels)
    reversed_fisher = reversed_logits_fisher_diag(qnode, weights, features, labels)
    assert vanilla.shape == reversed_fisher.shape == (1,)
    assert reversed_fisher[0] > 20.0 * vanilla[0]
    with pytest.raises(ValueError, match="project convention"):
        reversed_logits_fisher_diag(qnode, weights, features, np.zeros(4))


def test_task_relevant_allocation_prioritizes_old_task_direction():
    fishers = {
        "ZZ": np.array([10.0, 0.1]),
        "XX": np.array([0.1, 10.0]),
    }
    result = optimize_task_relevant_allocation(
        fishers,
        np.array([100.0, 1.0]),
        minimum_allocation=0.01,
    )
    allocation = dict(zip(result.bases, result.weights, strict=True))
    assert sum(allocation.values()) == pytest.approx(1.0)
    assert allocation["ZZ"] > 0.9
    assert allocation["XX"] >= 0.01 - 1e-9


def test_task_relevance_floor_is_finite_and_mean_one():
    relevance = normalize_task_relevance(np.array([0.0, 0.0, 2.0]), floor=0.01)
    assert np.all(relevance > 0.0)
    assert np.mean(relevance) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero task relevance"):
        normalize_task_relevance(np.zeros(3))


def test_exact_and_finite_shot_fisher_from_known_binary_table():
    table = {
        "base": np.array([[0.5, 0.5]]),
        "plus": np.array([[[0.7, 0.3]]]),
        "minus": np.array([[[0.3, 0.7]]]),
    }
    exact = exact_fisher_from_probability_table(table)
    assert exact == pytest.approx([0.16])
    estimate = finite_shot_fisher_from_probability_table(
        table,
        shots_per_circuit=20_000,
        repetitions=80,
        seed=7,
    )
    assert estimate.mean == pytest.approx(exact, abs=0.01)
    assert estimate.sample_std[0] > 0.0
    assert estimate.circuits_per_repetition == 3
    assert estimate.total_shots == 3 * 20_000 * 80
    assert estimate.estimates.shape == (80, 1)


def test_probability_table_validation_rejects_nonprobabilities():
    table = {
        "base": np.array([[0.6, 0.6]]),
        "plus": np.array([[[0.7, 0.3]]]),
        "minus": np.array([[[0.3, 0.7]]]),
    }
    with pytest.raises(ValueError, match="sum to one"):
        exact_fisher_from_probability_table(table)


def test_e010_reuses_verified_parent_and_changes_only_new_branches(monkeypatch):
    import experiments.e008_measqcl as e008
    import experiments.e010_physmeas_qcl as e010
    from src.continual_data import Task

    rng = np.random.default_rng(10)

    def make_task(name: str) -> Task:
        train = rng.normal(size=(8, 4))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test = rng.normal(size=(4, 4))
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        return Task(name, train, np.tile([-1, 1], 4), test, np.tile([-1, 1], 2))

    tasks = (make_task("old"), make_task("new"))
    monkeypatch.setattr(e008, "load_two_tasks", lambda **_kwargs: tasks)
    parent_result = e008.run_experiment(
        n_qubits=2,
        layers=1,
        epochs_per_task=1,
        fisher_samples=2,
        reference_shots=10,
        n_train=8,
        n_test=4,
        seed=3,
        verbose=False,
    )
    monkeypatch.setattr(e010, "load_two_tasks", lambda **_kwargs: tasks)
    result = e010.run_experiment(seed=3, parent_result=parent_result, verbose=False)
    assert set(result["histories"]) == set(e010.NEW_METHODS)
    assert result["training"]["phase1_replayed_and_verified_against_parent"] is True
    assert result["measurement_design"]["relevance_role"].startswith("used only")
    assert sum(result["measurement_design"]["allocation"].values()) == pytest.approx(1.0)
    for method in e010.NEW_METHODS:
        assert result["histories"][method][:2] == parent_result["histories"]["naive"][:2]
        normalized = result["measurement_design"]["normalized_method_fisher"][method]
        assert np.mean(normalized) == pytest.approx(1.0)

    invalid = copy.deepcopy(parent_result)
    invalid["data_sha256"] = "not-the-paired-split"
    with pytest.raises(ValueError, match="seed/data split"):
        e010.run_experiment(seed=3, parent_result=invalid, verbose=False)


def test_locality_library_tracks_pauli_weight_diameter_and_setting_reuse():
    from src.physmeas_observables import (
        family_resource_summary,
        phase_measurement_families,
    )

    families = phase_measurement_families()
    assert set(families) == {
        "readout",
        "one_local",
        "two_local",
        "hamiltonian",
        "nonlocal",
    }
    assert family_resource_summary(families["one_local"]) == {
        "n_observables": 12,
        "n_product_basis_settings": 3,
        "pauli_weights": [1] * 12,
        "support_diameters": [0] * 12,
    }
    nonlocal_measurement = families["nonlocal"][0]
    assert nonlocal_measurement.pauli == "XYYX"
    assert nonlocal_measurement.weight == 4
    assert nonlocal_measurement.diameter == 3
    assert family_resource_summary(families["hamiltonian"])[
        "n_product_basis_settings"
    ] == 3


def test_binary_pauli_fisher_matches_analytic_ry_result():
    from src.physmeas_observables import (
        make_pauli_expectation_qnode,
        pauli_fisher_diag,
    )

    qnode = make_pauli_expectation_qnode("Z", n_qubits=1, n_layers=1)
    features = np.asarray([[1.0, 0.0]])
    weights = pnp.array([[[0.61, 0.27]]], requires_grad=True)
    fisher = pauli_fisher_diag(qnode, weights, features, [0])
    assert fisher[0] == pytest.approx(1.0, abs=1e-7)
    assert fisher[1] == pytest.approx(0.0, abs=1e-9)


def test_phase_locality_experiment_branches_one_boundary_offline(monkeypatch):
    import experiments.e010_phase_locality as experiment
    from src.continual_data import Task

    rng = np.random.default_rng(19)

    def make_task(name: str) -> Task:
        train = rng.normal(size=(8, 16))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test = rng.normal(size=(4, 16))
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        return Task(name, train, np.tile([-1, 1], 4), test, np.tile([-1, 1], 2))

    phase = make_task("phase")
    images = (make_task("mnist"), make_task("fashion"))
    monkeypatch.setattr(experiment, "load_spt_atf", lambda **_kwargs: phase)
    monkeypatch.setattr(experiment, "load_two_tasks", lambda **_kwargs: images)
    methods = ("naive", "one_local", "nonlocal", "task_relevant_all")
    result = experiment.run_experiment(
        layers=1,
        epochs_per_task=1,
        fisher_samples=2,
        n_train=8,
        n_test=4,
        seed=2,
        methods=methods,
        verbose=False,
    )
    assert set(result["histories"]) == set(methods)
    common = [result["histories"][method][:2] for method in methods]
    assert all(history == common[0] for history in common[1:])
    assert result["training"]["phase_trained_once_then_branched"] is True
    assert result["claim_boundaries"]["input_locality_equivalence"] is False
    assert result["locality_analysis"]["domain"].startswith("Pauli support")
    for method in methods[1:]:
        fisher = result["locality_analysis"]["normalized_method_fisher"][method]
        assert np.mean(fisher) == pytest.approx(1.0)


def test_checked_e010_artifacts_match_sources_provenance_and_figures():
    from experiments.e010_finite_shot import _source_digest as finite_source_digest
    from experiments.e010_phase_locality import _source_digest as phase_source_digest
    from experiments.e010_physmeas_qcl import _source_digest as core_source_digest
    from scripts.run_e010_finite_shot_multiseed import __file__ as finite_aggregation_file
    from scripts.run_e010_multiseed import (
        _aggregation_digest as core_aggregation_digest,
        build_summary as build_core_summary,
    )
    from scripts.run_e010_phase_multiseed import (
        _aggregation_digest as phase_aggregation_digest,
        build_summary as build_phase_summary,
    )

    root = Path(__file__).resolve().parents[1]
    seeds = (42, 43, 44)
    core_paths = [root / f"results/e010_physmeas_seed{seed}.json" for seed in seeds]
    core_runs = [json.loads(path.read_text()) for path in core_paths]
    assert all(run["source_code_sha256"] == core_source_digest() for run in core_runs)
    rebuilt_core = build_core_summary(core_runs, core_paths)
    checked_core = json.loads((root / "results/e010_physmeas_summary.json").read_text())
    assert checked_core["aggregation_code_sha256"] == core_aggregation_digest()
    for key in (
        "source_code_sha256",
        "seeds",
        "aggregate_metrics",
        "task_relevant_allocation",
        "paired_differences",
    ):
        assert rebuilt_core[key] == checked_core[key]

    phase_paths = [
        root / f"results/e010_phase_locality_seed{seed}.json" for seed in seeds
    ]
    phase_runs = [json.loads(path.read_text()) for path in phase_paths]
    assert all(run["source_code_sha256"] == phase_source_digest() for run in phase_runs)
    rebuilt_phase = build_phase_summary(phase_runs, phase_paths)
    checked_phase = json.loads(
        (root / "results/e010_phase_locality_summary.json").read_text()
    )
    assert checked_phase["aggregation_code_sha256"] == phase_aggregation_digest()
    for key in (
        "source_code_sha256",
        "seeds",
        "aggregate_metrics",
        "task_relevant_allocation",
        "cosine_to_qfi",
    ):
        assert rebuilt_phase[key] == checked_phase[key]

    for seed in seeds:
        path = root / f"results/e010_finite_shot_seed{seed}.json"
        run = json.loads(path.read_text())
        assert run["source_code_sha256"] == finite_source_digest()
        extension = root / run["extension_result_file"]
        assert hashlib.sha256(extension.read_bytes()).hexdigest() == run[
            "extension_result_sha256"
        ]
        assert run["claim_boundaries"]["finite_shot_training_or_retention"] is False
        assert all(
            budget["optimizer_failures"] == 0 for budget in run["budgets"].values()
        )
    finite_summary = json.loads(
        (root / "results/e010_finite_shot_summary.json").read_text()
    )
    assert finite_summary["aggregation_code_sha256"] == hashlib.sha256(
        Path(finite_aggregation_file).read_bytes()
    ).hexdigest()
    finite_runs = [
        json.loads((root / f"results/e010_finite_shot_seed{seed}.json").read_text())
        for seed in seeds
    ]
    for budget, aggregate in finite_summary["aggregate"].items():
        for source_key, aggregate_key in (
            ("selected_profile_cosine_to_exact", "selected_profile_cosine_to_exact"),
            ("allocation_l1_error_to_exact", "allocation_l1_error_to_exact"),
        ):
            values = np.asarray(
                [run["budgets"][budget][source_key]["mean"] for run in finite_runs]
            )
            assert aggregate[aggregate_key]["seed_mean"] == {
                "mean": round(float(np.mean(values)), 6),
                "sample_std": round(float(np.std(values, ddof=1)), 6),
            }
        for basis in finite_runs[0]["bases"]:
            values = np.asarray(
                [
                    run["budgets"][budget]["allocation"][basis]["mean"]
                    for run in finite_runs
                ]
            )
            assert aggregate["allocation"][basis]["seed_mean"] == {
                "mean": round(float(np.mean(values)), 6),
                "sample_std": round(float(np.std(values, ddof=1)), 6),
            }
    for name in ("e010_physmeas_main.png", "e010_finite_shot.png"):
        assert (root / "figures" / name).stat().st_size > 20_000
