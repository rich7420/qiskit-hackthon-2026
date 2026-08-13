"""E013 physics and optimization contracts for learnable measurement Fisher consolidation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from autograd import value_and_grad
from pennylane import numpy as pnp
from scipy.optimize import check_grad

from src.measqcl_fisher import measurement_fisher_diag
from src.measqcl_learnable import (
    ShiftedDensityCache,
    _autograd_axis_objective,
    basis_fisher_diags_from_cache,
    cache_parameter_shift_density_matrices,
    canonical_product_axes,
    local_observables,
    optimize_learnable_measurements,
    product_projectors,
)
from src.measqcl_model import make_measurement_qnode, make_reduced_state_qnode


def test_fixed_spectrum_observables_and_projectors_are_physical():
    axes = canonical_product_axes(3, 2, initialization_noise=0.0)
    observables = local_observables(axes)
    expected = np.asarray(
        [
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
        ],
        dtype=complex,
    )
    for setting in range(3):
        for qubit in range(2):
            observable = observables[setting, qubit]
            np.testing.assert_allclose(observable, expected[setting], atol=1e-12)
            np.testing.assert_allclose(observable, observable.conj().T, atol=1e-12)
            np.testing.assert_allclose(observable @ observable, np.eye(2), atol=1e-12)
            assert np.trace(observable) == pytest.approx(0.0)
            np.testing.assert_allclose(np.linalg.eigvalsh(observable), [-1.0, 1.0])

    projectors = product_projectors(axes)
    for setting in range(3):
        np.testing.assert_allclose(
            np.sum(projectors[setting], axis=0), np.eye(4), atol=1e-12
        )
        for projector in projectors[setting]:
            np.testing.assert_allclose(projector, projector.conj().T, atol=1e-12)
            np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)


def test_cached_zz_fisher_matches_existing_pauli_qnode():
    features = np.asarray(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.1, -0.3, np.sqrt(0.41)],
        ]
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    weights = pnp.array([[[0.2, -0.1], [0.4, 0.3]]], requires_grad=True)
    density_qnode = make_reduced_state_qnode(
        n_qubits=2,
        n_layers=1,
        readout_wires=(0, 1),
    )
    cache = cache_parameter_shift_density_matrices(
        density_qnode,
        weights,
        features,
        [0, 1],
    )
    zz_axes = np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    cached = basis_fisher_diags_from_cache(cache, zz_axes)[0]
    zz_qnode = make_measurement_qnode(
        "ZZ",
        n_qubits=2,
        n_layers=1,
        readout_wires=(0, 1),
    )
    direct = measurement_fisher_diag(zz_qnode, weights, features, [0, 1])
    np.testing.assert_allclose(cached, direct, atol=1e-8, rtol=1e-7)


def test_antipodal_axis_only_relabels_outcomes_and_preserves_fisher():
    identity = np.eye(2, dtype=complex)
    base = 0.5 * (identity + 0.4 * np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    plus = 0.5 * (
        identity + 0.4 * np.asarray([[0.0, -1.0j], [1.0j, 0.0]])
    )
    minus = 0.5 * (
        identity - 0.4 * np.asarray([[0.0, -1.0j], [1.0j, 0.0]])
    )
    cache = ShiftedDensityCache(
        base=base[np.newaxis],
        plus=plus[np.newaxis, np.newaxis],
        minus=minus[np.newaxis, np.newaxis],
        anchor_indices=np.asarray([0]),
        weight_shape=(1,),
    )
    axis = np.asarray([[[0.2, 0.9, 0.3]]])
    fisher = basis_fisher_diags_from_cache(cache, axis)
    antipodal = basis_fisher_diags_from_cache(cache, -axis)
    np.testing.assert_allclose(fisher, antipodal, atol=1e-12)


def test_learned_axis_recovers_mixed_qubit_phase_sensitive_direction():
    identity = np.eye(2, dtype=complex)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    radius = 0.75
    cache = ShiftedDensityCache(
        base=(0.5 * (identity + radius * pauli_x))[np.newaxis],
        plus=(0.5 * (identity + radius * pauli_y))[np.newaxis, np.newaxis],
        minus=(0.5 * (identity - radius * pauli_y))[np.newaxis, np.newaxis],
        anchor_indices=np.asarray([0]),
        weight_shape=(1,),
    )
    initial = np.asarray([[[0.25, 0.70, 0.30]]])
    initial_fisher = basis_fisher_diags_from_cache(cache, initial)[0, 0]
    result = optimize_learnable_measurements(
        cache,
        np.asarray([1.0]),
        initial_axes=initial,
        n_settings=1,
        learn_allocation=False,
        minimum_allocation=0.0,
        diversity_coefficient=0.0,
        outer_iterations=10,
        axis_max_iterations=100,
    )
    assert result.basis_fishers[0, 0] > initial_fisher
    assert abs(result.axes[0, 0, 1]) > 0.999
    assert result.basis_fishers[0, 0] == pytest.approx(radius**2, abs=1e-6)
    assert result.allocation == pytest.approx([1.0])
    assert result.axis_solver_messages


def test_learnable_optimizer_returns_physical_simplex_and_frozen_cache():
    identity = np.eye(2, dtype=complex)
    paulis = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
        ],
        dtype=complex,
    )
    base = 0.5 * identity
    plus = np.stack([0.5 * (identity + 0.5 * pauli) for pauli in paulis])
    minus = np.stack([0.5 * (identity - 0.5 * pauli) for pauli in paulis])
    cache = ShiftedDensityCache(
        base=base[np.newaxis],
        plus=plus[np.newaxis],
        minus=minus[np.newaxis],
        anchor_indices=np.asarray([0]),
        weight_shape=(2,),
    )
    base_before = cache.base.copy()
    result = optimize_learnable_measurements(
        cache,
        np.asarray([5.0, 1.0]),
        n_settings=3,
        minimum_allocation=0.01,
        outer_iterations=2,
        axis_max_iterations=100,
        seed=9,
    )
    np.testing.assert_allclose(np.linalg.norm(result.axes, axis=-1), 1.0)
    assert np.sum(result.allocation) == pytest.approx(1.0)
    assert np.all(result.allocation >= 0.01 - 1e-9)
    assert np.all(result.basis_fishers >= 0.0)
    assert result.objective_evaluations > 0
    np.testing.assert_array_equal(cache.base, base_before)


def test_analytic_axis_gradient_matches_independent_finite_difference():
    identity = np.eye(2, dtype=complex)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    pauli_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    cache = ShiftedDensityCache(
        base=(0.5 * (identity + 0.2 * pauli_z))[np.newaxis],
        plus=np.stack(
            [
                0.5 * (identity + 0.5 * pauli_x),
                0.5 * (identity + 0.4 * pauli_y),
            ]
        )[np.newaxis],
        minus=np.stack(
            [
                0.5 * (identity - 0.5 * pauli_x),
                0.5 * (identity - 0.4 * pauli_y),
            ]
        )[np.newaxis],
        anchor_indices=np.asarray([0]),
        weight_shape=(2,),
    )
    axes = np.asarray([[[0.7, 0.2, 0.5]], [[0.1, 0.8, -0.3]]])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    basis = basis_fisher_diags_from_cache(cache, axes)
    scales = np.maximum(np.max(basis, axis=0), 1e-8)
    kwargs = {
        "axis_shape": axes.shape,
        "cache": cache,
        "allocation": np.asarray([0.4, 0.6]),
        "relevance": np.asarray([0.6, 0.4]),
        "objective_scale": scales,
        "diversity_coefficient": 1e-3,
        "radial_gauge_coefficient": 1.0,
        "epsilon": 1e-10,
    }
    def objective(flat):
        return _autograd_axis_objective(flat, **kwargs)

    gradient = value_and_grad(objective)
    error = check_grad(
        lambda flat: float(objective(flat)),
        lambda flat: np.asarray(gradient(flat)[1], dtype=float),
        axes.reshape(-1),
        epsilon=1e-6,
    )
    assert error < 1e-5


def test_e013_reuses_paired_boundary_and_changes_only_measurement_branches(monkeypatch):
    import experiments.e008_measqcl as e008
    import experiments.e010_physmeas_qcl as e010
    import experiments.e013_learnable_measqcl as e013
    from src.continual_data import Task

    rng = np.random.default_rng(31)

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
        seed=5,
        verbose=False,
    )
    monkeypatch.setattr(e010, "load_two_tasks", lambda **_kwargs: tasks)
    task_result = e010.run_experiment(
        seed=5,
        parent_result=parent_result,
        verbose=False,
    )
    monkeypatch.setattr(e013, "load_two_tasks", lambda **_kwargs: tasks)
    result = e013.run_experiment(
        seed=5,
        parent_result=parent_result,
        task_result=task_result,
        relevance_floor=1.0,
        outer_iterations=10,
        axis_max_iterations=500,
        verbose=False,
    )
    assert result["experiment"] == "e013_learnable_measurement_fisher_exact"
    assert set(result["histories"]) == set(e013.METHODS)
    assert result["training"]["phase1_replayed_and_verified_against_parents"] is True
    assert result["model"]["prediction_measurement_unchanged"] is True
    assert result["resource_accounting"]["axis_optimization_quantum_circuit_configurations"] == 0
    for method in e013.METHODS:
        design = result["measurement_design"]["optimization"][method]
        axes = np.asarray(design["axes"])
        allocation = np.asarray(design["allocation"])
        np.testing.assert_allclose(np.linalg.norm(axes, axis=-1), 1.0)
        assert np.sum(allocation) == pytest.approx(1.0)
        assert result["histories"][method][:2] == parent_result["histories"]["naive"][:2]
        assert np.mean(
            result["measurement_design"]["normalized_method_fisher"][method]
        ) == pytest.approx(1.0)
        hierarchy = result["measurement_design"]["diagnostics"][method][
            "information_hierarchy"
        ]
        assert hierarchy["passed"] is True
        assert hierarchy["minimum_readout_qfi_minus_basis_cfi"] >= -hierarchy["tolerance"]


def test_checked_e013_artifacts_match_sources_aggregation_and_figure():
    from experiments.e013_learnable_measqcl import _source_digest
    from scripts.run_e013_multiseed import (
        _aggregation_digest,
        build_summary,
    )

    root = Path(__file__).resolve().parents[1]
    seeds = (42, 43, 44)
    paths = [root / f"results/e013_learnable_measqcl_seed{seed}.json" for seed in seeds]
    runs = [json.loads(path.read_text()) for path in paths]
    assert all(run["source_code_sha256"] == _source_digest() for run in runs)
    rebuilt = build_summary(runs, paths)
    checked = json.loads(
        (root / "results/e013_learnable_measqcl_summary.json").read_text()
    )
    assert checked["aggregation_code_sha256"] == _aggregation_digest()
    for key in (
        "source_code_sha256",
        "seeds",
        "aggregate_metrics",
        "measurement_geometry",
        "paired_differences",
        "resource_accounting",
        "result_artifacts",
    ):
        assert rebuilt[key] == checked[key]
    invalid = copy.deepcopy(runs)
    invalid[1]["measurement_design"]["diversity_coefficient"] = 123.0
    with pytest.raises(ValueError, match="different formal E013 configuration"):
        build_summary(invalid, paths)
    assert checked["n_seeds"] == 3
    assert checked["resource_accounting"]["cached_quantum_circuit_configurations"] == 7712
    assert checked["claim_boundaries"]["hardware_result"] is False
    figure = root / "figures/e013_learnable_measqcl.png"
    assert figure.stat().st_size > 50_000
    provenance = json.loads((root / "results/e013_figure_provenance.json").read_text())
    summary_path = root / provenance["summary_file"]
    plot_source = root / provenance["plot_source_file"]
    figure_path = root / provenance["figure_file"]
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == provenance[
        "summary_file_sha256"
    ]
    assert hashlib.sha256(plot_source.read_bytes()).hexdigest() == provenance[
        "plot_source_sha256"
    ]
    assert hashlib.sha256(figure_path.read_bytes()).hexdigest() == provenance[
        "figure_file_sha256"
    ]
