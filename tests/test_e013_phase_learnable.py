"""E013 phase-first learnable full-output measurement contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_phase_extension_reuses_boundary_and_full_joint_measurement(monkeypatch):
    import experiments.e010_phase_locality as e010
    import experiments.e013_phase_learnable as e013
    from src.continual_data import Task

    rng = np.random.default_rng(113)

    def make_task(name: str) -> Task:
        train = rng.normal(size=(8, 16))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test = rng.normal(size=(4, 16))
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        return Task(name, train, np.tile([-1, 1], 4), test, np.tile([-1, 1], 2))

    phase = make_task("phase")
    images = (make_task("mnist"), make_task("fashion"))
    monkeypatch.setattr(e010, "load_spt_atf", lambda **_kwargs: phase)
    monkeypatch.setattr(e010, "load_two_tasks", lambda **_kwargs: images)
    parent = e010.run_experiment(
        layers=1,
        epochs_per_task=1,
        fisher_samples=2,
        n_train=8,
        n_test=4,
        seed=7,
        methods=("naive", "qewc"),
        verbose=False,
    )
    monkeypatch.setattr(e013, "load_spt_atf", lambda **_kwargs: phase)
    monkeypatch.setattr(e013, "load_two_tasks", lambda **_kwargs: images)
    result = e013.run_experiment(
        seed=7,
        parent_result=parent,
        outer_iterations=10,
        axis_max_iterations=3_000,
        verbose=False,
    )
    assert result["experiment"] == "e013_phase_first_learnable_full_output_exact"
    assert result["training"]["phase_replayed_and_verified_against_parent"] is True
    assert result["model"]["learned_measurement_domain"] == "all four output qubits"
    assert result["measurement_design"]["outcomes_per_setting"] == 16
    assert set(result["histories"]) == set(e013.METHODS)
    assert result["resource_accounting"]["axis_optimization_quantum_circuit_configurations"] == 0
    reference = parent["histories"]["naive"][:2]
    for method in e013.METHODS:
        assert result["histories"][method][:2] == reference
        assert np.mean(
            result["measurement_design"]["normalized_method_fisher"][method]
        ) == pytest.approx(1.0)
        diagnostic = result["measurement_design"]["diagnostics"][method]
        assert diagnostic["information_hierarchy"]["passed"] is True
        assert diagnostic["cosine_to_full_qfi"] <= 1.0 + 1e-9


def test_phase_fixed_joint_library_is_zzzz_xxxx_yyyy():
    from src.measqcl_learnable import canonical_product_axes

    axes = canonical_product_axes(3, 4, initialization_noise=0.0)
    np.testing.assert_allclose(axes[0], np.tile([0.0, 0.0, 1.0], (4, 1)))
    np.testing.assert_allclose(axes[1], np.tile([1.0, 0.0, 0.0], (4, 1)))
    np.testing.assert_allclose(axes[2], np.tile([0.0, 1.0, 0.0], (4, 1)))


def test_checked_phase_artifacts_match_sources_aggregation_and_figure():
    from experiments.e013_phase_learnable import _source_digest
    from scripts.run_e013_phase_multiseed import _aggregation_digest, build_summary

    root = Path(__file__).resolve().parents[1]
    paths = [root / f"results/e013_phase_learnable_seed{seed}.json" for seed in (42, 43, 44)]
    runs = [json.loads(path.read_text()) for path in paths]
    assert all(run["source_code_sha256"] == _source_digest() for run in runs)
    rebuilt = build_summary(runs, paths)
    checked = json.loads((root / "results/e013_phase_learnable_summary.json").read_text())
    assert checked["aggregation_code_sha256"] == _aggregation_digest()
    for key in (
        "source_code_sha256",
        "seeds",
        "result_artifacts",
        "aggregate_metrics",
        "measurement_geometry",
        "paired_differences",
        "paired_seed_metrics",
        "ceiling_effect_diagnostic",
    ):
        assert rebuilt[key] == checked[key]
    assert checked["n_seeds"] == 3
    assert checked["claim_boundaries"]["superiority_claimed"] is False
    figure = root / "figures/e013_phase_learnable.png"
    assert figure.stat().st_size > 50_000
    provenance = json.loads(
        (root / "results/e013_phase_figure_provenance.json").read_text()
    )
    for file_key, digest_key in (
        ("summary_file", "summary_file_sha256"),
        ("plot_source_file", "plot_source_sha256"),
        ("figure_file", "figure_file_sha256"),
    ):
        path = root / provenance[file_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[digest_key]

    training_figure = root / "figures/e013_phase_training.png"
    assert training_figure.stat().st_size > 50_000
    training_provenance = json.loads(
        (root / "results/e013_phase_training_figure_provenance.json").read_text()
    )
    assert training_provenance["seeds"] == [42, 43, 44]
    assert training_provenance["uncertainty"] == "sample standard deviation"
    for artifact in training_provenance["input_artifacts"]:
        path = root / artifact["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for file_key, digest_key in (
        ("plot_source_file", "plot_source_sha256"),
        ("figure_file", "figure_file_sha256"),
    ):
        path = root / training_provenance[file_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == training_provenance[digest_key]
