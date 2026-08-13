"""Contracts for the scope-aware E004–E008 comparison artifact and plot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _comparison() -> dict:
    return json.loads(
        (ROOT / "results/e004_e008_comparison.json").read_text(encoding="utf-8")
    )


def test_comparison_has_scoped_experiments_and_bounded_metrics():
    data = _comparison()
    assert list(data["experiments"]) == ["E004", "E005", "E006", "E007", "E008"]
    assert data["claim_boundaries"]["cross_experiment_ranking_valid"] is False
    assert data["experiments"]["E006"]["score"] == "held-out test balanced accuracy"
    assert data["experiments"]["E007"]["test_tuned"] is True
    assert data["experiments"]["E008"]["test_tuned"] is False
    assert data["experiments"]["E005"]["artifact_verified"] is False
    assert data["experiments"]["E005"]["test_tuned"] is None
    assert data["claim_boundaries"]["e005_artifacts_authenticated_to_final_source"] is False
    expected_methods = {
        "E004": {"Naive"},
        "E005": {"Baseline", "EWC", "QEWC"},
        "E006": {"Baseline", "Replay"},
        "E007": {"Sequential", "QEWC", "L2", "Adaptive", "QFI-TR", "CFI-TR"},
        "E008": {
            "Naive",
            "Output CFI",
            "Joint ZZ CFI",
            "Uniform XYZ",
            "MOF-EWC",
            "Readout QFI",
            "Full QFI",
        },
    }
    observed = {(point["experiment"], point["method"]) for point in data["points"]}
    assert len(observed) == len(data["points"])
    for experiment, methods in expected_methods.items():
        assert {method for block, method in observed if block == experiment} == methods
        metadata = data["experiments"][experiment]
        assert "artifact_verified" in metadata
        assert metadata["result_sources"]
        assert metadata["aggregation_rule"]
    for point in data["points"]:
        for metric in (
            "old_task_retention",
            "new_task_plasticity",
            "average_forgetting",
        ):
            estimate = point[metric]
            assert 0.0 <= estimate["mean"] <= 1.0
            assert 0.0 <= estimate["sample_std"] <= 1.0


def test_comparison_reproduces_key_checked_results():
    points = {
        (point["experiment"], point["method"]): point for point in _comparison()["points"]
    }
    assert points[("E004", "Naive")]["old_task_retention"]["mean"] == pytest.approx(0.67)
    assert points[("E005", "QEWC")]["old_task_retention"]["mean"] == pytest.approx(
        0.825833
    )
    assert points[("E006", "Replay")]["average_forgetting"] == pytest.approx(
        {"mean": 0.050183, "sample_std": 0.097461}
    )
    assert points[("E007", "QFI-TR")]["new_task_plasticity"]["mean"] == pytest.approx(
        0.507
    )
    assert points[("E008", "Joint ZZ CFI")]["old_task_retention"]["mean"] == pytest.approx(
        0.865
    )


def test_e006_test_metrics_recompute_from_committed_seed_artifacts():
    points = {
        point["method"]: point
        for point in _comparison()["points"]
        if point["experiment"] == "E006"
    }
    commit = _comparison()["experiments"]["E006"]["source_commit"]
    for method, label in (("baseline", "Baseline"), ("replay", "Replay")):
        retention = []
        plasticity = []
        forgetting = []
        for seed in (42, 43, 44):
            path = f"results/e006_advanced_seed{seed}.json"
            result = subprocess.run(
                ["git", "show", f"{commit}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = json.loads(result.stdout)["methods"][method]["metrics"]
            retention.append(
                statistics.mean(
                    metrics[f"task{task}"]["test_balanced_accuracy_final"]
                    for task in (1, 2)
                )
            )
            plasticity.append(metrics["task3"]["test_balanced_accuracy_final"])
            forgetting.append(
                statistics.mean(
                    metrics[f"task{task}"]["test_balanced_phase_end_drop"]
                    for task in (1, 2)
                )
            )
        expected = points[label]
        for field, values in (
            ("old_task_retention", retention),
            ("new_task_plasticity", plasticity),
            ("average_forgetting", forgetting),
        ):
            assert expected[field]["mean"] == pytest.approx(
                statistics.mean(values), abs=1e-6
            )
            assert expected[field]["sample_std"] == pytest.approx(
                statistics.stdev(values), abs=1e-6
            )


def test_e008_points_match_checked_summary():
    summary = json.loads(
        (ROOT / "results/e008_measqcl_summary.json").read_text(encoding="utf-8")
    )["aggregate_metrics"]
    methods = {
        "Naive": "naive",
        "Output CFI": "output_cfi",
        "Joint ZZ CFI": "zz_cfi",
        "Uniform XYZ": "uniform_xyz",
        "MOF-EWC": "mof_ewc",
        "Readout QFI": "readout_qewc",
        "Full QFI": "qewc",
    }
    points = {
        point["method"]: point
        for point in _comparison()["points"]
        if point["experiment"] == "E008"
    }
    for label, key in methods.items():
        source = summary[key]["test"]
        expected = points[label]
        for target, source_key in (
            ("old_task_retention", "task1_final_retention"),
            ("new_task_plasticity", "task2_final_adaptation"),
            ("average_forgetting", "task1_forgetting"),
        ):
            assert expected[target] == source[source_key]


def test_comparison_plot_renders_offline(tmp_path):
    output = tmp_path / "comparison.png"
    environment = os.environ.copy()
    homebrew_expat = Path("/opt/homebrew/opt/expat/lib")
    if homebrew_expat.exists():
        environment["DYLD_LIBRARY_PATH"] = str(homebrew_expat)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plot_e004_e008_comparison.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert output.stat().st_size > 20_000
