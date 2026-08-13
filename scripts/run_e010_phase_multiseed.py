"""Validate and aggregate the phase-first E010 locality experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e010_phase_locality import METHODS, _source_digest  # noqa: E402
from scripts.tune_e010_phase_train_only import (  # noqa: E402
    _source_digest as tuning_source_digest,
)

SEEDS = (42, 43, 44)
METRICS = (
    "phase_at_boundary",
    "phase_final_retention",
    "phase_forgetting",
    "mnist_final_adaptation",
    "fashion_final_adaptation",
    "average_final_accuracy",
)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6),
    }


def _aggregation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_summary(runs: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    if [run["training"]["seed"] for run in runs] != list(SEEDS):
        raise ValueError("formal phase-locality runs must be ordered seeds 42/43/44")
    tuning_path = ROOT / "results/e010_phase_train_only_tuning.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    if tuning["source_code_sha256"] != tuning_source_digest():
        raise ValueError("phase train-only calibration does not match current source")
    if tuning["test_evaluated"] or tuning["selected_layers"] != 3:
        raise ValueError("phase capacity calibration invariants failed")
    for run, path in zip(runs, paths, strict=True):
        if run["source_code_sha256"] != _source_digest():
            raise ValueError(f"{path} does not match current phase experiment source")
        if set(run["metrics"]) != set(METHODS):
            raise ValueError(f"{path} is missing a formal locality method")
        if run["data"]["test_used_for_selection"]:
            raise ValueError(f"{path} claims test-based selection")
        if run["model"]["layers"] != tuning["selected_layers"]:
            raise ValueError(f"{path} does not use calibrated capacity")
        if run["training"]["ewc_lambda"] != tuning["shared_lambda"]["value"]:
            raise ValueError(f"{path} does not use the prespecified shared lambda")
        if (
            run["locality_analysis"]["task_relevant_optimality_gap"]
            > run["locality_analysis"]["task_relevant_optimality_tolerance"]
        ):
            raise ValueError(f"{path} contains a non-converged measurement allocation")

    aggregate = {
        method: {
            split: {
                metric: _summary(
                    [run["metrics"][method][split][metric] for run in runs]
                )
                for metric in METRICS
            }
            for split in ("train", "test")
        }
        for method in METHODS
    }
    all_paulis = sorted(
        {
            pauli
            for run in runs
            for pauli in run["locality_analysis"]["task_relevant_allocation"]
        }
    )
    allocation = {
        pauli: _summary(
            [
                run["locality_analysis"]["task_relevant_allocation"].get(pauli, 0.0)
                for run in runs
            ]
        )
        for pauli in all_paulis
    }
    geometry = {
        method: _summary(
            [run["locality_analysis"]["cosine_to_qfi"][method] for run in runs]
        )
        for method in runs[0]["locality_analysis"]["cosine_to_qfi"]
    }
    resources = {
        family: runs[0]["locality_analysis"]["families"][family]["resources"]
        for family in runs[0]["locality_analysis"]["families"]
    }
    return {
        "schema_version": 1,
        "experiment": "e010_phase_first_locality_resolved_exact_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "aggregation_code_sha256": _aggregation_digest(),
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "uncertainty": "sample standard deviation across paired seeds (ddof=1)",
        "result_files": [str(path.relative_to(ROOT)) for path in paths],
        "train_only_calibration_file": str(tuning_path.relative_to(ROOT)),
        "configuration": {
            "data": runs[0]["data"],
            "model": runs[0]["model"],
            "training": {
                key: runs[0]["training"][key]
                for key in (
                    "optimizer",
                    "learning_rate",
                    "epochs_per_task",
                    "ewc_lambda",
                    "fisher_normalization",
                    "fisher_samples",
                )
            },
        },
        "aggregate_metrics": aggregate,
        "measurement_resources": resources,
        "task_relevant_allocation": allocation,
        "cosine_to_qfi": geometry,
        "paired_seed_metrics": {
            str(run["training"]["seed"]): {
                method: run["metrics"][method]["test"] for method in METHODS
            }
            for run in runs
        },
        "claim_boundaries": {
            "finite_shot_result": False,
            "hardware_result": False,
            "thermodynamic_phase_transition": False,
            "input_locality_equivalence": False,
            "statistical_significance_claimed": False,
            "quantum_advantage": False,
            "interpretation": (
                "three-seed output-Pauli-locality diagnostic on a four-qubit "
                "phase-memory task"
            ),
        },
    }


def main() -> None:
    paths = [ROOT / f"results/e010_phase_locality_seed{seed}.json" for seed in SEEDS]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = build_summary(runs, paths)
    output = ROOT / "results/e010_phase_locality_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for method in METHODS:
        metric = result["aggregate_metrics"][method]["test"]
        print(
            f"  {method:18s} phase={metric['phase_final_retention']['mean']:.3f} "
            f"Fashion={metric['fashion_final_adaptation']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
