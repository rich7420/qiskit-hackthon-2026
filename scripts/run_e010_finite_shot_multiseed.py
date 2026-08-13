"""Validate and aggregate E010 finite-shot measurement-selection audits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e010_finite_shot import _source_digest  # noqa: E402

SEEDS = (42, 43, 44)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "sample_std": round(float(np.std(array, ddof=1)), 6),
    }


def main() -> None:
    paths = [ROOT / f"results/e010_finite_shot_seed{seed}.json" for seed in SEEDS]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for run, path, seed in zip(runs, paths, SEEDS, strict=True):
        if run["seed"] != seed or run["source_code_sha256"] != _source_digest():
            raise ValueError(f"{path} does not match its seed/current finite-shot source")
        extension = ROOT / run["extension_result_file"]
        if hashlib.sha256(extension.read_bytes()).hexdigest() != run[
            "extension_result_sha256"
        ]:
            raise ValueError(f"{path} references a changed E010 exact artifact")
        for budget in run["budgets"].values():
            if budget["optimizer_failures"]:
                raise ValueError(f"{path} has finite-shot allocation failures")
    budgets = runs[0]["budgets"].keys()
    aggregate = {}
    for budget in budgets:
        aggregate[budget] = {
            "shots_per_probability_circuit": int(budget),
            "pilot_shots_per_seed_repetition": runs[0]["budgets"][budget][
                "pilot_shots_per_repetition"
            ],
            "allocation": {
                basis: {
                    "seed_mean": _summary(
                        [run["budgets"][budget]["allocation"][basis]["mean"] for run in runs]
                    ),
                    "mean_within_seed_sample_std": round(
                        float(
                            np.mean(
                                [
                                    run["budgets"][budget]["allocation"][basis][
                                        "sample_std"
                                    ]
                                    for run in runs
                                ]
                            )
                        ),
                        6,
                    ),
                }
                for basis in runs[0]["bases"]
            },
            "selected_profile_cosine_to_exact": {
                "seed_mean": _summary(
                    [
                        run["budgets"][budget]["selected_profile_cosine_to_exact"][
                            "mean"
                        ]
                        for run in runs
                    ]
                ),
                "mean_within_seed_sample_std": round(
                    float(
                        np.mean(
                            [
                                run["budgets"][budget][
                                    "selected_profile_cosine_to_exact"
                                ]["sample_std"]
                                for run in runs
                            ]
                        )
                    ),
                    6,
                ),
            },
            "allocation_l1_error_to_exact": {
                "seed_mean": _summary(
                    [
                        run["budgets"][budget]["allocation_l1_error_to_exact"]["mean"]
                        for run in runs
                    ]
                ),
                "mean_within_seed_sample_std": round(
                    float(
                        np.mean(
                            [
                                run["budgets"][budget]["allocation_l1_error_to_exact"][
                                    "sample_std"
                                ]
                                for run in runs
                            ]
                        )
                    ),
                    6,
                ),
            },
        }
    result = {
        "schema_version": 1,
        "experiment": "e010_conditional_finite_shot_basis_fisher_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "aggregation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "repetitions_per_seed_budget": runs[0]["budgets"][next(iter(budgets))][
            "repetitions"
        ],
        "result_files": [str(path.relative_to(ROOT)) for path in paths],
        "measurement_cost_definition": (
            "all anchors times base/+shift/-shift probability circuits times candidate "
            "bases times shots; pilot selection only"
        ),
        "aggregate": aggregate,
        "claim_boundaries": runs[0]["claim_boundaries"],
    }
    output = ROOT / "results/e010_finite_shot_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for budget, values in aggregate.items():
        cosine = values["selected_profile_cosine_to_exact"]
        print(
            f"  {budget:>4s} shots/circuit: cosine seed mean "
            f"{cosine['seed_mean']['mean']:.4f} +/- "
            f"{cosine['seed_mean']['sample_std']:.4f}"
        )


if __name__ == "__main__":
    main()
