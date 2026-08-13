"""Train-only capacity calibration for the E010 phase-first experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e010_phase_locality import _train_phase  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.measqcl_model import make_classifier_qnode  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

CANDIDATE_LAYERS = (2, 3, 5)
TARGET_TRAIN_ACCURACY = 0.99


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/measqcl_fisher.py",
        ROOT / "src/measqcl_task_relevance.py",
        ROOT / "src/physmeas_observables.py",
        ROOT / "src/measqcl_model.py",
        ROOT / "experiments/e010_phase_locality.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_calibration(seed: int = 42) -> dict:
    phase = load_spt_atf(n_train=400, n_test=200, seed=seed)
    images = load_two_tasks(n_features=2**N_QUBITS, n_train=400, n_test=200, seed=seed)
    tasks = (phase, *images)
    scans = []
    for layers in CANDIDATE_LAYERS:
        qnode, shape = make_classifier_qnode(N_QUBITS, layers)
        _, _, _, history = _train_phase(
            qnode=qnode,
            weight_shape=shape,
            tasks=tasks,
            learning_rate=0.02,
            epochs=40,
            seed=seed,
            record_test=False,
            verbose=False,
        )
        if any(row["test_accuracy"] is not None for row in history):
            raise RuntimeError("train-only calibration unexpectedly evaluated test data")
        scans.append(
            {
                "layers": layers,
                "n_parameters": int(2 * N_QUBITS * layers),
                "phase_train_accuracy": history[-1]["train_accuracy"]["phase"],
            }
        )
    eligible = [
        scan for scan in scans if scan["phase_train_accuracy"] >= TARGET_TRAIN_ACCURACY
    ]
    if not eligible:
        raise RuntimeError("no candidate meets the prespecified phase capacity target")
    selected = min(eligible, key=lambda scan: scan["layers"])
    return {
        "schema_version": 1,
        "experiment": "e010_phase_train_only_capacity_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "seed": seed,
        "test_evaluated": False,
        "candidate_layers": list(CANDIDATE_LAYERS),
        "selection_rule": (
            "smallest layer count reaching phase training accuracy >= 0.99 after "
            "40 epochs"
        ),
        "scans": scans,
        "selected_layers": selected["layers"],
        "shared_lambda": {
            "value": 0.1,
            "selection": (
                "inherited unchanged from the normalized-Fisher E008 train-only "
                "calibration; not tuned on E010 phase results"
            ),
        },
    }


def main() -> None:
    result = run_calibration()
    output = ROOT / "results/e010_phase_train_only_tuning.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
