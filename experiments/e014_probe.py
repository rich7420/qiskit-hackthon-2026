"""E014 GO/NO-GO: Feature-Sufficiency probe for measurement-side continual learning.

The cheap gate that decides whether MPI is worth building fully (mentor review, sec 22/32):

  1. Train the shared VQC on Task 1 only -> theta_1*.
  2. Freeze theta_1*.  Compute frozen probability features p(x; theta_1*) for ALL tasks.
  3. Fit an independent linear head (= learnable diagonal observable) per task on those
     frozen features, and report probe accuracy per task.

Interpretation:
  * High T2/T3 probe accuracy  -> the Task-1 representation already separates later tasks,
    so freezing theta and only swapping the readout can work  -> GO (build full MPI).
  * T2 ok but T3 poor          -> PARTIAL GO: need soft backbone adaptation (Variant C).
  * T2/T3 both poor            -> NO-GO for a purely frozen backbone.

Also reports the "head-only gain": full 2^n probs head vs a fixed few-observable readout
(the 2-wire marginal), i.e. does the full measurement distribution carry more reusable
task information than a conventional VQC readout.  This is the honest, defensible
replacement for the CFI/QFI-ratio analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import (  # noqa: E402
    fit_linear_head,
    make_probs_qnode,
    probs_features,
    train_backbone,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/e014_oiqcl.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _marginal_2wire(P: np.ndarray) -> np.ndarray:
    """Reduce the 2^n computational-basis probs to the 2-wire (qubits 0,1) marginal.

    Fixed 'few-observable' readout baseline: with n=4, groups the 16 outcomes by the top
    two bits into 4 marginal probabilities -- the information a conventional 2-qubit
    readout would see.  qubit 0 is the most significant bit in PennyLane's ordering.
    """
    n = int(np.log2(P.shape[1]))
    idx = np.arange(P.shape[1])
    top2 = idx >> (n - 2)  # value of the two most-significant bits, in {0,1,2,3}
    return np.stack([P[:, top2 == b].sum(axis=1) for b in range(4)], axis=1)


def run_probe(
    *,
    layers: int = 20,
    lr: float = 0.02,
    epochs: int = 20,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    task3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [task1, task2, task3]

    if verbose:
        print(f"E014 probe seed={seed}: train backbone on Task1 ({task1.name}), "
              f"freeze, fit linear heads on all tasks", flush=True)

    started = time.perf_counter()
    weights, _, weight_shape = train_backbone(
        task1, n_qubits=N_QUBITS, n_layers=layers, lr=lr, epochs=epochs,
        seed=seed, verbose=verbose,
    )
    probs_qnode, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers)

    per_task: dict[str, Any] = {}
    for key, task in zip(TASK_KEYS, tasks):
        P_tr = probs_features(probs_qnode, weights, task.X_train)
        P_te = probs_features(probs_qnode, weights, task.X_test)

        head_full = fit_linear_head(P_tr, task.y_train, task.name, seed=seed)
        M_tr, M_te = _marginal_2wire(P_tr), _marginal_2wire(P_te)
        head_marg = fit_linear_head(M_tr, task.y_train, task.name, seed=seed)

        per_task[key] = {
            "name": task.name,
            "probe_train_acc": round(head_full.accuracy(P_tr, task.y_train), 4),
            "probe_test_acc": round(head_full.accuracy(P_te, task.y_test), 4),
            "marginal_readout_test_acc": round(head_marg.accuracy(M_te, task.y_test), 4),
            "head_only_gain": round(
                head_full.accuracy(P_te, task.y_test) - head_marg.accuracy(M_te, task.y_test), 4
            ),
        }
        if verbose:
            r = per_task[key]
            print(f"    {key} ({task.name}): probe_test={r['probe_test_acc']:.3f}  "
                  f"marginal={r['marginal_readout_test_acc']:.3f}  "
                  f"gain={r['head_only_gain']:+.3f}", flush=True)

    elapsed = time.perf_counter() - started
    later_probe = float(np.mean([per_task[k]["probe_test_acc"] for k in ("task2", "task3")]))
    verdict = "GO" if later_probe >= 0.80 else ("PARTIAL_GO" if later_probe >= 0.65 else "NO_GO")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e014_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("pennylane", "numpy", "scipy", "scikit-learn")},
        },
        "config": {
            "method": "frozen theta_1* + independent per-task linear head over probs (Variant A probe)",
            "tasks": [t.name for t in tasks],
            "n_qubits": N_QUBITS, "layers": layers, "n_weights": int(np.prod(weight_shape)),
            "backbone_readout": "two Pauli-Z softmax + BCE (trained on Task1 only)",
            "probe_head": "logistic regression over full 2^n computational-basis probs",
            "marginal_readout": "logistic regression over 2-wire (qubits 0,1) marginal probs",
            "optimizer": "Adam", "learning_rate": lr, "epochs_task1": epochs,
            "n_train_per_task": n_train, "n_test_per_task": n_test, "seed": seed,
        },
        "per_task": per_task,
        "later_task_mean_probe_acc": round(later_probe, 4),
        "verdict": verdict,
        "verdict_thresholds": {"GO": ">=0.80", "PARTIAL_GO": ">=0.65", "NO_GO": "<0.65"},
        "elapsed_sec": round(elapsed, 1),
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e014_probe_seed{result['config']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_probe(
        layers=args.layers, lr=args.lr, epochs=args.epochs,
        n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )
    path = write_result(result, args.output)
    print(f"\nverdict: {result['verdict']}  "
          f"(later-task mean probe acc = {result['later_task_mean_probe_acc']:.3f})")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
