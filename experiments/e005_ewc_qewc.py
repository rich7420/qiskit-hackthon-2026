"""E005 - continual learning with consolidation: Baseline vs EWC (CFI) vs QEWC (QFI).

Reproduces the core comparison of arXiv:2607.16030 on the e004 three-task sequence
(MNIST -> Fashion-MNIST -> SPT/ATF). One QNN is trained sequentially under three methods,
each from the same initial weights:

- baseline: no consolidation (lambda = 0)
- ewc:  penalty weighted by the classical Fisher diagonal (CFI), lambda = 30
- qewc: penalty weighted by the quantum Fisher diagonal (QFI), lambda_q = 0.8

Test accuracy on all three tasks is recorded every epoch so retention of earlier tasks can
be compared across methods. Same learner as e002/e003/e004 (4 qubits, RY/RZ + CNOT, Adam).
This is a paper-inspired engineering reproduction, single seed per run (aggregate via
scripts/e005_run_multiseed.py).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e005_consolidation import EWC, quantum_fisher_diag  # noqa: E402
from src.continual_data import Task, load_two_tasks  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402 (single-Z, used only for the QFI metric)
from src.e005_softmax import (  # noqa: E402
    accuracy as softmax_accuracy,
    bce_loss,
    classical_fisher_diag,
    make_softmax_qnode,
)

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")
METHODS = ("baseline", "ewc", "qewc")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/e005_softmax.py",
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        ROOT / "src/qnn_pennylane.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _train_one_method(
    method: str,
    tasks: list[Task],
    *,
    clf_qnode,
    qfi_qnode,
    weight_shape,
    lam_ewc: float,
    lam_qewc: float,
    lr: float,
    epochs_per_task: int,
    qfi_samples: int,
    seed: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    lam = {"baseline": 0.0, "ewc": lam_ewc, "qewc": lam_qewc}[method]
    reg = EWC(lam)
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    history: list[dict[str, Any]] = []

    def snapshot(epoch: int, phase: int) -> None:
        history.append({
            "epoch": epoch, "phase": phase,
            "test_accuracy": {k: softmax_accuracy(clf_qnode, weights, t.X_test, t.y_test)
                              for k, t in zip(TASK_KEYS, tasks)},
        })

    snapshot(0, 0)
    for phase, task in enumerate(tasks, start=1):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase):
            return bce_loss(clf_qnode, W, Xtr, ytr) + reg.penalty(W.flatten(), phase)

        for _ in range(epochs_per_task):
            weights = optimizer.step(cost, weights)
            snapshot(history[-1]["epoch"] + 1, phase)
        if verbose:
            acc = history[-1]["test_accuracy"]
            print(f"    [{method}] after phase {phase} ({task.name}): "
                  + " ".join(f"{k}={acc[k]:.3f}" for k in TASK_KEYS), flush=True)

        # Consolidate this task (skip after the final task; baseline has lam=0 anyway).
        if method != "baseline" and phase < len(tasks):
            if method == "ewc":
                fisher = classical_fisher_diag(clf_qnode, weights, task.X_train, task.y_train)
            else:  # QFI is a state property -> use the single-Z qnode's metric tensor
                fisher = quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                             n_samples=qfi_samples, seed=seed)
            reg.consolidate(np.asarray(weights).flatten(), fisher)
    return history


def run_experiment(
    *,
    layers: int = 20,
    learning_rate: float = 0.02,
    epochs_per_task: int = 20,
    lam_ewc: float = 30.0,
    lam_qewc: float = 0.8,
    qfi_samples: int = 64,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    task3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [task1, task2, task3]
    boundaries = [epochs_per_task, 2 * epochs_per_task]
    # Softmax classifier (2 Z readouts + BCE) for training/CFI; single-Z qnode for the QFI
    # metric tensor (same ansatz -> same state -> same QFI, so readout is irrelevant there).
    clf_qnode, weight_shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    qfi_qnode, _ = make_qnode(n_qubits=N_QUBITS, n_layers=layers)

    if verbose:
        print(f"E005 seed={seed}: {[t.name for t in tasks]}  "
              f"(lam_ewc={lam_ewc}, lam_qewc={lam_qewc}, qfi_samples={qfi_samples})")

    started = time.perf_counter()
    method_histories: dict[str, list] = {}
    for method in METHODS:
        method_histories[method] = _train_one_method(
            method, tasks, clf_qnode=clf_qnode, qfi_qnode=qfi_qnode, weight_shape=weight_shape,
            lam_ewc=lam_ewc, lam_qewc=lam_qewc, lr=learning_rate,
            epochs_per_task=epochs_per_task, qfi_samples=qfi_samples, seed=seed, verbose=verbose,
        )
    train_time = time.perf_counter() - started

    # Per-method, per-earlier-task retention: test acc at its phase-end vs final.
    metrics: dict[str, Any] = {}
    for method, hist in method_histories.items():
        per_task = {}
        for idx, key in enumerate(TASK_KEYS, start=1):
            phase_end = hist[idx * epochs_per_task]["test_accuracy"][key]
            final = hist[-1]["test_accuracy"][key]
            per_task[key] = {"name": tasks[idx - 1].name,
                             "test_at_phase_end": round(phase_end, 4),
                             "test_final": round(final, 4),
                             "forgetting": round(phase_end - final, 4)}
        # Mean retained accuracy on the two earlier tasks (the headline number).
        retained = np.mean([per_task[k]["test_final"] for k in ("task1", "task2")])
        metrics[method] = {"tasks": per_task, "mean_earlier_task_final": round(float(retained), 4)}

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e005_ewc_qewc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scipy", "scikit-learn")}},
        "config": {
            "tasks": [t.name for t in tasks],
            "n_qubits": N_QUBITS, "layers": layers, "n_weights": int(np.prod(weight_shape)),
            "readout": "two Pauli-Z on qubits 0,1 -> softmax (Eq. 7)",
            "loss": "binary cross-entropy",
            "cfi": "empirical Fisher from softmax log-likelihood gradients (Eq. 9)",
            "qfi": "diagonal QFI via Fubini-Study metric tensor (Eq. 18), mini-batch averaged",
            "optimizer": "Adam", "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task, "task_boundaries": boundaries,
            "lambda_ewc": lam_ewc, "lambda_qewc": lam_qewc, "qfi_samples": qfi_samples,
            "alpha_rule": "alpha_j^(k) = j/(k-1) for k>1 (Eq. 30)",
            "deviations_from_paper": "independent RY/RZ (no weight sharing), 20 layers/160 params vs 30/120; Adam state continues across tasks",
            "n_train_per_task": n_train, "n_test_per_task": n_test, "seed": seed,
        },
        "histories": method_histories,
        "metrics": metrics,
        "train_time_sec": round(train_time, 1),
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e005_seed{result['config']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--lam-ewc", type=float, default=30.0)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--qfi-samples", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_experiment(
        layers=args.layers, learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task, lam_ewc=args.lam_ewc, lam_qewc=args.lam_qewc,
        qfi_samples=args.qfi_samples, n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )
    print("\n=== E005 retention (mean final acc on earlier tasks T1,T2) ===")
    for m in METHODS:
        mt = result["metrics"][m]
        print(f"  {m:9s} T1_final={mt['tasks']['task1']['test_final']:.3f} "
              f"T2_final={mt['tasks']['task2']['test_final']:.3f} "
              f"T3_final={mt['tasks']['task3']['test_final']:.3f} "
              f"mean_earlier={mt['mean_earlier_task_final']:.3f}")
    out = write_result(result, args.output)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
