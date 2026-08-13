"""E011 - schedule study: blocked (sequential) vs interleaved training, fixed epoch budget.

E004/E005 train the three-task sequence in *blocks* (MNIST x20 -> Fashion-MNIST x20 ->
SPT/ATF x20), which is what makes earlier tasks get forgotten. This experiment asks a
different question: if the *total* epoch budget per task is held fixed (still 20 each, 60
gradient steps total), does simply *interleaving* the tasks change the picture?

- blocked:           step order  [T1]*E , [T2]*E , [T3]*E         (== E005 baseline)
- interleaved:       step order  [T1, T2, T3] repeated E times
- qewc_interleaved:  interleaved order PLUS online QEWC consolidation (below)

The first two use the SAME initial weights, the SAME learner (4 qubits, RY/RZ + CNOT, 20
layers, Adam), and the SAME per-task epoch count -- only the ordering of the 60 gradient
steps differs. No consolidation: this isolates the effect of the schedule alone. Interleaving
is the classic continual-learning upper bound (it approximates joint training).

The third arm asks whether QEWC's consolidation adds anything ON TOP of interleaving. Because
an interleaved stream has no "task finished" moment where the blocked QEWC (E005) would
consolidate, it uses *online QEWC*: a single running QFI-weighted anchor, re-anchored to the
current weights after every full T1->T2->T3 round with the diagonal QFI accumulated across
rounds (Schwarz et al. online EWC, with the quantum Fisher of E005). Same 60-step budget.

Test accuracy on all three tasks is recorded after every gradient step so the arms can be
overlaid on a shared 0..60 epoch axis. Single seed per run (aggregate via
scripts/e011_run_multiseed.py).
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

from src.continual_data import Task, load_two_tasks  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402
from src.e005_consolidation import quantum_fisher_diag  # noqa: E402
from src.qnn_pennylane import make_qnode  # noqa: E402 (single-Z qnode, used only for the QFI metric)
from src.e005_softmax import (  # noqa: E402
    accuracy as softmax_accuracy,
    bce_loss,
    make_softmax_qnode,
)

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")
# "blocked"/"interleaved" go through _train_one_schedule; "qewc_interleaved" uses online QEWC.
SCHEDULES = ("blocked", "interleaved", "qewc_interleaved")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src/e005_softmax.py",
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/qnn_pennylane.py",
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _step_order(schedule: str, n_tasks: int, epochs_per_task: int) -> list[int]:
    """Return the sequence of task indices (0-based) for one full training run.

    Both schedules contain exactly `epochs_per_task` steps per task, so the total gradient
    budget is identical; only the ordering differs.
    """
    if schedule == "blocked":
        return [t for t in range(n_tasks) for _ in range(epochs_per_task)]
    if schedule == "interleaved":
        return [t for _ in range(epochs_per_task) for t in range(n_tasks)]
    raise ValueError(f"unknown schedule {schedule!r}")


def _train_one_schedule(
    schedule: str,
    tasks: list[Task],
    *,
    clf_qnode,
    weight_shape,
    lr: float,
    epochs_per_task: int,
    seed: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    history: list[dict[str, Any]] = []

    # Pre-wrap each task's training tensors once (non-trainable inputs).
    xtr = [pnp.array(t.X_train, requires_grad=False) for t in tasks]
    ytr = [pnp.array(t.y_train, requires_grad=False) for t in tasks]

    def snapshot(epoch: int, task_idx: int) -> None:
        history.append({
            "epoch": epoch,
            "task_trained": task_idx + 1 if task_idx >= 0 else 0,
            "test_accuracy": {k: softmax_accuracy(clf_qnode, weights, t.X_test, t.y_test)
                              for k, t in zip(TASK_KEYS, tasks)},
        })

    snapshot(0, -1)
    for epoch, t in enumerate(_step_order(schedule, len(tasks), epochs_per_task), start=1):
        def cost(W, X=xtr[t], y=ytr[t]):
            return bce_loss(clf_qnode, W, X, y)

        weights = optimizer.step(cost, weights)
        snapshot(epoch, t)
    if verbose:
        acc = history[-1]["test_accuracy"]
        print(f"    [{schedule}] final: "
              + " ".join(f"{k}={acc[k]:.3f}" for k in TASK_KEYS), flush=True)
    return history


def _online_penalty(weights_flat, anchor, fisher_diag, lam: float):
    """Online-EWC quadratic penalty (lam/2) * sum_i F_i (theta_i - anchor_i)^2.

    `anchor` / `fisher_diag` are constants (None before the first consolidation); returns a
    differentiable zero in that case so the cost stays autograd-friendly.
    """
    if anchor is None or lam == 0.0:
        return 0.0 * pnp.sum(weights_flat) * 0.0
    diff = weights_flat - anchor
    return 0.5 * lam * pnp.sum(fisher_diag * diff ** 2)


def _train_qewc_interleaved(
    tasks: list[Task],
    *,
    clf_qnode,
    qfi_qnode,
    weight_shape,
    lr: float,
    epochs_per_task: int,
    lam_qewc: float,
    qfi_samples: int,
    seed: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Interleaved schedule ([T1,T2,T3] per round) with online QEWC consolidation.

    After every full round the running QFI (diagonal, averaged over the three tasks at the
    current weights) is accumulated and the anchor is reset to the current weights, so the
    penalty in round r+1 discourages drift away from the round-r solution, weighted by all
    Fisher information gathered so far. Same 60-step budget as the pure schedules.
    """
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    history: list[dict[str, Any]] = []

    xtr = [pnp.array(t.X_train, requires_grad=False) for t in tasks]
    ytr = [pnp.array(t.y_train, requires_grad=False) for t in tasks]
    # QFI is averaged over the interleaved data distribution, so estimate it from a single
    # subsample of the pooled task data each round (one metric-tensor pass, not one per task).
    x_pool = np.concatenate([t.X_train for t in tasks], axis=0)

    n_weights = int(np.prod(weight_shape))
    fisher_run = np.zeros(n_weights)          # accumulated diagonal QFI (Schwarz online EWC)
    anchor = None                             # pnp constant once the first round consolidates

    def snapshot(epoch: int, task_idx: int) -> None:
        history.append({
            "epoch": epoch,
            "task_trained": task_idx + 1 if task_idx >= 0 else 0,
            "test_accuracy": {k: softmax_accuracy(clf_qnode, weights, t.X_test, t.y_test)
                              for k, t in zip(TASK_KEYS, tasks)},
        })

    snapshot(0, -1)
    epoch = 0
    for _round in range(epochs_per_task):
        for t in range(len(tasks)):
            def cost(W, X=xtr[t], y=ytr[t], anchor=anchor, fisher=fisher_run):
                return bce_loss(clf_qnode, W, X, y) + _online_penalty(
                    W.flatten(), anchor, fisher, lam_qewc)

            weights = optimizer.step(cost, weights)
            epoch += 1
            snapshot(epoch, t)
        # Consolidate at the round boundary: grow the running QFI, re-anchor to current weights.
        round_qfi = quantum_fisher_diag(qfi_qnode, weights, x_pool,
                                        n_samples=qfi_samples, seed=seed)
        fisher_run = fisher_run + round_qfi
        anchor = pnp.array(np.asarray(weights).flatten(), requires_grad=False)
    if verbose:
        acc = history[-1]["test_accuracy"]
        print(f"    [qewc_interleaved] final: "
              + " ".join(f"{k}={acc[k]:.3f}" for k in TASK_KEYS), flush=True)
    return history


def _metrics_for(history: list[dict[str, Any]], tasks: list[Task]) -> dict[str, Any]:
    """Per-task final accuracy and forgetting (best-so-far minus final)."""
    per_task = {}
    for idx, key in enumerate(TASK_KEYS):
        curve = [row["test_accuracy"][key] for row in history]
        best = max(curve)
        final = curve[-1]
        per_task[key] = {"name": tasks[idx].name,
                         "test_best": round(best, 4),
                         "test_final": round(final, 4),
                         "forgetting": round(best - final, 4)}
    retained = np.mean([per_task[k]["test_final"] for k in ("task1", "task2")])
    mean_all = np.mean([per_task[k]["test_final"] for k in TASK_KEYS])
    return {"tasks": per_task,
            "mean_earlier_task_final": round(float(retained), 4),
            "mean_all_task_final": round(float(mean_all), 4)}


def run_experiment(
    *,
    layers: int = 20,
    learning_rate: float = 0.02,
    epochs_per_task: int = 20,
    lam_qewc: float = 0.8,
    qfi_samples: int = 32,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    task3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [task1, task2, task3]
    boundaries = [epochs_per_task, 2 * epochs_per_task]  # only meaningful for the blocked run
    clf_qnode, weight_shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    qfi_qnode, _ = make_qnode(n_qubits=N_QUBITS, n_layers=layers)  # single-Z, QFI metric only

    if verbose:
        print(f"E011 seed={seed}: {[t.name for t in tasks]}  "
              f"(epochs_per_task={epochs_per_task}, blocked / interleaved / qewc_interleaved)")

    started = time.perf_counter()
    histories: dict[str, list] = {}
    metrics: dict[str, Any] = {}
    for schedule in SCHEDULES:
        if schedule == "qewc_interleaved":
            histories[schedule] = _train_qewc_interleaved(
                tasks, clf_qnode=clf_qnode, qfi_qnode=qfi_qnode, weight_shape=weight_shape,
                lr=learning_rate, epochs_per_task=epochs_per_task, lam_qewc=lam_qewc,
                qfi_samples=qfi_samples, seed=seed, verbose=verbose,
            )
        else:
            histories[schedule] = _train_one_schedule(
                schedule, tasks, clf_qnode=clf_qnode, weight_shape=weight_shape,
                lr=learning_rate, epochs_per_task=epochs_per_task, seed=seed, verbose=verbose,
            )
        metrics[schedule] = _metrics_for(histories[schedule], tasks)
    train_time = time.perf_counter() - started

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e011_interleaved",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scipy", "scikit-learn")}},
        "config": {
            "tasks": [t.name for t in tasks],
            "n_qubits": N_QUBITS, "layers": layers, "n_weights": int(np.prod(weight_shape)),
            "readout": "two Pauli-Z on qubits 0,1 -> softmax (Eq. 7)",
            "loss": "binary cross-entropy",
            "consolidation": "none for blocked/interleaved; online QEWC (running diagonal QFI, "
                             "re-anchored each round) for qewc_interleaved",
            "lambda_qewc": lam_qewc, "qfi_samples": qfi_samples,
            "optimizer": "Adam", "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task, "total_steps": epochs_per_task * len(tasks),
            "task_boundaries": boundaries, "schedules": list(SCHEDULES),
            "n_train_per_task": n_train, "n_test_per_task": n_test, "seed": seed,
        },
        "histories": histories,
        "metrics": metrics,
        "train_time_sec": round(train_time, 1),
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e011_seed{result['config']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--qfi-samples", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_experiment(
        layers=args.layers, learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task, lam_qewc=args.lam_qewc,
        qfi_samples=args.qfi_samples, n_train=args.n_train, n_test=args.n_test,
        seed=args.seed,
    )
    print("\n=== E011 blocked vs interleaved vs qewc_interleaved (test acc, single seed) ===")
    for s in SCHEDULES:
        mt = result["metrics"][s]
        print(f"  {s:16s} T1={mt['tasks']['task1']['test_final']:.3f} "
              f"T2={mt['tasks']['task2']['test_final']:.3f} "
              f"T3={mt['tasks']['task3']['test_final']:.3f} "
              f"mean_earlier={mt['mean_earlier_task_final']:.3f} "
              f"mean_all={mt['mean_all_task_final']:.3f}")
    out = write_result(result, args.output)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
