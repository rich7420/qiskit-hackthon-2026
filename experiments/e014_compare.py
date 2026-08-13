"""E014: five-method continual comparison for measurement-side CL (Task-IL).

Sequence: MNIST 0/1 -> Fashion-MNIST 0/1 -> SPT/ATF phases (same as e004/e005).
Task identity is known at test time (Task-Incremental Learning), stated up front.

Methods (mentor review, sec 15):
  1. sequential   -- shared softmax readout, theta continued, no isolation (CF baseline)
  2. qewc         -- shared readout + QFI-weighted EWC anchor on theta (reg baseline, e005)
  3. frozen_head  -- Variant A: theta frozen after T1, one isolated linear head per task
  4. free_head    -- Variant B: theta keeps training, isolated heads (rep-drift probe)
  5. anchor_head  -- Variant C: soft-L2-anchored theta + isolated heads (MAIN candidate)

Reports the accuracy matrix R (R[i][j] = test acc on task j after training through task i),
average accuracy ACC = mean_j R[T][j], and backward transfer BWT = mean_{j<T}(R[T][j]-R[j][j]).
For the isolated-head methods, -BWT_j = R[j][j]-R[T][j] is exactly the *representation
forgetting* of task j (old head, drifted backbone): the measurement side never forgets.
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
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e005_consolidation import EWC, quantum_fisher_diag  # noqa: E402
from src.e005_softmax import accuracy as softmax_accuracy  # noqa: E402
from src.e005_softmax import bce_loss, classical_fisher_diag, make_softmax_qnode  # noqa: E402
from src.e014_oiqcl import (  # noqa: E402
    fit_linear_head,
    make_probs_qnode,
    probs_features,
    train_backbone,
    train_task_isolated_head,
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
        ROOT / "src/e005_consolidation.py",
        ROOT / "src/continual_data.py",
        ROOT / "src/phase_data.py",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _acc_matrix_stats(R: list[list[float | None]]) -> dict[str, Any]:
    """ACC and BWT from a lower-triangular accuracy matrix (R[i][j] for i>=j)."""
    T = len(R)
    final_row = [R[T - 1][j] for j in range(T)]
    acc = float(np.mean(final_row))
    bwt = float(np.mean([R[T - 1][j] - R[j][j] for j in range(T - 1)])) if T > 1 else 0.0
    return {"ACC": round(acc, 4), "BWT": round(bwt, 4),
            "final_row": [round(v, 4) for v in final_row]}


def _run_shared_readout(method: str, tasks, *, layers, lr, epochs, lam_qewc,
                        qfi_samples, seed, verbose, lam_ewc=30.0, n_qubits=N_QUBITS) -> dict[str, Any]:
    """sequential / ewc / qewc: one shared softmax readout, theta continued across tasks.

    ewc consolidates with the classical (empirical) Fisher; qewc with the diagonal QFI.
    """
    clf_qnode, weight_shape = make_softmax_qnode(n_qubits=n_qubits, n_layers=layers)
    qfi_qnode, _ = make_softmax_qnode(n_qubits=n_qubits, n_layers=layers)  # state-equivalent
    reg = EWC({"sequential": 0.0, "ewc": lam_ewc, "qewc": lam_qewc}[method])
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    R: list[list[float | None]] = [[None] * len(tasks) for _ in tasks]

    for phase, task in enumerate(tasks):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase):
            return bce_loss(clf_qnode, W, Xtr, ytr) + reg.penalty(W.flatten(), phase + 1)

        for _ in range(epochs):
            weights = optimizer.step(cost, weights)
        for j, tj in enumerate(tasks):
            R[phase][j] = softmax_accuracy(clf_qnode, weights, tj.X_test, tj.y_test)
        if method in ("ewc", "qewc") and phase < len(tasks) - 1:
            if method == "ewc":
                fisher = classical_fisher_diag(clf_qnode, weights, task.X_train, task.y_train)
            else:
                fisher = quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                             n_samples=qfi_samples, seed=seed)
            reg.consolidate(np.asarray(weights).flatten(), fisher)
        if verbose:
            print(f"    [{method}] after T{phase + 1}: "
                  + " ".join(f"{k}={R[phase][j]:.3f}" for j, k in enumerate(TASK_KEYS)), flush=True)
    return {"R": [[round(v, 4) if v is not None else None for v in row] for row in R],
            **_acc_matrix_stats([[v for v in row] for row in R])}


def _run_isolated_head(method: str, tasks, *, layers, lr, epochs, alpha, seed, verbose,
                       n_qubits=N_QUBITS) -> dict[str, Any]:
    """frozen_head / free_head / anchor_head: shared backbone + one isolated head per task.

    theta is advanced by a quantum gradient step (Variants B/C) or frozen (A); each task's
    stored readout is a *converged classical* linear head (logistic regression) over the
    frozen probs -- the "trains in seconds, no quantum gradients" measurement side.  A head
    fit at task j on theta_j is kept fixed and re-evaluated on the *current* theta, so any
    later backbone drift surfaces as representation forgetting in R.
    """
    probs_qnode, _ = make_probs_qnode(n_qubits=n_qubits, n_layers=layers)
    heads: list = []  # frozen sklearn LinearHead per completed task (fit at theta_j)
    R: list[list[float | None]] = [[None] * len(tasks) for _ in tasks]
    weights = None

    for phase, task in enumerate(tasks):
        if phase == 0:
            # Solid shared representation from Task 1 (softmax/BCE backbone).
            weights, _, _ = train_backbone(task, n_qubits=n_qubits, n_layers=layers,
                                           lr=lr, epochs=epochs, seed=seed)
        elif method != "frozen_head":
            # Variant B: free theta; Variant C: soft-L2 anchor to previous theta.
            use_alpha = alpha if method == "anchor_head" else 0.0
            anchor = np.asarray(weights) if method == "anchor_head" else None
            weights, _, _ = train_task_isolated_head(
                probs_qnode, weights, task, train_theta=True, alpha=use_alpha,
                anchor=anchor, lr=lr, epochs=epochs, head_seed=seed + phase,
            )
        # Fit this task's converged linear head on the current backbone.
        P_tr = probs_features(probs_qnode, weights, task.X_train)
        heads.append(fit_linear_head(P_tr, task.y_train, task.name, seed=seed))
        # Task-IL eval: every seen task read out with its own frozen head on current theta.
        for j in range(phase + 1):
            P_te = probs_features(probs_qnode, weights, tasks[j].X_test)
            R[phase][j] = heads[j].accuracy(P_te, tasks[j].y_test)
        if verbose:
            seen = " ".join(f"{TASK_KEYS[j]}={R[phase][j]:.3f}" for j in range(phase + 1))
            print(f"    [{method}] after T{phase + 1}: {seen}", flush=True)
    return {"R": [[round(v, 4) if v is not None else None for v in row] for row in R],
            **_acc_matrix_stats(R)}


def run_experiment(*, layers=12, lr=0.05, epochs=20, alpha=5.0, lam_qewc=0.8, lam_ewc=30.0,
                   qfi_samples=64, n_train=800, n_test=200, seed=42, verbose=True) -> dict[str, Any]:
    task1, task2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    task3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [task1, task2, task3]
    if verbose:
        print(f"E014 compare seed={seed}: {[t.name for t in tasks]} (Task-IL)", flush=True)

    started = time.perf_counter()
    methods: dict[str, Any] = {}
    for m in ("sequential", "ewc", "qewc"):
        methods[m] = _run_shared_readout(m, tasks, layers=layers, lr=lr, epochs=epochs,
                                         lam_qewc=lam_qewc, lam_ewc=lam_ewc, qfi_samples=qfi_samples,
                                         seed=seed, verbose=verbose)
    for m in ("frozen_head", "free_head", "anchor_head"):
        methods[m] = _run_isolated_head(m, tasks, layers=layers, lr=lr, epochs=epochs,
                                        alpha=alpha, seed=seed, verbose=verbose)
    elapsed = time.perf_counter() - started

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "e014_compare",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scipy", "scikit-learn")}},
        "config": {
            "setting": "Task-Incremental Learning (task id known at test)",
            "tasks": [t.name for t in tasks], "task_keys": list(TASK_KEYS),
            "n_qubits": N_QUBITS, "layers": layers,
            "optimizer": "Adam", "learning_rate": lr, "epochs_per_task": epochs,
            "alpha_l2_anchor": alpha, "lambda_qewc": lam_qewc, "lambda_ewc": lam_ewc,
            "qfi_samples": qfi_samples,
            "isolated_head": "logistic-style linear head over 2^n probs (diagonal observable per class)",
            "n_train_per_task": n_train, "n_test_per_task": n_test, "seed": seed,
            "method_notes": {
                "sequential": "shared softmax readout, no isolation (CF baseline)",
                "ewc": "shared readout + classical-Fisher EWC anchor (e005)",
                "qewc": "shared readout + QFI-weighted EWC anchor (e005)",
                "frozen_head": "Variant A: theta frozen after T1 + isolated heads",
                "free_head": "Variant B: theta free + isolated heads (rep-drift probe)",
                "anchor_head": "Variant C: soft-L2 theta + isolated heads (MAIN)",
            },
        },
        "methods": methods,
        "elapsed_sec": round(elapsed, 1),
    }


def write_result(result: dict[str, Any], output: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    if output is None:
        output = RESULTS / f"e014_compare_seed{result['config']['seed']}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--lam-ewc", type=float, default=30.0)
    ap.add_argument("--qfi-samples", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_experiment(
        layers=args.layers, lr=args.lr, epochs=args.epochs, alpha=args.alpha,
        lam_qewc=args.lam_qewc, lam_ewc=args.lam_ewc, qfi_samples=args.qfi_samples,
        n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )
    path = write_result(result, args.output)
    print("\nACC / BWT by method:")
    for m, r in result["methods"].items():
        print(f"  {m:12s} ACC={r['ACC']:.3f}  BWT={r['BWT']:+.3f}  final_row={r['final_row']}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
