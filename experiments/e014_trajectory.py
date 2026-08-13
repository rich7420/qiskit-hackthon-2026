"""E014 trajectory recorder: per-epoch test accuracy on all three tasks, all methods.

Produces the same history schema as e005 (histories[method] = list of
{epoch, phase, test_accuracy:{task1,task2,task3}}) so the three-panel per-task curve plot
(scripts/plot_e014_trajectory.py) can render OI-QCL against the shared-readout baselines in
the e009/e013 style, with task boundaries and seed bands.

Method families:
  * sequential / qewc  -- one shared softmax readout; any task is evaluable at any epoch.
  * frozen/free/anchor -- isolated per-task heads.  A task's curve begins at its own
    boundary (its head is created then); before that it is null (unseen).  During a task's
    own phase the active head is refit (converged logistic regression) each epoch so the
    curve shows adaptation; it is frozen at the phase end and re-evaluated on the drifting
    (free/anchor) or fixed (frozen) backbone afterwards -- i.e. the retention curve.
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
    _head_logits,
    _probs_batched,
    _softmax_ce,
    head_accuracy_theta,
    init_head_weights,
    make_probs_qnode,
)
from src.phase_data import N_QUBITS, load_cluster_full, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")
SHARED = ("sequential", "ewc", "qewc")
ISOLATED = ("frozen_head", "free_head", "anchor_head")
METHODS = SHARED + ISOLATED


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "src/e014_oiqcl.py", ROOT / "src/e005_softmax.py",
                 ROOT / "src/e005_consolidation.py", Path(__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _shared_history(method, tasks, *, layers, lr, epochs, lam_qewc, qfi_samples, seed, lam_ewc=30.0):
    clf_qnode, weight_shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    qfi_qnode, _ = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    reg = EWC({"sequential": 0.0, "ewc": lam_ewc, "qewc": lam_qewc}[method])
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    history: list[dict[str, Any]] = []

    def snap(epoch, phase):
        history.append({"epoch": epoch, "phase": phase, "test_accuracy": {
            k: round(softmax_accuracy(clf_qnode, weights, t.X_test, t.y_test), 4)
            for k, t in zip(TASK_KEYS, tasks)}})

    snap(0, 0)
    for phase, task in enumerate(tasks):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase):
            return bce_loss(clf_qnode, W, Xtr, ytr) + reg.penalty(W.flatten(), phase + 1)

        for _ in range(epochs):
            weights = optimizer.step(cost, weights)
            snap(history[-1]["epoch"] + 1, phase + 1)
        if method in ("ewc", "qewc") and phase < len(tasks) - 1:
            if method == "ewc":
                fisher = classical_fisher_diag(clf_qnode, weights, task.X_train, task.y_train)
            else:
                fisher = quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                             n_samples=qfi_samples, seed=seed)
            reg.consolidate(np.asarray(weights).flatten(), fisher)
    return history


def _isolated_history(method, tasks, *, layers, lr, epochs, alpha, seed, readout_qubits=None):
    """Every curve is a genuine gradient learning curve on a common accuracy axis.

    Each task's readout is a learnable diagonal-observable head (W, b) trained by Adam --
    NOT a per-epoch converged classical fit -- so its within-phase curve rises from chance
    like the shared-readout baselines instead of jumping instantly high.  Faithful to
    min_{theta, W_t} L_t (mentor sec 29): Task 1 trains theta + W_1 jointly; later tasks
    add a fresh W_t while frozen (A) / free (B) / soft-anchored (C) drive theta; every old
    head is frozen at its boundary and re-evaluated on the current backbone (retention).
    """
    readout_wires = None if readout_qubits is None else range(readout_qubits)
    probs_qnode, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers, readout_wires=readout_wires)
    weight_shape = (layers, N_QUBITS, 2)
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    frozen_heads: dict[int, tuple] = {}  # task -> (W, b) numpy, frozen at its boundary
    history: list[dict[str, Any]] = []

    def snap(epoch, phase, live_head):
        acc: dict[str, Any] = {}
        for j, key in enumerate(TASK_KEYS):
            head = frozen_heads.get(j) or (live_head if j == phase else None)
            if head is None:  # unseen future task
                acc[key] = None
            else:
                W, b = head
                acc[key] = round(
                    head_accuracy_theta(probs_qnode, weights, W, b,
                                        tasks[j].X_test, tasks[j].y_test), 4)
        history.append({"epoch": epoch, "phase": phase, "test_accuracy": acc})

    snap(0, 0, None)
    for phase, task in enumerate(tasks):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = np.asarray(task.y_train)
        # A freezes theta after Task 1; B/C keep training it (C anchors softly).
        train_theta = (phase == 0) or (method != "frozen_head")
        use_alpha = alpha if (method == "anchor_head" and phase > 0) else 0.0
        anchor = np.asarray(weights) if (method == "anchor_head" and phase > 0) else None
        weights = pnp.array(np.asarray(weights), requires_grad=bool(train_theta))
        n_probs = 2 ** (readout_qubits if readout_qubits is not None else N_QUBITS)
        W_live, b_live = init_head_weights(n_probs, seed=seed + phase)
        optimizer = qml.AdamOptimizer(lr)

        def cost(weights, W, b, Xtr=Xtr, ytr=ytr, use_alpha=use_alpha, anchor=anchor):
            logits = _head_logits(_probs_batched(probs_qnode, weights, Xtr), W, b)
            loss = _softmax_ce(logits, ytr)
            if use_alpha > 0.0 and anchor is not None:
                loss = loss + use_alpha * pnp.sum((weights - anchor) ** 2)
            return loss

        for _ in range(epochs):
            weights, W_live, b_live = optimizer.step(cost, weights, W_live, b_live)
            live = (np.asarray(W_live), np.asarray(b_live))
            snap(history[-1]["epoch"] + 1, phase, live)
        frozen_heads[phase] = (np.asarray(W_live), np.asarray(b_live))  # freeze at boundary
    return history


def run(*, layers=12, lr=0.05, epochs=20, alpha=5.0, lam_qewc=0.8, qfi_samples=64,
        n_train=800, n_test=200, seed=42, t3="spt", readout_qubits=None,
        verbose=True) -> dict[str, Any]:
    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    _t3_loader = {"spt": load_spt_atf, "cluster_full": load_cluster_full}[t3]
    t3_task = _t3_loader(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [t1, t2, t3_task]
    started = time.perf_counter()
    histories: dict[str, Any] = {}
    for m in METHODS:
        if verbose:
            print(f"  seed {seed}: {m}", flush=True)
        if m in SHARED:
            histories[m] = _shared_history(m, tasks, layers=layers, lr=lr, epochs=epochs,
                                           lam_qewc=lam_qewc, qfi_samples=qfi_samples, seed=seed)
        else:
            histories[m] = _isolated_history(m, tasks, layers=layers, lr=lr, epochs=epochs,
                                             alpha=alpha, seed=seed, readout_qubits=readout_qubits)
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "e014_trajectory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "training": {"epochs_per_task": epochs, "layers": layers, "n_qubits": N_QUBITS,
                     "learning_rate": lr, "alpha_l2_anchor": alpha, "lambda_qewc": lam_qewc,
                     "qfi_samples": qfi_samples, "n_train": n_train, "n_test": n_test, "seed": seed},
        "tasks": [t.name for t in tasks], "task_keys": list(TASK_KEYS),
        "histories": histories, "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--qfi-samples", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t3", choices=["spt", "cluster_full"], default="spt")
    ap.add_argument("--readout-qubits", type=int, default=None)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(layers=args.layers, lr=args.lr, epochs=args.epochs, alpha=args.alpha,
                 lam_qewc=args.lam_qewc, qfi_samples=args.qfi_samples,
                 n_train=args.n_train, n_test=args.n_test, seed=args.seed,
                 t3=args.t3, readout_qubits=args.readout_qubits)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / f"e014_trajectory_seed{args.seed}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
