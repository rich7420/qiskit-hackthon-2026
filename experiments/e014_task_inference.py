"""E014: remove the task id at test time -- infer it by max-confidence head selection.

Task-IL (our main setting) hands the correct head at test. Here we test the harder
regime: pool all tasks' test sets, hide the task id, and for each sample pick the head that
is most confident, then predict with it. This measures how much accuracy MPI loses when
the task oracle is taken away.

For a pooled test sample x with frozen per-task heads {W_t} on the final backbone theta:

    t_hat(x) = argmax_t  max_c softmax(W_t p_theta(x))_c            (max-confidence routing)
    y_hat(x) = argmax_c (W_{t_hat} p_theta(x))_c

Reported per method (frozen/free/anchor), mean over seeds:
  * known_task_accuracy      -- Task-IL reference (task id given; = e014_compare final row)
  * task_inference_accuracy  -- fraction routed to the correct task's head (TIA)
  * task_agnostic_accuracy   -- end-to-end accuracy with inferred head (the honest number)
  * per-task TIA             -- where routing succeeds/fails
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

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import (  # noqa: E402
    _labels_to_classes,
    fit_linear_head,
    make_probs_qnode,
    probs_features,
    train_backbone,
    train_task_isolated_head,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
METHODS = ("frozen_head", "free_head", "anchor_head")
TASK_KEYS = ("task1", "task2", "task3")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "src/e014_oiqcl.py", Path(__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _train_variant(method, tasks, *, layers, lr, epochs, alpha, seed):
    """Return (probs_qnode, final_weights, [frozen LinearHead per task])."""
    probs_qnode, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers)
    weights = None
    heads = []
    for phase, task in enumerate(tasks):
        if phase == 0:
            weights, _, _ = train_backbone(task, n_qubits=N_QUBITS, n_layers=layers,
                                           lr=lr, epochs=epochs, seed=seed)
        elif method != "frozen_head":
            use_alpha = alpha if method == "anchor_head" else 0.0
            anchor = np.asarray(weights) if method == "anchor_head" else None
            weights, _, _ = train_task_isolated_head(
                probs_qnode, weights, task, train_theta=True, alpha=use_alpha,
                anchor=anchor, lr=lr, epochs=epochs, head_seed=seed + phase)
        P_tr = probs_features(probs_qnode, weights, task.X_train)
        heads.append(fit_linear_head(P_tr, task.y_train, task.name, seed=seed))
    return probs_qnode, weights, heads


def _evaluate(probs_qnode, weights, heads, tasks):
    """Task-IL reference + task-agnostic (max-confidence routed) metrics on pooled test."""
    T = len(heads)
    true_task = np.concatenate([np.full(len(t.X_test), j) for j, t in enumerate(tasks)])
    true_cls = _labels_to_classes(np.concatenate([t.y_test for t in tasks]))
    X_pool = np.concatenate([t.X_test for t in tasks])
    P = probs_features(probs_qnode, weights, X_pool)  # (N_pool, 2^n) on the final backbone

    conf = np.zeros((T, len(P)))
    preds = np.zeros((T, len(P)), dtype=int)
    for t, h in enumerate(heads):
        proba = h.clf.predict_proba(P)          # (N, C) calibrated-ish class probabilities
        conf[t] = proba.max(axis=1)             # confidence of head t on each sample
        preds[t] = h.clf.predict(P)             # head t's class prediction
    t_hat = conf.argmax(axis=0)                 # inferred task = most confident head
    routed_pred = preds[t_hat, np.arange(len(P))]

    # Alternative router: a dedicated linear task classifier over p_theta, trained on the
    # pooled TRAIN sets with task labels (task id is known at train time, hidden at test).
    P_tr_pool = np.concatenate([probs_features(probs_qnode, weights, t.X_train) for t in tasks])
    tr_task = np.concatenate([np.full(len(t.X_train), j) for j, t in enumerate(tasks)])
    router = LogisticRegression(max_iter=2000, C=1.0)
    router.fit(P_tr_pool, tr_task)
    t_hat_r = router.predict(P)
    routed_pred_r = preds[t_hat_r, np.arange(len(P))]

    known = float(np.mean([
        heads[j].accuracy(probs_features(probs_qnode, weights, tasks[j].X_test), tasks[j].y_test)
        for j in range(T)]))

    def _confusion(t_hat_arr):
        return [[int(np.sum((true_task == a) & (t_hat_arr == b))) for b in range(T)]
                for a in range(T)]

    return {
        "known_task_accuracy": round(known, 4),
        # max-confidence routing
        "task_inference_accuracy": round(float(np.mean(t_hat == true_task)), 4),
        "task_agnostic_accuracy": round(float(np.mean(routed_pred == true_cls)), 4),
        "per_task_task_inference": {
            TASK_KEYS[j]: round(float(np.mean(t_hat[true_task == j] == j)), 4) for j in range(T)},
        "task_confusion": _confusion(t_hat),
        # learned linear router
        "router_task_inference_accuracy": round(float(np.mean(t_hat_r == true_task)), 4),
        "router_task_agnostic_accuracy": round(float(np.mean(routed_pred_r == true_cls)), 4),
        "router_per_task_task_inference": {
            TASK_KEYS[j]: round(float(np.mean(t_hat_r[true_task == j] == j)), 4) for j in range(T)},
        "router_confusion": _confusion(t_hat_r),
    }


def run(*, layers=12, lr=0.05, epochs=20, alpha=5.0, n_train=800, n_test=200,
        seed=42, verbose=True) -> dict[str, Any]:
    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [t1, t2, t3]
    started = time.perf_counter()
    methods: dict[str, Any] = {}
    for m in METHODS:
        if verbose:
            print(f"  seed {seed}: {m}", flush=True)
        qnode, weights, heads = _train_variant(m, tasks, layers=layers, lr=lr,
                                               epochs=epochs, alpha=alpha, seed=seed)
        methods[m] = _evaluate(qnode, weights, heads, tasks)
        if verbose:
            r = methods[m]
            print(f"    known(Task-IL)={r['known_task_accuracy']:.3f}  "
                  f"| max-conf: TIA={r['task_inference_accuracy']:.3f} "
                  f"acc={r['task_agnostic_accuracy']:.3f}  "
                  f"| router: TIA={r['router_task_inference_accuracy']:.3f} "
                  f"acc={r['router_task_agnostic_accuracy']:.3f}", flush=True)
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "e014_task_inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"routing": "max-confidence head selection (task id hidden at test)",
                   "tasks": [t.name for t in tasks], "n_qubits": N_QUBITS, "layers": layers,
                   "epochs_per_task": epochs, "lr": lr, "alpha_l2_anchor": alpha,
                   "n_train": n_train, "n_test": n_test, "seed": seed},
        "methods": methods, "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(layers=args.layers, lr=args.lr, epochs=args.epochs, alpha=args.alpha,
                 n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / f"e014_task_inference_seed{args.seed}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
