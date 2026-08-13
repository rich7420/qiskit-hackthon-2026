"""e015 -- remove the task id at test time: route each window to a head (task-agnostic).

The Task-IL comparison (``e015_oiqcl_forecast_compare``) hands the correct head at test.  Here
we drop that oracle: pool all tasks' test windows, hide the task id, and for each window infer
which task's head to use, then forecast with it.  This is the forecasting port of e014's
``e014_task_inference`` -- the honest number when the task oracle is taken away.

Two routers over the frozen per-task heads {W_t} on the final backbone p_theta(x):
  * centroid   -- naive baseline: route to the nearest per-task mean probs vector (no labels,
                  the regression analog of e014's weak max-confidence router).
  * learned    -- a linear LogisticRegression task classifier over p_theta trained on the
                  pooled TRAIN windows with task labels (task id known at train, hidden at test);
                  this is the good router in e014 (~0.91 task-agnostic).

Reported per method (frozen/free/anchor), mean over seeds:
  * known_task_nmse           -- Task-IL reference (task id given; = e015_compare final row)
  * task_inference_accuracy   -- fraction of windows routed to the correct task's head (TIA)
  * task_agnostic_nmse        -- per-task NMSE with the inferred head, averaged (the honest number)
  * per-task TIA + confusion  -- where routing succeeds/fails

Run:
    python experiments/e015_router.py --tasks narma_5 damped_shm bessel_j2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.e009_data import load_task_sequence  # noqa: E402
from src.e014_oiqcl_forecast import probs_features, train_isolated_variant  # noqa: E402

RESULTS = ROOT / "results"
METHODS = ("frozen_head", "free_head", "anchor_head")


def _nmse(pred: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(pred, float) - y) ** 2) / (np.var(y) + 1e-12))


def _evaluate(probs_qnode, circ_w, heads, tasks) -> dict[str, Any]:
    T = len(heads)
    names = [t.name for t in tasks]

    # Pooled test windows on the final backbone; per-head prediction for every window.
    true_task = np.concatenate([np.full(len(t.X_test), j) for j, t in enumerate(tasks)])
    y_pool = np.concatenate([t.y_test for t in tasks])
    P = np.concatenate([probs_features(probs_qnode, circ_w, t.X_test) for t in tasks])
    per_head_pred = np.stack([h.predict(P) for h in heads], axis=0)  # (T, N_pool)

    # Pooled train windows (final backbone) with task labels: known at train, hidden at test.
    P_tr = [probs_features(probs_qnode, circ_w, t.X_train) for t in tasks]
    P_tr_pool = np.concatenate(P_tr)
    tr_task = np.concatenate([np.full(len(t.X_train), j) for j, t in enumerate(tasks)])

    centroids = np.stack([p.mean(axis=0) for p in P_tr], axis=0)          # (T, 2^n)
    t_hat_centroid = np.argmin(
        ((P[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)

    router = LogisticRegression(max_iter=2000, C=1.0)
    router.fit(P_tr_pool, tr_task)
    t_hat_router = router.predict(P)

    idx = np.arange(len(P))
    known_nmse = {names[j]: round(_nmse(per_head_pred[j][true_task == j], y_pool[true_task == j]), 4)
                  for j in range(T)}

    def _routed(t_hat):
        routed_pred = per_head_pred[t_hat, idx]
        tia = float(np.mean(t_hat == true_task))
        per_task_nmse = {names[j]: round(_nmse(routed_pred[true_task == j], y_pool[true_task == j]), 4)
                         for j in range(T)}
        per_task_tia = {names[j]: round(float(np.mean(t_hat[true_task == j] == j)), 4)
                        for j in range(T)}
        confusion = [[int(np.sum((true_task == a) & (t_hat == b))) for b in range(T)]
                     for a in range(T)]
        return {
            "task_inference_accuracy": round(tia, 4),
            "task_agnostic_nmse": round(float(np.mean(list(per_task_nmse.values()))), 4),
            "per_task_nmse": per_task_nmse,
            "per_task_task_inference": per_task_tia,
            "task_confusion": confusion,
        }

    return {
        "known_task_nmse": known_nmse,
        "known_task_avg_nmse": round(float(np.mean(list(known_nmse.values()))), 4),
        "centroid": _routed(t_hat_centroid),
        "router": _routed(t_hat_router),
    }


def run(*, tasks_names, layers=2, seq_len=8, lr=0.05, epochs=40, alpha=5.0, seed=42,
        verbose=True) -> dict[str, Any]:
    tasks = load_task_sequence(tasks_names, seq_len=seq_len)
    if verbose:
        print(f"e015 router seed={seed}: {tasks_names} (task id hidden at test)", flush=True)
    started = time.perf_counter()
    methods: dict[str, Any] = {}
    for m in METHODS:
        if verbose:
            print(f"  seed {seed}: {m}", flush=True)
        probs_qnode, circ_w, heads = train_isolated_variant(
            m, tasks, n_qubits=4, n_layers=layers, seq_len=seq_len, lr=lr, epochs=epochs,
            alpha=alpha, seed=seed)
        methods[m] = _evaluate(probs_qnode, circ_w, heads, tasks)
        if verbose:
            r = methods[m]
            print(f"    known(Task-IL) avg_nmse={r['known_task_avg_nmse']:.3f}  "
                  f"| centroid: TIA={r['centroid']['task_inference_accuracy']:.3f} "
                  f"nmse={r['centroid']['task_agnostic_nmse']:.3f}  "
                  f"| router: TIA={r['router']['task_inference_accuracy']:.3f} "
                  f"nmse={r['router']['task_agnostic_nmse']:.3f}", flush=True)
    return {
        "experiment": "e015_router",
        "config": {
            "setting": "task-agnostic (task id hidden at test), one-step forecasting, NMSE",
            "tasks": list(tasks_names), "seq_len": seq_len, "n_qubits": 4, "layers": layers,
            "optimizer": "Adam", "learning_rate": lr, "epochs_per_task": epochs,
            "alpha_l2_anchor": alpha, "seed": seed,
            "centroid_router": "nearest per-task mean-probs vector (unsupervised, no task labels)",
            "learned_router": "LogisticRegression over p_theta(x), pooled-train task labels",
        },
        "methods": methods,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(tasks_names=args.tasks, layers=args.layers, seq_len=args.seq_len, lr=args.lr,
                 epochs=args.epochs_per_task, alpha=args.alpha, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / f"e015_router_seed{args.seed}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n=== e015 task-agnostic routing (NMSE lower=better; TIA higher=better) ===")
    for m, r in result["methods"].items():
        print(f"  {m:12s} known={r['known_task_avg_nmse']:.3f}  "
              f"centroid: TIA={r['centroid']['task_inference_accuracy']:.3f} "
              f"nmse={r['centroid']['task_agnostic_nmse']:.3f}  "
              f"router: TIA={r['router']['task_inference_accuracy']:.3f} "
              f"nmse={r['router']['task_agnostic_nmse']:.3f}")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
