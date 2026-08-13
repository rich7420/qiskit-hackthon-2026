"""e015 -- OI-QCL (measurement-side CL) on quantum time-series forecasting (Task-IL).

Ports e014's Observable-Isolated Quantum Continual Learning from Task-IL classification to
one-step-ahead regression on the e009 forecasting series (narma_5 -> damped_shm -> bessel_j2).

Shared recurrent data-reuploading backbone theta; every task gets its own linear head
(= diagonal observable) over the full computational-basis probabilities p_theta(x).  Old
heads are frozen, so measurement-side forgetting is a structural zero; only backbone drift
can move an earlier task's error.  Task identity is known at test (Task-IL).

Methods
  sequential   -- e009 naive: shared tanh readout, theta continued, no isolation (CF baseline)
  qewc         -- e009 QFI-weighted EWC anchor on theta (strongest e009 regularizer baseline)
  frozen_head  -- Variant A: theta frozen after Task 1 + isolated per-task linear heads
  free_head    -- Variant B: theta keeps training + isolated heads (representation-drift probe)
  anchor_head  -- Variant C: soft-L2-anchored theta + isolated heads (MAIN candidate)

Reports the NMSE matrix R (R[i][j] = test NMSE on task j after training through task i, lower
better), plus:
  retention  = mean_{j<T} R[T][j]           (final error on earlier tasks)
  plasticity = R[T][T]                       (final error on the last task)
  forgetting = mean_{j<T} (R[T][j] - R[j][j])  (NMSE increase on earlier tasks; ~0 is good)
For the isolated-head methods forgetting is *exactly* representation forgetting: the frozen
head never forgets, so any increase is backbone drift (zero by construction for Variant A).

Run:
    python experiments/e015_oiqcl_forecast_compare.py --tasks narma_5 damped_shm bessel_j2 --seed 42
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

from src.e009_data import load_task_sequence  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402
from src.e014_oiqcl_forecast import (  # noqa: E402
    fit_linear_head,
    make_probs_forecaster,
    probs_features,
    train_backbone_forecast,
    train_task_isolated_head_forecast,
)

RESULTS = ROOT / "results"
SHARED_METHODS = ("sequential", "qewc")          # reuse the e009 engine (naive == sequential)
ISOLATED_METHODS = ("frozen_head", "free_head", "anchor_head")


def _matrix_stats(R: list[list[float | None]]) -> dict[str, Any]:
    """retention / plasticity / forgetting / avg from a lower-triangular NMSE matrix."""
    T = len(R)
    final = [R[T - 1][j] for j in range(T)]
    retention = float(np.mean(final[:-1])) if T > 1 else 0.0
    forgetting = float(np.mean([R[T - 1][j] - R[j][j] for j in range(T - 1)])) if T > 1 else 0.0
    return {
        "final_nmse_row": [round(v, 4) for v in final],
        "retention_earlier_nmse": round(retention, 4),
        "plasticity_final_nmse": round(float(final[-1]), 4),
        "avg_earlier_forgetting": round(forgetting, 4),
        "avg_final_nmse": round(float(np.mean(final)), 4),
    }


def _shared_R(method: str, tasks, *, layers, seq_len, lr, epochs, lam, seed):
    """Run an e009 shared-readout method and reduce its per-epoch history to an NMSE matrix.

    R[phase][j] = task j's test NMSE at the end of phase (phase p ends at epoch p*epochs).
    """
    e009_method = "naive" if method == "sequential" else method
    res = train_method(e009_method, tasks, n_layers=layers, seq_len=seq_len, lr=lr,
                       epochs=epochs, lam=lam, buffer_size=24, seed=seed, verbose=False)
    names = [t.name for t in tasks]
    R: list[list[float | None]] = [[None] * len(tasks) for _ in tasks]
    for phase in range(len(tasks)):
        snap = res["history"][(phase + 1) * epochs]["nmse"]
        for j, nm in enumerate(names):
            R[phase][j] = float(snap[nm])
    return {"R": [[round(v, 4) for v in row] for row in R], **_matrix_stats(R)}


def _isolated_R(method: str, tasks, *, layers, seq_len, lr, epochs, alpha, seed, verbose):
    """frozen_head / free_head / anchor_head: shared backbone + one isolated head per task."""
    probs_qnode, _ = make_probs_forecaster(n_qubits=4, n_layers=layers, seq_len=seq_len)
    heads: list = []
    circ_w = None
    R: list[list[float | None]] = [[None] * len(tasks) for _ in tasks]

    for phase, task in enumerate(tasks):
        if phase == 0:
            circ_w, _, _ = train_backbone_forecast(task, n_qubits=4, n_layers=layers,
                                                   seq_len=seq_len, lr=lr, epochs=epochs, seed=seed)
            P_tr = probs_features(probs_qnode, circ_w, task.X_train)
            heads.append(fit_linear_head(P_tr, task.y_train, task.name))
        else:
            train_theta = method != "frozen_head"
            use_alpha = alpha if method == "anchor_head" else 0.0
            anchor = np.asarray(circ_w) if method == "anchor_head" else None
            circ_w, head = train_task_isolated_head_forecast(
                probs_qnode, circ_w, task, train_theta=train_theta, alpha_anchor=use_alpha,
                anchor=anchor, lr=lr, epochs=epochs, head_seed=seed + phase,
            )
            heads.append(head)
        # Task-IL eval: every seen task read out with its own frozen head on the current theta.
        for j in range(phase + 1):
            P_te = probs_features(probs_qnode, circ_w, tasks[j].X_test)
            R[phase][j] = heads[j].nmse(P_te, tasks[j].y_test)
        if verbose:
            seen = " ".join(f"{tasks[j].name}={R[phase][j]:.3f}" for j in range(phase + 1))
            print(f"    [{method:11s}] after {task.name}: {seen}", flush=True)
    return {"R": [[round(v, 4) if v is not None else None for v in row] for row in R],
            **_matrix_stats(R)}


def run_experiment(*, tasks_names, layers=2, seq_len=8, lr=0.05, epochs=40, alpha=5.0,
                   lam=5.0, seed=42, verbose=True) -> dict[str, Any]:
    tasks = load_task_sequence(tasks_names, seq_len=seq_len)
    if verbose:
        print(f"e015 OI-QCL forecast seed={seed}: {tasks_names} (seq_len={seq_len}, Task-IL)",
              flush=True)
    started = time.perf_counter()
    methods: dict[str, Any] = {}
    for m in SHARED_METHODS:
        if verbose:
            print(f"  running {m} (e009 shared readout)...", flush=True)
        methods[m] = _shared_R(m, tasks, layers=layers, seq_len=seq_len, lr=lr, epochs=epochs,
                               lam=lam, seed=seed)
    for m in ISOLATED_METHODS:
        if verbose:
            print(f"  running {m} (OI-QCL)...", flush=True)
        methods[m] = _isolated_R(m, tasks, layers=layers, seq_len=seq_len, lr=lr, epochs=epochs,
                                 alpha=alpha, seed=seed, verbose=verbose)
    elapsed = time.perf_counter() - started

    return {
        "experiment": "e015_oiqcl_forecast_compare",
        "config": {
            "setting": "Task-Incremental Learning (task id known at test), one-step forecasting",
            "tasks": list(tasks_names), "seq_len": seq_len, "n_qubits": 4, "layers": layers,
            "optimizer": "Adam", "learning_rate": lr, "epochs_per_task": epochs,
            "alpha_l2_anchor": alpha, "lambda_qewc": lam, "seed": seed,
            "isolated_head": "ridge linear head over 2^n probs (diagonal observable per task)",
        },
        "methods": methods,
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_experiment(tasks_names=args.tasks, layers=args.layers, seq_len=args.seq_len,
                            lr=args.lr, epochs=args.epochs_per_task, alpha=args.alpha,
                            lam=args.lam, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or (RESULTS / f"e015_oiqcl_forecast_seed{args.seed}.json")
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n=== e015 OI-QCL forecast (lower NMSE = better; forgetting ~0 is good) ===")
    for m, r in result["methods"].items():
        print(f"  {m:12s} retention={r['retention_earlier_nmse']:.3f} "
              f"plasticity={r['plasticity_final_nmse']:.3f} "
              f"forgetting={r['avg_earlier_forgetting']:+.3f} "
              f"avg_final={r['avg_final_nmse']:.3f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
