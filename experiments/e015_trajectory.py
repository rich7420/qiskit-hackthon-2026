"""e015 -- per-epoch trajectory of OI-QCL vs naive/QEWC on forecasting (for the panel figure).

Records, at EVERY gradient step of the full sequential run, each task's test NMSE, so we can
draw the per-task trajectory panels (T1/T2/T3 vs epoch, mean +/- std over seeds, shaded =
task being trained).  Each task's curve is only recorded from the epoch that task STARTS
training (earlier epochs are None) -- so a panel is blank until its task begins.

Methods (same set as e015_oiqcl_forecast_compare):
  sequential / qewc          -- e009 shared tanh readout (reuse train_method's per-epoch history)
  frozen_head / free_head / anchor_head -- OI-QCL: shared backbone + per-task linear head over
                                           probs.  Current task read with its live gradient head;
                                           past tasks read with their FROZEN head at the current
                                           backbone (so representation drift shows up as forgetting).

Run:
    python experiments/e015_trajectory.py --tasks narma_5 damped_shm bessel_j2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_data import load_task_sequence  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402
from src.e014_oiqcl_forecast import (  # noqa: E402
    _probs_batched,
    make_probs_forecaster,
    probs_features,
)

RESULTS = ROOT / "results"
SHARED_METHODS = ("sequential", "qewc")
ISOLATED_METHODS = ("frozen_head", "free_head", "anchor_head")


def _nmse_wb(P: np.ndarray, W: np.ndarray, b: float, y: np.ndarray) -> float:
    pred = np.asarray(P, float) @ np.asarray(W, float) + float(b)
    y = np.asarray(y, float)
    return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def _shared_traj(method: str, tasks, *, layers, seq_len, lr, epochs, lam, seed):
    """Per-epoch test NMSE for an e009 shared-readout method, clipped to each task's onset."""
    e009_method = "naive" if method == "sequential" else method
    res = train_method(e009_method, tasks, n_layers=layers, seq_len=seq_len, lr=lr,
                       epochs=epochs, lam=lam, buffer_size=24, seed=seed, verbose=False)
    names = [t.name for t in tasks]
    hist = {n: [] for n in names}
    # history[0] is the pre-training snapshot; step k (1-based) is epoch k. Total T*epochs steps.
    for k in range(1, len(tasks) * epochs + 1):
        snap = res["history"][k]["nmse"]
        for j, n in enumerate(names):
            hist[n].append(float(snap[n]) if k > j * epochs else None)  # only from task onset
    return hist


def _isolated_traj(method: str, tasks, *, layers, seq_len, lr, epochs, alpha, seed):
    """Per-epoch test NMSE for an OI-QCL variant, recording each task from its own onset."""
    probs_qnode, cs = make_probs_forecaster(n_qubits=4, n_layers=layers, seq_len=seq_len)
    n_probs = 2 ** 4
    names = [t.name for t in tasks]
    hist = {n: [] for n in names}
    rng = np.random.default_rng(seed)
    circ_w = pnp.array(0.1 * rng.standard_normal(cs), requires_grad=True)
    frozen_heads: dict[int, tuple[np.ndarray, float]] = {}   # past task -> (W, b)

    for phase, task in enumerate(tasks):
        hrng = np.random.default_rng(seed + phase)
        W = pnp.array(0.01 * hrng.standard_normal(n_probs), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
        train_theta = (phase == 0) or (method != "frozen_head")
        use_alpha = alpha if (phase > 0 and method == "anchor_head") else 0.0
        anchor = np.asarray(circ_w) if use_alpha > 0.0 else None
        circ_w = pnp.array(np.asarray(circ_w), requires_grad=bool(train_theta))
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)
        opt = qml.AdamOptimizer(lr)

        def cost_full(circ_w, W, b, Xtr=Xtr, ytr=ytr, use_alpha=use_alpha, anchor=anchor):
            pred = _probs_batched(probs_qnode, circ_w, Xtr) @ W + b
            loss = pnp.mean((pred - ytr) ** 2)
            if use_alpha > 0.0 and anchor is not None:
                loss = loss + use_alpha * pnp.sum((circ_w - anchor) ** 2)
            return loss

        def cost_headonly(W, b, circ_w=circ_w, Xtr=Xtr, ytr=ytr):
            pred = _probs_batched(probs_qnode, circ_w, Xtr) @ W + b
            return pnp.mean((pred - ytr) ** 2)

        for _ in range(epochs):
            if train_theta:
                circ_w, W, b = opt.step(cost_full, circ_w, W, b)
            else:
                W, b = opt.step(cost_headonly, W, b)
            # Snapshot every task's test NMSE at the current backbone.
            for j, tj in enumerate(tasks):
                if j > phase:
                    hist[tj.name].append(None)                       # not started yet
                elif j == phase:
                    P_te = probs_features(probs_qnode, circ_w, tj.X_test)
                    hist[tj.name].append(_nmse_wb(P_te, np.asarray(W), float(b), tj.y_test))
                else:
                    Wj, bj = frozen_heads[j]                          # frozen head, drifting backbone
                    P_te = probs_features(probs_qnode, circ_w, tj.X_test)
                    hist[tj.name].append(_nmse_wb(P_te, Wj, bj, tj.y_test))
        frozen_heads[phase] = (np.asarray(W, float).copy(), float(b))
    return hist


def run(*, tasks_names, layers=2, seq_len=8, lr=0.05, epochs=40, alpha=5.0, lam=5.0,
        seed=42, verbose=True) -> dict[str, Any]:
    tasks = load_task_sequence(tasks_names, seq_len=seq_len)
    if verbose:
        print(f"e015 trajectory seed={seed}: {tasks_names} ({len(tasks)}x{epochs} epochs)", flush=True)
    started = time.perf_counter()
    methods: dict[str, Any] = {}
    for m in SHARED_METHODS:
        if verbose:
            print(f"  {m}...", flush=True)
        methods[m] = _shared_traj(m, tasks, layers=layers, seq_len=seq_len, lr=lr, epochs=epochs,
                                  lam=lam, seed=seed)
    for m in ISOLATED_METHODS:
        if verbose:
            print(f"  {m}...", flush=True)
        methods[m] = _isolated_traj(m, tasks, layers=layers, seq_len=seq_len, lr=lr, epochs=epochs,
                                    alpha=alpha, seed=seed)
    return {
        "experiment": "e015_trajectory",
        "config": {"tasks": list(tasks_names), "seq_len": seq_len, "layers": layers, "lr": lr,
                   "epochs_per_task": epochs, "alpha_l2_anchor": alpha, "lambda_qewc": lam,
                   "seed": seed, "n_qubits": 4},
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
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(tasks_names=args.tasks, layers=args.layers, seq_len=args.seq_len, lr=args.lr,
                 epochs=args.epochs_per_task, alpha=args.alpha, lam=args.lam, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / f"e015_trajectory_seed{args.seed}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
