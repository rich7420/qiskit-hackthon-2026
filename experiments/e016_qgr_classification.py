"""E016 - Quantum Generative Replay ported to the e005 CLASSIFICATION benchmark.

Ports the QGR idea from the e009 forecasting study (PR #13) onto the three-task image /
phase classification sequence (MNIST 0/1 -> Fashion-MNIST 0/1 -> SPT/ATF), the same benchmark
as e005 (PR #5). The point of QGR is: the frozen quantum model IS the memory of an old task,
and we rehearse on data GENERATED from it instead of storing raw samples. Classification has no
temporal axis to autoregressively roll out, so we realize "generation" two faithful ways:

  qgr_seed      : keep a few real seed feature vectors per old task + a frozen classifier
                  snapshot; synthesize new vectors by convex-mixing seeds + noise (renormalized
                  to valid amplitude states) and PSEUDO-LABEL them with the frozen snapshot.
                  Mirrors forecasting-QGR, which keeps a few real seed windows and generates
                  from a frozen model.
  qgr_inversion : fully data-free. For each old class, start from a random amplitude vector and
                  gradient-ascend the frozen classifier's confidence for that class (renormalized
                  each step) to synthesize class prototypes. No raw sample is ever stored --
                  the circuit params are the only memory (closest to the QGR headline).

Compared against the e005 methods on the identical learner / data / schedule:
  baseline : sequential fine-tuning, no protection (forgetting reference)
  ewc      : classical-Fisher (CFI) elastic weight consolidation
  qewc     : quantum-Fisher (QFI) elastic weight consolidation
  replay   : store a small balanced buffer of REAL old-task samples (raw-data upper bound)

Higher test accuracy is better (opposite sign to the forecasting NMSE). Retention = mean final
accuracy on the two earlier tasks; plasticity = final accuracy on the last task.

Run:
    python experiments/e016_qgr_classification.py --seed 42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from importlib.metadata import version
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
from src.qnn_pennylane import make_qnode  # noqa: E402 (single-Z, QFI metric only)
from src.e005_softmax import (  # noqa: E402
    accuracy as softmax_accuracy,
    bce_loss,
    classical_fisher_diag,
    make_softmax_qnode,
    scores,
)

RESULTS = ROOT / "results"
TASK_KEYS = ("task1", "task2", "task3")
METHODS = ("baseline", "ewc", "qewc", "replay", "qgr_seed", "qgr_inversion")
REG_METHODS = ("ewc", "qewc")
REPLAY_METHODS = ("replay", "qgr_seed", "qgr_inversion")


# --------------------------------------------------------------------------- generation

def _pseudo_label(clf_qnode, theta_s, X) -> np.ndarray:
    """Label synthetic vectors with a frozen classifier snapshot: argmax score -> {-1,+1}."""
    logits = np.asarray(scores(clf_qnode, pnp.array(theta_s, requires_grad=False),
                               pnp.array(X, requires_grad=False)))
    if logits.ndim == 1:  # single sample -> (2,)
        logits = logits[None, :]
    pred = np.argmax(logits, axis=-1)  # {0,1}
    return np.where(pred == 0, -1, 1)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


def generate_seed_based(clf_qnode, theta_s, seeds_X, n_gen, rng, noise=0.05):
    """Convex-mix stored seed vectors + Gaussian noise, renormalize, pseudo-label with theta_s."""
    m = len(seeds_X)
    i = rng.integers(0, m, size=n_gen)
    j = rng.integers(0, m, size=n_gen)
    t = rng.uniform(size=(n_gen, 1))
    X = t * seeds_X[i] + (1.0 - t) * seeds_X[j]
    X = X + noise * rng.standard_normal(X.shape)
    X = _normalize_rows(X)
    y = _pseudo_label(clf_qnode, theta_s, X)
    return X, y


def generate_inversion(clf_qnode, theta_s, n_per_class, rng, dim, steps=30, lr=0.2):
    """Data-free: gradient-ascend a random amplitude vector toward each frozen-class prototype."""
    theta_c = pnp.array(theta_s, requires_grad=False)

    def neg_logprob(x, c):
        s = scores(clf_qnode, theta_c, x)  # (2,)
        return -(s[c] - pnp.log(pnp.sum(pnp.exp(s))))

    grad = qml.grad(neg_logprob, argnums=0)
    Xs, ys = [], []
    for c in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(dim)
            x = pnp.array(x / np.linalg.norm(x), requires_grad=True)
            for _ in range(steps):
                g = np.asarray(grad(x, c))
                x = pnp.array(np.asarray(x) - lr * g, requires_grad=True)  # descend -logprob
                x = pnp.array(np.asarray(x) / (np.linalg.norm(np.asarray(x)) + 1e-12),
                              requires_grad=True)
            Xs.append(np.asarray(x))
            ys.append(-1 if c == 0 else 1)
    return np.asarray(Xs), np.asarray(ys)


# --------------------------------------------------------------------------- training

def _build_replay_set(method, phase, generators, replay_buf, clf_qnode, rng, *, n_gen, dim,
                      inv_steps, inv_lr, noise):
    """Assemble (Rx, Ry) from every stored old-task generator/buffer for this phase."""
    if phase <= 1:
        return None, None
    if method == "replay":
        if not replay_buf:
            return None, None
        Rx = np.concatenate([b[0] for b in replay_buf])
        Ry = np.concatenate([b[1] for b in replay_buf])
    elif method == "qgr_seed":
        gx, gy = [], []
        for theta_s, seeds_X in generators:
            X, y = generate_seed_based(clf_qnode, theta_s, seeds_X, n_gen, rng, noise=noise)
            gx.append(X)
            gy.append(y)
        Rx, Ry = np.concatenate(gx), np.concatenate(gy)
    else:  # qgr_inversion
        gx, gy = [], []
        for (theta_s,) in generators:
            X, y = generate_inversion(clf_qnode, theta_s, max(1, n_gen // 2), rng, dim,
                                      steps=inv_steps, lr=inv_lr)
            gx.append(X)
            gy.append(y)
        Rx, Ry = np.concatenate(gx), np.concatenate(gy)
    return (pnp.array(Rx, requires_grad=False), pnp.array(Ry, requires_grad=False))


def _consolidate(method, phase, tasks, weights, reg, generators, replay_buf, clf_qnode,
                 qfi_qnode, rng, *, qfi_samples, seed, n_seeds, buffer_size):
    """After finishing a task (phase < n), stash whatever this method needs to protect it."""
    task = tasks[phase - 1]
    if method in REG_METHODS:
        if method == "ewc":
            fisher = classical_fisher_diag(clf_qnode, weights, task.X_train, task.y_train)
        else:
            fisher = quantum_fisher_diag(qfi_qnode, weights, task.X_train,
                                         n_samples=qfi_samples, seed=seed)
        reg.consolidate(np.asarray(weights).flatten(), fisher)
    elif method == "replay":
        idx = rng.choice(len(task.X_train), size=min(buffer_size, len(task.X_train)),
                         replace=False)
        replay_buf.append((task.X_train[idx].copy(), task.y_train[idx].copy()))
    elif method == "qgr_seed":
        # Keep a balanced handful of REAL seed vectors + a frozen classifier snapshot.
        y = task.y_train
        seeds = []
        for lab in (-1, 1):
            pool = np.where(y == lab)[0]
            take = rng.choice(pool, size=min(n_seeds, len(pool)), replace=False)
            seeds.extend(take.tolist())
        generators.append((np.asarray(weights).copy(), task.X_train[np.array(seeds)].copy()))
    elif method == "qgr_inversion":
        generators.append((np.asarray(weights).copy(),))  # data-free: params only


def train_method(method, tasks, *, clf_qnode, qfi_qnode, weight_shape, lam_ewc, lam_qewc,
                 replay_weight, lr, epochs_per_task, qfi_samples, buffer_size, n_seeds,
                 n_gen, inv_steps, inv_lr, noise, seed, verbose):
    lam = {"ewc": lam_ewc, "qewc": lam_qewc}.get(method, 0.0)
    reg = EWC(lam)
    weights = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
                        requires_grad=True)
    optimizer = qml.AdamOptimizer(lr)
    rng = np.random.default_rng(seed + 1000)  # generation/buffer sampling stream
    dim = int(np.asarray(tasks[0].X_train).shape[1])
    generators: list[tuple] = []
    replay_buf: list[tuple] = []
    history: list[dict[str, Any]] = []

    def snapshot(epoch, phase):
        history.append({"epoch": epoch, "phase": phase,
                        "test_accuracy": {k: softmax_accuracy(clf_qnode, weights, t.X_test, t.y_test)
                                          for k, t in zip(TASK_KEYS, tasks)}})

    snapshot(0, 0)
    for phase, task in enumerate(tasks, start=1):
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)
        Rx, Ry = (None, None)
        if method in REPLAY_METHODS:
            Rx, Ry = _build_replay_set(method, phase, generators, replay_buf, clf_qnode, rng,
                                       n_gen=n_gen, dim=dim, inv_steps=inv_steps, inv_lr=inv_lr,
                                       noise=noise)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase, Rx=Rx, Ry=Ry):
            loss = bce_loss(clf_qnode, W, Xtr, ytr)
            if method in REG_METHODS:
                loss = loss + reg.penalty(W.flatten(), phase)
            if Rx is not None:
                loss = loss + replay_weight * bce_loss(clf_qnode, W, Rx, Ry)
            return loss

        for _ in range(epochs_per_task):
            weights = optimizer.step(cost, weights)
            snapshot(history[-1]["epoch"] + 1, phase)
        if verbose:
            acc = history[-1]["test_accuracy"]
            print(f"    [{method:13s}] after phase {phase} ({task.name}): "
                  + " ".join(f"{k}={acc[k]:.3f}" for k in TASK_KEYS), flush=True)

        if phase < len(tasks):
            _consolidate(method, phase, tasks, weights, reg, generators, replay_buf, clf_qnode,
                         qfi_qnode, rng, qfi_samples=qfi_samples, seed=seed, n_seeds=n_seeds,
                         buffer_size=buffer_size)
    return history


# --------------------------------------------------------------------------- experiment

def run_experiment(*, layers=20, learning_rate=0.02, epochs_per_task=20, lam_ewc=30.0,
                   lam_qewc=0.8, replay_weight=1.0, qfi_samples=64, n_train=800, n_test=200,
                   buffer_size=48, n_seeds=16, n_gen=48, inv_steps=30, inv_lr=0.2, noise=0.05,
                   seed=42, methods=METHODS, verbose=True):
    task1, task2 = load_two_tasks(n_features=2 ** N_QUBITS, n_train=n_train, n_test=n_test,
                                  seed=seed)
    task3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [task1, task2, task3]
    boundaries = [epochs_per_task, 2 * epochs_per_task]
    clf_qnode, weight_shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    qfi_qnode, _ = make_qnode(n_qubits=N_QUBITS, n_layers=layers)

    if verbose:
        print(f"E016 seed={seed}: {[t.name for t in tasks]}  methods={list(methods)}")

    started = time.perf_counter()
    histories: dict[str, list] = {}
    for method in methods:
        t0 = time.perf_counter()
        histories[method] = train_method(
            method, tasks, clf_qnode=clf_qnode, qfi_qnode=qfi_qnode, weight_shape=weight_shape,
            lam_ewc=lam_ewc, lam_qewc=lam_qewc, replay_weight=replay_weight, lr=learning_rate,
            epochs_per_task=epochs_per_task, qfi_samples=qfi_samples, buffer_size=buffer_size,
            n_seeds=n_seeds, n_gen=n_gen, inv_steps=inv_steps, inv_lr=inv_lr, noise=noise,
            seed=seed, verbose=verbose)
        if verbose:
            print(f"    ({method} done in {time.perf_counter()-t0:.0f}s)", flush=True)
    train_time = time.perf_counter() - started

    metrics: dict[str, Any] = {}
    for method, hist in histories.items():
        per_task = {}
        for idx, key in enumerate(TASK_KEYS, start=1):
            phase_end = hist[idx * epochs_per_task]["test_accuracy"][key]
            final = hist[-1]["test_accuracy"][key]
            per_task[key] = {"name": tasks[idx - 1].name,
                             "test_at_phase_end": round(phase_end, 4),
                             "test_final": round(final, 4),
                             "forgetting": round(phase_end - final, 4)}
        retention = float(np.mean([per_task[k]["test_final"] for k in ("task1", "task2")]))
        forgetting = float(np.mean([per_task[k]["forgetting"] for k in ("task1", "task2")]))
        metrics[method] = {
            "tasks": per_task,
            "retention_earlier_acc": round(retention, 4),      # higher = better retained
            "plasticity_final_acc": round(per_task["task3"]["test_final"], 4),
            "avg_earlier_forgetting": round(forgetting, 4),    # lower = better
            "avg_final_acc": round(float(np.mean([per_task[k]["test_final"] for k in TASK_KEYS])), 4),
        }

    return {
        "experiment": "e016_qgr_classification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {
            "tasks": [t.name for t in tasks], "methods": list(methods),
            "n_qubits": N_QUBITS, "layers": layers, "n_weights": int(np.prod(weight_shape)),
            "optimizer": "Adam", "learning_rate": learning_rate,
            "epochs_per_task": epochs_per_task, "task_boundaries": boundaries,
            "lambda_ewc": lam_ewc, "lambda_qewc": lam_qewc, "qfi_samples": qfi_samples,
            "replay_weight": replay_weight, "buffer_size": buffer_size,
            "qgr_seed_n_seeds_per_class": n_seeds, "qgr_n_generated_per_task": n_gen,
            "qgr_inversion_steps": inv_steps, "qgr_inversion_lr": inv_lr, "qgr_seed_noise": noise,
            "n_train_per_task": n_train, "n_test_per_task": n_test, "seed": seed,
        },
        "histories": histories,
        "metrics": metrics,
        "train_time_sec": round(train_time, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--lam-ewc", type=float, default=30.0)
    ap.add_argument("--lam-qewc", type=float, default=0.8)
    ap.add_argument("--replay-weight", type=float, default=1.0)
    ap.add_argument("--qfi-samples", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--buffer-size", type=int, default=48)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--n-gen", type=int, default=48)
    ap.add_argument("--inv-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = run_experiment(
        layers=args.layers, learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task, lam_ewc=args.lam_ewc, lam_qewc=args.lam_qewc,
        replay_weight=args.replay_weight, qfi_samples=args.qfi_samples, n_train=args.n_train,
        n_test=args.n_test, buffer_size=args.buffer_size, n_seeds=args.n_seeds, n_gen=args.n_gen,
        inv_steps=args.inv_steps, seed=args.seed, methods=args.methods)

    print("\n=== E016 QGR-on-classification (test accuracy; higher = better) ===")
    print(f"  {'method':14s} {'retention(T1,T2)':>16s} {'plasticity(T3)':>15s} "
          f"{'forgetting':>11s} {'avg':>6s}")
    for m in args.methods:
        mt = result["metrics"][m]
        print(f"  {m:14s} {mt['retention_earlier_acc']:>16.3f} {mt['plasticity_final_acc']:>15.3f} "
              f"{mt['avg_earlier_forgetting']:>+11.3f} {mt['avg_final_acc']:>6.3f}")

    RESULTS.mkdir(exist_ok=True)
    out = args.output if args.output else RESULTS / f"e016_qgr_classification_seed{args.seed}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
