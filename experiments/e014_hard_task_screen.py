"""E014 hard-task screening: find a T3 replacement where the quantum circuit is NECESSARY.

We want a task with a genuine gap: a matched classical LINEAR head on the raw 2^n amplitude
input fails, while a trained VQC (which builds quadratic/nonlocal features of the amplitudes)
succeeds -- and where DEPTH matters (shallow fails, deeper works). For each candidate task we
report: classical linear accuracy, and quantum accuracy at shallow (L=2) and deeper (L=8) VQCs.

Candidates:
  cluster_deep  -- current SPT/ATF: cluster-Ising ground states sampled DEEP in each phase
                   (h far apart). Baseline; expected classically easy.
  cluster_near  -- same model sampled NEAR the transition: local features overlap, only the
                   nonlocal string order separates the phases.
  parity        -- random real states labelled by sign<x|ZZZZ|x> (a global stabilizer).
                   Quadratic/nonlocal in amplitudes -> a linear head cannot compute it.
  string        -- random real states labelled by sign<x|X Z Z X|x> (a nonlocal string operator).
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e005_softmax import accuracy as softmax_accuracy  # noqa: E402
from src.e005_softmax import bce_loss, make_softmax_qnode  # noqa: E402
from src.phase_data import _ground_state  # noqa: E402

RESULTS = ROOT / "results"
N_QUBITS = 4
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(ops):
    m = np.array([[1]], dtype=complex)
    for o in ops:
        m = np.kron(m, o)
    return m


PARITY_OP = _kron([_Z, _Z, _Z, _Z]).real
STRING_OP = _kron([_X, _Z, _Z, _X]).real


def _balanced_states_by_observable(op, n_per_class, rng):
    """Random real 4-qubit states labelled by sign of <x|op|x>, balanced classes."""
    pos, neg = [], []
    while len(pos) < n_per_class or len(neg) < n_per_class:
        x = rng.standard_normal(2**N_QUBITS)
        x /= np.linalg.norm(x)
        val = float(x @ op @ x)
        if val > 0.05 and len(pos) < n_per_class:
            pos.append(x)
        elif val < -0.05 and len(neg) < n_per_class:
            neg.append(x)
    X = np.array(pos + neg)
    y = np.array([1] * n_per_class + [-1] * n_per_class)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _cluster_states(h_lo, h_hi, n, rng):
    hs = rng.uniform(h_lo, h_hi, size=n)
    return np.array([_ground_state(float(h)) for h in hs])


def _make_task(name, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    ntr, nte = n_train // 2, n_test // 2
    if name in ("cluster_deep", "cluster_near"):
        (a, b, c, d) = ((0.0, 0.5, 2.5, 3.0) if name == "cluster_deep"
                        else (0.7, 0.95, 1.05, 1.3))
        Xtr = np.vstack([_cluster_states(a, b, ntr, rng), _cluster_states(c, d, ntr, rng)])
        Xte = np.vstack([_cluster_states(a, b, nte, rng), _cluster_states(c, d, nte, rng)])
        ytr = np.array([1] * ntr + [-1] * ntr)
        yte = np.array([1] * nte + [-1] * nte)
        ptr, pte = rng.permutation(len(Xtr)), rng.permutation(len(Xte))
        return Xtr[ptr], ytr[ptr], Xte[pte], yte[pte]
    op = PARITY_OP if name == "parity" else STRING_OP
    Xtr, ytr = _balanced_states_by_observable(op, ntr, rng)
    Xte, yte = _balanced_states_by_observable(op, nte, rng)
    return Xtr, ytr, Xte, yte


def _train_vqc(Xtr, ytr, Xte, yte, layers, lr, epochs, seed):
    qnode, shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    w = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    Xp, yp = pnp.array(Xtr, requires_grad=False), pnp.array(ytr, requires_grad=False)
    for _ in range(epochs):
        w = opt.step(lambda W: bce_loss(qnode, W, Xp, yp), w)
    return float(softmax_accuracy(qnode, w, Xte, yte))


def run(*, tasks=("cluster_deep", "cluster_near", "parity", "string"),
        depths=(2, 8), n_train=600, n_test=200, lr=0.05, epochs=25, seed=42) -> dict[str, Any]:
    started = time.perf_counter()
    out: dict[str, Any] = {}
    for name in tasks:
        Xtr, ytr, Xte, yte = _make_task(name, n_train, n_test, seed)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, (ytr + 1) // 2)
        classical = float(np.mean(clf.predict(Xte) == (yte + 1) // 2))
        quantum = {f"L{L}": round(_train_vqc(Xtr, ytr, Xte, yte, L, lr, epochs, seed), 4)
                   for L in depths}
        gap = round(max(quantum.values()) - classical, 4)
        out[name] = {"classical_linear": round(classical, 4), "quantum": quantum, "gap": gap}
        print(f"  {name:13s} classical={classical:.3f}  "
              + "  ".join(f"q{k}={v:.3f}" for k, v in quantum.items())
              + f"  gap={gap:+.3f}", flush=True)
    return {
        "experiment": "e014_hard_task_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"depths": list(depths), "n_train": n_train, "n_test": n_test,
                   "lr": lr, "epochs": epochs, "seed": seed, "n_qubits": N_QUBITS,
                   "note": "classical = logistic head on raw 2^n amplitudes; quantum = trained VQC"},
        "tasks": out, "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 8])
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(depths=tuple(args.depths), n_train=args.n_train, n_test=args.n_test,
                 epochs=args.epochs, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / "e014_hard_task_screen.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
