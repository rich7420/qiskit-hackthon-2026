"""E014 real-dataset screen: which real phase-classification datasets does classical lose on?

For each real spin-model phase task (cluster-Ising and transverse-field Ising ground states,
deep-in-phase and near-critical), we compare three classifiers:
  * classical_full  -- logistic head on the FULL 2^n amplitude vector (sees everything)
  * classical_local -- logistic head on a few LOCAL observables (<Z_i>, <X_i>, <Z_iZ_{i+1}>),
                       the literature protocol where nonlocal (string) order genuinely matters
  * quantum         -- a trained VQC (L4) on the amplitude-embedded state

Reading: classical loses only where classical_local is low but quantum (and full) are high --
i.e. the discriminating feature is NONLOCAL, invisible to a local classifier but reachable by a
circuit that reconstructs nonlocal correlations. A linear head on the full amplitudes is not a
"local" classifier, so it usually still wins -- which is exactly why the topological-ML
literature restricts inputs to local observables.
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
from src.phase_data import _ground_state as _cluster_ground_state  # noqa: E402

N = 4
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _op(single: dict[int, np.ndarray]) -> np.ndarray:
    m = np.array([[1]], dtype=complex)
    for q in range(N):
        m = np.kron(m, single.get(q, _I))
    return m


# Local observable operators (real symmetric): <Z_i>, <X_i>, <Z_i Z_{i+1}>.
_LOCAL_OPS = ([_op({i: _Z}).real for i in range(N)]
              + [_op({i: _X}).real for i in range(N)]
              + [_op({i: _Z, i + 1: _Z}).real for i in range(N - 1)])


def _local_features(states: np.ndarray) -> np.ndarray:
    return np.array([[float(s @ O @ s) for O in _LOCAL_OPS] for s in states])


def _tfim_ground_state(h: float) -> np.ndarray:
    """Transverse-field Ising, open chain: H = -sum Z_i Z_{i+1} - h sum X_i."""
    H = np.zeros((2**N, 2**N))
    for i in range(N - 1):
        H -= _op({i: _Z, i + 1: _Z}).real
    for i in range(N):
        H -= h * _op({i: _X}).real
    w, v = np.linalg.eigh(H)
    s = np.real(v[:, 0])
    s /= np.linalg.norm(s)
    if s[int(np.argmax(np.abs(s)))] < 0:
        s = -s
    return s


def _states(gen, lo, hi, n, rng):
    return np.array([gen(float(h)) for h in rng.uniform(lo, hi, size=n)])


TASKS = {
    "cluster_deep": (_cluster_ground_state, (0.0, 0.5), (2.5, 3.0)),
    "cluster_near": (_cluster_ground_state, (0.7, 0.95), (1.05, 1.3)),
    "cluster_full": (_cluster_ground_state, (0.0, 0.95), (1.05, 3.0)),  # full diagram, both sides of h_c~1
    "tfim_deep": (_tfim_ground_state, (0.1, 0.5), (1.6, 2.4)),
    "tfim_near": (_tfim_ground_state, (0.8, 0.95), (1.05, 1.2)),
}


def _make(name, n_train, n_test, seed):
    gen, (a, b), (c, d) = TASKS[name]
    rng = np.random.default_rng(seed)
    ntr, nte = n_train // 2, n_test // 2
    Xtr = np.vstack([_states(gen, a, b, ntr, rng), _states(gen, c, d, ntr, rng)])
    Xte = np.vstack([_states(gen, a, b, nte, rng), _states(gen, c, d, nte, rng)])
    ytr = np.array([1] * ntr + [-1] * ntr)
    yte = np.array([1] * nte + [-1] * nte)
    ptr, pte = rng.permutation(len(Xtr)), rng.permutation(len(Xte))
    return Xtr[ptr], ytr[ptr], Xte[pte], yte[pte]


def _classical(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000).fit(Xtr, (ytr + 1) // 2)
    return float(np.mean(clf.predict(Xte) == (yte + 1) // 2))


def _quantum(Xtr, ytr, Xte, yte, layers, epochs, seed):
    qnode, shape = make_softmax_qnode(n_qubits=N, n_layers=layers)
    w = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    Xp, yp = pnp.array(Xtr, requires_grad=False), pnp.array(ytr, requires_grad=False)
    for _ in range(epochs):
        w = opt.step(lambda W: bce_loss(qnode, W, Xp, yp), w)
    return float(softmax_accuracy(qnode, w, Xte, yte))


def run(*, n_train=400, n_test=200, layers=4, epochs=25, seed=42) -> dict[str, Any]:
    started = time.perf_counter()
    out: dict[str, Any] = {}
    for name in TASKS:
        Xtr, ytr, Xte, yte = _make(name, n_train, n_test, seed)
        cf = _classical(Xtr, ytr, Xte, yte)
        cl = _classical(_local_features(Xtr), ytr, _local_features(Xte), yte)
        q = _quantum(Xtr, ytr, Xte, yte, layers, epochs, seed)
        loses = bool(cl < 0.65 <= q)  # classical LOCAL fails but quantum succeeds
        out[name] = {"classical_full": round(cf, 4), "classical_local": round(cl, 4),
                     "quantum_L%d" % layers: round(q, 4), "classical_local_loses": loses}
        print(f"  {name:13s} classical_full={cf:.3f}  classical_local={cl:.3f}  "
              f"quantum_L{layers}={q:.3f}  {'<-- local loses' if loses else ''}", flush=True)
    return {
        "experiment": "e014_real_dataset_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"n_train": n_train, "n_test": n_test, "layers": layers, "epochs": epochs,
                   "seed": seed, "n_qubits": N,
                   "local_observables": "<Z_i>,<X_i>,<Z_iZ_{i+1}>",
                   "note": "classical_full sees the whole state; classical_local is the literature protocol"},
        "tasks": out, "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(n_train=args.n_train, n_test=args.n_test, layers=args.layers,
                 epochs=args.epochs, seed=args.seed)
    (ROOT / "results").mkdir(exist_ok=True)
    out = args.output or ROOT / "results/e014_real_dataset_screen.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
