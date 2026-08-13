"""E014 the quantum-necessity punchline: classical baseline with LOCAL-observable input on T3.

Continual sequence MNIST -> Fashion -> cluster-Ising. The classical multi-head control sees:
  * T1, T2 (classical image data): the raw amplitude features (full) -- fair, it's classical data.
  * T3 (cluster-Ising quantum data): only LOCAL observables (<X_i>,<Z_i>,<X_iX>,<Y_iY>,<Z_iZ>),
    the hardware-honest protocol (you cannot classically tomograph the full 2^n state; a real
    device only yields local measurements, and full-state readout costs exponentially many shots).

Near the SPT transition the phases have overlapping local observables and differ only in nonlocal
string order, so the local classical head fails on T3, dragging its continual average below OI-QCL
(which amplitude-embeds the state and lets the circuit reconstruct the nonlocal order). This is the
"quantum matters" comparison: OI-QCL beats the classical control BECAUSE of the quantum task.
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import _labels_to_classes  # noqa: E402
from src.phase_data import N_QUBITS, load_cluster_full  # noqa: E402

N = N_QUBITS
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _op(single):
    m = np.array([[1]], dtype=complex)
    for q in range(N):
        m = np.kron(m, single.get(q, _I))
    return m


# Local 1- and 2-body observables (real-valued on real ground states):
# <X_i>, <Z_i>, and nearest-neighbour <X_iX_{i+1}>, <Y_iY_{i+1}>, <Z_iZ_{i+1}>.
_LOCAL_OPS = (
    [_op({i: _X}) for i in range(N)]
    + [_op({i: _Z}) for i in range(N)]
    + [_op({i: _X, i + 1: _X}) for i in range(N - 1)]
    + [_op({i: _Y, i + 1: _Y}) for i in range(N - 1)]
    + [_op({i: _Z, i + 1: _Z}) for i in range(N - 1)]
)
_LOCAL_OPS = [np.real_if_close(O) for O in _LOCAL_OPS]


def _local_features(states):
    return np.array([[float(np.real(s @ O @ s)) for O in _LOCAL_OPS] for s in states])


def _acc(Xtr, ytr, Xte, yte, seed):
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, _labels_to_classes(ytr))
    return float(np.mean(clf.predict(Xte) == _labels_to_classes(yte)))


def run(*, n_train=800, n_test=200, seeds=(42, 43, 44), verbose=True) -> dict[str, Any]:
    started = time.perf_counter()
    rows = {"t1": [], "t2": [], "t3_amplitude": [], "t3_local": []}
    for seed in seeds:
        t1, t2 = load_two_tasks(n_features=2**N, n_train=n_train, n_test=n_test, seed=seed)
        t3 = load_cluster_full(n_train=n_train, n_test=n_test, n_qubits=N, seed=seed)
        rows["t1"].append(_acc(t1.X_train, t1.y_train, t1.X_test, t1.y_test, seed))
        rows["t2"].append(_acc(t2.X_train, t2.y_train, t2.X_test, t2.y_test, seed))
        rows["t3_amplitude"].append(_acc(t3.X_train, t3.y_train, t3.X_test, t3.y_test, seed))
        rows["t3_local"].append(_acc(_local_features(t3.X_train), t3.y_train,
                                     _local_features(t3.X_test), t3.y_test, seed))
        if verbose:
            print(f"  seed {seed}: T1={rows['t1'][-1]:.3f} T2={rows['t2'][-1]:.3f} "
                  f"T3(amp)={rows['t3_amplitude'][-1]:.3f} T3(local)={rows['t3_local'][-1]:.3f}",
                  flush=True)

    def ms(v):
        a = np.asarray(v)
        return round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)

    per = {k: ms(v) for k, v in rows.items()}
    avg_amp = ms([(a + b + c) / 3 for a, b, c in zip(rows["t1"], rows["t2"], rows["t3_amplitude"])])
    avg_local = ms([(a + b + c) / 3 for a, b, c in zip(rows["t1"], rows["t2"], rows["t3_local"])])
    return {
        "experiment": "e014_classical_local_t3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("numpy", "scikit-learn")}},
        "config": {"tasks": ["MNIST 0/1", "Fashion-MNIST 0/1", "cluster-Ising"],
                   "T1_T2_input": "raw amplitude features", "seeds": list(seeds),
                   "T3_local_observables": "<X_i>,<Z_i>,<X_iX>,<Y_iY>,<Z_iZ> (17 features)",
                   "n_train": n_train, "n_test": n_test,
                   "oiqcl_reference_m2_cluster": {"frozen_A": 0.939, "anchor_C": 0.944}},
        "per_task_mean_sd": per,
        "classical_avg_ACC_amplitudeT3": {"mean": avg_amp[0], "sd": avg_amp[1]},
        "classical_avg_ACC_localT3": {"mean": avg_local[0], "sd": avg_local[1]},
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(n_train=args.n_train, n_test=args.n_test, seeds=tuple(args.seeds))
    (ROOT / "results").mkdir(exist_ok=True)
    out = args.output or ROOT / "results/e014_classical_local_t3.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nclassical avg (T3 amplitude) = {result['classical_avg_ACC_amplitudeT3']['mean']:.3f}")
    print(f"classical avg (T3 LOCAL)     = {result['classical_avg_ACC_localT3']['mean']:.3f}")
    print(f"OI-QCL (m2, cluster)         = 0.939 (A) / 0.944 (C)")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
