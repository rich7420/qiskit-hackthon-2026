"""E014 readout-width ablation: how SMALL can the per-task observable be?

The concern: a full linear head over 2^n probabilities (34 params/task) is so expressive that
the shared circuit becomes vestigial -- effectively "one linear model per task". Here we shrink
the readout to m<=n qubits (a diagonal observable on 2^m outcomes = C*2^m + C params) while
KEEPING AmplitudeEmbedding and the shared frozen backbone fixed, and watch how Task-IL accuracy
falls -- i.e. how much task information a small, genuinely-lightweight observable can still read
off the SHARED representation. If a small readout still beats QEWC, the shared circuit is doing
real work; if it collapses, the full-probs head was carrying the method (motivating a soft-
adapting circuit, Variant C, to make a small observable sufficient).
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

from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import (  # noqa: E402
    fit_linear_head,
    make_probs_qnode,
    probs_features,
    train_backbone,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")
QEWC_ACC = 0.819


def _source_digest() -> str:
    digest = hashlib.sha256()
    for p in (Path(__file__), ROOT / "src/e014_oiqcl.py"):
        digest.update(p.read_bytes())
    return digest.hexdigest()


def _marginal(P: np.ndarray, n: int, m: int) -> np.ndarray:
    """Marginalize full 2^n probs to the first m qubits (2^m outcomes).

    qubit 0 is the most-significant bit in PennyLane's ordering, so the first m qubits are
    selected by the top m bits of the outcome index.
    """
    if m >= n:
        return P
    idx = np.arange(P.shape[1])
    grp = idx >> (n - m)
    return np.stack([P[:, grp == b].sum(axis=1) for b in range(2**m)], axis=1)


def run(*, layers=12, lr=0.05, epochs=20, n_train=800, n_test=200,
        widths=(1, 2, 3, 4), seeds=(42, 43, 44), verbose=True) -> dict[str, Any]:
    started = time.perf_counter()
    probs_qnode, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers)
    acc_by_m: dict[int, list[float]] = {m: [] for m in widths}

    for seed in seeds:
        t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
        t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
        tasks = [t1, t2, t3]
        # Frozen backbone: train theta on Task 1 only, then freeze (Variant A).
        weights, _, _ = train_backbone(t1, n_qubits=N_QUBITS, n_layers=layers,
                                       lr=lr, epochs=epochs, seed=seed)
        full = {"tr": [probs_features(probs_qnode, weights, t.X_train) for t in tasks],
                "te": [probs_features(probs_qnode, weights, t.X_test) for t in tasks]}
        for m in widths:
            accs = []
            for j, t in enumerate(tasks):
                Ptr = _marginal(full["tr"][j], N_QUBITS, m)
                Pte = _marginal(full["te"][j], N_QUBITS, m)
                head = fit_linear_head(Ptr, t.y_train, t.name, seed=seed)
                accs.append(head.accuracy(Pte, t.y_test))
            acc_by_m[m].append(float(np.mean(accs)))

    per_width = {}
    for m in widths:
        a = np.asarray(acc_by_m[m])
        per_width[str(m)] = {
            "readout_qubits": m, "readout_dim": 2**m,
            "head_params_per_task": 2 * (2**m) + 2,  # C*2^m + C, C=2
            "ACC_mean": round(float(a.mean()), 4),
            "ACC_sd": round(float(a.std(ddof=1)), 4),
            "beats_qewc": bool(a.mean() > QEWC_ACC),
        }
        if verbose:
            r = per_width[str(m)]
            print(f"  m={m} ({r['readout_dim']:2d}-dim, {r['head_params_per_task']:2d} params/task): "
                  f"ACC={r['ACC_mean']:.3f} ± {r['ACC_sd']:.3f}  "
                  f"{'> QEWC' if r['beats_qewc'] else '< QEWC'}", flush=True)

    return {
        "schema_version": SCHEMA_VERSION, "experiment": "e014_readout_ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"variant": "frozen theta_1 (Variant A), AmplitudeEmbedding kept",
                   "layers": layers, "n_qubits": N_QUBITS, "seeds": list(seeds),
                   "epochs_task1": epochs, "n_train": n_train, "n_test": n_test,
                   "qewc_reference": QEWC_ACC},
        "per_width": per_width,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(layers=args.layers, epochs=args.epochs, n_train=args.n_train,
                 n_test=args.n_test, widths=tuple(args.widths), seeds=tuple(args.seeds))
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / "e014_readout_ablation.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
