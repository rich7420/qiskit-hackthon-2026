"""E014 noisy readout: how much does gate+readout noise cost MPI?

Train the shared backbone noiselessly on Task 1 (Variant A), then read out the per-task
linear-head MPI classifier under two conditions -- noiseless (default.qubit) and noisy
(default.mixed, config {'bit':0,'phase':0,'depol':0.01,'meas':0.02}) -- fitting each task's
head on the probabilities actually observed. This is the "train on the simulator, deploy
under noise" comparison, reported for the full 2^n readout and the m=2 local readout (the
small readout applies the measurement error to fewer qubits, so it should be more robust).
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
from src.e014_noise import DEFAULT_NOISE, make_noisy_probs_qnode  # noqa: E402
from src.e014_oiqcl import (  # noqa: E402
    fit_linear_head,
    make_probs_qnode,
    probs_features,
    train_backbone,
)
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
TASK_KEYS = ("task1", "task2", "task3")


def _oiqcl_frozen_acc(qnode, weights, tasks, seed):
    """Frozen-backbone per-task acc: fit each task's head on this qnode's probs."""
    per, accs = {}, []
    for k, t in zip(TASK_KEYS, tasks):
        head = fit_linear_head(probs_features(qnode, weights, t.X_train), t.y_train, t.name, seed=seed)
        a = head.accuracy(probs_features(qnode, weights, t.X_test), t.y_test)
        per[k] = round(a, 4)
        accs.append(a)
    return round(float(np.mean(accs)), 4), per


def run(*, layers=12, lr=0.05, epochs=20, n_train=400, n_test=200, readouts=(None, 2),
        noise=None, seeds=(42, 43, 44), verbose=True) -> dict[str, Any]:
    cfg = {**DEFAULT_NOISE, **(noise or {})}
    started = time.perf_counter()
    agg: dict[str, dict[str, list]] = {}
    for seed in seeds:
        t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
        t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
        tasks = [t1, t2, t3]
        weights, _, _ = train_backbone(t1, n_qubits=N_QUBITS, n_layers=layers, lr=lr,
                                       epochs=epochs, seed=seed)  # noiseless Task-1 training
        for m in readouts:
            rw = None if m is None else range(m)
            clean, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers, readout_wires=rw)
            noisy, _ = make_noisy_probs_qnode(n_qubits=N_QUBITS, n_layers=layers, noise=cfg,
                                              readout_wires=rw)
            tag = "full" if m is None else f"m{m}"
            agg.setdefault(tag, {"noiseless": [], "noisy": []})
            a_clean, _ = _oiqcl_frozen_acc(clean, weights, tasks, seed)
            a_noisy, _ = _oiqcl_frozen_acc(noisy, weights, tasks, seed)
            agg[tag]["noiseless"].append(a_clean)
            agg[tag]["noisy"].append(a_noisy)
            if verbose:
                print(f"  seed {seed} readout={tag:4s}: noiseless={a_clean:.3f}  "
                      f"noisy={a_noisy:.3f}  drop={a_clean - a_noisy:+.3f}", flush=True)

    def ms(v):
        a = np.asarray(v)
        return round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)

    summary = {tag: {"noiseless": ms(d["noiseless"]), "noisy": ms(d["noisy"]),
                     "drop": round(ms(d["noiseless"])[0] - ms(d["noisy"])[0], 4)}
               for tag, d in agg.items()}
    return {
        "experiment": "e014_noise_compare",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"noise": cfg, "layers": layers, "epochs_task1": epochs, "n_qubits": N_QUBITS,
                   "n_train": n_train, "n_test": n_test, "seeds": list(seeds),
                   "variant": "MPI frozen theta_1 (A); noiseless training, noisy readout",
                   "device": "default.qubit (train/noiseless) + default.mixed (noisy readout)"},
        "summary": summary, "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--depol", type=float, default=0.01)
    ap.add_argument("--meas", type=float, default=0.02)
    ap.add_argument("--bit", type=float, default=0.0)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    noise = {"bit": args.bit, "phase": args.phase, "depol": args.depol, "meas": args.meas}
    result = run(layers=args.layers, epochs=args.epochs, n_train=args.n_train,
                 n_test=args.n_test, noise=noise, seeds=tuple(args.seeds))
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / "e014_noise_compare.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nsummary (mean over seeds):")
    for tag, d in result["summary"].items():
        print(f"  readout {tag:4s}: noiseless={d['noiseless'][0]:.3f}  noisy={d['noisy'][0]:.3f}  "
              f"drop={d['drop']:+.3f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
