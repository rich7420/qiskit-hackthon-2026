"""E014 gate-count ablation: how few gates does OI-QCL actually need?

Because the per-task advantage of OI-QCL comes from the measurement-side readout (a linear
head over probs), not from a deep circuit, we sweep the ansatz depth L and measure Task-IL
ACC of the isolated-head variants against the true circuit cost (2-qubit gates + depth from
qml.specs, which includes the AmplitudeEmbedding decomposition). Expectation: ACC is nearly
flat in L, so the circuit can be stripped to very few layers.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e014_compare import _run_isolated_head  # noqa: E402
from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import make_probs_qnode  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
DEFAULT_DEPTHS = (1, 2, 3, 4, 8, 12)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "src/e014_oiqcl.py", ROOT / "experiments/e014_compare.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _circuit_cost(layers: int) -> dict[str, int]:
    """True gate cost (incl. AmplitudeEmbedding decomposition) via qml.specs."""
    qnode, shape = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers)
    x = pnp.array(np.ones(2**N_QUBITS) / np.sqrt(2**N_QUBITS), requires_grad=False)
    w = pnp.array(np.zeros(shape), requires_grad=False)
    specs = qml.specs(qnode)(x, w)
    res = specs["resources"]
    gt = dict(res.gate_types)
    two_qubit = sum(v for k, v in gt.items() if k in ("CNOT", "CZ", "IsingXX", "SWAP"))
    return {"total_gates": int(res.num_gates), "depth": int(res.depth),
            "two_qubit_gates": int(two_qubit),
            "ansatz_cnots": (N_QUBITS - 1) * layers,
            "ansatz_rotations": 2 * N_QUBITS * layers,
            "theta_params": int(np.prod(shape))}


def run(*, depths=DEFAULT_DEPTHS, lr=0.05, epochs=20, alpha=5.0,
        n_train=800, n_test=200, seeds=(42, 43, 44), verbose=True) -> dict[str, Any]:
    started = time.perf_counter()
    per_depth: dict[str, Any] = {}
    for L in depths:
        cost = _circuit_cost(L)
        accs = {"frozen_head": [], "anchor_head": []}
        for seed in seeds:
            t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
            t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
            tasks = [t1, t2, t3]
            for m in ("frozen_head", "anchor_head"):
                r = _run_isolated_head(m, tasks, layers=L, lr=lr, epochs=epochs,
                                       alpha=alpha, seed=seed, verbose=False)
                accs[m].append(r["ACC"])
        summary = {m: {"ACC_mean": round(float(np.mean(v)), 4),
                       "ACC_sd": round(float(np.std(v, ddof=1)), 4)} for m, v in accs.items()}
        per_depth[str(L)] = {"layers": L, "cost": cost, "acc": summary}
        if verbose:
            print(f"  L={L:2d}  depth={cost['depth']:3d} 2q_gates={cost['two_qubit_gates']:3d} "
                  f"θ={cost['theta_params']:3d} | frozen ACC={summary['frozen_head']['ACC_mean']:.3f} "
                  f"anchor ACC={summary['anchor_head']['ACC_mean']:.3f}", flush=True)

    return {
        "schema_version": SCHEMA_VERSION, "experiment": "e014_depth_ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("pennylane", "numpy", "scikit-learn")}},
        "config": {"depths": list(depths), "seeds": list(seeds), "n_qubits": N_QUBITS,
                   "lr": lr, "epochs_per_task": epochs, "alpha_l2_anchor": alpha,
                   "n_train": n_train, "n_test": n_test,
                   "note": "cost includes AmplitudeEmbedding decomposition (fixed) + ansatz"},
        "per_depth": per_depth,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(depths=tuple(args.depths), epochs=args.epochs, n_train=args.n_train,
                 n_test=args.n_test, seeds=tuple(args.seeds))
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / "e014_depth_ablation.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
