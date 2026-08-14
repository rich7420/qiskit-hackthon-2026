"""E014 matched-capacity classical baseline: is MPI "just a classical multi-head"?

MPI puts a per-task linear head on the QUANTUM probabilities p_theta(x) in R^{2^n}. The
matched control is the SAME head capacity on the RAW input instead: a per-task logistic head
on the amplitude-embedding feature vector x_tilde in R^{2^n} (the same 16-dim vector that
would be fed to AmplitudeEmbedding), with NO quantum circuit at all. Same dimensionality, same
linear head, same Task-IL / task-agnostic protocols. If MPI >> this, the quantum feature
map p_theta carries extra separable, forgetting-resistant information; if ~=, the circuit adds
nothing on this benchmark.

Reports (mean over seeds), for the classical multi-head:
  * per-task Task-IL accuracy + average (task id given)  -> compare vs MPI frozen/anchor ACC
  * task-agnostic accuracy via a classical linear router over x_tilde (task id inferred)
No quantum gradients, no circuit: pure sklearn on the raw features.
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
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = 1
TASK_KEYS = ("task1", "task2", "task3")


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "src/continual_data.py", ROOT / "src/phase_data.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fit_head(X, y, seed):
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X, _labels_to_classes(y))
    return clf


def run(*, n_train=800, n_test=200, seed=42, verbose=True) -> dict[str, Any]:
    t1, t2 = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
    t3 = load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    tasks = [t1, t2, t3]
    started = time.perf_counter()

    # Per-task classical heads on the RAW amplitude vectors (no circuit).
    heads = [_fit_head(t.X_train, t.y_train, seed) for t in tasks]
    per_task = {}
    for j, (t, h) in enumerate(zip(tasks, heads)):
        acc = float(np.mean(h.predict(t.X_test) == _labels_to_classes(t.y_test)))
        per_task[TASK_KEYS[j]] = {"name": t.name, "taskIL_acc": round(acc, 4)}
    taskIL_ACC = float(np.mean([per_task[k]["taskIL_acc"] for k in TASK_KEYS]))

    # Task-agnostic: classical linear router over raw x_tilde, then the routed head.
    X_pool = np.concatenate([t.X_test for t in tasks])
    true_task = np.concatenate([np.full(len(t.X_test), j) for j, t in enumerate(tasks)])
    true_cls = _labels_to_classes(np.concatenate([t.y_test for t in tasks]))
    router = LogisticRegression(max_iter=2000, C=1.0)
    router.fit(np.concatenate([t.X_train for t in tasks]),
               np.concatenate([np.full(len(t.X_train), j) for j, t in enumerate(tasks)]))
    t_hat = router.predict(X_pool)
    preds = np.stack([h.predict(X_pool) for h in heads])  # (T, N)
    routed = preds[t_hat, np.arange(len(X_pool))]
    agnostic = float(np.mean(routed == true_cls))
    router_tia = float(np.mean(t_hat == true_task))

    if verbose:
        print(f"  seed {seed}: classical multi-head  Task-IL ACC={taskIL_ACC:.3f}  "
              f"agnostic={agnostic:.3f}  router-TIA={router_tia:.3f}", flush=True)

    return {
        "schema_version": SCHEMA_VERSION, "experiment": "e014_classical_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": _source_digest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("numpy", "scikit-learn")}},
        "config": {"model": "per-task logistic head on raw 2^n amplitude input (NO quantum circuit)",
                   "matched_to": "MPI head over quantum probs p_theta(x) (same dim, same head)",
                   "tasks": [t.name for t in tasks], "n_qubits": N_QUBITS,
                   "n_train": n_train, "n_test": n_test, "seed": seed},
        "per_task": per_task,
        "taskIL_ACC": round(taskIL_ACC, 4),
        "task_agnostic_accuracy": round(agnostic, 4),
        "router_task_inference_accuracy": round(router_tia, 4),
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    out = args.output or RESULTS / f"e014_classical_baseline_seed{args.seed}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
