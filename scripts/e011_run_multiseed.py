"""Run E011 (blocked vs interleaved) for several seeds and aggregate mean +/- sample SD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e011_interleaved import (  # noqa: E402
    SCHEDULES,
    TASK_KEYS,
    run_experiment,
    write_result,
)

RESULTS = ROOT / "results"
DEFAULT_SEEDS = (42, 43, 44)


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(np.mean(arr)), 6),
            "sample_std": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0}


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = {r["source_code_sha256"] for r in results}
    if len(hashes) != 1:
        raise ValueError("seeded results came from different source revisions")
    epochs = [row["epoch"] for row in results[0]["histories"]["blocked"]]

    agg_hist = {s: [] for s in SCHEDULES}
    for s in SCHEDULES:
        for i, epoch in enumerate(epochs):
            agg_hist[s].append({
                "epoch": epoch,
                "test_accuracy": {
                    k: _mean_std([r["histories"][s][i]["test_accuracy"][k] for r in results])
                    for k in TASK_KEYS
                },
            })

    agg_metrics = {s: {k: {
        "name": results[0]["metrics"][s]["tasks"][k]["name"],
        "test_final": _mean_std([r["metrics"][s]["tasks"][k]["test_final"] for r in results]),
        "forgetting": _mean_std([r["metrics"][s]["tasks"][k]["forgetting"] for r in results]),
    } for k in TASK_KEYS} for s in SCHEDULES}
    agg_earlier = {
        s: _mean_std([r["metrics"][s]["mean_earlier_task_final"] for r in results])
        for s in SCHEDULES
    }
    agg_all = {
        s: _mean_std([r["metrics"][s]["mean_all_task_final"] for r in results])
        for s in SCHEDULES
    }

    return {
        "schema_version": 1,
        "experiment": "e011_interleaved_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": next(iter(hashes)),
        "seeds": [r["config"]["seed"] for r in results],
        "n_seeds": len(results),
        "uncertainty": "sample standard deviation across seeds (ddof=1)",
        "config": results[0]["config"],
        "aggregate_histories": agg_hist,
        "aggregate_metrics": agg_metrics,
        "mean_earlier_task_final": agg_earlier,
        "mean_all_task_final": agg_all,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02, dest="learning_rate")
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--summary-output", type=Path, default=RESULTS / "e011_summary.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        result = run_experiment(
            layers=args.layers, learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            n_train=args.n_train, n_test=args.n_test, seed=seed, verbose=True,
        )
        write_result(result, RESULTS / f"e011_seed{seed}.json")
        results.append(result)
        print(f"  seed {seed} done")

    summary = build_summary(results)
    out = args.summary_output if args.summary_output.is_absolute() else ROOT / args.summary_output
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("\n=== mean final accuracy across seeds ===")
    for s in SCHEDULES:
        me = summary["mean_earlier_task_final"][s]
        ma = summary["mean_all_task_final"][s]
        print(f"  {s:12s} earlier(T1,T2)={me['mean']:.3f} +/- {me['sample_std']:.3f}   "
              f"all(T1-T3)={ma['mean']:.3f} +/- {ma['sample_std']:.3f}")


if __name__ == "__main__":
    main()
