"""e009 multi-seed: continual-learning forecasting across seeds (mean +/- sample SD)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e009_continual_forecasting import METHODS, train_method  # noqa: E402
from src.e009_data import load_task_sequence  # noqa: E402

RESULTS = ROOT / "results"
FIELDS = ("retention_earlier_nmse", "plasticity_final_nmse", "avg_earlier_forgetting",
          "avg_final_nmse")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--buffer-size", type=int, default=24)
    args = ap.parse_args()

    raw = {m: {f: [] for f in FIELDS} for m in METHODS}
    curves = {m: [] for m in METHODS}   # per-seed histories, for the mean curve figure
    t0 = time.perf_counter()
    for seed in args.seeds:
        tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
        for m in METHODS:
            r = train_method(m, tasks, n_layers=args.layers, seq_len=args.seq_len, lr=0.05,
                             epochs=args.epochs_per_task, lam=args.lam,
                             buffer_size=args.buffer_size, seed=seed)
            for f in FIELDS:
                raw[m][f].append(r[f])
            curves[m].append(r["history"])
        print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    def ms(v):
        a = np.array(v)
        return {"mean": round(float(a.mean()), 4),
                "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}
    summary = {m: {f: ms(raw[m][f]) for f in FIELDS} for m in METHODS}

    # Mean per-task NMSE curve across seeds (aligned by epoch).
    mean_curves = {}
    for m in METHODS:
        hs = curves[m]
        epochs = [row["epoch"] for row in hs[0]]
        mean_curves[m] = [
            {"epoch": epochs[i], "phase": hs[0][i]["phase"],
             "nmse": {t: round(float(np.mean([h[i]["nmse"][t] for h in hs])), 5)
                      for t in args.tasks}}
            for i in range(len(epochs))]

    out = {"experiment": "e009_continual_forecasting_multiseed", "tasks": args.tasks,
           "seeds": args.seeds, "epochs_per_task": args.epochs_per_task,
           "summary": summary, "mean_curves": mean_curves}
    (RESULTS / "e009_multiseed.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== e009 multi-seed (mean +/- sd; lower NMSE better) ===")
    for m in METHODS:
        s = summary[m]
        print(f"  {m:7s} retention={s['retention_earlier_nmse']['mean']:.3f}"
              f"+/-{s['retention_earlier_nmse']['sd']:.3f}  "
              f"plasticity={s['plasticity_final_nmse']['mean']:.3f}"
              f"+/-{s['plasticity_final_nmse']['sd']:.3f}  "
              f"avg_final={s['avg_final_nmse']['mean']:.3f}")
    print(f"Wrote {RESULTS / 'e009_multiseed.json'}")


if __name__ == "__main__":
    main()
