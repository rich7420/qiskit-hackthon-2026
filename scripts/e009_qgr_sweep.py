"""Sweep the QGR rollout length to test whether longer rollouts reduce retention variance.

For each gen_len, run QGR across seeds and report retention/plasticity mean +/- sample SD. Longer
autoregressive rollouts produce more (and longer-horizon) synthetic rehearsal data per generator,
which should stabilize the rehearsal signal and lower the retention variance.

Run:
    python scripts/e009_qgr_sweep.py --gen-lens 16 32 48 64 --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e009_continual_forecasting import train_method  # noqa: E402
from src.e009_data import load_task_sequence  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--gen-lens", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-gen-seeds", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    args = ap.parse_args()

    rows = []
    t0 = time.perf_counter()
    for g in args.gen_lens:
        ret, pla, avg = [], [], []
        for seed in args.seeds:
            tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
            r = train_method("qgr", tasks, n_layers=args.layers, seq_len=args.seq_len, lr=0.05,
                             epochs=args.epochs_per_task, lam=5.0, buffer_size=24, seed=seed,
                             gen_len=g, n_gen_seeds=args.n_gen_seeds)
            ret.append(r["retention_earlier_nmse"])
            pla.append(r["plasticity_final_nmse"])
            avg.append(r["avg_final_nmse"])
        row = {"gen_len": g,
               "retention_mean": round(float(np.mean(ret)), 4),
               "retention_sd": round(float(np.std(ret, ddof=1)), 4),
               "plasticity_mean": round(float(np.mean(pla)), 4),
               "plasticity_sd": round(float(np.std(pla, ddof=1)), 4),
               "avg_mean": round(float(np.mean(avg)), 4)}
        rows.append(row)
        print(f"  gen_len={g:3d}: retention {row['retention_mean']:.3f} +/- {row['retention_sd']:.3f}"
              f"   plasticity {row['plasticity_mean']:.3f} +/- {row['plasticity_sd']:.3f}"
              f"   avg {row['avg_mean']:.3f}   ({time.perf_counter()-t0:.0f}s)", flush=True)

    out = {"experiment": "e009_qgr_genlen_sweep", "tasks": args.tasks, "seeds": args.seeds,
           "n_gen_seeds": args.n_gen_seeds, "rows": rows}
    (RESULTS / "e009_qgr_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n=== does longer rollout reduce retention SD? ===")
    for r in rows:
        print(f"  gen_len {r['gen_len']:3d}  retention_SD = {r['retention_sd']:.3f}")
    print(f"Wrote {RESULTS / 'e009_qgr_sweep.json'}")


if __name__ == "__main__":
    main()
