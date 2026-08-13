"""Retune the QEWC anchor strength (lam) per ansatz config — fair best-vs-best comparison.

The gate ablation showed the 13-param `aggressive` circuit is over-regularized by the default
lam=5.0 (excellent retention, poor plasticity), while the 21-param `baseline` was tuned at 5.0.
This sweeps lam for the chosen configs under qewc so each is compared at its OWN best operating
point — the honest "does the smaller (24-CNOT) circuit match the 64-CNOT baseline?" test.
Records retention / plasticity / avg (5 seeds) per (config, lam).

Run:
    python scripts/e009_gate_lam_sweep.py --configs aggressive baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_data import load_task_sequence  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402
from scripts.e009_gate_ablation import CONFIGS, _ANSATZ_KEYS, _agg, circuit_cost  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["aggressive", "baseline"], choices=list(CONFIGS))
    ap.add_argument("--lams", nargs="+", type=float, default=[0.3, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_gate_lam_sweep.json")
    args = ap.parse_args()

    tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
    print(f"lam sweep (qewc): configs={args.configs} lams={args.lams} seeds={args.seeds}\n", flush=True)
    t0 = time.perf_counter()
    rows = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        ansatz = {k: cfg[k] for k in _ANSATZ_KEYS}
        cost = circuit_cost(cfg, args.seq_len)
        rows[name] = {"config": cfg, "cost": cost, "lams": {}}
        for lam in args.lams:
            ret, plas, avg = [], [], []
            for s in args.seeds:
                r = train_method("qewc", tasks, n_layers=cfg["n_layers"], seq_len=args.seq_len,
                                 lr=args.lr, epochs=args.epochs_per_task, lam=lam,
                                 buffer_size=args.buffer_size, seed=s, ansatz=ansatz)
                ret.append(r["retention_earlier_nmse"])
                plas.append(r["plasticity_final_nmse"])
                avg.append(r["avg_final_nmse"])
            rows[name]["lams"][f"{lam:g}"] = {"retention": _agg(ret), "plasticity": _agg(plas),
                                              "avg": _agg(avg)}
            print(f"[{name:11s} lam={lam:4.1f}] CNOT={cost['cnot']:2d} | "
                  f"ret={_agg(ret)['mean']:.3f} plas={_agg(plas)['mean']:.3f} "
                  f"avg={_agg(avg)['mean']:.3f}", flush=True)

    out = {"experiment": "e009_gate_lam_sweep", "method": "qewc", "seeds": args.seeds,
           "tasks": args.tasks, "seq_len": args.seq_len, "configs": rows,
           "run_time_sec": round(time.perf_counter() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    outp = args.output if args.output.is_absolute() else ROOT / args.output
    outp.write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== best lam per config (min avg NMSE) ===")
    for name in args.configs:
        lams = rows[name]["lams"]
        best = min(lams, key=lambda L: lams[L]["avg"]["mean"])
        b = lams[best]
        print(f"{name:11s} CNOT={rows[name]['cost']['cnot']:2d}  best lam={best}: "
              f"ret={b['retention']['mean']:.3f} plas={b['plasticity']['mean']:.3f} "
              f"avg={b['avg']['mean']:.3f}")
    print(f"\nWrote {outp} ({out['run_time_sec']}s)")


if __name__ == "__main__":
    main()
