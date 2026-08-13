"""Noise-robustness of continual-learning methods on the quantum forecaster.

Compares naive / QEWC / QGR under phase-flip + bit-flip + measurement (readout) error against the
noiseless reference, across seeds. Noise uses the density-matrix simulator (default.mixed); the
QEWC QFI stays noiseless (it is a pure-state importance weight, not a measured quantity). Records
retention / plasticity / avg NMSE and the noise-induced degradation per method.

Run:
    python scripts/e009_noise_compare.py --seeds 42 43 44
    python scripts/e009_noise_compare.py --conditions noisy --bit 0.02 --phase 0.02 --meas 0.05
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

from src.e009_data import load_task_sequence  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402

RESULTS = ROOT / "results"
METHODS = ("naive", "qewc", "qgr")


def _agg(vals) -> dict:
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=0)), 4),
            "vals": [round(float(v), 4) for v in a]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=20)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--bit", type=float, default=0.01)     # bit-flip (X) gate error per step
    ap.add_argument("--phase", type=float, default=0.01)   # phase-flip (Z) gate error per step
    ap.add_argument("--depol", type=float, default=0.0)    # depolarizing (generic X/Y/Z) per step
    ap.add_argument("--meas", type=float, default=0.02)    # readout / measurement bit-flip
    ap.add_argument("--ansatz-encoding", default="ry_rz", choices=["ry_rz", "ry"])
    ap.add_argument("--ansatz-entangler", default="ring", choices=["ring", "chain", "none"])
    ap.add_argument("--conditions", nargs="+", default=["noiseless", "noisy"],
                    choices=["noiseless", "noisy"])
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_noise_compare.json")
    args = ap.parse_args()

    tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
    ansatz = {"encoding": args.ansatz_encoding, "entangler": args.ansatz_entangler}
    noise = {"bit": args.bit, "phase": args.phase, "depol": args.depol, "meas": args.meas}
    cond_noise = {"noiseless": None, "noisy": noise}
    print(f"noise compare: methods={METHODS} conds={args.conditions} seeds={args.seeds} "
          f"epochs={args.epochs_per_task} gen_len={args.gen_len} noise={noise}\n", flush=True)

    task_names = [t.name for t in tasks]
    t0 = time.perf_counter()
    rows = {}
    for cond in args.conditions:   # noiseless first (fast), then noisy (density matrix)
        for m in METHODS:
            ret, plas, avg, hists = [], [], [], []
            for s in args.seeds:
                r = train_method(m, tasks, n_layers=args.layers, seq_len=args.seq_len, lr=args.lr,
                                 epochs=args.epochs_per_task, lam=args.lam,
                                 buffer_size=args.buffer_size, seed=s, noise=cond_noise[cond],
                                 ansatz=ansatz, gen_len=args.gen_len)
                ret.append(r["retention_earlier_nmse"])
                plas.append(r["plasticity_final_nmse"])
                avg.append(r["avg_final_nmse"])
                hists.append(r["history"])
                print(f"    [{cond:9s} {m:5s} seed {s}] ret={r['retention_earlier_nmse']:.3f} "
                      f"plas={r['plasticity_final_nmse']:.3f} avg={r['avg_final_nmse']:.3f}", flush=True)
            # per-epoch test-NMSE curves (mean +/- SD over seeds) for the forgetting figure
            epochs = [h["epoch"] for h in hists[0]]
            curves = {}
            for t in task_names:
                arr = np.array([[hh["nmse"][t] for hh in H] for H in hists])   # (seed, epoch)
                curves[t] = {"mean": arr.mean(0).round(4).tolist(), "sd": arr.std(0).round(4).tolist()}
            rows[f"{cond}:{m}"] = {"cond": cond, "method": m, "retention": _agg(ret),
                                   "plasticity": _agg(plas), "avg": _agg(avg),
                                   "curves": {"epochs": epochs, "nmse": curves}}

    out = {"experiment": "e009_noise_compare", "seeds": args.seeds, "tasks": args.tasks,
           "epochs_per_task": args.epochs_per_task, "gen_len": args.gen_len, "noise": noise,
           "ansatz": ansatz, "rows": rows, "run_time_sec": round(time.perf_counter() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    outp = args.output if args.output.is_absolute() else ROOT / args.output
    outp.write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== noise robustness (avg NMSE, lower = better) ===")
    print(f"{'method':6s} {'noiseless':>16s} {'noisy':>16s} {'degradation':>12s}")
    for m in METHODS:
        nl, nz = rows.get(f"noiseless:{m}"), rows.get(f"noisy:{m}")
        if nl and nz:
            d = nz["avg"]["mean"] - nl["avg"]["mean"]
            print(f"{m:6s} {nl['avg']['mean']:.3f}±{nl['avg']['sd']:.3f}   "
                  f"{nz['avg']['mean']:.3f}±{nz['avg']['sd']:.3f}   {d:+.3f}")
    print(f"\nWrote {outp} ({out['run_time_sec']}s)")


if __name__ == "__main__":
    main()
