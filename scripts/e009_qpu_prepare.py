"""Step A of the QPU pilot — sim-train the simplified-architecture models, save what hardware needs.

- Simplified ansatz (aggressive: 1 layer, CNOT chain, single-axis RY encoding; 24 CNOT, depth 41) —
  the hardware-friendly circuit from the gate ablation.
- Methods QEWC (lam=0.1, its retuned best on this small model) and QGR (gen_len=24), across seeds.
- Fixes a subset of test windows/task, extracts the final trained weights, and records the
  sim-noiseless and sim-noisy NMSE on that subset (the two references for the sim-vs-QPU figure).

Output: results/e009_qpu_models.json (weights + test subset + sim references).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_data import load_task_sequence  # noqa: E402
from src.e009_qtsf import make_forecaster, nmse  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402

RESULTS = ROOT / "results"
ANSATZ = {"entangler": "chain", "encoding": "ry"}   # aggressive (simplified) architecture
N_LAYERS, SEQ_LEN = 1, 8
LAM = {"qewc": 0.1, "qgr": 5.0}       # qewc retuned best for the 13-param model; qgr ignores lam
GEN_LEN = 24
NOISE = {"depol": 0.01, "meas": 0.02}   # same as the e009 noise study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--methods", nargs="+", default=["qewc", "qgr"])
    ap.add_argument("--windows", type=int, default=15)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--entangler", default="chain", choices=["ring", "chain", "none"])
    ap.add_argument("--encoding", default="ry", choices=["ry", "ry_rz"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--qewc-lam", type=float, default=0.1)   # aggressive best; use 5.0 for baseline arch
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_qpu_models.json")
    args = ap.parse_args()

    ansatz = {"entangler": args.entangler, "encoding": args.encoding}
    n_layers = args.layers
    lam_map = {"qewc": args.qewc_lam, "qgr": 5.0}
    tasks = load_task_sequence(args.tasks, seq_len=SEQ_LEN)
    # fixed, evenly-spaced test-window subset per task (same windows for sim and hardware)
    subsets = {}
    for t in tasks:
        idx = np.linspace(0, len(t.X_test) - 1, args.windows).round().astype(int)
        subsets[t.name] = {"X": t.X_test[idx].tolist(), "y": t.y_test[idx].tolist()}

    qn_clean, cs, hs = make_forecaster(n_qubits=4, n_layers=n_layers, seq_len=SEQ_LEN, **ansatz)
    qn_noisy = make_forecaster(n_qubits=4, n_layers=n_layers, seq_len=SEQ_LEN, noise=NOISE, **ansatz)[0]

    def sub_nmse(qn, cw, hw):
        return {t.name: round(nmse(qn, cw, hw, np.array(subsets[t.name]["X"]),
                                   np.array(subsets[t.name]["y"])), 4) for t in tasks}

    models = {}
    for m in args.methods:
        for s in args.seeds:
            r = train_method(m, tasks, n_layers=n_layers, seq_len=SEQ_LEN, lr=0.05,
                             epochs=args.epochs_per_task, lam=lam_map[m], buffer_size=24, seed=s,
                             ansatz=ansatz, gen_len=GEN_LEN)
            cw, hw = np.array(r["circ_w"]), np.array(r["head_w"])
            models[f"{m}:{s}"] = {"method": m, "seed": s, "circ_w": r["circ_w"], "head_w": r["head_w"],
                                  "sim_noiseless_nmse": sub_nmse(qn_clean, cw, hw),
                                  "sim_noisy_nmse": sub_nmse(qn_noisy, cw, hw)}
            print(f"  [{m}:{s}] sim-noiseless={models[f'{m}:{s}']['sim_noiseless_nmse']} "
                  f"sim-noisy={models[f'{m}:{s}']['sim_noisy_nmse']}", flush=True)

    out = {"experiment": "e009_qpu_models", "ansatz": ansatz, "n_layers": n_layers,
           "seq_len": SEQ_LEN, "seeds": args.seeds, "methods": args.methods,
           "windows_per_task": args.windows, "tasks": [t.name for t in tasks],
           "test_subset": subsets, "noise": NOISE, "models": models}
    RESULTS.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {args.output}  ({len(models)} models, {args.windows} windows/task)")


if __name__ == "__main__":
    main()
