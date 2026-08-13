"""e009 gate-count ablation: can a smaller circuit keep NMSE on the forecasting benchmark?

For each ansatz variant (entangler ring/chain/none x encoding two-axis/single-axis x n_layers)
record the REAL circuit cost via `qml.specs` (total gates, two-qubit/CNOT count, depth, params)
and the continual-learning NMSE (retention / plasticity / avg) across seeds. The point is a
Pareto view of NMSE vs CNOT count, so we pick the smallest circuit that still holds quality
(CLAUDE.md: always compare circuit depth + two-qubit gate count before claiming an improvement).

Run:
    python scripts/e009_gate_ablation.py                          # baseline vs aggressive, naive, 5 seeds
    python scripts/e009_gate_ablation.py --configs baseline chain one_layer aggressive
    python scripts/e009_gate_ablation.py --method qewc            # confirm under the protected method
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_data import load_task_sequence  # noqa: E402
from src.e009_qtsf import make_forecaster  # noqa: E402
from experiments.e009_continual_forecasting import train_method  # noqa: E402

RESULTS = ROOT / "results"

# ansatz variants: (n_layers, entangler, encoding). "baseline" reproduces the current model
# (2 layers, CNOT ring, two-axis RY+RZ encoding = 256 gates / 64 CNOT / depth 112).
CONFIGS = {
    "baseline":   dict(n_layers=2, entangler="ring",  encoding="ry_rz"),  # 256 / 64 CNOT
    "chain":      dict(n_layers=2, entangler="chain", encoding="ry_rz"),  # drop CNOT wrap  -> 48 CNOT
    "enc_ry":     dict(n_layers=2, entangler="ring",  encoding="ry"),     # single-axis encode
    "one_layer":  dict(n_layers=1, entangler="ring",  encoding="ry_rz"),  # half the block  -> 32 CNOT
    "aggressive": dict(n_layers=1, entangler="chain", encoding="ry"),     # smallest entangled -> 24 CNOT
    "minimal":    dict(n_layers=1, entangler="none",  encoding="ry"),     # no entanglement (floor) -> 0 CNOT
}
_ANSATZ_KEYS = ("entangler", "encoding")


def circuit_cost(cfg: dict, seq_len: int) -> dict:
    """Real per-forward-pass circuit resources via qml.specs (data-independent counts)."""
    ansatz = {k: cfg[k] for k in _ANSATZ_KEYS}
    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=cfg["n_layers"], seq_len=seq_len, **ansatz)
    r = qml.specs(qnode)(np.zeros((1, seq_len)), np.zeros(cs)).resources
    return {"total_gates": int(r.num_gates), "cnot": int(r.gate_sizes.get(2, 0)),
            "depth": int(r.depth), "n_params": int(np.prod(cs)) + int(np.prod(hs)),
            "gate_types": {k: int(v) for k, v in r.gate_types.items()}}


def _agg(vals) -> dict:
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=0)), 4),
            "vals": [round(float(v), 4) for v in a]}


def run_config(cfg: dict, tasks, seeds, method: str, *, seq_len, lr, epochs, lam, buffer_size) -> dict:
    ansatz = {k: cfg[k] for k in _ANSATZ_KEYS}
    ret, plas, avg, forg = [], [], [], []
    for s in seeds:
        r = train_method(method, tasks, n_layers=cfg["n_layers"], seq_len=seq_len, lr=lr,
                         epochs=epochs, lam=lam, buffer_size=buffer_size, seed=s, ansatz=ansatz)
        ret.append(r["retention_earlier_nmse"])
        plas.append(r["plasticity_final_nmse"])
        avg.append(r["avg_final_nmse"])
        forg.append(r["avg_earlier_forgetting"])
    return {"retention": _agg(ret), "plasticity": _agg(plas), "avg": _agg(avg), "forgetting": _agg(forg)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["baseline", "aggressive"], choices=list(CONFIGS))
    ap.add_argument("--method", default="naive", choices=["naive", "l2", "ewc", "qewc", "replay"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_gate_ablation.json")
    args = ap.parse_args()

    tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
    print(f"e009 gate ablation: method={args.method} seeds={args.seeds} tasks={args.tasks}\n", flush=True)
    t0 = time.perf_counter()
    rows = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        cost = circuit_cost(cfg, args.seq_len)
        perf = run_config(cfg, tasks, args.seeds, args.method, seq_len=args.seq_len, lr=args.lr,
                          epochs=args.epochs_per_task, lam=args.lam, buffer_size=args.buffer_size)
        rows[name] = {"config": cfg, "cost": cost, "perf": perf}
        print(f"[{name:11s}] gates={cost['total_gates']:3d} CNOT={cost['cnot']:2d} "
              f"depth={cost['depth']:3d} params={cost['n_params']:2d} | "
              f"ret={perf['retention']['mean']:.3f}±{perf['retention']['sd']:.3f} "
              f"plas={perf['plasticity']['mean']:.3f} avg={perf['avg']['mean']:.3f}", flush=True)

    out = {"experiment": "e009_gate_ablation", "method": args.method, "seeds": args.seeds,
           "tasks": args.tasks, "seq_len": args.seq_len, "configs": rows,
           "run_time_sec": round(time.perf_counter() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    outp = args.output if args.output.is_absolute() else ROOT / args.output
    outp.write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== gate ablation summary (NMSE lower = better) ===")
    print(f"{'config':11s} {'gates':>5} {'CNOT':>4} {'depth':>5} {'par':>3} | "
          f"{'retention':>15} {'plas':>6} {'avg':>6}   vs baseline")
    base = rows.get("baseline")
    for name in args.configs:
        c, p = rows[name]["cost"], rows[name]["perf"]
        delta = ""
        if base and name != "baseline":
            da = p["avg"]["mean"] - base["perf"]["avg"]["mean"]
            dc = base["cost"]["cnot"] - c["cnot"]
            delta = f"  Δavg {da:+.3f}, -{dc} CNOT ({100 * dc / max(base['cost']['cnot'], 1):.0f}%)"
        print(f"{name:11s} {c['total_gates']:5d} {c['cnot']:4d} {c['depth']:5d} {c['n_params']:3d} | "
              f"{p['retention']['mean']:.3f}±{p['retention']['sd']:.3f} "
              f"{p['plasticity']['mean']:6.3f} {p['avg']['mean']:6.3f}{delta}")
    print(f"\nWrote {outp}  ({out['run_time_sec']}s)")


if __name__ == "__main__":
    main()
