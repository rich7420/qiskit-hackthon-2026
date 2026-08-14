"""Per-task forgetting curves — Baseline / EWC / QEWC / QGR (simulator, mean +/- SD over seeds).

The classic E009 catastrophic-forgetting view in the polished house style (each task shown only
from its own training onset; dotted task boundaries). Runs the four methods across seeds, records
per-epoch test NMSE, and plots the 3-panel figure.

Run:
    python scripts/e009_forgetting_curves.py --seeds 42 43 44
    python scripts/e009_forgetting_curves.py --plot-only
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
FIGURES = ROOT / "figures"

# method: (label, color, linestyle, lw)
STYLE = {
    "naive": ("Baseline (naive)", "0.5", "--", 1.8),
    "ewc": ("EWC (classical Fisher)", "#4477AA", "-.", 2.0),
    "qewc": ("QEWC (quantum Fisher)", "#228833", "-.", 2.0),
    "qgr": ("QGR (quantum generative replay)", "#CC3311", "-", 2.6),
}

# Architecture per method: QGR uses the simplified circuit we deploy (single-axis RY, 1 layer, CNOT
# chain); the regularizers stay on the original two-axis baseline (QEWC at its baseline-tuned lam).
ARCH = {
    "naive": {"n_layers": 2, "ansatz": {}, "gen_len": 48, "lam": 5.0},
    "ewc":   {"n_layers": 2, "ansatz": {}, "gen_len": 48, "lam": 5.0},
    "qewc":  {"n_layers": 2, "ansatz": {}, "gen_len": 48, "lam": 5.0},
    "qgr":   {"n_layers": 1, "ansatz": {"entangler": "chain", "encoding": "ry"}, "gen_len": 24, "lam": 5.0},
}


def run(seeds, tasks, *, seq_len, lr, epochs, buffer_size):
    task_names = [t.name for t in tasks]
    rows = {}
    for m in STYLE:
        cfg = ARCH[m]
        hists = []
        for s in seeds:
            r = train_method(m, tasks, n_layers=cfg["n_layers"], seq_len=seq_len, lr=lr, epochs=epochs,
                             lam=cfg["lam"], buffer_size=buffer_size, seed=s,
                             ansatz=cfg["ansatz"], gen_len=cfg["gen_len"])
            hists.append(r["history"])
            print(f"    [{m:5s} seed {s}] final={r['final_nmse']}", flush=True)
        epochs_axis = [h["epoch"] for h in hists[0]]
        curves = {}
        for t in task_names:
            arr = np.array([[hh["nmse"][t] for hh in H] for H in hists])
            curves[t] = {"mean": arr.mean(0).round(4).tolist(), "sd": arr.std(0).round(4).tolist()}
        rows[m] = {"epochs": epochs_axis, "nmse": curves}
        print(f"  [{m}] aggregated", flush=True)
    return {"experiment": "e009_forgetting_curves", "seeds": seeds, "tasks": task_names,
            "epochs_per_task": epochs, "rows": rows}


def plot(data, output):
    import matplotlib.pyplot as plt

    tasks, ept = data["tasks"], data["epochs_per_task"]
    total = ept * len(tasks)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.6, 2.7 * len(tasks)), sharex=True)

    for i, (ax, task) in enumerate(zip(axes, tasks), start=1):
        start = max((i - 1) * ept, 1)   # draw each task only from its own training onset
        for m, (label, color, ls, lw) in STYLE.items():
            row = data["rows"].get(m)
            if not row:
                continue
            ep = np.array(row["epochs"])
            r2 = 1.0 - np.array(row["nmse"][task]["mean"])   # accuracy view: R^2 = 1 - NMSE
            sd = np.array(row["nmse"][task]["sd"])
            k = ep >= start
            ax.plot(ep[k], r2[k], ls, color=color, lw=lw, label=label)
            ax.fill_between(ep[k], np.clip(r2[k] - sd[k], 0, 1.03), np.clip(r2[k] + sd[k], 0, 1.03),
                            color=color, alpha=0.13)
        for b in range(ept, total, ept):
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.axvspan((i - 1) * ept, i * ept, color="green", alpha=0.06)   # this task being trained
        ax.set_xlim(0, total)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(f"T{i}\nR² (=1−NMSE)")
        ax.set_title(f"Task {i}: {task}", loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.18)

    axes[0].legend(loc="center", fontsize=8.5, ncol=2)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(f"Advanced methods on quantum forecasting — accuracy (R²) view: "
                 f"Baseline / EWC / QEWC / QGR (seeds {data['seeds']})", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--qgr-arch", default="aggressive", choices=["aggressive", "baseline"],
                    help="QGR arm's architecture; 'baseline' = same two-axis circuit as the others")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.qgr_arch == "baseline":   # fair same-architecture comparison (QGR on the two-axis baseline)
        ARCH["qgr"] = {"n_layers": 2, "ansatz": {}, "gen_len": 48, "lam": 5.0}
    suffix = "" if args.qgr_arch == "aggressive" else "_twoaxis"
    json_path = args.json or RESULTS / f"e009_forgetting_curves{suffix}.json"
    output_path = args.output or FIGURES / f"e009_forgetting_curves{suffix}.png"

    if args.plot_only:
        data = json.loads(json_path.read_text())
    else:
        tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
        print(f"e009 forgetting curves ({args.qgr_arch} QGR): {list(STYLE)} x {len(args.seeds)} seeds", flush=True)
        t0 = time.perf_counter()
        data = run(args.seeds, tasks, seq_len=args.seq_len, lr=args.lr,
                   epochs=args.epochs_per_task, buffer_size=args.buffer_size)
        data["qgr_arch"] = args.qgr_arch
        data["run_time_sec"] = round(time.perf_counter() - t0, 1)
        RESULTS.mkdir(exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {json_path} ({data['run_time_sec']}s)")

    plot(data, output_path)


if __name__ == "__main__":
    main()
