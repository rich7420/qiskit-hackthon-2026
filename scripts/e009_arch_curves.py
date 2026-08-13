"""Per-task R^2 training curves — does the simplified (single-axis) architecture beat the
original architecture under EWC/QEWC?

Reference-style 3-panel figure: R^2 = 1 - NMSE vs epoch, mean +/- SD over seeds, sequential
training on narma_5 -> damped_shm -> bessel_j2. Each task is shown ONLY from its own training
onset (no pre-training segment). Arms:

  Baseline (naive, orig. arch)          two-axis encoding, 2 layers, ring   (256 gates / 64 CNOT)
  EWC (orig. arch)
  QEWC (orig. arch)
  Simplified (single-axis) + QEWC       single-axis encoding = enc_ry       (224 gates / 64 CNOT)   <- highlight

Run:
    python scripts/e009_arch_curves.py                     # 4 arms x5 seeds -> JSON + PNG
    python scripts/e009_arch_curves.py --plot-only         # re-plot from saved JSON
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

# (key, label, method, ansatz, color, linestyle, lw, highlight)
ARMS = [
    ("naive_orig", "Baseline (naive, orig. arch)", "naive", {}, "0.5", "--", 1.6, False),
    ("ewc_orig", "EWC (orig. arch)", "ewc", {}, "#4477AA", "-.", 2.0, False),
    ("qewc_orig", "QEWC (orig. arch)", "qewc", {}, "#228833", "-.", 2.0, False),
    ("qewc_simpl", "Simplified (single-axis) + QEWC", "qewc", {"encoding": "ry"},
     "#CC3311", "-", 2.6, True),
]


def run(seeds, tasks, *, n_layers, seq_len, lr, epochs, lam, buffer_size):
    task_names = [t.name for t in tasks]
    out = {}
    for key, label, method, ansatz, *_ in ARMS:
        # stack per-seed test-NMSE histories -> R^2 mean/sd per task per epoch
        per_seed = []
        for s in seeds:
            r = train_method(method, tasks, n_layers=n_layers, seq_len=seq_len, lr=lr,
                             epochs=epochs, lam=lam, buffer_size=buffer_size, seed=s, ansatz=ansatz)
            per_seed.append(r["history"])
            print(f"    [{key:11s} seed {s}] done", flush=True)
        epochs_axis = [h["epoch"] for h in per_seed[0]]
        r2 = {}
        for t in task_names:
            arr = np.array([[1.0 - hh["nmse"][t] for hh in hist] for hist in per_seed])  # (seed, epoch)
            r2[t] = {"mean": arr.mean(0).round(4).tolist(), "sd": arr.std(0).round(4).tolist()}
        out[key] = {"label": label, "epochs": epochs_axis, "r2": r2}
        print(f"  [{key}] aggregated", flush=True)
    return {"experiment": "e009_arch_curves", "seeds": seeds, "tasks": task_names,
            "epochs_per_task": epochs, "arms": out}


def plot(data, output):
    import matplotlib.pyplot as plt

    tasks = data["tasks"]
    ept = data["epochs_per_task"]
    total = ept * len(tasks)
    style = {key: (ls, color, lw, hl, label)
             for key, label, _m, _a, color, ls, lw, hl in ARMS}

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.6, 2.7 * len(tasks)))

    for i, (ax, task) in enumerate(zip(axes, tasks)):
        onset = i * ept                      # show each task only from its own training onset
        start = max(onset, 1)                # skip the pre-training epoch-0 snapshot
        for key, arm in data["arms"].items():
            ls, color, lw, hl, label = style[key]
            ep = np.array(arm["epochs"])
            mean = np.array(arm["r2"][task]["mean"])
            sd = np.array(arm["r2"][task]["sd"])
            m = ep >= start
            ax.plot(ep[m], mean[m], ls, color=color, lw=lw, label=label, zorder=4 if hl else 3)
            ax.fill_between(ep[m], np.clip(mean[m] - sd[m], 0, 1), np.clip(mean[m] + sd[m], 0, 1),
                            color=color, alpha=0.16 if hl else 0.12)
        for b in range(ept, total, ept):     # task-boundary lines within the visible range
            if b >= start:
                ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.axvspan(onset, onset + ept, color="green", alpha=0.05)   # this task's training window
        ax.set_xlim(onset, total)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"T{i + 1}\nR² (=1−NMSE)")
        ax.set_title(f"Task {i + 1}: {task}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower right", fontsize=8.5, ncol=2)
    axes[-1].set_xlabel("Epoch (sequential training; shown from each task's training onset)")
    fig.suptitle(f"Simplified single-axis architecture vs baseline / EWC / QEWC "
                 f"— R² view (seeds {data['seeds']})", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--tasks", nargs="+", default=["narma_5", "damped_shm", "bessel_j2"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs-per-task", type=int, default=40)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--buffer-size", type=int, default=24)
    ap.add_argument("--json", type=Path, default=RESULTS / "e009_arch_curves.json")
    ap.add_argument("--output", type=Path, default=FIGURES / "e009_arch_curves.png")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        data = json.loads(args.json.read_text())
    else:
        tasks = load_task_sequence(args.tasks, seq_len=args.seq_len)
        print(f"e009 arch curves: {len(ARMS)} arms x {len(args.seeds)} seeds", flush=True)
        t0 = time.perf_counter()
        data = run(args.seeds, tasks, n_layers=args.layers, seq_len=args.seq_len, lr=args.lr,
                   epochs=args.epochs_per_task, lam=args.lam, buffer_size=args.buffer_size)
        data["run_time_sec"] = round(time.perf_counter() - t0, 1)
        RESULTS.mkdir(exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {args.json} ({data['run_time_sec']}s)")

    plot(data, args.output)


if __name__ == "__main__":
    main()
