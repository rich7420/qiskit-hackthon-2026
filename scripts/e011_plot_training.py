"""Plot E011 training curves in the E009 style: per-task rows, test accuracy over the whole
run, method lines with mean +/- SD bands, and a green span marking each task's own training
block in the *blocked/sequential* schedule.

Four arms on one figure (schedule x consolidation), all with identical params:
- blocked            (E011, == E005 baseline)   naive sequential, forgets
- QEWC blocked       (E005)                      QFI consolidation between blocks
- QEWC interleaved   (E011)                      interleaved order + online QFI penalty
- interleaved        (E011)                      upper bound: revisits every task each round

NOTE on the shading: the green band = "this row's task is the one being trained" only holds for
the sequential arms (blocked, QEWC-blocked). The interleaved arms train ALL tasks every round,
so for them the whole axis is effectively 'active'; the band is drawn as the blocked-schedule
reference. Reads results/e005_summary.json and results/e011_summary.json.

Run:
    python scripts/e011_plot_training.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E005 = ROOT / "results" / "e005_summary.json"
E011 = ROOT / "results" / "e011_summary.json"
OUT = ROOT / "figures" / "e011_training.png"

TASK_KEYS = ("task1", "task2", "task3")
# label, color, linestyle, (source, history_key) -- palette echoes E009 (naive/EWC/QEWC/replay).
SERIES = [
    ("blocked (no CL)", "0.45", "--", ("e011", "blocked")),
    ("QEWC blocked (E005)", "#4477AA", "-.", ("e005", "qewc")),
    ("QEWC interleaved (online QFI)", "#228833", "-.", ("e011", "qewc_interleaved")),
    ("interleaved (upper bound)", "#EE6677", "-", ("e011", "interleaved")),
]


def _curve(summaries, src, hkey, task_key):
    rows = summaries[src]["aggregate_histories"][hkey]
    ep = np.array([r["epoch"] for r in rows])
    mean = np.array([r["test_accuracy"][task_key]["mean"] for r in rows])
    std = np.array([r["test_accuracy"][task_key]["sample_std"] for r in rows])
    return ep, mean, std


def main() -> None:
    if not E005.exists():
        raise SystemExit(f"missing {E005} -- run scripts/e005_run_multiseed.py first")
    summaries = {"e005": json.loads(E005.read_text()), "e011": json.loads(E011.read_text())}

    cfg = summaries["e011"]["config"]
    epochs_per_task = cfg["epochs_per_task"]
    boundaries = cfg["task_boundaries"]
    names = [summaries["e011"]["aggregate_metrics"]["blocked"][k]["name"] for k in TASK_KEYS]
    total_epochs = summaries["e011"]["aggregate_histories"]["blocked"][-1]["epoch"]
    seeds = summaries["e011"]["seeds"]

    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, names), start=1):
        key = f"task{i}"
        # Green span = this task's own training block in the blocked/sequential schedule.
        ax.axvspan((i - 1) * epochs_per_task, i * epochs_per_task, color="green", alpha=0.06)
        for label, color, ls, (src, hkey) in SERIES:
            ep, mean, std = _curve(summaries, src, hkey, key)
            ax.plot(ep, mean, ls, color=color, lw=2.2, label=label)
            ax.fill_between(ep, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.15)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls="--", lw=1.0)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.3, 1.03)
        ax.set_ylabel(f"T{i}\ntest accuracy")
        ax.set_title(f"Task {i}: {name}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=10, ncol=2)
    axes[-1].set_xlabel("Epoch (20 per task; green = task's block in the blocked/sequential "
                        "schedule — interleaved arms train every task each round)")
    fig.suptitle("E011: schedule x consolidation training curves — blocked vs QEWC-blocked vs "
                 f"QEWC-interleaved vs interleaved (seeds {seeds}, test accuracy)", fontsize=13)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
