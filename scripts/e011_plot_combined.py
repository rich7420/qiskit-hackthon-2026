"""Overlay the three continual-learning regimes on one three-panel figure:

- Blocked / sequential  (E011, == E005 baseline)  -- forgets earlier tasks
- QEWC (QFI consolidation, E005)                   -- fixes forgetting WITHOUT revisiting tasks
- Interleaved (E011)                               -- upper bound: revisits every task each round

All three share the exact same learner, tasks, epoch budget (20/task) and seeds, so they are
directly comparable. This places the continual method (QEWC) between the naive blocked schedule
(lower reference) and the interleaved schedule (joint-training upper bound).

Reads results/e005_summary.json and results/e011_summary.json.

Run:
    python scripts/e011_plot_combined.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E005 = ROOT / "results" / "e005_summary.json"
E011 = ROOT / "results" / "e011_summary.json"
OUT = ROOT / "figures" / "e011_combined.png"

TASK_KEYS = ("task1", "task2", "task3")
# label, color, linestyle, (source_summary_key, history_key)
SERIES = [
    ("Blocked / sequential (baseline)", "0.45", "--", ("e011", "blocked")),
    ("QEWC (QFI consolidation, E005)", "#4477AA", "-", ("e005", "qewc")),
    ("Interleaved (upper bound)", "#228833", "-", ("e011", "interleaved")),
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

    boundaries = summaries["e011"]["config"]["task_boundaries"]
    names = [summaries["e011"]["aggregate_metrics"]["blocked"][k]["name"] for k in TASK_KEYS]
    total_epochs = summaries["e011"]["aggregate_histories"]["blocked"][-1]["epoch"]
    seeds = summaries["e011"]["seeds"]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, names), start=1):
        key = f"task{i}"
        for label, color, ls, (src, hkey) in SERIES:
            ep, mean, std = _curve(summaries, src, hkey, key)
            ax.plot(ep, mean, ls, color=color, lw=2.0, label=label)
            ax.fill_between(ep, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.18)
        for b in boundaries:
            ax.axvline(b, color="0.35", ls=":", lw=1.0)
        ax.set_xlim(0, total_epochs)
        ax.set_ylim(0.35, 1.03)
        ax.set_ylabel(f"T{i}\naccuracy")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=8.5, ncol=1)
    axes[-1].set_xlabel("Epoch (gradient step; 20 per task)")
    fig.suptitle("Blocked vs QEWC vs Interleaved — fixed 20 epochs/task\n"
                 f"MNIST -> Fashion-MNIST -> SPT/ATF, seeds {seeds} (test accuracy)",
                 fontsize=12)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
