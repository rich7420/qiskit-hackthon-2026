"""Plot e015 per-task trajectory panels (T1/T2/T3 vs epoch) from e015_trajectory_seed*.json.

Style matches the e009/e014 reference figures: one panel per task, x = sequential epoch, each
method a mean line +/- std band over seeds, dotted task boundaries, and a light shaded band on
the region where THAT panel's task is being trained.  Each task's curve starts only at its
training onset (None before), so panels are blank until their task begins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

METHODS = ("sequential", "qewc", "frozen_head", "free_head", "anchor_head")
STYLE = {
    "sequential":  dict(label="naive (no CL)",              color="#7f7f7f", ls="--",  lw=2.0),
    "qewc":        dict(label="QEWC (quantum Fisher)",      color="#2ca02c", ls="-.",  lw=2.0),
    "frozen_head": dict(label="Frozen theta + heads (A)",   color="#1f77b4", ls="-",   lw=2.2),
    "free_head":   dict(label="Free theta + heads (B)",     color="#ff7f0e", ls="-",   lw=2.0),
    "anchor_head": dict(label="Anchor theta + heads (C)",   color="#d62728", ls="-",   lw=2.2),
}


def _stack(seed_hists, method, task_name, n_epochs_total):
    """(n_seeds, n_epochs) array with np.nan where a seed has None (pre-onset)."""
    rows = []
    for h in seed_hists:
        seq = h["methods"][method][task_name]
        row = [np.nan if v is None else float(v) for v in seq]
        row = row[:n_epochs_total] + [np.nan] * (n_epochs_total - len(row))
        rows.append(row)
    return np.array(rows)


def main() -> None:
    files = sorted(RESULTS.glob("e015_trajectory_seed*.json"))
    if not files:
        sys.exit("no results/e015_trajectory_seed*.json")
    runs = [json.loads(f.read_text()) for f in files]
    seeds = [r["config"]["seed"] for r in runs]
    tasks = runs[0]["config"]["tasks"]
    epochs = runs[0]["config"]["epochs_per_task"]
    T = len(tasks)
    total = T * epochs
    print(f"plotting {len(runs)} seeds {seeds}: tasks={tasks}, {epochs} epochs/task")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        sys.exit(f"matplotlib unavailable: {e}")

    x = np.arange(1, total + 1)
    fig, axes = plt.subplots(T, 1, figsize=(11, 3.2 * T), sharex=True)
    for i, (ax, task) in enumerate(zip(axes, tasks)):
        # light shade where THIS task is being trained
        ax.axvspan(i * epochs, (i + 1) * epochs, color="#2ca02c", alpha=0.06, zorder=0)
        for k in range(1, T):  # dotted task boundaries
            ax.axvline(k * epochs, color="0.5", ls=":", lw=1.0, zorder=1)
        for m in METHODS:
            arr = _stack(runs, m, task, total)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
            defined = ~np.isnan(mean)
            s = STYLE[m]
            ax.plot(x[defined], mean[defined], color=s["color"], ls=s["ls"], lw=s["lw"],
                    label=s["label"], zorder=3)
            ax.fill_between(x[defined], (mean - std)[defined], (mean + std)[defined],
                            color=s["color"], alpha=0.15, zorder=2)
        ax.set_ylabel(f"T{i + 1}\ntest NMSE")
        ax.set_title(f"Task {i + 1}: {task}", loc="left", fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, total)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(ncol=2, fontsize=9, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("Epoch (sequential training; shaded = task being trained)")
    fig.suptitle(f"e015 OI-QCL on forecasting (Task-IL) — naive vs QEWC vs OI-QCL "
                 f"A/B/C\n{' -> '.join(tasks)}, seeds {seeds} (test NMSE, lower=better)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "e015_trajectory.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
