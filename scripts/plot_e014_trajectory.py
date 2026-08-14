"""Three-panel per-task test-accuracy curves for E014 (e009/e013 style).

MPI (measurement-side) vs shared-readout baselines across the sequential
MNIST -> Fashion -> SPT/ATF run, with task boundaries and three-seed mean/SD bands.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
TASK_KEYS = ("task1", "task2", "task3")
TASK_TITLES = ("MNIST 0/1", "Fashion-MNIST 0/1", "SPT/ATF phases")
TEMPLATE = "results/e014_trajectory_seed{seed}.json"
OUTPUT = ROOT / "figures/e014_trajectory.png"
PROVENANCE = ROOT / "results/e014_trajectory_figure_provenance.json"

# method -> (label, color, linestyle)
METHODS = {
    "sequential": ("Sequential (naive)", "#777777", "--"),
    "ewc": ("EWC (classical Fisher)", "#F28E2B", (0, (4, 1, 1, 1))),
    "qewc": ("QEWC (QFI consolidation, E005)", "#4C78A8", ":"),
    "frozen_head": ("MPI frozen θ + heads (A)", "#59A14F", "-"),
    "free_head": ("MPI free θ + heads (B)", "#B07AA1", "-."),
    "anchor_head": ("MPI anchor θ + heads (C)", "#E15759", "-"),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    runs, paths = {}, []
    for seed in SEEDS:
        p = ROOT / TEMPLATE.format(seed=seed)
        runs[seed] = json.loads(p.read_text(encoding="utf-8"))
        paths.append(p)
    return runs, paths


def _series(run, method, task_key):
    """Return (epochs, values) with None -> nan so matplotlib breaks the line."""
    rows = run["histories"][method]
    ep = np.asarray([r["epoch"] for r in rows], dtype=int)
    val = np.asarray([np.nan if r["test_accuracy"][task_key] is None
                      else r["test_accuracy"][task_key] for r in rows], dtype=float)
    return ep, val


def main() -> None:
    runs, input_paths = _load()
    epochs_per_task = int(runs[SEEDS[0]]["training"]["epochs_per_task"])
    ep_ref, _ = _series(runs[SEEDS[0]], "sequential", "task1")
    total = int(ep_ref[-1])
    boundaries = [epochs_per_task * i for i in range(1, len(TASK_KEYS))]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    for ti, (ax, key, title) in enumerate(zip(axes, TASK_KEYS, TASK_TITLES), start=1):
        # A task is undefined before it is first trained: draw each panel only from the
        # epoch its own task begins (T1: 0, T2: 20, T3: 40) so no method shows a
        # meaningless pre-task (chance-level) curve.
        start = (ti - 1) * epochs_per_task
        drawn = ep_ref >= start
        for method, (label, color, ls) in METHODS.items():
            stack = np.vstack([_series(runs[s], method, key)[1] for s in SEEDS])
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1) if len(SEEDS) > 1 else np.zeros_like(mean)
            ax.plot(ep_ref[drawn], mean[drawn], color=color, linestyle=ls,
                    linewidth=2.0, label=label)
            ax.fill_between(ep_ref[drawn], np.clip(mean[drawn] - sd[drawn], 0, 1),
                            np.clip(mean[drawn] + sd[drawn], 0, 1), color=color, alpha=0.13)
        for b in boundaries:
            ax.axvline(b, color="0.5", linestyle=":", linewidth=1.0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(f"T{ti}\naccuracy")
        ax.set_ylim(0.35, 1.03)
        ax.set_xlim(0, total)
        ax.grid(True, alpha=0.15)
        if ti == 1:
            ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    axes[-1].set_xlabel(f"Epoch (gradient step; {epochs_per_task} per task)")
    fig.suptitle("MPI (measurement-side) vs θ-protection — MNIST → Fashion → SPT/ATF\n"
                 f"seeds {list(SEEDS)} (test accuracy; MPI uses task id / Task-IL — "
                 "fair no-oracle comparison: figures/e014_fair_compare.png)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    PROVENANCE.write_text(json.dumps({
        "figure": OUTPUT.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "methods": list(METHODS),
        "inputs": {p.name: _digest(p) for p in input_paths},
    }, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} and {PROVENANCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
