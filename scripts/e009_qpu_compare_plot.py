"""3-panel sim-vs-QPU comparison — forecast accuracy (R^2 = 1 - NMSE) per task.

One panel per task; grouped bars per method (QEWC / QGR) for sim-noiseless / sim-noisy / QPU
(real hardware), mean +/- SD over seeds. Shows the degradation ideal-sim -> simulated-noise ->
real-hardware. Reads results/e009_qpu_models.json (sim references) + results/e009_qpu_hardware.json.

Run:
    python scripts/e009_qpu_compare_plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COND = [("sim_noiseless", "sim (noiseless)", "#4477AA"),
        ("sim_noisy", "sim (noisy)", "#EE7733"),
        ("qpu", "QPU (hardware)", "#CC3311")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "results" / "e009_qpu_models.json")
    ap.add_argument("--hardware", type=Path, default=ROOT / "results" / "e009_qpu_hardware.json")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "e009_qpu_compare.png")
    args = ap.parse_args()

    models = json.loads(args.models.read_text())
    hw = json.loads(args.hardware.read_text()) if args.hardware.exists() else {"hardware_nmse": {}}
    hw_nmse, backend = hw.get("hardware_nmse", {}), hw.get("backend", "QPU")
    tasks, seeds, methods = models["tasks"], models["seeds"], models["methods"]

    def acc(method, cond, task):   # R^2 = 1 - NMSE across seeds
        vals = []
        for s in seeds:
            key = f"{method}:{s}"
            nm = hw_nmse.get(key, {}).get(task) if cond == "qpu" \
                else models["models"][key][f"{cond}_nmse"][task]
            if nm is not None:
                vals.append(1.0 - nm)
        a = np.asarray(vals, float)
        return (a.mean(), a.std()) if len(a) else (np.nan, 0.0)

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.8), sharey=True)
    x = np.arange(len(methods))
    w = 0.26

    for ax, task in zip(axes, tasks):
        for j, (ck, clabel, color) in enumerate(COND):
            means = [acc(m, ck, task)[0] for m in methods]
            sds = [acc(m, ck, task)[1] for m in methods]
            ax.bar(x + (j - 1) * w, means, w, yerr=sds, capsize=3, color=color,
                   edgecolor="k", label=clabel)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_title(task, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("forecast accuracy   R² = 1 − NMSE   (higher = better)")
    axes[0].legend(fontsize=8.5, loc="lower left")

    fig.suptitle(f"Simplified forecaster: simulator vs real QPU ({backend}) — "
                 f"per-task accuracy ({len(seeds)} seeds, {models['windows_per_task']} windows/task)",
                 fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
