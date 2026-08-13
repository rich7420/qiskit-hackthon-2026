"""Catastrophic-forgetting curves under noise — per-task test NMSE vs epoch (naive / QEWC / QGR).

3-panel per-task figure in the E009 forgetting style, for the noise experiment. Each method's test
NMSE is tracked across the whole sequential run; the jumps at the task boundaries are the
forgetting. Default = the noisy condition; use --condition noiseless for the reference.

Run:
    python scripts/e009_noise_curves_plot.py
    python scripts/e009_noise_curves_plot.py --condition noiseless
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# method: (label, color, linestyle, lw) — matches the E009 house style
STYLE = {
    "naive": ("naive (no CL)", "0.5", "--", 1.8),
    "qewc": ("QEWC (quantum Fisher)", "#228833", "-.", 2.1),
    "qgr": ("QGR (quantum generative replay)", "#CC3311", "-", 2.6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "e009_noise_compare_A_depol.json")
    ap.add_argument("--condition", default="noisy", choices=["noisy", "noiseless"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.input.read_text())
    rows, seeds, noise = d["rows"], d["seeds"], d["noise"]
    tasks, ept = d["tasks"], d["epochs_per_task"]
    total = ept * len(tasks)
    cond = args.condition
    out = args.output or ROOT / "figures" / f"e009_noise_curves_{cond}.png"

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.6, 2.7 * len(tasks)), sharex=True)

    for i, (ax, task) in enumerate(zip(axes, tasks), start=1):
        start = max((i - 1) * ept, 1)   # show each task only from its own training onset (skip epoch 0)
        for m, (label, color, ls, lw) in STYLE.items():
            row = rows.get(f"{cond}:{m}")
            if not row:
                continue
            ep = np.array(row["curves"]["epochs"])
            mean = np.array(row["curves"]["nmse"][task]["mean"])
            sd = np.array(row["curves"]["nmse"][task]["sd"])
            k = ep >= start
            ax.plot(ep[k], mean[k], ls, color=color, lw=lw, label=label)
            ax.fill_between(ep[k], np.clip(mean[k] - sd[k], 0, None), mean[k] + sd[k],
                            color=color, alpha=0.13)
        for b in range(ept, total, ept):
            ax.axvline(b, color="0.4", ls=":", lw=1.3)   # dotted task boundaries
        ax.set_xlim(0, total)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"T{i}\ntest NMSE")
        ax.set_title(f"Task {i}: {task}", loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.18)

    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[-1].set_xlabel(f"Epoch (sequential training; {ept} per task, dotted = task boundary)")
    ns = f"depol={noise.get('depol')}" if noise.get("depol") else \
         f"bit+phase={noise.get('bit')}/{noise.get('phase')}"
    tag = "UNDER NOISE" if cond == "noisy" else "(noiseless reference)"
    fig.suptitle(f"Catastrophic forgetting {tag} — naive vs QEWC vs QGR "
                 f"({ns} + readout={noise.get('meas')}, seeds {seeds}, test NMSE)", fontsize=10.5)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
