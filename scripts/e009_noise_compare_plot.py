"""Plot noise robustness: noiseless vs noisy NMSE per method (naive / QEWC / QGR).

Three panels (avg / retention / plasticity), grouped bars (noiseless vs noisy) with SD error bars.
The gap between the two bars is the noise-induced degradation — which method survives noise best.

Run:
    python scripts/e009_noise_compare_plot.py
    python scripts/e009_noise_compare_plot.py --input results/e009_noise_compare_A_depol.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "e009_noise_compare_A_depol.json")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "e009_noise_compare.png")
    args = ap.parse_args()

    d = json.loads(args.input.read_text())
    rows, seeds, noise = d["rows"], d["seeds"], d["noise"]
    methods = []
    for k in rows:
        if rows[k]["method"] not in methods:
            methods.append(rows[k]["method"])

    def val(cond, m, key, sub):
        return rows.get(f"{cond}:{m}", {}).get(key, {}).get(sub, float("nan"))

    metrics = [("avg", "average NMSE"), ("retention", "retention (earlier tasks)"),
               ("plasticity", "plasticity (new task)")]
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    x = np.arange(len(methods))
    w = 0.38

    for ax, (key, title) in zip(axes, metrics):
        nl = [val("noiseless", m, key, "mean") for m in methods]
        nz = [val("noisy", m, key, "mean") for m in methods]
        ax.bar(x - w / 2, nl, w, yerr=[val("noiseless", m, key, "sd") for m in methods],
               capsize=3, color="#4477AA", edgecolor="k", label="noiseless")
        ax.bar(x + w / 2, nz, w, yerr=[val("noisy", m, key, "sd") for m in methods],
               capsize=3, color="#CC3311", edgecolor="k", label="noisy")
        for xi, (a, b) in enumerate(zip(nl, nz)):
            if b == b and a == a:
                ax.annotate(f"+{b - a:.2f}", (xi + w / 2, b), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, fontweight="bold", color="#CC3311")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() if m != "naive" else "Baseline" for m in methods])
        ax.set_title(title)
        ax.set_ylabel("NMSE  (lower = better)")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")

    ns = f"depol={noise.get('depol')}/step" if noise.get("depol") else \
         f"bit={noise.get('bit')}, phase={noise.get('phase')}/step"
    fig.suptitle(f"Noise robustness of continual-learning methods — {ns} + readout={noise.get('meas')} "
                 f"({len(seeds)} seeds, {d['epochs_per_task']} ep/task)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
