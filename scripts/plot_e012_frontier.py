"""Plot the E012 stability-plasticity frontiers (retention vs adaptation as lambda varies).

Each curve is one consolidation method; points are the shared lambda grid averaged over seeds.
The lambda=0.1 marker is the single operating point E008 used to rank all methods -- the plot
shows why that point misranks QEWC/Readout-QFI, which reach higher retention at slightly larger
lambda (i.e. on their own frontier knee).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
METHODS = ["output_cfi", "zz_cfi", "uniform_xyz", "mof_ewc", "readout_qewc", "qewc"]
LABEL = {
    "output_cfi": "EWC (output-CFI)", "zz_cfi": "Joint-ZZ CFI",
    "uniform_xyz": "Uniform XYZ", "mof_ewc": "MOF-EWC",
    "readout_qewc": "Readout-QFI", "qewc": "QEWC (full-QFI)",
}
COLOR = {
    "output_cfi": "#1f77b4", "zz_cfi": "#2ca02c", "uniform_xyz": "#9467bd",
    "mof_ewc": "#8c564b", "readout_qewc": "#ff7f0e", "qewc": "#d62728",
}
E008_LAMBDA = 0.1


def frontier_curve(data_by_seed, method, split):
    """Return (lambdas, mean retention, mean adaptation) averaged over seeds per lambda."""
    lambdas = [pt["lambda"] for pt in data_by_seed[SEEDS[0]]["frontier"][method]]
    ret, adapt = [], []
    for i in range(len(lambdas)):
        r = [data_by_seed[s]["frontier"][method][i]["metrics"][split]["task1_final_retention"] for s in SEEDS]
        a = [data_by_seed[s]["frontier"][method][i]["metrics"][split]["task2_final_adaptation"] for s in SEEDS]
        ret.append(np.mean(r))
        adapt.append(np.mean(a))
    return np.array(lambdas), np.array(ret), np.array(adapt)


def main() -> None:
    data = {s: json.loads((ROOT / f"results/e012_frontier_seed{s}.json").read_text()) for s in SEEDS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for m in METHODS:
            lam, ret, adapt = frontier_curve(data, m, split)
            ax.plot(adapt, ret, "-o", color=COLOR[m], ms=4, lw=1.6,
                    label=LABEL[m], alpha=0.9)
            k = int(np.argmin(np.abs(lam - E008_LAMBDA)))
            ax.plot(adapt[k], ret[k], "s", color=COLOR[m], ms=11,
                    mfc="none", mew=2.0)
        ax.set_xlabel("Task-2 adaptation (plasticity) →")
        ax.set_title(f"{split} split")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()  # high plasticity on the left, so upper-left = best frontier
    axes[0].set_ylabel("Task-1 retention (stability) →")
    axes[1].plot([], [], "s", color="k", ms=11, mfc="none", mew=2.0,
                 label=f"E008 point (λ={E008_LAMBDA})")
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "E012 axis C: stability-plasticity frontiers (seeds 42/43/44 mean).  "
        "Upper-left dominates.  Squares = E008's single λ=0.1 ranking point.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "figures/e012_frontier.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
