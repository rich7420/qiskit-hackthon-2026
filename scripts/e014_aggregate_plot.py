"""Aggregate E014 multi-seed comparison results and render the headline figure.

Reads results/e014_compare_seed{42,43,44}.json, prints a mean +/- sample-SD table of ACC
and BWT per method, and draws a grouped bar chart (ACC with SD whiskers) that visually
separates the shared-readout baselines from the isolated-head (measurement-side) methods.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEEDS = (42, 43, 44)
METHOD_ORDER = ("sequential", "ewc", "qewc", "frozen_head", "free_head", "anchor_head")
LABELS = {
    "sequential": "Sequential\n(naive)",
    "ewc": "EWC",
    "qewc": "QEWC",
    "frozen_head": "Frozen θ\n+ heads (A)",
    "free_head": "Free θ\n+ heads (B)",
    "anchor_head": "Anchor θ\n+ heads (C)",
}


def _load() -> list[dict]:
    runs = []
    for s in SEEDS:
        path = RESULTS / f"e014_compare_seed{s}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    if not runs:
        sys.exit("no e014_compare_seed*.json found; run experiments/e014_compare.py first")
    return runs


def _summary(runs: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for m in METHOD_ORDER:
        accs = np.array([r["methods"][m]["ACC"] for r in runs], dtype=float)
        bwts = np.array([r["methods"][m]["BWT"] for r in runs], dtype=float)
        ddof = 1 if len(runs) > 1 else 0
        out[m] = {
            "ACC": (float(accs.mean()), float(accs.std(ddof=ddof))),
            "BWT": (float(bwts.mean()), float(bwts.std(ddof=ddof))),
        }
    return out


def main() -> None:
    runs = _load()
    summ = _summary(runs)

    print(f"E014 five-method comparison (mean +/- sample-SD over {len(runs)} seed(s))\n")
    print(f"{'method':14s} {'ACC':>16s} {'BWT':>16s}")
    for m in METHOD_ORDER:
        a_m, a_s = summ[m]["ACC"]
        b_m, b_s = summ[m]["BWT"]
        print(f"{m:14s} {a_m:.3f} +/- {a_s:.3f}   {b_m:+.3f} +/- {b_s:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(METHOD_ORDER))
    accs = [summ[m]["ACC"][0] for m in METHOD_ORDER]
    errs = [summ[m]["ACC"][1] for m in METHOD_ORDER]
    colors = ["#b0b0b0", "#a0a0a0", "#808080", "#2c7fb8", "#7fcdbb", "#e6550d"]
    ax.bar(x, accs, yerr=errs, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Average accuracy (ACC), Task-IL")
    ax.set_ylim(0.4, 1.0)
    ax.axvspan(-0.5, 2.5, color="0.95", zorder=0)
    ax.text(1.0, 0.98, "shared readout", ha="center", va="top", transform=ax.get_xaxis_transform(),
            fontsize=9, color="0.4")
    ax.text(4.0, 0.98, "measurement-side (ours)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9, color="0.4")
    for xi, (a, e) in enumerate(zip(accs, errs)):
        ax.text(xi, a + e + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("OI-QCL: measurement-side continual learning beats θ-protection")
    fig.tight_layout()

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "e014_compare.png"
    fig.savefig(out, dpi=150)
    prov = FIGURES / "e014_compare_provenance.json"
    prov.write_text(json.dumps({
        "figure": out.name, "seeds": [r["config"]["seed"] for r in runs],
        "source_results": [f"results/e014_compare_seed{r['config']['seed']}.json" for r in runs],
        "summary": {m: {"ACC_mean": summ[m]["ACC"][0], "ACC_sd": summ[m]["ACC"][1],
                        "BWT_mean": summ[m]["BWT"][0], "BWT_sd": summ[m]["BWT"][1]}
                    for m in METHOD_ORDER},
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)} and {prov.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
