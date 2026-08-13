"""e016 comparison figure: QGR (seed / inversion) vs EWC / QEWC on the classification benchmark.

Two panels: (left) retention vs plasticity scatter (upper-right = best, since higher accuracy is
better); (right) per-task final test accuracy grouped bars. Reads the seed-42 result JSON.

Run:
    python scripts/e016_qgr_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "e016_qgr_classification_seed42.json"
OUT = ROOT / "figures" / "e016_qgr_compare.png"

STYLE = {
    "baseline": ("Baseline (naive seq.)", "0.55", "X"),
    "ewc": ("EWC (classical Fisher)", "#4477AA", "o"),
    "qewc": ("QEWC (quantum Fisher)", "#228833", "s"),
    "replay": ("replay (stores raw data)", "#EE6677", "D"),
    "qgr_seed": ("QGR-seed (quantum gen.)", "#CC3311", "*"),
    "qgr_inversion": ("QGR-inversion (data-free)", "#AA3377", "P"),
}


def main() -> None:
    data = json.loads(RESULT.read_text())
    metrics = data["metrics"]
    tasks = data["config"]["tasks"]
    seed = data["config"]["seed"]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- retention ranking (plasticity saturates at 1.000 for every method: T3 is perfectly
    #      separable, so retention on the two earlier tasks is the only discriminating axis) ----
    ms = [m for m in STYLE if m in metrics]
    order = sorted(ms, key=lambda m: metrics[m]["retention_earlier_acc"])
    ys = np.arange(len(order))
    rets = [metrics[m]["retention_earlier_acc"] for m in order]
    axL.barh(ys, rets, color=[STYLE[m][1] for m in order], edgecolor="k", linewidth=0.6)
    axL.set_yticks(ys)
    axL.set_yticklabels([STYLE[m][0] for m in order], fontsize=9)
    axL.set_xlim(0.5, 0.9)
    axL.set_xlabel("retention — mean earlier-task (T1,T2) final acc  (higher → better)")
    axL.set_title("Retention ranking  (plasticity = 1.000 for all methods)")
    for y, m in zip(ys, order):
        axL.annotate(f"{metrics[m]['retention_earlier_acc']:.3f}",
                     (metrics[m]["retention_earlier_acc"], y), textcoords="offset points",
                     xytext=(4, 0), va="center", fontsize=9,
                     fontweight="bold" if m.startswith("qgr") else "normal")
    axL.grid(axis="x", alpha=0.25)

    # ---- per-task final accuracy grouped bars ----
    keys = ("task1", "task2", "task3")
    x = np.arange(len(keys))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        vals = [metrics[m]["tasks"][k]["test_final"] for k in keys]
        axR.bar(x + i * w, vals, w, label=STYLE[m][0], color=STYLE[m][1],
                edgecolor="k", linewidth=0.5)
    axR.set_xticks(x + 0.4 - w / 2)
    axR.set_xticklabels([f"T{i+1}\n{tasks[i]}" for i in range(3)], fontsize=9)
    axR.set_ylabel("final test accuracy")
    axR.set_ylim(0.4, 1.0)
    axR.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.6)
    axR.set_title("Final accuracy per task (T1,T2 earlier; T3 newest)")
    axR.legend(fontsize=8, ncol=1, loc="lower left")
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Quantum Generative Replay vs EWC/QEWC — classification benchmark (seed {seed})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
