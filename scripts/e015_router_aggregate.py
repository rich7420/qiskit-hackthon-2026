"""Aggregate e015 task-agnostic router seeds: oracle vs centroid vs learned-router.

Reads results/e015_router_seed*.json, averages over seeds, prints a table, and writes
figures/e015_router.png (grouped bars: Task-IL oracle NMSE vs router NMSE per method,
annotated with router task-inference accuracy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ("frozen_head", "free_head", "anchor_head")
LABELS = {"frozen_head": "Frozen theta (A)", "free_head": "Free theta (B)",
          "anchor_head": "Anchor theta (C)"}


def main() -> None:
    files = sorted(RESULTS.glob("e015_router_seed*.json"))
    if not files:
        sys.exit("no results/e015_router_seed*.json")
    runs = [json.loads(f.read_text()) for f in files]
    seeds = [r["config"]["seed"] for r in runs]
    print(f"aggregating {len(runs)} seeds: {seeds}\n")

    def col(m, path):
        vals = []
        for r in runs:
            d = r["methods"][m]
            for k in path:
                d = d[k]
            vals.append(d)
        return float(np.mean(vals)), float(np.std(vals))

    print(f"{'method':16s} {'oracle NMSE':>14s} {'centroid TIA/NMSE':>20s} {'router TIA/NMSE':>20s}")
    agg = {}
    for m in METHODS:
        okn = col(m, ["known_task_avg_nmse"])
        cti = col(m, ["centroid", "task_inference_accuracy"])
        cnm = col(m, ["centroid", "task_agnostic_nmse"])
        rti = col(m, ["router", "task_inference_accuracy"])
        rnm = col(m, ["router", "task_agnostic_nmse"])
        agg[m] = {"oracle": okn, "router_tia": rti, "router_nmse": rnm}
        print(f"{m:16s} {okn[0]:.3f}±{okn[1]:.3f}   "
              f"{cti[0]:.2f}/{cnm[0]:.3f}±{cnm[1]:.3f}   "
              f"{rti[0]:.2f}/{rnm[0]:.3f}±{rnm[1]:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n(matplotlib unavailable: {e}; skipped figure)")
        return

    x = np.arange(len(METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    oracle = [agg[m]["oracle"][0] for m in METHODS]
    router = [agg[m]["router_nmse"][0] for m in METHODS]
    router_err = [agg[m]["router_nmse"][1] for m in METHODS]
    ax.bar(x - w / 2, oracle, w, label="Task-IL oracle (id known)", color="#2a7")
    ax.bar(x + w / 2, router, w, yerr=router_err, capsize=3,
           label="Learned router (id hidden)", color="#48a")
    for i, m in enumerate(METHODS):
        tia = agg[m]["router_tia"][0]
        ax.annotate(f"TIA={tia:.2f}", (i + w / 2, router[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("Task-agnostic test NMSE (lower = better)")
    ax.set_title("e015 OI-QCL forecasting: remove the test task id\n"
                 "frozen/anchor degrade gracefully (heads interchangeable); free-theta does not")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "e015_router.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
