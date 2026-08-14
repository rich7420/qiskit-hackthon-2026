"""Fair task-agnostic comparison: no method uses a task oracle at test.

baseline / EWC / QEWC use a single shared readout (never needed a task id); MPI is
evaluated with the learned linear router over p_θ(x) (task id inferred, not given). This
puts every method on equal footing. Bars = final average test accuracy, 3-seed mean +/- SD.
Hollow caps on the MPI bars mark the Task-IL ceiling (task id given) for reference.

Sources: results/e014_compare_seed*.json (ACC of sequential/ewc/qewc, already task-agnostic)
and results/e014_task_inference_seed*.json (router_task_agnostic_accuracy of A/C).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SEEDS = (42, 43, 44)


def _mean_sd(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def main() -> None:
    cmp_runs = [json.loads((ROOT / f"results/e014_compare_seed{s}.json").read_text()) for s in SEEDS]
    ti_runs = [json.loads((ROOT / f"results/e014_task_inference_seed{s}.json").read_text()) for s in SEEDS]
    cb_runs = [json.loads((ROOT / f"results/e014_classical_baseline_seed{s}.json").read_text()) for s in SEEDS]

    # (label, task-agnostic accuracy series, color, Task-IL ceiling series, hatch)
    bars = [
        ("Sequential\n(naive)", [r["methods"]["sequential"]["ACC"] for r in cmp_runs], "#b0b0b0", None, None),
        ("EWC", [r["methods"]["ewc"]["ACC"] for r in cmp_runs], "#a0a0a0", None, None),
        ("QEWC", [r["methods"]["qewc"]["ACC"] for r in cmp_runs], "#808080", None, None),
        ("MPI (A)\n+ router", [r["methods"]["frozen_head"]["router_task_agnostic_accuracy"] for r in ti_runs],
         "#59A14F", [r["methods"]["frozen_head"]["known_task_accuracy"] for r in ti_runs], None),
        ("MPI (C)\n+ router", [r["methods"]["anchor_head"]["router_task_agnostic_accuracy"] for r in ti_runs],
         "#E15759", [r["methods"]["anchor_head"]["known_task_accuracy"] for r in ti_runs], None),
        ("Classical\nmulti-head\n(raw, no circuit)", [r["task_agnostic_accuracy"] for r in cb_runs],
         "#9467bd", [r["taskIL_ACC"] for r in cb_runs], "///"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(bars))
    for xi, (label, series, color, ceil, hatch) in enumerate(bars):
        m, sd = _mean_sd(series)
        ax.bar(xi, m, 0.62, yerr=sd, capsize=4, color=color, zorder=3, hatch=hatch,
               edgecolor="white" if hatch else color)
        ax.text(xi, m + sd + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")
        if ceil is not None:
            cm, _ = _mean_sd(ceil)
            ax.hlines(cm, xi - 0.31, xi + 0.31, color="#333", lw=1.6, linestyle=(0, (2, 1)), zorder=4)
            ax.text(xi, cm + 0.006, f"Task-IL {cm:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333")

    ax.axhline(0.5, ls="--", color="0.6", lw=1)
    ax.text(len(bars) - 0.55, 0.508, "binary chance", fontsize=8, color="0.4", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("Average test accuracy (no task oracle)")
    ax.set_ylim(0.4, 1.03)
    ax.axvspan(-0.5, 2.5, color="0.96", zorder=0)
    ax.axvspan(4.5, 5.5, color="#f3eefa", zorder=0)
    ax.text(1.0, 1.005, "θ-protection (single head)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color="0.4")
    ax.text(3.5, 1.005, "MPI + router (quantum)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color="0.4")
    ax.text(5.0, 1.005, "classical control", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color="#6a4a9a")
    ax.set_title("Fair comparison (no task oracle) + matched classical control\n"
                 "MPI beats θ-protection; a classical multi-head on the raw input wins —\n"
                 "no quantum advantage on this classically-easy benchmark",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_fair_compare.png"
    fig.savefig(out, dpi=155)
    prov = ROOT / "results/e014_fair_compare_provenance.json"
    prov.write_text(json.dumps({"figure": out.name, "seeds": list(SEEDS),
        "inputs": [f"results/e014_compare_seed{s}.json" for s in SEEDS]
                  + [f"results/e014_task_inference_seed{s}.json" for s in SEEDS]}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} and {prov.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
