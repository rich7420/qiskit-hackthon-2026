"""Fair task-agnostic comparison: no method uses a task oracle at test.

baseline / EWC / QEWC use a single shared readout (never needed a task id); OI-QCL is
evaluated with the learned linear router over p_θ(x) (task id inferred, not given). This
puts every method on equal footing. Bars = final average test accuracy, 3-seed mean +/- SD.
Hollow caps on the OI-QCL bars mark the Task-IL ceiling (task id given) for reference.

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

    # (label, task-agnostic accuracy series, color, optional Task-IL ceiling series)
    bars = [
        ("Sequential\n(naive)", [r["methods"]["sequential"]["ACC"] for r in cmp_runs], "#b0b0b0", None),
        ("EWC", [r["methods"]["ewc"]["ACC"] for r in cmp_runs], "#a0a0a0", None),
        ("QEWC", [r["methods"]["qewc"]["ACC"] for r in cmp_runs], "#808080", None),
        ("OI-QCL (A)\n+ router", [r["methods"]["frozen_head"]["router_task_agnostic_accuracy"] for r in ti_runs],
         "#59A14F", [r["methods"]["frozen_head"]["known_task_accuracy"] for r in ti_runs]),
        ("OI-QCL (C)\n+ router", [r["methods"]["anchor_head"]["router_task_agnostic_accuracy"] for r in ti_runs],
         "#E15759", [r["methods"]["anchor_head"]["known_task_accuracy"] for r in ti_runs]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bars))
    for xi, (label, series, color, ceil) in enumerate(bars):
        m, sd = _mean_sd(series)
        ax.bar(xi, m, 0.62, yerr=sd, capsize=4, color=color, zorder=3)
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
    ax.set_ylim(0.4, 1.02)
    ax.axvspan(-0.5, 2.5, color="0.96", zorder=0)
    ax.text(1.0, 0.99, "shared readout (single head)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9, color="0.4")
    ax.text(3.5, 0.99, "measurement-side + learned router", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9, color="0.4")
    ax.set_title("Fair comparison — no method gets the task id at test\n"
                 "OI-QCL routes with a linear classifier over p(x); baselines use one shared head",
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()

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
