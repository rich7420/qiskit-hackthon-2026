"""Visualize the max-confidence task-inference results (task id removed at test).

Left: per-method accuracy with the task id given (Task-IL) vs inferred by max-confidence
routing (task-agnostic), 3-seed mean +/- SD.  Right: task-routing confusion matrix (mean
over seeds, row-normalized) for the anchor variant -- shows *why* accuracy drops: MNIST/
Fashion get confused and the over-confident SPT head (task 3) grabs many of them.
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
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"
SEEDS = (42, 43, 44)
METHODS = ("frozen_head", "free_head", "anchor_head")
LABELS = {"frozen_head": "Frozen θ (A)", "free_head": "Free θ (B)", "anchor_head": "Anchor θ (C)"}
TASK_TICKS = ("T1 MNIST", "T2 Fashion", "T3 SPT")


def _load():
    return [json.loads((ROOT / f"results/e014_task_inference_seed{s}.json").read_text())
            for s in SEEDS]


def _stat(runs, method, key):
    a = np.array([r["methods"][method][key] for r in runs], float)
    return a.mean(), a.std(ddof=1)


def main() -> None:
    runs = _load()
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Left: Task-IL vs task-agnostic accuracy ----
    x = np.arange(len(METHODS))
    w = 0.36
    known = [_stat(runs, m, "known_task_accuracy") for m in METHODS]
    agn = [_stat(runs, m, "task_agnostic_accuracy") for m in METHODS]
    tia = [_stat(runs, m, "task_inference_accuracy") for m in METHODS]
    axb.bar(x - w / 2, [k[0] for k in known], w, yerr=[k[1] for k in known], capsize=4,
            color="#59A14F", label="Task-IL (task id given)")
    axb.bar(x + w / 2, [a[0] for a in agn], w, yerr=[a[1] for a in agn], capsize=4,
            color="#E15759", label="Task-agnostic (max-confidence routing)")
    for xi, (k, a, t) in enumerate(zip(known, agn, tia)):
        axb.text(xi - w / 2, k[0] + 0.015, f"{k[0]:.2f}", ha="center", fontsize=9)
        axb.text(xi + w / 2, a[0] + 0.015, f"{a[0]:.2f}", ha="center", fontsize=9)
        axb.annotate(f"−{k[0]-a[0]:.2f}", (xi, (k[0] + a[0]) / 2), ha="center", va="center",
                     fontsize=8.5, color="#555")
        axb.text(xi + w / 2, a[0] - 0.06, f"TIA {t[0]:.2f}", ha="center", fontsize=7.5,
                 color="#8a2b2c")
    axb.axhline(0.5, ls="--", color="0.5", lw=1)
    axb.text(2.55, 0.51, "binary chance", fontsize=7.5, color="0.4", va="bottom", ha="right")
    axb.set_xticks(x)
    axb.set_xticklabels([LABELS[m] for m in METHODS])
    axb.set_ylabel("Test accuracy")
    axb.set_ylim(0.4, 1.02)
    axb.set_title("Removing the task id hurts: max-confidence routing\n"
                  "(TIA = fraction routed to the correct task's head)", fontsize=10.5)
    axb.legend(loc="lower left", fontsize=8.5)
    axb.grid(axis="y", alpha=0.15)

    # ---- Right: routing confusion matrix (anchor, mean over seeds, row-normalized) ----
    method = "anchor_head"
    conf = np.mean([np.array(r["methods"][method]["task_confusion"], float) for r in runs], axis=0)
    conf_norm = conf / conf.sum(axis=1, keepdims=True)
    im = axc.imshow(conf_norm, cmap="Reds", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            axc.text(j, i, f"{conf_norm[i, j]:.2f}", ha="center", va="center", fontsize=11,
                     color="white" if conf_norm[i, j] > 0.5 else "#333")
    axc.set_xticks(range(3), TASK_TICKS, fontsize=8.5)
    axc.set_yticks(range(3), TASK_TICKS, fontsize=8.5)
    axc.set_xlabel("routed to head (inferred task)")
    axc.set_ylabel("true task")
    axc.set_title(f"Task-routing confusion — {LABELS[method]}\n"
                  "diagonal = correct routing; T3 head over-grabs T1/T2", fontsize=10.5)
    fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04, label="routing fraction")

    fig.suptitle("OI-QCL under task-agnostic evaluation (task id hidden), seeds 42/43/44",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_task_inference.png"
    fig.savefig(out, dpi=155)
    prov = ROOT / "results/e014_task_inference_figure_provenance.json"
    prov.write_text(json.dumps({"figure": out.name, "seeds": list(SEEDS),
        "inputs": [f"results/e014_task_inference_seed{s}.json" for s in SEEDS]}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} and {prov.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
