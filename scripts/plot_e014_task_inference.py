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

    # ---- Left: Task-IL vs two task-agnostic routers ----
    x = np.arange(len(METHODS))
    w = 0.27
    known = [_stat(runs, m, "known_task_accuracy") for m in METHODS]
    maxc = [_stat(runs, m, "task_agnostic_accuracy") for m in METHODS]
    rout = [_stat(runs, m, "router_task_agnostic_accuracy") for m in METHODS]
    maxc_tia = [_stat(runs, m, "task_inference_accuracy") for m in METHODS]
    rout_tia = [_stat(runs, m, "router_task_inference_accuracy") for m in METHODS]
    axb.bar(x - w, [k[0] for k in known], w, yerr=[k[1] for k in known], capsize=3,
            color="#59A14F", label="Task-IL (task id given)")
    axb.bar(x, [a[0] for a in maxc], w, yerr=[a[1] for a in maxc], capsize=3,
            color="#E15759", label="max-confidence routing")
    axb.bar(x + w, [a[0] for a in rout], w, yerr=[a[1] for a in rout], capsize=3,
            color="#4C78A8", label="learned linear router")
    for xi in range(len(METHODS)):
        axb.text(xi - w, known[xi][0] + 0.012, f"{known[xi][0]:.2f}", ha="center", fontsize=8)
        axb.text(xi, maxc[xi][0] + 0.012, f"{maxc[xi][0]:.2f}", ha="center", fontsize=8)
        axb.text(xi + w, rout[xi][0] + 0.012, f"{rout[xi][0]:.2f}", ha="center", fontsize=8)
        axb.text(xi, maxc[xi][0] - 0.05, f"TIA\n{maxc_tia[xi][0]:.2f}", ha="center",
                 fontsize=6.8, color="#8a2b2c", va="top")
        axb.text(xi + w, rout[xi][0] - 0.05, f"TIA\n{rout_tia[xi][0]:.2f}", ha="center",
                 fontsize=6.8, color="#2a4b6a", va="top")
    axb.axhline(0.5, ls="--", color="0.5", lw=1)
    axb.text(2.55, 0.51, "binary chance", fontsize=7.5, color="0.4", va="bottom", ha="right")
    axb.set_xticks(x)
    axb.set_xticklabels([LABELS[m] for m in METHODS])
    axb.set_ylabel("Test accuracy")
    axb.set_ylim(0.4, 1.02)
    axb.set_title("Task id hidden at test: a learned router over p(x) nearly\n"
                  "recovers Task-IL accuracy (TIA = correct-task routing rate)", fontsize=10.5)
    axb.legend(loc="lower left", fontsize=8)
    axb.grid(axis="y", alpha=0.15)

    # ---- Right: learned-router confusion (anchor, mean over seeds, row-normalized) ----
    method = "anchor_head"
    conf = np.mean([np.array(r["methods"][method]["router_confusion"], float) for r in runs], axis=0)
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
    axc.set_title(f"Learned-router confusion — {LABELS[method]}\n"
                  "diagonal = correct routing (near-perfect)", fontsize=10.5)
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
