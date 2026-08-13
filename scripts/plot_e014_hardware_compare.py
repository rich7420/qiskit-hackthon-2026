"""OI-QCL vs QEWC on real IBM hardware — per-task + average accuracy (frozen-A vs QFI theta)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
KEYS = ("task1", "task2", "task3")


def main() -> None:
    o = json.loads((ROOT / "results/e014_hardware_ibm_marrakesh_alltasks.json").read_text())
    q = json.loads((ROOT / "results/e014_hardware_ibm_marrakesh_qewc_alltasks.json").read_text())
    backend = o["backend"]
    shots = o["config"]["shots"]
    n_test = o["config"]["n_test"]
    names = [o["per_task"][k]["name"] for k in KEYS]
    oi = [o["per_task"][k]["backend_acc"] for k in KEYS]
    qe = [q["per_task"][k]["backend_acc"] for k in KEYS]
    oi.append(float(np.mean(oi)))
    qe.append(float(np.mean(qe)))
    labels = [f"T{i+1}\n{n}" for i, n in enumerate(names)] + ["average"]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, oi, w, color="#59A14F", label="OI-QCL (measurement-side)")
    ax.bar(x + w / 2, qe, w, color="#4C78A8", label="QEWC (θ-protection)")
    for xi, (a, b) in enumerate(zip(oi, qe)):
        ax.text(xi - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=8.5,
                weight="bold" if xi == len(labels) - 1 else "normal")
        ax.text(xi + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=8.5,
                weight="bold" if xi == len(labels) - 1 else "normal")
    ax.axhline(0.5, ls="--", color="0.6", lw=1)
    ax.text(len(labels) - 0.5, 0.51, "binary chance", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.axvline(len(KEYS) - 0.5, color="0.8", lw=1)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Test accuracy on QPU (Task-IL)")
    ax.set_ylim(0.2, 1.05)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Continual learning on real IBM hardware ({backend}) — OI-QCL vs QEWC\n"
                 f"avg {oi[-1]:.2f} vs {qe[-1]:.2f} (gap +{oi[-1]-qe[-1]:.2f}); QEWC forgets T1 "
                 "and its QFI mis-anchors under noise\n"
                 f"n_test={n_test}/task, {shots} shots, 1 seed — per-task numbers are noisy; "
                 "the average is the signal", fontsize=9.5)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_hardware_compare.png"
    fig.savefig(out, dpi=155)
    (ROOT / "results/e014_hardware_compare_figure_provenance.json").write_text(json.dumps(
        {"figure": out.name, "backend": backend,
         "oiqcl": "results/e014_hardware_ibm_marrakesh_alltasks.json",
         "qewc": "results/e014_hardware_ibm_marrakesh_qewc_alltasks.json"}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
