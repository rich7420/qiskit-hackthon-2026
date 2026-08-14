"""Per-task MPI accuracy: noiseless sim vs Aer(shots) vs real QPU (frozen-A, all 3 tasks).

Reads results/e014_hardware_aer_alltasks.json (sim + Aer) and, if present, any real-backend
results/e014_hardware_<backend>_alltasks.json to add the QPU bars.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TASK_KEYS = ("task1", "task2", "task3")


def main() -> None:
    aer = json.loads((ROOT / "results/e014_hardware_aer_alltasks.json").read_text())
    names = [aer["per_task"][k]["name"] for k in TASK_KEYS]
    sim = [aer["per_task"][k]["sim_acc"] for k in TASK_KEYS]
    aer_acc = [aer["per_task"][k]["backend_acc"] for k in TASK_KEYS]
    shots = aer["config"]["shots"]

    # optional real-QPU file(s)
    qpu = None
    qpu_name = None
    for f in sorted(glob.glob(str(ROOT / "results/e014_hardware_*_alltasks.json"))):
        d = json.loads(Path(f).read_text())
        if d["backend"] != "aer":
            qpu = [d["per_task"][k]["backend_acc"] for k in TASK_KEYS]
            qpu_name = d["backend"]

    series = [("noiseless sim", sim, "#4C78A8"),
              (f"Aer ({shots} shots)", aer_acc, "#59A14F")]
    if qpu is not None:
        series.append((f"QPU: {qpu_name}", qpu, "#E15759"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TASK_KEYS))
    w = 0.8 / len(series)
    for i, (lab, vals, col) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(x + off, vals, w, color=col, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", color="0.6", lw=1)
    ax.set_xticks(x, [f"T{i+1}\n{n}" for i, n in enumerate(names)], fontsize=9)
    ax.set_ylabel("Test accuracy (frozen-A, Task-IL)")
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9)
    sub = "sim vs Aer" if qpu is None else f"sim vs Aer vs {qpu_name}"
    ax.set_title("MPI on IBM stack — per-task readout, shared frozen backbone (%s)\n"
                 "L=%d, readout=%d qubits, n_test=%d, %d shots; measurement-side readout is "
                 "shot-noise robust" % (sub, aer["config"]["layers"],
                 len(aer["config"]["readout_qubits"]), aer["config"]["n_test"], shots), fontsize=10)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_hardware.png"
    fig.savefig(out, dpi=155)
    (ROOT / "results/e014_hardware_figure_provenance.json").write_text(json.dumps(
        {"figure": out.name, "aer": "results/e014_hardware_aer_alltasks.json",
         "qpu": qpu_name}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}" + (f" (incl. QPU {qpu_name})" if qpu else " (Aer only)"))


if __name__ == "__main__":
    main()
