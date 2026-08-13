"""Plot OI-QCL accuracy vs readout width — how small the per-task observable can be."""

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


def main() -> None:
    d = json.loads((ROOT / "results/e014_readout_ablation.json").read_text())
    pw = d["per_width"]
    qewc = d["config"]["qewc_reference"]
    ms = sorted(int(k) for k in pw)
    params = [pw[str(m)]["head_params_per_task"] for m in ms]
    mean = np.array([pw[str(m)]["ACC_mean"] for m in ms])
    sd = np.array([pw[str(m)]["ACC_sd"] for m in ms])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(ms, mean, "o-", color="#59A14F", lw=2, label="OI-QCL frozen θ (A)")
    ax.fill_between(ms, mean - sd, mean + sd, color="#59A14F", alpha=0.15)
    for m, y, p in zip(ms, mean, params):
        ax.annotate(f"{y:.2f}\n{p}p", (m, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5)
    ax.axhline(qewc, ls=":", color="#4C78A8", lw=1.5)
    ax.text(ms[-1], qewc - 0.006, f"QEWC {qewc:.2f}", ha="right", va="top",
            fontsize=8.5, color="#4C78A8")
    ax.set_xticks(ms)
    ax.set_xticklabels([f"{m} qubit\n({2**m}-dim, {2*2**m+2}p/task)" for m in ms], fontsize=8.5)
    ax.set_xlabel("readout width (qubits measured) — AmplitudeEmbedding + L=12 kept fixed")
    ax.set_ylabel("Average accuracy (ACC), Task-IL")
    ax.set_ylim(0.78, 1.0)
    ax.grid(alpha=0.15)
    ax.set_title("A tiny per-task observable still beats θ-protection\n"
                 "even a 6-param readout on 1 qubit gets 0.90 — the shared circuit carries the "
                 "task info,\nso the head is a genuine lightweight observable, not a per-task model",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_readout_ablation.png"
    fig.savefig(out, dpi=155)
    prov = ROOT / "results/e014_readout_ablation_figure_provenance.json"
    prov.write_text(json.dumps({"figure": out.name,
                                "input": "results/e014_readout_ablation.json"}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
