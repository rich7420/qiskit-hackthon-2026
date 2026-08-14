"""Plot MPI accuracy vs ansatz depth / gate count — how few gates keep the effect."""

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
QEWC_ACC = 0.819  # e014_compare 3-seed reference


def main() -> None:
    d = json.loads((ROOT / "results/e014_depth_ablation.json").read_text())
    pd = d["per_depth"]
    Ls = sorted(int(k) for k in pd)
    depth = [pd[str(L)]["cost"]["depth"] for L in Ls]
    twoq = [pd[str(L)]["cost"]["two_qubit_gates"] for L in Ls]
    theta = [pd[str(L)]["cost"]["theta_params"] for L in Ls]

    fig, ax = plt.subplots(figsize=(9, 5))
    for m, color, lab in [("frozen_head", "#59A14F", "MPI frozen θ (A)"),
                          ("anchor_head", "#E15759", "MPI anchor θ (C)")]:
        mean = np.array([pd[str(L)]["acc"][m]["ACC_mean"] for L in Ls])
        sd = np.array([pd[str(L)]["acc"][m]["ACC_sd"] for L in Ls])
        ax.plot(Ls, mean, "o-", color=color, lw=2, label=lab)
        ax.fill_between(Ls, mean - sd, mean + sd, color=color, alpha=0.15)

    ax.axhline(QEWC_ACC, ls=":", color="#4C78A8", lw=1.5)
    ax.text(Ls[-1], QEWC_ACC - 0.005, f"QEWC (L=12) {QEWC_ACC:.2f}", ha="right", va="top",
            fontsize=8.5, color="#4C78A8")
    ax.set_xticks(Ls)
    ax.set_xlabel("ansatz depth L  (layers of RY/RZ + CNOT ladder)")
    ax.set_ylabel("Average accuracy (ACC), Task-IL")
    ax.set_ylim(0.75, 1.0)
    ax.grid(alpha=0.15)
    # second x-axis: 2-qubit gate count
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(Ls)
    secax.set_xticklabels([f"{q}cx\n{dp}dp" for q, dp in zip(twoq, depth)], fontsize=8)
    secax.set_xlabel("ansatz 2-qubit gates (cx) / circuit depth (dp)  "
                     "— AmplitudeEmbedding state-prep counted separately", fontsize=8.5)
    ax.set_title("MPI accuracy is nearly flat in circuit depth — the effect is in the readout\n"
                 "L=4 (12 CNOTs, 3× fewer) matches L=12; even L=1 (3 CNOTs) already beats QEWC by ~0.10",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_depth_ablation.png"
    fig.savefig(out, dpi=155)
    prov = ROOT / "results/e014_depth_ablation_figure_provenance.json"
    prov.write_text(json.dumps({"figure": out.name,
                                "input": "results/e014_depth_ablation.json"}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
