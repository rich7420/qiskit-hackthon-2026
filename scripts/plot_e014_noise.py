"""Bar figure: MPI noiseless vs noisy readout (depol+meas), full vs m=2 readout."""

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
    d = json.loads((ROOT / "results/e014_noise_compare.json").read_text())
    summ, cfg = d["summary"], d["config"]["noise"]
    tags = list(summ.keys())  # e.g. ["full", "m2"]
    labels = {"full": "full readout\n(16-dim, 34p)", "m2": "m=2 readout\n(4-dim, 10p)"}

    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(tags))
    w = 0.36
    clean = [summ[t]["noiseless"][0] for t in tags]
    clean_sd = [summ[t]["noiseless"][1] for t in tags]
    noisy = [summ[t]["noisy"][0] for t in tags]
    noisy_sd = [summ[t]["noisy"][1] for t in tags]
    ax.bar(x - w / 2, clean, w, yerr=clean_sd, capsize=4, color="#4C78A8", label="noiseless")
    ax.bar(x + w / 2, noisy, w, yerr=noisy_sd, capsize=4, color="#E15759", label="noisy")
    for xi, t in enumerate(tags):
        ax.text(xi - w / 2, clean[xi] + 0.012, f"{clean[xi]:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, noisy[xi] + 0.012, f"{noisy[xi]:.3f}", ha="center", fontsize=9)
        ax.text(xi, min(clean[xi], noisy[xi]) - 0.05, f"drop {summ[t]['drop']:+.3f}",
                ha="center", fontsize=8.5, color="#555")
    ax.set_xticks(x, [labels.get(t, t) for t in tags])
    ax.set_ylabel("Average accuracy (frozen-A, Task-IL)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("MPI is robust to gate + readout noise\n"
                 f"noise = depol {cfg['depol']}, meas {cfg['meas']} "
                 f"(bit {cfg['bit']}, phase {cfg['phase']}); noiseless training, noisy readout",
                 fontsize=10.5)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_noise.png"
    fig.savefig(out, dpi=155)
    (ROOT / "results/e014_noise_figure_provenance.json").write_text(
        json.dumps({"figure": out.name, "input": "results/e014_noise_compare.json"}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
