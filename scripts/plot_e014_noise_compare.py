"""Grouped bars: all-6-method noiseless vs noisy ACC (depol+meas), reduced-config 3-seed."""

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
METHODS = ("sequential", "ewc", "qewc", "frozen_head", "free_head", "anchor_head")
LABELS = {"sequential": "Sequential", "ewc": "EWC", "qewc": "QEWC",
          "frozen_head": "Frozen θ (A)", "free_head": "Free θ (B)", "anchor_head": "Anchor θ (C)"}


def _load(pat):
    return [json.loads((ROOT / pat.format(s=s)).read_text()) for s in SEEDS]


def _ms(runs, m):
    a = np.array([r["methods"][m]["ACC"] for r in runs])
    return a.mean(), a.std(ddof=1)


def main() -> None:
    nl = _load("results/e014_cmp_red_noiseless_seed{s}.json")
    ny = _load("results/e014_cmp_red_noisy_seed{s}.json")
    cfg = ny[0]["config"]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(METHODS))
    w = 0.38
    for xi, m in enumerate(METHODS):
        a, asd = _ms(nl, m)
        b, bsd = _ms(ny, m)
        c1 = "#4C78A8" if m in ("sequential", "ewc", "qewc") else "#59A14F"
        ax.bar(xi - w / 2, a, w, yerr=asd, capsize=3, color=c1, alpha=0.55,
               label="noiseless" if xi == 0 else None)
        ax.bar(xi + w / 2, b, w, yerr=bsd, capsize=3, color=c1,
               label="noisy" if xi == 0 else None, hatch="//", edgecolor="white")
        ax.text(xi, max(a, b) + max(asd, bsd) + 0.015, f"−{a-b:.2f}", ha="center",
                fontsize=8, color="#8a2b2c" if (a - b) > 0.05 else "#555")
    ax.axhline(0.5, ls="--", color="0.6", lw=1)
    ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=9)
    ax.set_ylabel("Average accuracy (ACC), Task-IL")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.axvspan(-0.5, 2.5, color="0.96", zorder=0)
    ax.text(1.0, 0.98, "θ-protection", ha="center", va="top", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color="0.4")
    ax.text(4.0, 0.98, "OI-QCL (measurement-side)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8.5, color="0.4")
    ax.set_title("All methods under noise (depol %.2f, meas %.2f) — OI-QCL is the most robust\n"
                 "QEWC degrades most (−0.10; its QFI is a pure-state property); OI-QCL A/C drop ~0.01\n"
                 "reduced config (%dL/%dep/n=%d, 3 seeds; default.mixed ~250x slower than pure)"
                 % (cfg["noise"]["depol"], cfg["noise"]["meas"], cfg["layers"],
                    cfg["epochs_per_task"], cfg["n_train_per_task"]), fontsize=10)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_noise_compare.png"
    fig.savefig(out, dpi=155)
    (ROOT / "results/e014_noise_compare_figure_provenance.json").write_text(json.dumps(
        {"figure": out.name, "seeds": list(SEEDS),
         "inputs": [f"results/e014_cmp_red_{k}_seed{s}.json" for k in ("noiseless", "noisy")
                    for s in SEEDS]}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
