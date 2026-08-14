"""Deep vs shallow on real hardware — simplified vs baseline architecture, QPU accuracy per task.

Per task, R^2 (= 1 - NMSE) for QEWC & QGR under the simplified-arch QPU run (transpiled 2q-depth 17)
vs the baseline-arch QPU run (deep 64-CNOT). Same backend. Shows the gate-ablation payoff on-device:
the shallow circuit keeps accuracy while the deep baseline is degraded by hardware noise. Adds the
sim-noiseless reference (both archs train to similar ideal accuracy — so the on-hardware gap is noise).

Run:
    python scripts/e009_qpu_arch_compare_plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def _load(models_p, hw_p):
    m = json.loads(models_p.read_text())
    h = json.loads(hw_p.read_text()) if hw_p.exists() else {"hardware_nmse": {}, "transpiled_2q_depth": "?"}
    return m, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simplified-models", type=Path, default=R / "e009_qpu_models.json")
    ap.add_argument("--simplified-hw", type=Path, default=R / "e009_qpu_hardware.json")
    ap.add_argument("--baseline-models", type=Path, default=R / "e009_qpu_models_baseline.json")
    ap.add_argument("--baseline-hw", type=Path, default=R / "e009_qpu_hardware_baseline.json")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "e009_qpu_arch_compare.png")
    args = ap.parse_args()

    sm, sh = _load(args.simplified_models, args.simplified_hw)
    bm, bh = _load(args.baseline_models, args.baseline_hw)
    tasks, seeds, methods = sm["tasks"], sm["seeds"], sm["methods"]
    backend = sh.get("backend", "QPU")

    def acc_hw(hwd, method, task):
        vals = [1.0 - hwd.get(f"{method}:{s}", {}).get(task, np.nan) for s in seeds]
        a = np.asarray([v for v in vals if v == v], float)
        return (a.mean(), a.std()) if len(a) else (np.nan, 0.0)

    def acc_simnl(models, method, task):
        vals = [1.0 - models["models"][f"{method}:{s}"]["sim_noiseless_nmse"][task] for s in seeds]
        a = np.asarray(vals, float)
        return a.mean(), a.std()

    # (key, mean_fn, label, color)
    BARS = [
        ("simnl", lambda m, t: acc_simnl(sm, m, t), "sim (noiseless)", "#4477AA"),
        ("shallow", lambda m, t: acc_hw(sh.get("hardware_nmse", {}), m, t),
         f"simplified QPU (2q-depth {sh.get('transpiled_2q_depth')})", "#CC3311"),
        ("deep", lambda m, t: acc_hw(bh.get("hardware_nmse", {}), m, t),
         f"baseline QPU (2q-depth {bh.get('transpiled_2q_depth')})", "0.35"),
    ]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 4.6), sharey=True)
    x = np.arange(len(methods))
    w = 0.26

    for ax, task in zip(axes, tasks):
        for j, (key, fn, label, color) in enumerate(BARS):
            raw = [fn(m, task) for m in methods]
            means = [max(mn, -0.58) for mn, _ in raw]     # clip floor so a collapsed bar stays readable
            sds = [min(sd, 0.45) for _, sd in raw]         # cap error bars (deep circuit is very noisy)
            bars = ax.bar(x + (j - 1) * w, means, w, yerr=sds, capsize=3, color=color,
                          edgecolor="k", label=label)
            for (mn, _), xi in zip(raw, x):
                if mn < -0.05:                             # annotate the true (off-scale) collapsed R^2
                    ax.annotate(f"{mn:.1f}", (xi + (j - 1) * w, -0.55), ha="center", va="bottom",
                                fontsize=7, color=color, fontweight="bold")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-0.6, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_title(task, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("forecast accuracy   R² = 1 − NMSE   (higher = better)")
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Deep vs shallow circuit on real hardware ({backend}) — "
                 f"simplified survives, baseline is noise-degraded ({len(seeds)} seeds)", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
