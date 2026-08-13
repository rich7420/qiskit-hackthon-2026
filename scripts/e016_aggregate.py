"""Aggregate e016 QGR-classification across seeds (mean +/- sample SD) and redraw the figure.

Reads results/e016_qgr_classification_seed{S}.json for each seed, writes
results/e016_qgr_classification_summary.json and figures/e016_qgr_compare_multiseed.png.

Run:
    python scripts/e016_aggregate.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "e016_qgr_classification_summary.json"
OUT_FIG = ROOT / "figures" / "e016_qgr_compare_multiseed.png"

METRICS = ("retention_earlier_acc", "plasticity_final_acc", "avg_earlier_forgetting",
           "avg_final_acc")
STYLE = {
    "baseline": ("Baseline (naive seq.)", "0.55"),
    "ewc": ("EWC (classical Fisher)", "#4477AA"),
    "qewc": ("QEWC (quantum Fisher)", "#228833"),
    "replay": ("replay (stores raw data)", "#EE6677"),
    "qgr_seed": ("QGR-seed (quantum gen.)", "#CC3311"),
    "qgr_inversion": ("QGR-inversion (data-free)", "#AA3377"),
}


def _ms(v):
    a = np.asarray(v, float)
    return {"mean": round(float(a.mean()), 4),
            "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    per = {}  # method -> metric -> [values across seeds]
    methods = None
    for s in args.seeds:
        data = json.loads((RESULTS / f"e016_qgr_classification_seed{s}.json").read_text())
        methods = methods or list(data["metrics"].keys())
        for m, mt in data["metrics"].items():
            per.setdefault(m, {k: [] for k in METRICS})
            for k in METRICS:
                per[m][k].append(mt[k])

    summary = {m: {k: _ms(per[m][k]) for k in METRICS} for m in methods}
    OUT_JSON.write_text(json.dumps(
        {"experiment": "e016_qgr_classification_multiseed", "seeds": args.seeds,
         "summary": summary}, indent=2) + "\n")

    # Figure: retention (mean +/- SD) horizontal bars, methods sorted by retention.
    ms = [m for m in STYLE if m in summary]
    order = sorted(ms, key=lambda m: summary[m]["retention_earlier_acc"]["mean"])
    ys = np.arange(len(order))
    means = [summary[m]["retention_earlier_acc"]["mean"] for m in order]
    sds = [summary[m]["retention_earlier_acc"]["sd"] for m in order]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.1})
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(ys, means, xerr=sds, color=[STYLE[m][1] for m in order], edgecolor="k",
            linewidth=0.6, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_yticks(ys)
    ax.set_yticklabels([STYLE[m][0] for m in order], fontsize=9)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("retention — mean earlier-task (T1,T2) final acc  (higher → better)")
    ax.set_title(f"QGR vs EWC/QEWC on classification — retention "
                 f"(mean ± SD, {len(args.seeds)} seeds)\nplasticity = 1.000 for all methods")
    for y, m in zip(ys, order):
        v = summary[m]["retention_earlier_acc"]
        ax.annotate(f"{v['mean']:.3f}±{v['sd']:.3f}", (v['mean'] + v['sd'], y),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=9,
                    fontweight="bold" if m.startswith("qgr") else "normal")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")

    print(f"Wrote {OUT_JSON}\nWrote {OUT_FIG}\n")
    print(f"=== e016 multi-seed ({len(args.seeds)} seeds; acc, higher better) ===")
    print(f"  {'method':14s} {'retention':>16s} {'plasticity':>16s} {'avg':>16s}")
    for m in methods:
        s = summary[m]
        r, p, a = s["retention_earlier_acc"], s["plasticity_final_acc"], s["avg_final_acc"]
        print(f"  {m:14s} {r['mean']:.3f}±{r['sd']:.3f}   {p['mean']:.3f}±{p['sd']:.3f}   "
              f"{a['mean']:.3f}±{a['sd']:.3f}")


if __name__ == "__main__":
    main()
