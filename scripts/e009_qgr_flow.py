"""QGR mechanism — a small, honest schematic for the pitch (freeze -> rollout -> rehearse).

Draws the three-stage Quantum Generative Replay loop exactly as we implement it in
experiments/e009_continual_forecasting.py (train_method, method="qgr") and src/e009_qtsf.py
(rollout). The middle panel is a REAL rollout of the deployed QGR model loaded from
results/e009_qpu_models.json — we freeze the trained forecaster and autoregressively generate a
synthetic old-task series, so the sparkline is genuine output, not a drawing.

Run:
    python scripts/e009_qgr_flow.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e009_qtsf import make_forecaster, rollout  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# project palette: freeze=blue, rollout=QGR-red, rehearse=green
ACCENT = {"freeze": "#4477AA", "rollout": "#CC3311", "rehearse": "#228833"}


def real_rollout(models_path: Path, model_key="qgr:42", task="damped_shm", gen_len=24, window_idx=0):
    """Load the deployed QGR model and autoregressively roll out a genuine synthetic series."""
    d = json.loads(models_path.read_text())
    m = d["models"][model_key]
    qn, _, _ = make_forecaster(n_qubits=4, n_layers=d["n_layers"], seq_len=d["seq_len"], **d["ansatz"])
    circ_w, head_w = np.array(m["circ_w"]), np.array(m["head_w"])
    seed_window = np.array(d["test_subset"][task]["X"][window_idx])
    seq = rollout(qn, circ_w, head_w, seed_window, gen_len)   # [seed (seq_len) | generated (gen_len)]
    n_params = int(circ_w.size + head_w.size)
    return seed_window, np.asarray(seq), n_params, task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=RESULTS / "e009_qpu_models.json")
    ap.add_argument("--task", default="damped_shm")
    ap.add_argument("--window-idx", type=int, default=0)
    ap.add_argument("--output", type=Path, default=FIGURES / "e009_qgr_flow.png")
    args = ap.parse_args()

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    seed_window, seq, n_params, task = real_rollout(args.models, task=args.task, window_idx=args.window_idx)
    L = len(seed_window)

    plt.rcParams.update({"font.size": 11})
    fig = plt.figure(figsize=(11.8, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "Quantum Generative Replay (QGR): rehearse the frozen forecaster's own dynamics",
            ha="center", va="center", fontsize=14.5, fontweight="bold")
    ax.text(0.5, 0.89, f"the old task's memory lives in {n_params} circuit parameters — no stored data",
            ha="center", va="center", fontsize=10.5, color="0.35", style="italic")

    # (x_center, width, key, number, header, body-lines)
    boxes = [
        (0.150, 0.235, "freeze", "1", "FREEZE",
         ["After Task $k$, snapshot the", "trained forecaster as a", r"frozen generator  $\Phi_{\Theta_k}$"]),
        (0.500, 0.330, "rollout", "2", "ROLLOUT  (autoregressive)",
         [r"seed window $\rightarrow$ predict next", r"$\rightarrow$ append $\rightarrow$ slide $\rightarrow$ repeat"]),
        (0.850, 0.235, "rehearse", "3", "REHEARSE",
         [r"Train Task $k{+}1$ on new data", "+ replayed synthetic old data",
          r"$L=\mathrm{MSE}_{\mathrm{new}}+\mathrm{MSE}_{\mathrm{replay}}$"]),
    ]
    y0, y1 = 0.28, 0.80
    for xc, w, key, num, head, body in boxes:
        acc = ACCENT[key]
        ax.add_patch(FancyBboxPatch((xc - w / 2, y0), w, y1 - y0,
                                    boxstyle="round,pad=0.006,rounding_size=0.022",
                                    linewidth=1.9, edgecolor=acc,
                                    facecolor=mcolors.to_rgba(acc, 0.08), zorder=2))
        # number badge (marker stays circular regardless of axes aspect) + header
        bx = xc - w / 2 + 0.026
        by = y1 - 0.055
        ax.plot([bx], [by], marker="o", markersize=19, color=acc, zorder=4)
        ax.text(bx, by, num, ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=5)
        ax.text(bx + 0.032, by, head, ha="left", va="center", color=acc, fontsize=11.5, fontweight="bold")
        # body text
        ty = y1 - 0.135
        for line in body:
            ax.text(xc, ty, line, ha="center", va="center", fontsize=10.3, color="0.15")
            ty -= 0.058

    # ---- real rollout sparkline inside the ROLLOUT box ----
    inset = fig.add_axes([0.378, 0.315, 0.244, 0.155])
    xs = np.arange(len(seq))
    inset.plot(xs[:L], seq[:L], "-", color="#333333", lw=2.0, marker="o", ms=2.5)
    inset.plot(xs[L - 1:], seq[L - 1:], "--", color=ACCENT["rollout"], lw=2.0)
    inset.axvline(L - 1, color="0.6", lw=0.8, ls=":")
    inset.text(0.02, 0.90, "seed", transform=inset.transAxes, fontsize=8, color="#333333", fontweight="bold")
    inset.text(0.98, 0.90, "generated", transform=inset.transAxes, fontsize=8, ha="right",
               color=ACCENT["rollout"], fontweight="bold")
    inset.set_title(f"real QGR rollout ({task})", fontsize=8.5, color="0.3", pad=2)
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_edgecolor("0.7")

    # ---- forward arrows between the stages ----
    for xa, xb in [(0.150 + 0.235 / 2, 0.500 - 0.330 / 2), (0.500 + 0.330 / 2, 0.850 - 0.235 / 2)]:
        ax.add_patch(FancyArrowPatch((xa + 0.004, 0.545), (xb - 0.004, 0.545),
                                     arrowstyle="-|>", mutation_scale=20, lw=2.2, color="0.35"))

    # ---- loop-back arrow: repeats at each task boundary; generators accumulate ----
    ax.add_patch(FancyArrowPatch((0.850, y0 - 0.002), (0.150, y0 - 0.002),
                                 connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                 mutation_scale=18, lw=1.6, color="0.55", linestyle=(0, (5, 3))))
    ax.text(0.5, 0.075, "repeat at every task boundary — one frozen generator kept per past task",
            ha="center", va="center", fontsize=9.5, color="0.4")
    ax.text(0.5, 0.028, "function-space rehearsal  (vs QEWC, which anchors parameters in parameter space)",
            ha="center", va="center", fontsize=9.5, color=ACCENT["rehearse"], fontweight="bold")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}  (seed_len={L}, gen_len={len(seq) - L}, params={n_params})")


if __name__ == "__main__":
    main()
