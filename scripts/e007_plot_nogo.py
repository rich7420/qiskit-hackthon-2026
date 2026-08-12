"""Plot the e007 directional NO-GO: three falsifying lines of evidence in one figure.

(a) Global QFI is near-isotropic (effective rank ~ full) vs the more anisotropic CFI.
(b) The directional coefficient beta2 flips sign across seeds -> noise, not signal.
(c) Exact Bures state distance (global/readout) does not beat Euclidean parameter movement.

Reads results/e007_nogo_summary.json. Run:
    python scripts/e007_plot_nogo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e007_nogo_summary.json"
OUT = ROOT / "figures" / "e007_nogo.png"


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    P = s["evidence_1_isotropy"]["n_weights"]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) effective rank: global QFI vs CFI, out of P
    gr = s["evidence_1_isotropy"]["global_qfi_effective_rank"]
    cr = s["evidence_1_isotropy"]["cfi_effective_rank"]
    a.bar(["Global QFI", "CFI\n(task-relevant)"], [gr, cr],
          color=["tab:gray", "tab:blue"], edgecolor="k")
    a.axhline(P, ls="--", color="red", lw=1.3, label=f"full rank = {P} (isotropic)")
    a.set_ylabel("effective rank")
    a.set_title("(a) Global QFI is near-isotropic")
    a.legend(fontsize=9, loc="lower right")
    for i, v in enumerate([gr, cr]):
        a.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=10)

    # (b) per-seed beta2 sign grid (predictors x seeds)
    preds = ["Q_global", "Q_readout", "Q_cfi"]
    seeds = [r["seed"] for r in s["per_seed"]]
    signs = np.array([[np.sign(r[f"{p}_beta2"]) for r in s["per_seed"]] for p in preds])
    cmap = ListedColormap(["#d9534f", "#f0f0f0", "#5cb85c"])  # -, 0, +
    b.imshow(signs, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    b.set_xticks(range(len(seeds)), seeds)
    b.set_yticks(range(len(preds)), ["global", "readout", "CFI"])
    b.set_xlabel("seed")
    b.set_title(r"(b) Directional $\beta_2$ sign flips across seeds")
    for i in range(len(preds)):
        for j in range(len(seeds)):
            b.text(j, i, "+" if signs[i, j] > 0 else "−", ha="center", va="center",
                   fontsize=13, fontweight="bold")

    # (c) predictor correlation with forgetting
    e3 = s["evidence_3_exact_bures_vs_euclid"]
    names = ["Euclidean\n||Δθ||", "Bures\nglobal", "Bures\nreadout"]
    vals = [e3["corr_euclid"], e3["corr_bures_global"], e3["corr_bures_readout"]]
    c.bar(names, vals, color=["tab:green", "tab:gray", "tab:blue"], edgecolor="k")
    c.set_ylim(0.7, 0.95)
    c.set_ylabel("correlation with forgetting")
    c.set_title("(c) Exact state distance < Euclidean")
    for i, v in enumerate(vals):
        c.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("E007 — forgetting is radial (step size), not directional (quantum geometry): "
                 "directional Q-MemGuard is NO-GO", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
