"""Render MPI's accuracy metrics as a clean formula image (mathtext, white background)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

ROWS = [
    (r"$\mathrm{Acc}_j \;=\; \dfrac{1}{N_j}\,\sum_i \mathbf{1}\!\left[\,\hat{y}_i = y_i\,\right]$",
     "per-task test accuracy"),
    (r"$\mathrm{ACC} \;=\; \dfrac{1}{T}\,\sum_{j} R_{T,j}$",
     "average accuracy  (final row of the accuracy matrix)"),
    (r"$\mathrm{BWT} \;=\; \dfrac{1}{T-1}\,\sum_{j<T}\!\left(R_{T,j} - R_{j,j}\right)$",
     "backward transfer  ( < 0 = forgetting; MPI-A = 0 )"),
]


def main() -> None:
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(11, 6.2))
    for ax, (formula, label) in zip(axes, ROWS):
        ax.axis("off")
        ax.text(0.02, 0.62, formula, fontsize=30, va="center", ha="left", color="#1a1a1a")
        ax.text(0.02, 0.06, label, fontsize=14, va="bottom", ha="left", color="#777",
                style="italic")
    fig.text(0.5, 0.965, "MPI — accuracy metrics", fontsize=17, weight="bold", ha="center")
    fig.text(0.5, 0.02,
             r"$R_{i,j}$ = test accuracy on task $j$ after training through task $i$   ·   "
             r"$\hat{y}=\arg\max_c\,[W^{(t)}p_\theta(x)]_c$",
             fontsize=12.5, ha="center", color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_metrics.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
