"""One-slide MPI flow figure (ours only, positive framing). Minimal text, very large fonts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def _box(ax, x, y, w, h, text, fc, ec, fs, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight="bold", color=tc)


def _arrow(ax, xy1, xy2, color, lw=3.2):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=28, color=color,
                                 lw=lw, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.0))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0.5, 8.2)
    ax.axis("off")

    green, gedge, gtext = "#d5f0d5", "#1e7a34", "#12561f"

    yc = 4.6
    _box(ax, 0.3, yc - 0.8, 1.7, 1.6, "x", "#eef2f7", "#333", 24, tc="#333")
    _box(ax, 2.8, yc - 1.3, 3.3, 2.6, "U(θ)\nshared", green, gedge, 25, tc=gtext)
    _box(ax, 6.9, yc - 1.0, 1.9, 2.0, "probs\np(x)", "#fbe0c8", "#E15759", 19, tc="#8a2b2c")
    _arrow(ax, (2.0, yc), (2.8, yc), gedge)
    _arrow(ax, (6.1, yc), (6.9, yc), gedge)

    for hy, lab, fc in [(6.05, "W₁ · task 1", "#bfe6b8"),
                        (4.05, "W₂ · task 2", "#bcd4ef"),
                        (2.05, "W₃ · task 3", "#d3d3ec")]:
        _arrow(ax, (8.9, yc), (9.8, hy + 0.55), "#888", lw=2.4)
        _box(ax, 9.8, hy, 3.4, 1.1, lab, fc, "#555", 18)

    ax.text(11.5, 7.5, "one observable per task  (frozen)", ha="center", fontsize=16,
            weight="bold", color=gtext)
    ax.text(6.6, 0.9, "✓  each task keeps its own way to measure  →  nothing to overwrite",
            ha="center", fontsize=16.5, weight="bold", color=gtext)
    fig.suptitle("MPI — one shared quantum circuit, one measurement per task",
                 fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_mechanism.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
