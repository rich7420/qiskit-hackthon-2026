"""One-slide mechanism figure: theta-protection vs MPI. Minimal text, very large fonts."""

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
                                fc=fc, ec=ec, lw=2.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight="bold", color=tc)


def _arrow(ax, xy1, xy2, color, lw=3.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=26, color=color,
                                 lw=lw, zorder=2))


def _left(ax):
    ax.set_title("θ-protection  (EWC / QEWC)", fontsize=21, weight="bold", color="#b02a1f", pad=14)
    _box(ax, 0.3, 4.2, 2.0, 1.6, "x", "#eef2f7", "#333", 20, tc="#333")
    _box(ax, 3.2, 3.7, 3.0, 2.6, "U(θ)", "#fadbd8", "#b02a1f", 26, tc="#7b1a12")
    _box(ax, 7.2, 4.2, 2.4, 1.6, "one\nreadout", "#efefef", "#555", 16, tc="#333")
    _arrow(ax, (2.3, 5.0), (3.2, 5.0), "#b02a1f")
    _arrow(ax, (6.2, 5.0), (7.2, 5.0), "#b02a1f")
    ax.text(4.7, 3.2, "rewritten every task", ha="center", fontsize=15, color="#7b1a12")
    ax.text(5.0, 1.7, "✗  FORGETS", ha="center", fontsize=34, weight="bold", color="#b02a1f")


def _right(ax):
    ax.set_title("MPI  (ours)", fontsize=21, weight="bold", color="#1e7a34", pad=14)
    _box(ax, 0.3, 4.2, 1.8, 1.6, "x", "#eef2f7", "#333", 20, tc="#333")
    _box(ax, 2.9, 3.7, 2.9, 2.6, "U(θ)\nshared", "#d5f0d5", "#1e7a34", 21, tc="#12561f")
    _arrow(ax, (2.1, 5.0), (2.9, 5.0), "#1e7a34")
    for i, (hy, lab, fc) in enumerate([(6.7, "W₁", "#bfe6b8"), (4.6, "W₂", "#bcd4ef"),
                                       (2.5, "W₃", "#d3d3ec")]):
        _arrow(ax, (5.9, 5.0), (6.7, hy + 0.55), "#888", lw=2.2)
        _box(ax, 6.7, hy, 1.9, 1.1, lab, fc, "#555", 19)
    ax.text(7.65, 8.3, "one per task", ha="center", fontsize=15, color="#12561f")
    ax.text(5.0, 1.7, "✓  REMEMBERS", ha="center", fontsize=34, weight="bold", color="#1e7a34")


def main() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax in (axl, axr):
        ax.set_xlim(0, 10)
        ax.set_ylim(0.8, 9)
        ax.axis("off")
    _left(axl)
    _right(axr)
    fig.suptitle("Where does the memory live?  θ  vs  the measurement",
                 fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_mechanism.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
