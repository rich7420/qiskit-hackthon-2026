"""Small one-slide mechanism figure: theta-protection vs MPI (where does memory live?).

Big, explicit fonts for a projected slide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def _box(ax, x, y, w, h, text, fc, ec="#333", fs=15, lw=2.2, weight="bold", tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight=weight, color=tc)


def _arrow(ax, xy1, xy2, color="#333", lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=20, color=color,
                                 lw=lw, zorder=2))


def _tasks(ax):
    for ty, lab in [(8.3, "Task 1"), (6.0, "Task 2"), (3.7, "Task 3")]:
        _box(ax, 0.2, ty - 0.6, 1.9, 1.2, lab, "#eef2f7", fs=13, weight="normal")


def _left(ax):
    ax.set_title("θ-protection  (EWC / QEWC)", fontsize=17, weight="bold", color="#b02a1f", pad=12)
    _tasks(ax)
    _box(ax, 3.0, 3.1, 3.4, 4.1, "U(θ)\nREWRITTEN\nevery task", "#fadbd8", ec="#b02a1f",
         fs=15, tc="#7b1a12")
    for ty in (8.3, 6.0, 3.7):
        _arrow(ax, (2.1, ty), (3.0, 5.15), color="#b02a1f")
    _box(ax, 7.2, 4.4, 2.6, 1.6, "ONE shared\nreadout", "#efefef", fs=13.5, weight="normal")
    _arrow(ax, (6.4, 5.2), (7.2, 5.2))
    ax.text(5.0, 2.4, "memory lives in θ  →  it drifts", ha="center", fontsize=14.5,
            weight="bold", color="#b02a1f")
    ax.text(8.5, 3.5, "✗ old tasks\noverwritten", ha="center", va="top", fontsize=14,
            weight="bold", color="#b02a1f")


def _right(ax):
    ax.set_title("MPI  (Measurement-based Parameter Isolation) — ours", fontsize=17,
                 weight="bold", color="#1e7a34", pad=12)
    _tasks(ax)
    _box(ax, 3.0, 3.1, 2.8, 4.1, "U(θ)\nSHARED\n(frozen)", "#d5f0d5", ec="#1e7a34", fs=15, tc="#12561f")
    for ty in (8.3, 6.0, 3.7):
        _arrow(ax, (2.1, ty), (3.0, 5.15), color="#1e7a34")
    _box(ax, 6.0, 4.5, 1.3, 1.3, "probs\np(x)", "#fbe0c8", ec="#E15759", fs=12.5, tc="#8a2b2c")
    _arrow(ax, (5.8, 5.15), (6.0, 5.15))
    for i, (hy, lab, fc) in enumerate([(8.1, "W₁", "#bfe6b8"), (5.85, "W₂", "#bcd4ef"),
                                       (3.6, "W₃", "#d3d3ec")]):
        _arrow(ax, (7.3, 5.15), (8.0, hy + 0.4), color="#888", lw=1.8)
        _box(ax, 8.0, hy, 1.8, 0.8, f"{lab} · task {i+1}", fc, fs=12.5, weight="normal")
    ax.text(8.9, 9.3, "one frozen\nobservable per task", ha="center", fontsize=12, color="#444")
    ax.text(5.0, 2.4, "memory lives in the measurement", ha="center", fontsize=14.5,
            weight="bold", color="#1e7a34")
    ax.text(5.0, 1.3, "✓ each task keeps its own way to measure  →  it remembers",
            ha="center", fontsize=14, weight="bold", color="#1e7a34")


def main() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax in (axl, axr):
        ax.set_xlim(0, 10.2)
        ax.set_ylim(0.5, 10)
        ax.axis("off")
    _left(axl)
    _right(axr)
    fig.suptitle("Where should a quantum model's continual memory live?",
                 fontsize=19, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_mechanism.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
