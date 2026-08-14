"""Small one-slide mechanism figure: theta-protection vs MPI (where does memory live?)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def _box(ax, x, y, w, h, text, fc, ec="#333", fs=10, lw=1.6, weight="normal", tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight=weight, color=tc)


def _arrow(ax, xy1, xy2, color="#333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=13, color=color,
                                 lw=lw, zorder=2))


def _panel(ax, title, good):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", color="#2f6b2f" if good else "#8a2b2c")

    # three tasks on the left
    for i, (ty, lab) in enumerate([(8.2, "Task 1"), (6.0, "Task 2"), (3.8, "Task 3")]):
        _box(ax, 0.3, ty - 0.55, 1.7, 1.1, lab, "#eef2f7", fs=9)

    if not good:
        # theta-protection: all tasks REWRITE one circuit -> one drifting readout -> forgets
        _box(ax, 3.2, 3.3, 3.0, 3.9, "U(θ)\nrewritten\nevery task", "#fde2e1", ec="#c0392b",
             fs=11, weight="bold")
        for ty in (8.2, 6.0, 3.8):
            _arrow(ax, (2.0, ty), (3.2, 5.2), color="#c0392b")
        _box(ax, 7.0, 4.5, 2.4, 1.5, "one shared\nreadout", "#f2f2f2", fs=9.5)
        _arrow(ax, (6.2, 5.25), (7.0, 5.25))
        ax.text(8.2, 3.3, "✗ old tasks\noverwritten\n(forgetting)", ha="center", va="top",
                fontsize=10, color="#c0392b", weight="bold")
        ax.text(4.7, 2.6, "memory lives in θ → drifts", ha="center", fontsize=9.5, color="#8a2b2c")
    else:
        # MPI: shared frozen circuit + per-task frozen observables
        _box(ax, 3.2, 3.3, 2.6, 3.9, "U(θ)\nshared\nrepresentation", "#e7f6e7", ec="#2f6b2f",
             fs=10.5, weight="bold")
        for ty in (8.2, 6.0, 3.8):
            _arrow(ax, (2.0, ty), (3.2, 5.2), color="#2f6b2f")
        _box(ax, 6.0, 4.7, 1.1, 1.0, "probs\np(x)", "#fde8cf", ec="#E15759", fs=8.5, weight="bold")
        _arrow(ax, (5.8, 5.2), (6.0, 5.2))
        for i, (hy, lab, fc) in enumerate([(8.0, "W₁", "#c7e9c0"), (5.9, "W₂", "#c6dbef"),
                                           (3.8, "W₃", "#dadaeb")]):
            _arrow(ax, (7.1, 5.2), (7.9, hy + 0.3), color="#999", lw=1.2)
            _box(ax, 7.9, hy, 1.5, 0.62, f"{lab}  (task {i+1})", fc, fs=8.5)
        ax.text(6.7, 8.9, "one observable per task\n(frozen)", ha="center", fontsize=8.5, color="#444")
        ax.text(4.6, 2.7, "memory lives in the measurement", ha="center", fontsize=9.5, color="#2f6b2f")
        ax.text(6.9, 1.7, "✓ each task keeps its own\nway to measure → remembers",
                ha="center", va="top", fontsize=9.5, color="#2f6b2f", weight="bold")


def main() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _panel(axl, "θ-protection  (EWC / QEWC)", good=False)
    _panel(axr, "MPI — Measurement-based Parameter Isolation  (ours)", good=True)
    fig.suptitle("Where should a quantum model's continual memory live?",
                 fontsize=13.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(exist_ok=True)
    out = FIG / "e014_mechanism.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
