"""Bures-Budgeted QEWC (Quantum-State Trust-Region QEWC).

Replaces the always-on QEWC penalty with a per-task QFI drift budget. For each earlier task j
store (theta*_j, F_j, B_j) and define the trust region

    R_j(theta) = (theta - theta*_j)^T F_j (theta - theta*_j) <= B_j     (diagonal QFI)

which is a Bures/Fubini-Study ball in the local regime. A candidate optimizer step s0 (the
actual Adam step, not the raw gradient) is accepted freely while every R_j stays within budget;
otherwise it is scaled along its own direction to the largest beta in [0, 1] that keeps all
budgets satisfied (exact quadratic root -- the step-scaling MVP). Sweeping B_j traces a
quantum-state-drift-budget stability-plasticity frontier (B->0 frozen, B->inf naive).

This is a rehearsal-free quantum-state trust region, distinct from soft QEWC (always-on penalty)
and from quantum GEM (linear old-task gradient constraints on stored samples).
"""

from __future__ import annotations

import numpy as np


class BuresBudget:
    """Per-task QFI trust region with optimizer-aware step scaling."""

    def __init__(self, budget: float):
        self.budget = float(budget)
        self.anchors: list[tuple[np.ndarray, np.ndarray]] = []  # (theta*_flat, fisher_flat)

    def add(self, theta_star, fisher_diag) -> None:
        self.anchors.append((np.asarray(theta_star, float).ravel().copy(),
                             np.asarray(fisher_diag, float).ravel().copy()))

    def r_value(self, theta) -> float:
        """Max R_j(theta) over stored tasks (0 if none)."""
        if not self.anchors:
            return 0.0
        t = np.asarray(theta, float).ravel()
        return max(float(np.sum(F * (t - ts) ** 2)) for ts, F in self.anchors)

    def scale_step(self, theta_before, s0):
        """Return (beta, intervened): largest beta<=1 with all R_j(theta+beta*s0) <= budget.

        Solves a beta^2 + b beta + c <= 0 per task with a=s0^T F s0, b=2 delta^T F s0,
        c=R_j(theta)-B, and takes the min feasible beta across tasks.
        """
        if not self.anchors:
            return 1.0, False
        tb = np.asarray(theta_before, float).ravel()
        s = np.asarray(s0, float).ravel()
        beta = 1.0
        for ts, F in self.anchors:
            d = tb - ts
            a = float(np.sum(F * s * s))
            b = 2.0 * float(np.sum(F * d * s))
            c = float(np.sum(F * d * d)) - self.budget
            if c > 0.0:                       # already outside -> cannot move outward
                bj = 0.0
            elif a < 1e-18:                   # no curvature along s
                bj = 1.0 if b <= 0.0 else min(1.0, max(0.0, -c / b))
            else:
                root = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
                bj = max(0.0, min(1.0, root))
            beta = min(beta, bj)
        return beta, beta < 1.0
