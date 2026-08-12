"""H2 — adaptive step-size control for continual learning (e007 pivot).

Since forgetting is radial (step magnitude), not directional, the question becomes *when* to
constrain, not *where*. The Adaptive Norm Trust Region calibrates a per-old-task step budget
eps_j and pulls a candidate update back to the boundary only when it exceeds the budget:

    ||dtheta_raw||^2 <= eps_j  ->  accept unchanged (zero constraint, full plasticity)
    otherwise                 ->  dtheta_safe = sqrt(eps_j / ||dtheta_raw||^2) * dtheta_raw

No new attenuation hyperparameter: eps_j comes from calibration, the pullback is exact.
"""

from __future__ import annotations

import numpy as np


def calibrate_step_budget(accuracy_fn, theta_star, weight_shape, *, scales, n_dirs: int = 4,
                          acc_tol: float = 0.02, seed: int = 0) -> float:
    """Largest per-step ||dtheta||^2 whose random perturbation keeps old-task acc within tol."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta_star, dtype=float).ravel()
    acc0 = accuracy_fn(theta.reshape(weight_shape))
    budget = scales[0] ** 2
    for s in scales:
        drops = []
        for _ in range(n_dirs):
            u = rng.standard_normal(theta.size)
            u /= np.linalg.norm(u)
            drops.append(acc0 - accuracy_fn((theta + s * u).reshape(weight_shape)))
        if np.mean(drops) <= acc_tol:
            budget = s ** 2
        else:
            break
    return float(budget)


class AdaptiveNormGuard:
    """Event-triggered norm trust region; budget = tightest calibrated old-task budget."""

    def __init__(self):
        self.eps: float | None = None

    def add_budget(self, eps: float) -> None:
        self.eps = eps if self.eps is None else min(self.eps, eps)

    def redirect(self, delta_raw):
        """Return (delta, intervened). Safe steps pass through untouched."""
        d = np.asarray(delta_raw, dtype=float)
        if self.eps is None:
            return d, False
        n2 = float(np.sum(d**2))
        if n2 <= self.eps:
            return d, False
        return d * np.sqrt(self.eps / n2), True


class AdaptiveTrustRegion:
    """Cumulative trust region: keep theta within a calibrated ball ||theta - theta*_j||^2 <= eps_j.

    Forgetting is cumulative, so the constraint is on total displacement from each old optimum,
    not per-step size. Inside every ball -> free learning; outside -> project back to the
    boundary (toward the violated optimum). No attenuation hyperparameter.
    """

    def __init__(self):
        self.anchors: list[tuple[np.ndarray, float]] = []

    def add(self, theta_star, eps: float) -> None:
        self.anchors.append((np.asarray(theta_star, dtype=float).ravel().copy(), float(eps)))

    def project(self, theta_flat):
        """Return (theta, intervened) projected into the intersection of the trust balls."""
        th = np.asarray(theta_flat, dtype=float).copy()
        if not self.anchors:
            return th, False
        intervened = False
        for _ in range(8):  # alternating projection onto the ball intersection
            moved = False
            for ts, eps in self.anchors:
                d2 = float(np.sum((th - ts) ** 2))
                if d2 > eps:
                    th = ts + np.sqrt(eps / d2) * (th - ts)
                    moved = True
                    intervened = True
            if not moved:
                break
        return th, intervened


def clip_step(delta_raw, max_norm: float):
    """Fixed per-step norm clip (baseline). Return (delta, clipped)."""
    d = np.asarray(delta_raw, dtype=float)
    n = float(np.linalg.norm(d))
    if n <= max_norm:
        return d, False
    return d * (max_norm / n), True
