"""Elastic Weight Consolidation for the QNN — QFI-QEWC and the EWC penalty (arXiv:2607.16030).

- Quantum Fisher (QFI, Eq. 18): F_ii = 4(<d_i psi|d_i psi> - |<psi|d_i psi>|^2), averaged
  over the task data (Eq. 16); computed via PennyLane's Fubini-Study metric tensor (QFI/4).
- The EWC/QEWC penalty (Eq. 2/21) anchors weights near earlier task optima:
      (lam/2) * sum_j alpha_j^(k) * sum_i F_i^(j) (theta_i - theta*_{j,i})^2
  with the multi-task weighting alpha_j^(k) = j/(k-1) for k>1 (Eq. 30).

The classical Fisher (CFI, Eq. 9) lives with the classifier it is defined on:
`src.e005_softmax.classical_fisher_diag` (softmax log-likelihood gradients).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def quantum_fisher_diag(qnode, weights, X, n_samples: int = 64, seed: int = 0) -> np.ndarray:
    """Task-averaged diagonal QFI, shape (n_weights,), estimated over a data subsample."""
    n_weights = int(np.asarray(weights).size)
    metric = qml.metric_tensor(qnode, approx="diag")  # Fubini-Study metric = QFI / 4
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    acc = np.zeros(n_weights)
    for i in idx:
        g = np.asarray(metric(pnp.array(X[i], requires_grad=False), weights)).reshape(
            n_weights, n_weights
        )
        acc += 4.0 * np.diag(g)
    return acc / len(idx)


class EWC:
    """Accumulates (theta*, Fisher) anchors and returns the consolidation penalty."""

    def __init__(self, lam: float):
        self.lam = float(lam)
        self.anchors: list[tuple[np.ndarray, np.ndarray]] = []

    def consolidate(self, theta_star: np.ndarray, fisher: np.ndarray) -> None:
        self.anchors.append((np.asarray(theta_star, dtype=float).copy(),
                             np.asarray(fisher, dtype=float).copy()))

    def penalty(self, weights_flat, task_index: int):
        """(lam/2) * sum_j alpha_j^(k) * sum_i F_i^(j) (theta - theta*_j)^2, k = task_index."""
        if self.lam == 0.0 or not self.anchors:
            return 0.0 * pnp.sum(weights_flat) * 0.0
        total = 0.0
        for j, (theta_star, fisher) in enumerate(self.anchors, start=1):
            alpha = j / (task_index - 1) if task_index > 1 else 1.0
            total = total + alpha * pnp.sum(fisher * (weights_flat - theta_star) ** 2)
        return 0.5 * self.lam * total
