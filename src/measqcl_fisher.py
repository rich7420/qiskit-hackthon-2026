"""Measurement-specific Fisher information and allocation optimization for MeasQCL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize


@dataclass(frozen=True)
class AllocationResult:
    """Optimized measurement mixture and diagnostics."""

    bases: tuple[str, ...]
    weights: np.ndarray
    objective: float
    iterations: int


def select_anchor_indices(
    n_examples: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Select one deterministic no-replacement anchor subset shared by all estimators."""
    if n_examples <= 0 or n_samples <= 0:
        raise ValueError("n_examples and n_samples must be positive")
    size = min(n_examples, n_samples)
    return np.random.default_rng(seed).choice(n_examples, size=size, replace=False)


def measurement_fisher_diag(
    qnode,
    weights,
    features: np.ndarray,
    indices: Sequence[int],
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Average diagonal CFI of exact joint outcome probabilities over anchor inputs."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    chosen = np.asarray(indices, dtype=int)
    if chosen.ndim != 1 or len(chosen) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.min(chosen) < 0 or np.max(chosen) >= len(features):
        raise ValueError("anchor index is outside the feature array")

    trainable = pnp.array(weights, requires_grad=True)
    n_weights = int(np.asarray(trainable).size)
    jacobian = qml.jacobian(qnode, argnums=1)
    fisher = np.zeros(n_weights, dtype=float)
    for index in chosen:
        sample = pnp.array(features[index], requires_grad=False)
        probabilities = np.asarray(qnode(sample, trainable), dtype=float)
        derivatives = np.asarray(jacobian(sample, trainable), dtype=float).reshape(
            len(probabilities), n_weights
        )
        fisher += np.sum(
            derivatives**2 / (probabilities[:, np.newaxis] + epsilon),
            axis=0,
        )
    fisher /= len(chosen)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < -1e-9):
        raise ValueError("measurement Fisher must be finite and non-negative")
    return np.clip(fisher, 0.0, None)


def normalize_fisher_mass(fisher: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize a non-negative Fisher diagonal to mean parameter importance one."""
    values = np.asarray(fisher, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("fisher must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-9):
        raise ValueError("fisher must be finite and non-negative")
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= epsilon:
        raise ValueError("cannot normalize a zero-mass Fisher diagonal")
    return len(values) * values / total


def reduced_state_qfi_diag(
    qnode,
    weights,
    features: np.ndarray,
    indices: Sequence[int],
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Diagonal SLD QFI of a reduced state using exact parameter-shift derivatives.

    This is the measurement-independent upper bound for POVMs restricted to the QNode's
    returned subsystem. It is distinct from the full-state QFI and need not be jointly
    attainable by one measurement in the multiparameter setting.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    chosen = np.asarray(indices, dtype=int)
    if chosen.ndim != 1 or len(chosen) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.min(chosen) < 0 or np.max(chosen) >= len(features):
        raise ValueError("anchor index is outside the feature array")

    base_weights = np.asarray(weights, dtype=float)
    n_weights = base_weights.size
    qfi = np.zeros(n_weights, dtype=float)
    for index in chosen:
        sample = np.asarray(features[index])
        density = np.asarray(qnode(sample, base_weights), dtype=complex)
        density = 0.5 * (density + density.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(density)
        eigenvalues = np.clip(eigenvalues.real, 0.0, None)
        denominator = eigenvalues[:, np.newaxis] + eigenvalues[np.newaxis, :]
        valid = denominator > epsilon

        for parameter in range(n_weights):
            plus = base_weights.copy().reshape(-1)
            minus = base_weights.copy().reshape(-1)
            plus[parameter] += np.pi / 2.0
            minus[parameter] -= np.pi / 2.0
            derivative = 0.5 * (
                np.asarray(qnode(sample, plus.reshape(base_weights.shape)), dtype=complex)
                - np.asarray(qnode(sample, minus.reshape(base_weights.shape)), dtype=complex)
            )
            derivative = 0.5 * (derivative + derivative.conj().T)
            eigenbasis_derivative = eigenvectors.conj().T @ derivative @ eigenvectors
            contributions = np.zeros_like(denominator)
            contributions[valid] = (
                2.0
                * np.abs(eigenbasis_derivative[valid]) ** 2
                / denominator[valid]
            )
            qfi[parameter] += float(np.sum(contributions).real)
    qfi /= len(chosen)
    if not np.all(np.isfinite(qfi)) or np.any(qfi < -1e-8):
        raise ValueError("reduced-state QFI must be finite and non-negative")
    return np.clip(qfi, 0.0, None)


def accessible_fisher_diag(
    fishers: Mapping[str, np.ndarray],
    allocation: Mapping[str, float],
) -> np.ndarray:
    """Return the CFI of the randomized measurement, ``sum_m q_m F_m``."""
    if not fishers or set(fishers) != set(allocation):
        raise ValueError("fishers and allocation must contain the same bases")
    weights = np.asarray([allocation[basis] for basis in fishers], dtype=float)
    if np.any(weights < -1e-12) or not np.isclose(np.sum(weights), 1.0, atol=1e-9):
        raise ValueError("allocation must lie on the probability simplex")
    matrix = np.stack([np.asarray(fishers[basis], dtype=float) for basis in fishers])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-9):
        raise ValueError("all Fisher diagonals must be finite, non-negative, and aligned")
    return np.clip(weights, 0.0, None) @ np.clip(matrix, 0.0, None)


def optimize_measurement_allocation(
    fishers: Mapping[str, np.ndarray],
    *,
    epsilon: float = 1e-10,
) -> AllocationResult:
    """Maximize log coverage of the accessible diagonal Fisher over the simplex."""
    if not fishers or epsilon <= 0.0:
        raise ValueError("fishers must be non-empty and epsilon must be positive")
    bases = tuple(fishers)
    matrix = np.stack([np.asarray(fishers[basis], dtype=float) for basis in bases])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-9):
        raise ValueError("all Fisher diagonals must be finite, non-negative, and aligned")
    if float(np.sum(matrix)) <= epsilon:
        raise ValueError("cannot optimize zero Fisher information")

    def objective(allocation: np.ndarray) -> float:
        coverage = epsilon + allocation @ matrix
        return -float(np.sum(np.log(coverage)))

    def gradient(allocation: np.ndarray) -> np.ndarray:
        coverage = epsilon + allocation @ matrix
        return -np.sum(matrix / coverage[np.newaxis, :], axis=1)

    initial = np.full(len(bases), 1.0 / len(bases))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(bases),
        constraints={"type": "eq", "fun": lambda q: np.sum(q) - 1.0, "jac": lambda q: np.ones_like(q)},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    allocation = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    allocation /= np.sum(allocation)
    if not result.success or not np.isclose(np.sum(allocation), 1.0, atol=1e-9):
        raise RuntimeError(f"measurement allocation optimization failed: {result.message}")
    return AllocationResult(
        bases=bases,
        weights=allocation,
        objective=-objective(allocation),
        iterations=int(result.nit),
    )


def allocate_integer_shots(
    allocation: Mapping[str, float],
    total_shots: int,
) -> dict[str, int]:
    """Convert a measurement mixture to a deterministic largest-remainder shot plan."""
    if total_shots <= 0 or not allocation:
        raise ValueError("total_shots and allocation must be positive")
    bases = tuple(allocation)
    weights = np.asarray([allocation[basis] for basis in bases], dtype=float)
    if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-9):
        raise ValueError("allocation must lie on the probability simplex")
    exact = total_shots * weights
    shots = np.floor(exact).astype(int)
    remainder = total_shots - int(np.sum(shots))
    order = np.argsort(-(exact - shots), kind="stable")
    shots[order[:remainder]] += 1
    return dict(zip(bases, shots.tolist(), strict=True))


def fisher_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity of two nonzero Fisher diagonals."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Fisher diagonals must have the same one-dimensional shape")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero Fisher diagonal")
    return float(np.dot(first, second) / denominator)


def parameter_shift_evaluation_budget(
    n_parameters: int,
    n_samples: int,
    n_settings: int,
) -> int:
    """Logical exact-probability evaluations: one forward plus two shifts per parameter."""
    if min(n_parameters, n_samples, n_settings) <= 0:
        raise ValueError("budget dimensions must be positive")
    return n_samples * n_settings * (1 + 2 * n_parameters)
