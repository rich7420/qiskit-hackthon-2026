"""Task-relevant and finite-shot measurement design for PhysMeas-QCL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize, minimize_scalar

from src.e005_softmax import scores
@dataclass(frozen=True)
class ShotFisherEstimate:
    """Repeated finite-shot estimates and their explicit measurement cost."""

    mean: np.ndarray
    sample_std: np.ndarray
    repetitions: int
    shots_per_circuit: int
    circuits_per_repetition: int
    total_shots: int
    estimates: np.ndarray


@dataclass(frozen=True)
class TaskAllocationResult:
    """Task-relevant allocation with an auditable numerical optimality certificate."""

    bases: tuple[str, ...]
    weights: np.ndarray
    objective: float
    iterations: int
    solver: str
    optimality_gap: float
    tolerance: float


def reversed_logits_fisher_diag(qnode, weights, features, labels) -> np.ndarray:
    """Return the EWC-DR diagonal importance from per-example reversed logits.

    The classifier itself is unchanged. Logits are negated only while computing the
    squared log-likelihood gradients used as task-relevance weights.
    """
    x = pnp.array(features, requires_grad=False)
    raw_labels = np.asarray(labels)
    if not np.all(np.isin(raw_labels, (-1, 1))):
        raise ValueError("labels must use the project convention {-1, +1}")
    y = ((raw_labels + 1) // 2).astype(int)
    if np.asarray(x).ndim != 2 or len(x) == 0 or y.shape != (len(x),):
        raise ValueError("features and labels must be aligned non-empty arrays")
    onehot = pnp.array(np.eye(2)[y], requires_grad=False)

    def reversed_log_likelihood(candidate):
        logits = -scores(qnode, candidate, x)
        log_norm = pnp.log(pnp.sum(pnp.exp(logits), axis=-1))
        return pnp.sum(logits * onehot, axis=-1) - log_norm

    jacobian = np.asarray(qml.jacobian(reversed_log_likelihood)(weights), dtype=float)
    fisher = np.mean(jacobian.reshape(len(x), -1) ** 2, axis=0)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < -1e-10):
        raise ValueError("reversed-logits importance must be finite and non-negative")
    return np.clip(fisher, 0.0, None)


def normalize_task_relevance(relevance: np.ndarray, *, floor: float = 1e-3) -> np.ndarray:
    """Normalize relevance to mean one while retaining a small exploration floor."""
    values = np.asarray(relevance, dtype=float)
    if values.ndim != 1 or len(values) == 0 or floor < 0.0:
        raise ValueError("relevance must be one-dimensional and floor non-negative")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-10):
        raise ValueError("relevance must be finite and non-negative")
    values = np.clip(values, 0.0, None)
    maximum = float(np.max(values))
    if maximum <= 0.0:
        raise ValueError("cannot normalize zero task relevance")
    values = values + floor * maximum
    return len(values) * values / np.sum(values)


def optimize_task_relevant_allocation(
    fishers: dict[str, np.ndarray],
    relevance: np.ndarray,
    *,
    epsilon: float = 1e-10,
    minimum_allocation: float = 0.0,
    relevance_floor: float = 1e-3,
    optimality_tolerance: float = 1e-5,
    max_iterations: int = 5000,
) -> TaskAllocationResult:
    """Maximize task-weighted log coverage over a measurement simplex.

    Relevance affects only measurement selection. The returned accessible Fisher remains
    a physical mixture ``sum_m q_m F_m`` and can be normalized independently for EWC.
    """
    if (
        not fishers
        or epsilon <= 0.0
        or minimum_allocation < 0.0
        or relevance_floor < 0.0
        or optimality_tolerance <= 0.0
        or max_iterations <= 0
    ):
        raise ValueError("fishers, epsilon, and minimum_allocation are invalid")
    bases = tuple(fishers)
    matrix = np.stack([np.asarray(fishers[basis], dtype=float) for basis in bases])
    weights = normalize_task_relevance(relevance, floor=relevance_floor)
    if matrix.ndim != 2 or matrix.shape[1] != len(weights):
        raise ValueError("Fisher diagonals and task relevance must align")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-10):
        raise ValueError("Fisher diagonals must be finite and non-negative")
    if len(bases) * minimum_allocation >= 1.0:
        raise ValueError("minimum allocation leaves no feasible simplex")
    # Per-parameter positive rescaling changes the log objective only by an additive
    # constant in the epsilon->0 limit, while preventing tiny Fisher columns from
    # making the simplex optimization numerically singular. Directions unseen by
    # every candidate measurement cannot affect allocation and are removed.
    column_scale = np.max(matrix, axis=0)
    active = column_scale > epsilon
    if not np.any(active):
        raise ValueError("cannot optimize a zero Fisher library")
    optimization_matrix = matrix[:, active] / column_scale[active]
    optimization_weights = weights[active]

    def objective(allocation: np.ndarray) -> float:
        coverage = epsilon + allocation @ optimization_matrix
        return -float(np.sum(optimization_weights * np.log(coverage)))

    def gradient(allocation: np.ndarray) -> np.ndarray:
        coverage = epsilon + allocation @ optimization_matrix
        return -np.sum(
            optimization_weights[np.newaxis, :]
            * optimization_matrix
            / coverage[np.newaxis, :],
            axis=1,
        )

    initial = np.full(len(bases), 1.0 / len(bases))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(minimum_allocation, 1.0)] * len(bases),
        constraints={
            "type": "eq",
            "fun": lambda q: np.sum(q) - 1.0,
            "jac": lambda q: np.ones_like(q),
        },
        options={"ftol": min(1e-12, optimality_tolerance * 1e-3), "maxiter": max_iterations},
    )
    allocation = np.asarray(result.x, dtype=float)
    iterations = int(result.nit)
    solver = "SLSQP"
    residual = 1.0 - len(bases) * minimum_allocation

    def certified_gap(candidate: np.ndarray) -> float:
        mixture = (candidate - minimum_allocation) / residual
        coverage = epsilon + candidate @ optimization_matrix
        allocation_gradient = np.sum(
            optimization_weights[np.newaxis, :]
            * optimization_matrix
            / coverage[np.newaxis, :],
            axis=1,
        )
        return residual * float(
            np.max(allocation_gradient) - np.dot(allocation_gradient, mixture)
        )

    feasible = (
        np.all(allocation >= minimum_allocation - 1e-9)
        and np.isclose(np.sum(allocation), 1.0, atol=1e-9)
    )
    gap = certified_gap(allocation) if feasible else np.inf
    if not result.success or not feasible or gap > optimality_tolerance:
        # SLSQP can report a false linesearch failure for sparse Fisher libraries with
        # many nearly-zero directions. Frank-Wolfe stays on the simplex and exploits
        # the fact that this is a concave maximization with a one-dimensional exact
        # line search at every step.
        mixture = np.full(len(bases), 1.0 / len(bases))

        def from_mixture(candidate: np.ndarray) -> np.ndarray:
            return minimum_allocation + residual * candidate

        for fallback_iteration in range(1, max_iterations + 1):
            current = from_mixture(mixture)
            coverage = epsilon + current @ optimization_matrix
            allocation_gradient = np.sum(
                optimization_weights[np.newaxis, :]
                * optimization_matrix
                / coverage[np.newaxis, :],
                axis=1,
            )
            vertex_index = int(np.argmax(allocation_gradient))
            vertex = np.zeros_like(mixture)
            vertex[vertex_index] = 1.0
            direction = vertex - mixture
            gap = residual * float(np.dot(allocation_gradient, direction))
            if gap <= optimality_tolerance:
                break

            def line_objective(step: float) -> float:
                return objective(from_mixture(mixture + step * direction))

            line = minimize_scalar(
                line_objective,
                bounds=(0.0, 1.0),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if not line.success:
                raise RuntimeError(
                    f"task-relevant allocation failed: {result.message}; "
                    f"fallback line search: {line.message}"
                )
            mixture = mixture + float(line.x) * direction
        allocation = from_mixture(mixture)
        iterations += fallback_iteration
        solver = "Frank-Wolfe exact-line-search fallback"
        gap = certified_gap(allocation)
        if gap > optimality_tolerance:
            raise RuntimeError(
                "task-relevant allocation failed to reach its declared tolerance; "
                f"gap={gap}, tolerance={optimality_tolerance}"
            )
    allocation = np.clip(allocation, minimum_allocation, 1.0)
    allocation /= np.sum(allocation)
    return TaskAllocationResult(
        bases=bases,
        weights=allocation,
        objective=-objective(allocation),
        iterations=iterations,
        solver=solver,
        optimality_gap=certified_gap(allocation),
        tolerance=optimality_tolerance,
    )


def parameter_shift_probability_table(
    qnode,
    weights,
    features: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate exact base/plus/minus outcome probabilities for finite-shot reuse."""
    chosen = np.asarray(indices, dtype=int)
    data = np.asarray(features)
    base_weights = np.asarray(weights, dtype=float)
    if chosen.ndim != 1 or len(chosen) == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if np.min(chosen) < 0 or np.max(chosen) >= len(data):
        raise ValueError("anchor index is outside the feature array")
    n_parameters = base_weights.size
    base_rows: list[np.ndarray] = []
    plus_rows: list[np.ndarray] = []
    minus_rows: list[np.ndarray] = []
    for index in chosen:
        sample = data[index]
        base_rows.append(np.asarray(qnode(sample, base_weights), dtype=float))
        sample_plus = []
        sample_minus = []
        for parameter in range(n_parameters):
            plus = base_weights.reshape(-1).copy()
            minus = base_weights.reshape(-1).copy()
            plus[parameter] += np.pi / 2.0
            minus[parameter] -= np.pi / 2.0
            sample_plus.append(
                np.asarray(qnode(sample, plus.reshape(base_weights.shape)), dtype=float)
            )
            sample_minus.append(
                np.asarray(qnode(sample, minus.reshape(base_weights.shape)), dtype=float)
            )
        plus_rows.append(np.stack(sample_plus))
        minus_rows.append(np.stack(sample_minus))
    table = {
        "base": np.stack(base_rows),
        "plus": np.stack(plus_rows),
        "minus": np.stack(minus_rows),
    }
    _validate_probability_table(table)
    return table


def _validate_probability_table(table: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if set(table) != {"base", "plus", "minus"}:
        raise ValueError("probability table must contain base, plus, and minus")
    base = np.asarray(table["base"], dtype=float)
    plus = np.asarray(table["plus"], dtype=float)
    minus = np.asarray(table["minus"], dtype=float)
    if base.ndim != 2 or plus.ndim != 3 or plus.shape != minus.shape:
        raise ValueError("probability table has incompatible dimensions")
    if plus.shape[0] != base.shape[0] or plus.shape[2] != base.shape[1]:
        raise ValueError("probability table sample/outcome dimensions do not align")
    for values in (base, plus, minus):
        if not np.all(np.isfinite(values)) or np.any(values < -1e-10):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(np.sum(values, axis=-1), 1.0, atol=1e-8):
            raise ValueError("outcome probabilities must sum to one")
    return base, plus, minus


def exact_fisher_from_probability_table(table: dict[str, Any], *, epsilon: float = 1e-10) -> np.ndarray:
    """Recover the exact diagonal CFI from a parameter-shift probability table."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    base, plus, minus = _validate_probability_table(table)
    derivative = 0.5 * (plus - minus)
    return np.mean(np.sum(derivative**2 / (base[:, np.newaxis, :] + epsilon), axis=2), axis=0)


def finite_shot_fisher_from_probability_table(
    table: dict[str, Any],
    *,
    shots_per_circuit: int,
    repetitions: int,
    seed: int,
    pseudocount: float = 0.5,
    subtract_derivative_noise: bool = True,
) -> ShotFisherEstimate:
    """Estimate diagonal CFI by repeated multinomial parameter-shift sampling.

    ``shots_per_circuit`` applies to every base and shifted probability circuit. The
    returned total therefore includes all anchors, parameters, shifts, and repetitions.
    A first-order non-negative correction removes the squared-derivative sampling floor.
    """
    if shots_per_circuit <= 0 or repetitions <= 0 or pseudocount < 0.0:
        raise ValueError("shots, repetitions, and pseudocount are invalid")
    base, plus, minus = _validate_probability_table(table)
    n_samples, n_parameters, n_outcomes = plus.shape
    rng = np.random.default_rng(seed)
    estimates = []
    denominator = shots_per_circuit + pseudocount * n_outcomes
    for _ in range(repetitions):
        base_hat = np.empty_like(base)
        plus_hat = np.empty_like(plus)
        minus_hat = np.empty_like(minus)
        for sample in range(n_samples):
            base_hat[sample] = (
                rng.multinomial(shots_per_circuit, base[sample]) + pseudocount
            ) / denominator
            for parameter in range(n_parameters):
                plus_hat[sample, parameter] = (
                    rng.multinomial(shots_per_circuit, plus[sample, parameter])
                    + pseudocount
                ) / denominator
                minus_hat[sample, parameter] = (
                    rng.multinomial(shots_per_circuit, minus[sample, parameter])
                    + pseudocount
                ) / denominator
        derivative = 0.5 * (plus_hat - minus_hat)
        numerator = derivative**2
        if subtract_derivative_noise:
            noise = 0.25 * (
                plus_hat * (1.0 - plus_hat)
                + minus_hat * (1.0 - minus_hat)
            ) / shots_per_circuit
            numerator = np.clip(numerator - noise, 0.0, None)
        estimates.append(
            np.mean(
                np.sum(numerator / np.maximum(base_hat[:, np.newaxis, :], 1e-12), axis=2),
                axis=0,
            )
        )
    values = np.stack(estimates)
    circuits = n_samples * (1 + 2 * n_parameters)
    return ShotFisherEstimate(
        mean=np.mean(values, axis=0),
        sample_std=np.std(values, axis=0, ddof=1) if repetitions > 1 else np.zeros(n_parameters),
        repetitions=repetitions,
        shots_per_circuit=shots_per_circuit,
        circuits_per_repetition=circuits,
        total_shots=circuits * shots_per_circuit * repetitions,
        estimates=values,
    )
