"""Learnable product measurements for task-boundary Fisher consolidation.

The classifier is frozen before this module is used.  Its reduced density matrices at
the optimum and at every parameter-shift point are cached once.  Measurement-basis
optimization then becomes a classical contraction problem: it never reruns the VQC.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from src.measqcl_fisher import accessible_fisher_diag
from src.measqcl_task_relevance import (
    normalize_task_relevance,
    optimize_task_relevant_allocation,
)

_PAULIS = np.asarray(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=complex,
)
_IDENTITY = np.eye(2, dtype=complex)
_AUTOGRAD_PAULIS = anp.asarray(_PAULIS)
_AUTOGRAD_IDENTITY = anp.eye(2, dtype=complex)


@dataclass(frozen=True)
class ShiftedDensityCache:
    """Reduced states needed to evaluate arbitrary measurement-basis Fisher profiles."""

    base: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    anchor_indices: np.ndarray
    weight_shape: tuple[int, ...]

    @property
    def n_anchors(self) -> int:
        return int(self.base.shape[0])

    @property
    def n_parameters(self) -> int:
        return int(self.plus.shape[1])

    @property
    def n_measured_qubits(self) -> int:
        dimension = int(self.base.shape[-1])
        return dimension.bit_length() - 1


@dataclass(frozen=True)
class LearnableMeasurementResult:
    """Optimized axes, allocation, Fisher profile, and auditable solver diagnostics."""

    axes: np.ndarray
    allocation: np.ndarray
    basis_fishers: np.ndarray
    accessible_fisher: np.ndarray
    objective: float
    information_objective: float
    diversity_penalty: float
    outer_iterations: int
    axis_iterations: int
    objective_evaluations: int
    allocation_solver: str
    allocation_optimality_gap: float
    allocation_optimality_tolerance: float
    axis_solver_messages: tuple[str, ...]
    axis_physical_gradient_norms: tuple[float, ...]
    axis_stationarity_tolerance: float
    history: tuple[dict[str, float], ...]


def _validate_density_matrix(density: np.ndarray, *, atol: float = 1e-8) -> None:
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density matrices must be square")
    dimension = density.shape[0]
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("density-matrix dimension must be a power of two")
    if not np.all(np.isfinite(density)):
        raise ValueError("density matrices must be finite")
    if not np.allclose(density, density.conj().T, atol=atol, rtol=0.0):
        raise ValueError("density matrices must be Hermitian")
    if not np.isclose(np.trace(density), 1.0, atol=atol, rtol=0.0):
        raise ValueError("density matrices must have unit trace")
    if float(np.min(np.linalg.eigvalsh(density))) < -atol:
        raise ValueError("density matrices must be positive semidefinite")


def cache_parameter_shift_density_matrices(
    qnode,
    weights,
    features: np.ndarray,
    indices: Sequence[int],
) -> ShiftedDensityCache:
    """Cache base and +/- pi/2 reduced states for a frozen classifier boundary."""
    data = np.asarray(features)
    chosen = np.asarray(indices, dtype=int)
    base_weights = np.asarray(weights, dtype=float)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if chosen.ndim != 1 or len(chosen) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.min(chosen) < 0 or np.max(chosen) >= len(data):
        raise ValueError("anchor index is outside the feature array")
    if base_weights.size == 0 or not np.all(np.isfinite(base_weights)):
        raise ValueError("weights must be a finite non-empty array")

    n_parameters = int(base_weights.size)
    base_rows: list[np.ndarray] = []
    plus_rows: list[np.ndarray] = []
    minus_rows: list[np.ndarray] = []
    for index in chosen:
        sample = data[index]
        base_density = np.asarray(qnode(sample, base_weights), dtype=complex)
        _validate_density_matrix(base_density)
        base_rows.append(base_density)
        sample_plus: list[np.ndarray] = []
        sample_minus: list[np.ndarray] = []
        for parameter in range(n_parameters):
            plus = base_weights.reshape(-1).copy()
            minus = base_weights.reshape(-1).copy()
            plus[parameter] += np.pi / 2.0
            minus[parameter] -= np.pi / 2.0
            plus_density = np.asarray(
                qnode(sample, plus.reshape(base_weights.shape)), dtype=complex
            )
            minus_density = np.asarray(
                qnode(sample, minus.reshape(base_weights.shape)), dtype=complex
            )
            _validate_density_matrix(plus_density)
            _validate_density_matrix(minus_density)
            if plus_density.shape != base_density.shape or minus_density.shape != base_density.shape:
                raise ValueError("all cached density matrices must have the same shape")
            sample_plus.append(plus_density)
            sample_minus.append(minus_density)
        plus_rows.append(np.stack(sample_plus))
        minus_rows.append(np.stack(sample_minus))
    return ShiftedDensityCache(
        base=np.stack(base_rows),
        plus=np.stack(plus_rows),
        minus=np.stack(minus_rows),
        anchor_indices=chosen.copy(),
        weight_shape=tuple(base_weights.shape),
    )


def normalize_axes(raw_axes: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Map unconstrained three-vectors to physical unit Bloch axes."""
    axes = np.asarray(raw_axes, dtype=float)
    if axes.ndim != 3 or axes.shape[-1] != 3 or epsilon <= 0.0:
        raise ValueError("axes must have shape [settings, qubits, 3]")
    if not np.all(np.isfinite(axes)):
        raise ValueError("axes must be finite")
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("every raw axis must be nonzero")
    return axes / norms


def local_observables(axes: np.ndarray) -> np.ndarray:
    """Return H=n_x X+n_y Y+n_z Z for every fixed-spectrum local measurement."""
    physical_axes = normalize_axes(axes)
    return np.einsum("kqc,cab->kqab", physical_axes, _PAULIS)


def product_projectors(axes: np.ndarray) -> np.ndarray:
    """Return all joint projectors for each local product-measurement setting.

    Outcome ordering follows computational-basis integer order.  Bit zero corresponds
    to the +1 eigenspace of the local observable and bit one to the -1 eigenspace.
    """
    observables = local_observables(axes)
    n_settings, n_qubits = observables.shape[:2]
    n_outcomes = 2**n_qubits
    dimension = n_outcomes
    projectors = np.empty((n_settings, n_outcomes, dimension, dimension), dtype=complex)
    for setting in range(n_settings):
        for outcome in range(n_outcomes):
            joint = np.asarray([[1.0 + 0.0j]])
            for qubit in range(n_qubits):
                bit = (outcome >> (n_qubits - qubit - 1)) & 1
                sign = 1.0 if bit == 0 else -1.0
                local = 0.5 * (_IDENTITY + sign * observables[setting, qubit])
                joint = np.kron(joint, local)
            projectors[setting, outcome] = joint
    return projectors


def probability_tables_from_cache(
    cache: ShiftedDensityCache,
    axes: np.ndarray,
    *,
    probability_tolerance: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Contract cached states with arbitrary product projectors."""
    physical_axes = normalize_axes(axes)
    if physical_axes.shape[1] != cache.n_measured_qubits:
        raise ValueError("measurement axes and cached subsystem dimensions do not align")
    projectors = product_projectors(physical_axes)
    base = np.einsum("aij,koji->kao", cache.base, projectors).real
    plus = np.einsum("apij,koji->kapo", cache.plus, projectors).real
    minus = np.einsum("apij,koji->kapo", cache.minus, projectors).real
    for name, values in (("base", base), ("plus", plus), ("minus", minus)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} probabilities must be finite")
        if np.min(values) < -probability_tolerance or np.max(values) > 1.0 + probability_tolerance:
            raise ValueError(f"{name} probabilities must lie in [0, 1]")
        if not np.allclose(
            np.sum(values, axis=-1), 1.0, atol=probability_tolerance, rtol=0.0
        ):
            raise ValueError(f"{name} probabilities must sum to one")
    return {
        "base": np.clip(base, 0.0, 1.0),
        "plus": np.clip(plus, 0.0, 1.0),
        "minus": np.clip(minus, 0.0, 1.0),
    }


def basis_fisher_diags_from_cache(
    cache: ShiftedDensityCache,
    axes: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Return one exact parameter-shift CFI diagonal per product-measurement setting."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    table = probability_tables_from_cache(cache, axes)
    derivatives = 0.5 * (table["plus"] - table["minus"])
    fishers = np.mean(
        np.sum(
            derivatives**2 / (table["base"][:, :, np.newaxis, :] + epsilon),
            axis=-1,
        ),
        axis=1,
    )
    if fishers.shape[1] != cache.n_parameters:
        raise ValueError("computed Fisher and cached classifier parameters do not align")
    if not np.all(np.isfinite(fishers)) or np.any(fishers < -1e-8):
        raise ValueError("measurement Fisher must be finite and non-negative")
    return np.clip(fishers, 0.0, None)


def measurement_diversity_penalty(axes: np.ndarray) -> float:
    """Antipodal-invariant overlap penalty for redundant measurement settings."""
    physical_axes = normalize_axes(axes)
    n_settings = physical_axes.shape[0]
    if n_settings < 2:
        return 0.0
    overlaps = []
    for first in range(n_settings):
        for second in range(first + 1, n_settings):
            dot = np.sum(physical_axes[first] * physical_axes[second], axis=-1)
            overlaps.append(float(np.mean(dot**2)))
    return float(np.mean(overlaps))


def weighted_log_coverage(
    basis_fishers: np.ndarray,
    allocation: np.ndarray,
    relevance: np.ndarray,
    *,
    epsilon: float = 1e-10,
    column_scale: np.ndarray | None = None,
) -> float:
    """Scale-stable task-weighted diagonal information-coverage objective."""
    matrix = np.asarray(basis_fishers, dtype=float)
    mixture = np.asarray(allocation, dtype=float)
    weights = normalize_task_relevance(relevance, floor=0.0)
    if matrix.ndim != 2 or matrix.shape[1] != len(weights):
        raise ValueError("basis Fisher and relevance dimensions must align")
    if mixture.shape != (matrix.shape[0],):
        raise ValueError("allocation must contain one weight per measurement setting")
    if np.any(mixture < -1e-12) or not np.isclose(np.sum(mixture), 1.0, atol=1e-9):
        raise ValueError("allocation must lie on the probability simplex")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-10):
        raise ValueError("basis Fisher must be finite and non-negative")
    if column_scale is None:
        scales = np.max(matrix, axis=0)
    else:
        scales = np.asarray(column_scale, dtype=float)
        if scales.shape != (matrix.shape[1],):
            raise ValueError("column scale must contain one value per classifier parameter")
        if not np.all(np.isfinite(scales)) or np.any(scales < 0.0):
            raise ValueError("column scale must be finite and non-negative")
    active = scales > epsilon
    if not np.any(active):
        raise ValueError("cannot score a zero Fisher library")
    scaled = matrix[:, active] / scales[active]
    coverage = epsilon + mixture @ scaled
    return float(np.sum(weights[active] * np.log(coverage)))


def canonical_product_axes(
    n_settings: int,
    n_qubits: int,
    *,
    seed: int = 0,
    initialization_noise: float = 1e-2,
) -> np.ndarray:
    """Initialize settings near Z, X, Y, then deterministic random directions."""
    if n_settings <= 0 or n_qubits <= 0 or initialization_noise < 0.0:
        raise ValueError("settings/qubits must be positive and noise non-negative")
    canonical = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
    )
    rng = np.random.default_rng(seed)
    if n_settings <= len(canonical):
        initial = canonical[:n_settings].copy()
    else:
        extra = rng.normal(size=(n_settings - len(canonical), 3))
        extra /= np.linalg.norm(extra, axis=-1, keepdims=True)
        initial = np.concatenate([canonical, extra], axis=0)
    axes = np.repeat(initial[:, np.newaxis, :], n_qubits, axis=1)
    if initialization_noise:
        axes = axes + initialization_noise * rng.normal(size=axes.shape)
    return normalize_axes(axes)


def _autograd_axis_objective(
    flat_coordinates: np.ndarray,
    *,
    axis_shape: tuple[int, ...],
    cache: ShiftedDensityCache,
    allocation: np.ndarray,
    relevance: np.ndarray,
    objective_scale: np.ndarray,
    diversity_coefficient: float,
    radial_gauge_coefficient: float,
    epsilon: float,
):
    """Differentiable exact-CFI objective on seam-free Cartesian Bloch axes."""
    raw_axes = anp.reshape(flat_coordinates, axis_shape)
    squared_norms = anp.sum(raw_axes**2, axis=-1, keepdims=True)
    axes = raw_axes / anp.sqrt(squared_norms)
    objective = _autograd_physical_axis_objective(
        axes,
        cache=cache,
        allocation=allocation,
        relevance=relevance,
        objective_scale=objective_scale,
        diversity_coefficient=diversity_coefficient,
        epsilon=epsilon,
    )
    radial_gauge = anp.mean((squared_norms - 1.0) ** 2)
    return objective + radial_gauge_coefficient * radial_gauge


def _autograd_physical_axis_objective(
    axes,
    *,
    cache: ShiftedDensityCache,
    allocation: np.ndarray,
    relevance: np.ndarray,
    objective_scale: np.ndarray,
    diversity_coefficient: float,
    epsilon: float,
):
    """Score already-physical Bloch axes without choosing a coordinate chart."""
    observables = anp.einsum("kqc,cab->kqab", axes, _AUTOGRAD_PAULIS)
    n_settings, n_qubits = axes.shape[:2]
    n_outcomes = 2**n_qubits
    setting_projectors = []
    for setting in range(n_settings):
        outcomes = []
        for outcome in range(n_outcomes):
            joint = anp.asarray([[1.0 + 0.0j]])
            for qubit in range(n_qubits):
                bit = (outcome >> (n_qubits - qubit - 1)) & 1
                sign = 1.0 if bit == 0 else -1.0
                local = 0.5 * (
                    _AUTOGRAD_IDENTITY + sign * observables[setting, qubit]
                )
                joint = anp.kron(joint, local)
            outcomes.append(joint)
        setting_projectors.append(anp.stack(outcomes))
    projectors = anp.stack(setting_projectors)
    base = anp.real(anp.einsum("aij,koji->kao", cache.base, projectors))
    plus = anp.real(anp.einsum("apij,koji->kapo", cache.plus, projectors))
    minus = anp.real(anp.einsum("apij,koji->kapo", cache.minus, projectors))
    derivatives = 0.5 * (plus - minus)
    fishers = anp.mean(
        anp.sum(
            derivatives**2 / (base[:, :, anp.newaxis, :] + epsilon),
            axis=-1,
        ),
        axis=1,
    )
    active = objective_scale > epsilon
    scaled = fishers[:, active] / objective_scale[active]
    coverage = epsilon + anp.asarray(allocation) @ scaled
    information = anp.sum(anp.asarray(relevance)[active] * anp.log(coverage))
    overlaps = []
    for first in range(n_settings):
        for second in range(first + 1, n_settings):
            dot = anp.sum(axes[first] * axes[second], axis=-1)
            overlaps.append(anp.mean(dot**2))
    diversity = anp.mean(anp.stack(overlaps)) if overlaps else 0.0
    return -information + diversity_coefficient * diversity


def _tangent_frames(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal tangent frame at every Bloch direction."""
    physical = normalize_axes(axes)
    reference = np.zeros_like(physical)
    least_aligned = np.argmin(np.abs(physical), axis=-1)
    np.put_along_axis(reference, least_aligned[..., None], 1.0, axis=-1)
    first = np.cross(physical, reference)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second = np.cross(physical, first)
    return first, second


def _autograd_tangent_objective(
    flat_coordinates: np.ndarray,
    *,
    coordinate_shape: tuple[int, ...],
    base_axes: np.ndarray,
    first_tangent: np.ndarray,
    second_tangent: np.ndarray,
    cache: ShiftedDensityCache,
    allocation: np.ndarray,
    relevance: np.ndarray,
    objective_scale: np.ndarray,
    diversity_coefficient: float,
    epsilon: float,
):
    """Score a local tangent step followed by a sphere retraction."""
    coordinates = anp.reshape(flat_coordinates, coordinate_shape)
    candidate = (
        anp.asarray(base_axes)
        + coordinates[..., :1] * anp.asarray(first_tangent)
        + coordinates[..., 1:] * anp.asarray(second_tangent)
    )
    axes = candidate / anp.sqrt(anp.sum(candidate**2, axis=-1, keepdims=True))
    return _autograd_physical_axis_objective(
        axes,
        cache=cache,
        allocation=allocation,
        relevance=relevance,
        objective_scale=objective_scale,
        diversity_coefficient=diversity_coefficient,
        epsilon=epsilon,
    )


def optimize_learnable_measurements(
    cache: ShiftedDensityCache,
    relevance: np.ndarray,
    *,
    initial_axes: np.ndarray | None = None,
    n_settings: int = 3,
    learn_allocation: bool = True,
    minimum_allocation: float = 0.01,
    relevance_floor: float = 1e-3,
    diversity_coefficient: float = 1e-3,
    outer_iterations: int = 3,
    axis_max_iterations: int = 50,
    seed: int = 0,
    epsilon: float = 1e-10,
    axis_stationarity_tolerance: float = 1e-3,
) -> LearnableMeasurementResult:
    """Learn local product axes, alternating with a certified convex allocation solve."""
    if (
        n_settings <= 0
        or minimum_allocation < 0.0
        or relevance_floor < 0.0
        or diversity_coefficient < 0.0
        or outer_iterations <= 0
        or axis_max_iterations <= 0
        or epsilon <= 0.0
        or axis_stationarity_tolerance <= 0.0
    ):
        raise ValueError("invalid measurement-optimization configuration")
    if n_settings * minimum_allocation >= 1.0:
        raise ValueError("minimum allocation leaves no feasible simplex")
    normalized_relevance = normalize_task_relevance(relevance, floor=relevance_floor)
    if len(normalized_relevance) != cache.n_parameters:
        raise ValueError("task relevance and cached classifier parameters must align")
    if initial_axes is None:
        axes = canonical_product_axes(
            n_settings,
            cache.n_measured_qubits,
            seed=seed,
        )
    else:
        axes = normalize_axes(initial_axes)
        if axes.shape != (n_settings, cache.n_measured_qubits, 3):
            raise ValueError("initial axes do not match settings and measured qubits")
    initial_fishers = basis_fisher_diags_from_cache(cache, axes, epsilon=epsilon)
    initial_scale = np.max(initial_fishers, axis=0)
    largest_initial = float(np.max(initial_scale))
    if largest_initial <= epsilon:
        raise ValueError("initial measurement library has zero Fisher information")
    # This scale must remain fixed while axes move.  Recomputing a per-column maximum
    # at every step would divide away the information gain that basis learning seeks.
    objective_scale = np.maximum(initial_scale, largest_initial * 1e-8)
    allocation = np.full(n_settings, 1.0 / n_settings)
    axis_iterations = 0
    objective_evaluations = 0
    history: list[dict[str, float]] = []
    allocation_solver = "fixed-uniform"
    allocation_gap = 0.0
    allocation_tolerance = 0.0
    axis_solver_messages: list[str] = []
    axis_physical_gradient_norms: list[float] = []

    converged_jointly = False
    for outer in range(outer_iterations):
        fixed_allocation = allocation.copy()
        axis_shape = axes.shape
        physical_objective = value_and_grad(
            lambda flat_axes: _autograd_axis_objective(
                flat_axes,
                axis_shape=axis_shape,
                cache=cache,
                allocation=fixed_allocation,
                relevance=normalized_relevance,
                objective_scale=objective_scale,
                diversity_coefficient=diversity_coefficient,
                radial_gauge_coefficient=0.0,
                epsilon=epsilon,
            )
        )

        total_solver_iterations = 0
        previous_objective = np.inf
        while True:
            _, raw_gradient = physical_objective(axes.reshape(-1))
            objective_evaluations += 1
            physical_gradient = np.asarray(raw_gradient, dtype=float).reshape(axis_shape)
            physical_gradient -= np.sum(
                physical_gradient * axes, axis=-1, keepdims=True
            ) * axes
            maximum_physical_gradient = float(
                np.max(np.linalg.norm(physical_gradient, axis=-1))
            )
            if maximum_physical_gradient <= axis_stationarity_tolerance:
                break
            if total_solver_iterations >= axis_max_iterations:
                raise RuntimeError(
                    "manifold measurement-axis optimization reached its iteration limit: "
                    f"iterations={total_solver_iterations}, "
                    f"max_physical_gradient={maximum_physical_gradient:.8g}"
                )
            base_axes = axes.copy()
            first_tangent, second_tangent = _tangent_frames(base_axes)
            coordinate_shape = axis_shape[:-1] + (2,)
            tangent_objective = value_and_grad(
                lambda flat_coordinates: _autograd_tangent_objective(
                    flat_coordinates,
                    coordinate_shape=coordinate_shape,
                    base_axes=base_axes,
                    first_tangent=first_tangent,
                    second_tangent=second_tangent,
                    cache=cache,
                    allocation=fixed_allocation,
                    relevance=normalized_relevance,
                    objective_scale=objective_scale,
                    diversity_coefficient=diversity_coefficient,
                    epsilon=epsilon,
                )
            )

            def local_objective(flat_coordinates: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal objective_evaluations
                objective_evaluations += 1
                value, gradient = tangent_objective(flat_coordinates)
                return float(value), np.asarray(gradient, dtype=float)

            remaining = axis_max_iterations - total_solver_iterations
            result = minimize(
                local_objective,
                np.zeros(np.prod(coordinate_shape), dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=[(-0.5, 0.5)] * int(np.prod(coordinate_shape)),
                options={
                    "maxiter": min(100, remaining),
                    "ftol": 1e-12,
                    "gtol": 1e-7,
                    "maxls": 40,
                },
            )
            if not np.isfinite(result.fun):
                raise RuntimeError("local manifold measurement-axis solve became non-finite")
            local_solution = np.asarray(result.x, dtype=float)
            if result.fun >= previous_objective - 1e-14 and np.max(np.abs(local_solution)) < 1e-10:
                origin = np.zeros_like(local_solution)
                origin_value, origin_gradient = local_objective(origin)
                fallback_step = 1.0
                fallback_found = False
                for _ in range(40):
                    trial = np.clip(
                        -fallback_step * origin_gradient,
                        -0.5,
                        0.5,
                    )
                    trial_value, _ = local_objective(trial)
                    if trial_value < origin_value - 1e-14:
                        local_solution = trial
                        result.fun = trial_value
                        fallback_found = True
                        break
                    fallback_step *= 0.5
                if not fallback_found:
                    raise RuntimeError(
                        "local manifold measurement-axis solve made no progress: "
                        f"max_physical_gradient={maximum_physical_gradient:.8g}, "
                        f"objective={origin_value:.8g}"
                    )
            coordinates = local_solution.reshape(coordinate_shape)
            candidate = (
                base_axes
                + coordinates[..., :1] * first_tangent
                + coordinates[..., 1:] * second_tangent
            )
            axes = normalize_axes(candidate)
            used_iterations = max(1, int(result.nit))
            total_solver_iterations += used_iterations
            previous_objective = float(result.fun)
        axis_iterations += total_solver_iterations
        fishers = basis_fisher_diags_from_cache(cache, axes, epsilon=epsilon)
        if learn_allocation:
            named_fishers = {
                f"setting_{setting}": fishers[setting] for setting in range(n_settings)
            }
            allocation_result = optimize_task_relevant_allocation(
                named_fishers,
                normalized_relevance,
                epsilon=epsilon,
                minimum_allocation=minimum_allocation,
                relevance_floor=0.0,
            )
            allocation = allocation_result.weights
            allocation_solver = allocation_result.solver
            allocation_gap = allocation_result.optimality_gap
            allocation_tolerance = allocation_result.tolerance
        information = weighted_log_coverage(
            fishers,
            allocation,
            normalized_relevance,
            epsilon=epsilon,
            column_scale=objective_scale,
        )
        diversity = measurement_diversity_penalty(axes)
        final_objective = value_and_grad(
            lambda flat_axes: _autograd_axis_objective(
                flat_axes,
                axis_shape=axis_shape,
                cache=cache,
                allocation=allocation,
                relevance=normalized_relevance,
                objective_scale=objective_scale,
                diversity_coefficient=diversity_coefficient,
                radial_gauge_coefficient=0.0,
                epsilon=epsilon,
            )
        )
        _, final_raw_gradient = final_objective(axes.reshape(-1))
        final_gradient = np.asarray(final_raw_gradient, dtype=float).reshape(axis_shape)
        final_gradient -= np.sum(final_gradient * axes, axis=-1, keepdims=True) * axes
        final_gradient_norm = float(np.max(np.linalg.norm(final_gradient, axis=-1)))
        axis_physical_gradient_norms.append(final_gradient_norm)
        axis_solver_messages.append(
            "CONVERGENCE: JOINT ALLOCATION AND MANIFOLD PHYSICAL GRADIENT"
            if final_gradient_norm <= axis_stationarity_tolerance
            else "CONTINUE: ALLOCATION CHANGED THE PHYSICAL AXIS GRADIENT"
        )
        history.append(
            {
                "outer_iteration": float(outer + 1),
                "information_objective": information,
                "diversity_penalty": diversity,
                "total_objective": information - diversity_coefficient * diversity,
                "post_allocation_physical_gradient": final_gradient_norm,
            }
        )
        if final_gradient_norm <= axis_stationarity_tolerance:
            converged_jointly = True
            break

    if not converged_jointly:
        raise RuntimeError(
            "measurement axes/allocation did not converge jointly: "
            f"final_physical_gradient={axis_physical_gradient_norms[-1]:.8g}, "
            f"tolerance={axis_stationarity_tolerance:.8g}"
        )

    named_fishers = {
        f"setting_{setting}": fishers[setting] for setting in range(n_settings)
    }
    named_allocation = {
        f"setting_{setting}": float(allocation[setting]) for setting in range(n_settings)
    }
    accessible = accessible_fisher_diag(named_fishers, named_allocation)
    information = weighted_log_coverage(
        fishers,
        allocation,
        normalized_relevance,
        epsilon=epsilon,
        column_scale=objective_scale,
    )
    diversity = measurement_diversity_penalty(axes)
    return LearnableMeasurementResult(
        axes=axes,
        allocation=allocation,
        basis_fishers=fishers,
        accessible_fisher=accessible,
        objective=information - diversity_coefficient * diversity,
        information_objective=information,
        diversity_penalty=diversity,
        outer_iterations=len(history),
        axis_iterations=axis_iterations,
        objective_evaluations=objective_evaluations,
        allocation_solver=allocation_solver,
        allocation_optimality_gap=allocation_gap,
        allocation_optimality_tolerance=allocation_tolerance,
        axis_solver_messages=tuple(axis_solver_messages),
        axis_physical_gradient_norms=tuple(axis_physical_gradient_norms),
        axis_stationarity_tolerance=axis_stationarity_tolerance,
        history=tuple(history),
    )
