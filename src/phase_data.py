"""Four-qubit cluster-Ising phase data for the E004 quantum-native task.

The Hamiltonian and sampling bands follow Appendix D of arXiv:2607.16030: open boundary
conditions, SPT samples at h in [0.0, 0.5], and ATF samples at h in [2.5, 3.0]. The open
chain has a two-dimensional ground space. We select its even-Z-parity representative so
the returned pure state is deterministic rather than an arbitrary LAPACK basis vector.
"""

from __future__ import annotations

import numpy as np

from src.continual_data import Task

N_QUBITS = 4
SPT_RANGE = (0.0, 0.5)
ATF_RANGE = (2.5, 3.0)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _op(single: dict[int, np.ndarray], n_qubits: int) -> np.ndarray:
    operator = np.array([[1]], dtype=complex)
    for qubit in range(n_qubits):
        operator = np.kron(operator, single.get(qubit, _I))
    return operator


def _hamiltonian(h: float, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Return the open-boundary cluster-Ising Hamiltonian."""
    if n_qubits != N_QUBITS:
        raise ValueError(f"the E004 phase task is defined for {N_QUBITS} qubits")
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for center in range(1, n_qubits - 1):
        matrix -= _op(
            {center - 1: _X, center: _Z, center + 1: _X}, n_qubits
        )
    for left in range(n_qubits - 1):
        matrix += h * _op({left: _Y, left + 1: _Y}, n_qubits)
    if not np.allclose(matrix.imag, 0.0, atol=1e-12):
        raise RuntimeError("cluster-Ising Hamiltonian unexpectedly became complex")
    return matrix.real


def _ground_state(h: float, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Return a deterministic even-parity representative of the ground space."""
    matrix = _hamiltonian(h, n_qubits)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ground_mask = np.isclose(eigenvalues, eigenvalues[0], atol=1e-10, rtol=0.0)
    ground_basis = eigenvectors[:, ground_mask]

    parity = _op({qubit: _Z for qubit in range(n_qubits)}, n_qubits).real
    parity_in_ground_space = ground_basis.T @ parity @ ground_basis
    parity_values, parity_vectors = np.linalg.eigh(parity_in_ground_space)
    even_mask = np.isclose(parity_values, 1.0, atol=1e-10, rtol=0.0)
    if not np.any(even_mask):
        raise RuntimeError("ground space has no even-Z-parity representative")
    even_basis = ground_basis @ parity_vectors[:, even_mask]

    # At h=0 the even-parity ground space is still two-dimensional. Project the
    # lexicographically first computational basis vector with nonzero overlap into
    # that subspace, which is independent of the eigenbasis returned by LAPACK.
    projector = even_basis @ even_basis.T
    state = None
    for column in range(projector.shape[1]):
        candidate = projector[:, column]
        if np.linalg.norm(candidate) > 1e-10:
            state = candidate
            break
    if state is None:
        raise RuntimeError("failed to select a deterministic ground state")
    state = np.asarray(np.real_if_close(state), dtype=float)
    state /= np.linalg.norm(state)

    # Fix the remaining global sign for byte-for-byte reproducibility.
    pivot = int(np.argmax(np.abs(state)))
    if state[pivot] < 0.0:
        state = -state
    return state


def _sample_balanced_split(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or count % 2:
        raise ValueError("phase-task train and test sizes must be positive even numbers")
    per_class = count // 2
    parameters = np.concatenate(
        [
            rng.uniform(*SPT_RANGE, per_class),
            rng.uniform(*ATF_RANGE, per_class),
        ]
    )
    labels = np.concatenate(
        [-np.ones(per_class, dtype=int), np.ones(per_class, dtype=int)]
    )
    order = rng.permutation(count)
    parameters, labels = parameters[order], labels[order]
    states = np.stack([_ground_state(value) for value in parameters])
    return states, labels


def load_spt_atf(
    n_train: int = 800,
    n_test: int = 200,
    n_qubits: int = N_QUBITS,
    seed: int = 42,
) -> Task:
    """Return balanced train/test ground states labelled SPT (-1) and ATF (+1)."""
    if n_qubits != N_QUBITS:
        raise ValueError(f"the E004 phase task requires n_qubits={N_QUBITS}")
    rng = np.random.default_rng(seed)
    X_train, y_train = _sample_balanced_split(n_train, rng)
    X_test, y_test = _sample_balanced_split(n_test, rng)
    return Task("SPT/ATF phases", X_train, y_train, X_test, y_test)


# Complete cluster-Ising phase diagram: both sides of the SPT transition h_c ~ 1, sampled
# across the WHOLE diagram (incl. near-critical), not just deep in each phase. Near
# criticality the SPT and trivial phases have overlapping LOCAL observables and differ only
# in nonlocal string order, so a local-observable classical baseline fails there.
CLUSTER_SPT_FULL = (0.0, 0.95)
CLUSTER_TRIVIAL_FULL = (1.05, 3.0)


def _sample_balanced_range(count, rng, lo_range, hi_range):
    if count <= 0 or count % 2:
        raise ValueError("phase-task train and test sizes must be positive even numbers")
    per = count // 2
    params = np.concatenate([rng.uniform(*lo_range, per), rng.uniform(*hi_range, per)])
    labels = np.concatenate([-np.ones(per, dtype=int), np.ones(per, dtype=int)])
    order = rng.permutation(count)
    params, labels = params[order], labels[order]
    states = np.stack([_ground_state(v) for v in params])
    return states, labels


def load_cluster_full(
    n_train: int = 800,
    n_test: int = 200,
    n_qubits: int = N_QUBITS,
    seed: int = 42,
) -> Task:
    """Complete cluster-Ising phase task: SPT (h<h_c, -1) vs trivial (h>h_c, +1) sampled
    across the full diagram incl. near-critical. Harder than the deep SPT/ATF cut: a
    local-observable classical baseline drops to chance near criticality (only nonlocal
    string order separates the phases there), while the amplitude-embedded circuit succeeds.
    """
    if n_qubits != N_QUBITS:
        raise ValueError(f"the cluster-Ising phase task requires n_qubits={N_QUBITS}")
    rng = np.random.default_rng(seed)
    X_train, y_train = _sample_balanced_range(n_train, rng, CLUSTER_SPT_FULL, CLUSTER_TRIVIAL_FULL)
    X_test, y_test = _sample_balanced_range(n_test, rng, CLUSTER_SPT_FULL, CLUSTER_TRIVIAL_FULL)
    return Task("cluster-Ising", X_train, y_train, X_test, y_test)
