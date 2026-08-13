"""Locality-resolved Pauli-observable Fisher information for PhysMeas-QCL.

For a Pauli string P with outcomes +/-1, its binary measurement probabilities are
``(1 +/- <P>)/2``.  The diagonal CFI is therefore exactly
``(d_i <P>)^2 / (1 - <P>^2)``.  This is not merely a moment proxy; the exact identity
depends on P being a binary Pauli observable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.measqcl_model import _apply_vqc, _validate_model


@dataclass(frozen=True)
class PauliMeasurement:
    """One implementable Pauli observable and a compatible product-basis setting."""

    name: str
    pauli: str
    setting: str
    family: str

    @property
    def weight(self) -> int:
        return sum(letter != "I" for letter in self.pauli)

    @property
    def diameter(self) -> int:
        support = [index for index, letter in enumerate(self.pauli) if letter != "I"]
        return 0 if len(support) < 2 else support[-1] - support[0]


def _measurement(name: str, pauli: str, setting: str, family: str) -> PauliMeasurement:
    if not pauli or len(pauli) != len(setting):
        raise ValueError("Pauli observable and setting must have equal nonzero length")
    if set(pauli).difference("IXYZ") or set(setting).difference("XYZ"):
        raise ValueError("Pauli strings use I/X/Y/Z and settings use X/Y/Z")
    if set(pauli) == {"I"}:
        raise ValueError("identity is not an informative measurement")
    if any(letter != "I" and letter != basis for letter, basis in zip(pauli, setting)):
        raise ValueError("measurement setting is incompatible with its Pauli observable")
    return PauliMeasurement(name=name, pauli=pauli, setting=setting, family=family)


def phase_measurement_families(n_qubits: int = 4) -> dict[str, tuple[PauliMeasurement, ...]]:
    """Return prespecified output-observable families for the four-qubit phase task.

    ``hamiltonian`` mirrors the cluster-Ising XZX and nearest-neighbour YY terms.
    ``nonlocal`` contains their weight-four stabilizer-product correlation XYYX.
    Settings are recorded separately so shot cost can count compatible basis reuse.
    """
    if n_qubits != 4:
        raise ValueError("the current phase-observable library is defined for four qubits")
    readout = (
        _measurement("Z0", "ZIII", "ZZZZ", "readout"),
        _measurement("Z1", "IZII", "ZZZZ", "readout"),
    )
    one_local = tuple(
        _measurement(
            f"{basis}{wire}",
            "".join(basis if index == wire else "I" for index in range(4)),
            basis * 4,
            "one_local",
        )
        for basis in "XYZ"
        for wire in range(4)
    )
    two_local = tuple(
        _measurement(
            f"{basis}{left}{left + 1}",
            "".join(
                basis if index in (left, left + 1) else "I" for index in range(4)
            ),
            basis * 4,
            "two_local",
        )
        for basis in "XYZ"
        for left in range(3)
    )
    hamiltonian = (
        _measurement("cluster_012", "XZXI", "XZXX", "hamiltonian"),
        _measurement("cluster_123", "IXZX", "XXZX", "hamiltonian"),
        _measurement("ising_01", "YYII", "YYYY", "hamiltonian"),
        _measurement("ising_12", "IYYI", "YYYY", "hamiltonian"),
        _measurement("ising_23", "IIYY", "YYYY", "hamiltonian"),
    )
    nonlocal_family = (
        _measurement("cluster_product", "XYYX", "XYYX", "nonlocal"),
    )
    return {
        "readout": readout,
        "one_local": one_local,
        "two_local": two_local,
        "hamiltonian": hamiltonian,
        "nonlocal": nonlocal_family,
    }


def make_pauli_expectation_qnode(
    pauli: str,
    n_qubits: int = 4,
    n_layers: int = 3,
    *,
    diff_method: str = "backprop",
):
    """Return ``<P>`` after the shared amplitude encoding and VQC."""
    _validate_model(n_qubits, n_layers, (0,))
    if len(pauli) != n_qubits or set(pauli).difference("IXYZ") or set(pauli) == {"I"}:
        raise ValueError("pauli must be a non-identity I/X/Y/Z string on all qubits")
    device = qml.device("default.qubit", wires=n_qubits)
    pauli_types = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    @qml.qnode(device, diff_method=diff_method)
    def qnode(features, weights):
        _apply_vqc(features, weights, n_qubits, n_layers)
        factors = [
            pauli_types[letter](wire)
            for wire, letter in enumerate(pauli)
            if letter != "I"
        ]
        observable = factors[0] if len(factors) == 1 else qml.prod(*factors)
        return qml.expval(observable)

    return qnode


def pauli_fisher_diag(
    qnode,
    weights,
    features: np.ndarray,
    indices: Sequence[int],
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Return exact binary-outcome CFI diagonal for one Pauli observable."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    chosen = np.asarray(indices, dtype=int)
    if chosen.ndim != 1 or len(chosen) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.min(chosen) < 0 or np.max(chosen) >= len(features):
        raise ValueError("anchor index is outside the feature array")
    trainable = pnp.array(weights, requires_grad=True)
    jacobian = qml.jacobian(qnode, argnums=1)
    fisher = np.zeros(trainable.size, dtype=float)
    for index in chosen:
        sample = pnp.array(features[index], requires_grad=False)
        expectation = float(qnode(sample, trainable))
        if abs(expectation) > 1.0 + 1e-8:
            raise ValueError("Pauli expectation must lie in [-1, 1]")
        gradient = np.asarray(jacobian(sample, trainable), dtype=float).reshape(-1)
        fisher += gradient**2 / max(1.0 - expectation**2, epsilon)
    fisher /= len(chosen)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < -1e-9):
        raise ValueError("Pauli Fisher must be finite and non-negative")
    return np.clip(fisher, 0.0, None)


def uniform_family_fisher(fishers: dict[str, np.ndarray]) -> np.ndarray:
    """CFI of choosing one observable uniformly from a prespecified family."""
    if not fishers:
        raise ValueError("observable family must not be empty")
    matrix = np.stack([np.asarray(values, dtype=float) for values in fishers.values()])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-9):
        raise ValueError("observable Fishers must be aligned, finite, and non-negative")
    return np.mean(np.clip(matrix, 0.0, None), axis=0)


def family_resource_summary(
    measurements: Sequence[PauliMeasurement],
) -> dict[str, int | list[int]]:
    """Count observables and compatible product-basis settings separately."""
    if not measurements:
        raise ValueError("measurement family must not be empty")
    return {
        "n_observables": len(measurements),
        "n_product_basis_settings": len({measurement.setting for measurement in measurements}),
        "pauli_weights": [measurement.weight for measurement in measurements],
        "support_diameters": [measurement.diameter for measurement in measurements],
    }
