"""Shared VQC state preparation and measurement QNodes for MeasQCL."""

from __future__ import annotations

from collections.abc import Sequence

import pennylane as qml

DEFAULT_BASES = ("ZZ", "XX", "YY")


def _validate_model(
    n_qubits: int,
    n_layers: int,
    readout_wires: Sequence[int],
) -> tuple[int, ...]:
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive")
    wires = tuple(int(wire) for wire in readout_wires)
    if not wires or len(set(wires)) != len(wires):
        raise ValueError("readout_wires must be non-empty and unique")
    if min(wires) < 0 or max(wires) >= n_qubits:
        raise ValueError("readout_wires must refer to model qubits")
    return wires


def _apply_vqc(features, weights, n_qubits: int, n_layers: int) -> None:
    """Prepare the amplitude-encoded state shared by every MeasQCL QNode."""
    qml.AmplitudeEmbedding(
        features,
        wires=range(n_qubits),
        normalize=True,
        pad_with=0.0,
    )
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])


def _rotate_into_measurement_basis(basis: str, wires: tuple[int, ...]) -> None:
    if len(basis) != len(wires) or set(basis).difference("XYZ"):
        raise ValueError("basis must contain one X, Y, or Z for each readout wire")
    for pauli, wire in zip(basis, wires, strict=True):
        if pauli == "X":
            qml.Hadamard(wires=wire)
        elif pauli == "Y":
            # S-dagger followed by H maps a Pauli-Y eigenbasis measurement to Z.
            qml.adjoint(qml.S)(wires=wire)
            qml.Hadamard(wires=wire)


def make_classifier_qnode(
    n_qubits: int = 6,
    n_layers: int = 2,
    readout_wires: Sequence[int] = (0, 1),
    diff_method: str = "backprop",
):
    """Return the two-score classifier QNode and its RY/RZ weight shape."""
    wires = _validate_model(n_qubits, n_layers, readout_wires)
    if len(wires) != 2:
        raise ValueError("the binary softmax classifier requires two readout wires")
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, diff_method=diff_method)
    def qnode(features, weights):
        _apply_vqc(features, weights, n_qubits, n_layers)
        return tuple(qml.expval(qml.PauliZ(wire)) for wire in wires)

    return qnode, (n_layers, n_qubits, 2)


def make_measurement_qnode(
    basis: str,
    n_qubits: int = 6,
    n_layers: int = 2,
    readout_wires: Sequence[int] = (0, 1),
    diff_method: str = "parameter-shift",
):
    """Return joint outcome probabilities for one Pauli basis on the readout subsystem."""
    wires = _validate_model(n_qubits, n_layers, readout_wires)
    basis = basis.upper()
    if len(basis) != len(wires) or set(basis).difference("XYZ"):
        raise ValueError("basis must contain one X, Y, or Z for each readout wire")
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, diff_method=diff_method)
    def qnode(features, weights):
        _apply_vqc(features, weights, n_qubits, n_layers)
        _rotate_into_measurement_basis(basis, wires)
        return qml.probs(wires=wires)

    return qnode


def make_qfi_qnode(
    n_qubits: int = 6,
    n_layers: int = 2,
    diff_method: str = "backprop",
):
    """Return a state-equivalent scalar QNode accepted by ``qml.metric_tensor``."""
    _validate_model(n_qubits, n_layers, (0,))
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, diff_method=diff_method)
    def qnode(features, weights):
        _apply_vqc(features, weights, n_qubits, n_layers)
        return qml.expval(qml.PauliZ(0))

    return qnode


def make_reduced_state_qnode(
    n_qubits: int = 6,
    n_layers: int = 2,
    readout_wires: Sequence[int] = (0, 1),
):
    """Return the reduced density matrix on the physically measured subsystem."""
    wires = _validate_model(n_qubits, n_layers, readout_wires)
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def qnode(features, weights):
        _apply_vqc(features, weights, n_qubits, n_layers)
        return qml.density_matrix(wires=wires)

    return qnode
