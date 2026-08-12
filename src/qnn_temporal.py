"""Compact hybrid quantum temporal model with recurrent parameter sharing."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp


def make_temporal_qnode(
    n_qubits: int = 4,
    n_layers: int = 1,
    n_steps: int = 12,
    diff_method: str = "backprop",
):
    """Return a QNode that re-uploads a sequence through one shared quantum block.

    The quantum state is initialized once per sequence. At every time step the scalar input is
    angle-encoded on all qubits, followed by the same trainable rotations and CNOT ring. Reusing
    the parameters while preserving the state gives the circuit its recurrent temporal structure.
    """
    if min(n_qubits, n_layers, n_steps) <= 0:
        raise ValueError("n_qubits, n_layers, and n_steps must be positive")
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, diff_method=diff_method)
    def qnode(sequences, circuit_weights):
        for step in range(n_steps):
            value = sequences[..., step]
            for qubit in range(n_qubits):
                forward_scale = (qubit + 1) / n_qubits
                reverse_scale = (n_qubits - qubit) / n_qubits
                qml.RY(forward_scale * value, wires=qubit)
                qml.RZ(reverse_scale * value, wires=qubit)
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(circuit_weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(circuit_weights[layer, qubit, 1], wires=qubit)
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        return tuple(qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits))

    return qnode, (n_layers, n_qubits, 2), (n_qubits + 1,)


def predict(qnode, circuit_weights, head_weights, sequences):
    """Map quantum readouts through a trainable classical tanh head."""
    readouts = pnp.stack(qnode(sequences, circuit_weights), axis=-1)
    return pnp.tanh(pnp.dot(readouts, head_weights[:-1]) + head_weights[-1])
