"""A paper-inspired, gradient-trainable amplitude QNN in PennyLane.

Four qubits amplitude-encode a 16-dimensional vector. Independent RY/RZ parameters,
nearest-neighbor CNOTs, and one Z expectation form a deliberately small binary classifier.
This is not an exact reproduction of the cited paper architecture.
"""

from __future__ import annotations

import pennylane as qml


def make_qnode(n_qubits: int = 4, n_layers: int = 6, diff_method: str = "backprop"):
    """Return (qnode, weight_shape).

    ``qnode(features, weights)`` accepts a batch of shape ``(batch, 2**n_qubits)`` and
    returns a length-``batch`` array of <Z_0> expectations. ``weight_shape`` is
    ``(n_layers, n_qubits, 2)`` for the RY/RZ angles.
    """
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive")
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))

    return qnode, (n_layers, n_qubits, 2)
