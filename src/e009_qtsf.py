"""Quantum temporal forecasting model for e009 (recurrent data re-uploading).

A compact hybrid model in the spirit of the provided datasets' companion architecture
(data re-uploading quantum sequence models, arXiv:2605.06734, Peng & Chen et al.). One shared
quantum block is re-applied at every time step with the state persisting across steps, giving a
recurrent temporal circuit; the final Pauli-Z readouts feed a trainable linear+tanh head that
outputs the one-step-ahead scalar forecast.

- 4 qubits, `n_layers` shared RY/RZ + CNOT-ring blocks, state persists across `seq_len` steps.
- Weights: quantum block (n_layers, 4, 2) + classical head (4 + 1). Backprop, exact statevector.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def make_forecaster(n_qubits: int = 4, n_layers: int = 2, seq_len: int = 8,
                    diff_method: str = "backprop"):
    """Return (qnode, circ_shape, head_shape).

    qnode(window, circ_w) re-uploads a length-`seq_len` window through one shared block and
    returns the `n_qubits` Pauli-Z expectations.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def qnode(window, circ_w):
        for step in range(seq_len):
            v = window[..., step]
            for q in range(n_qubits):
                qml.RY(((q + 1) / n_qubits) * v, wires=q)
                qml.RZ(((n_qubits - q) / n_qubits) * v, wires=q)
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(circ_w[layer, q, 0], wires=q)
                    qml.RZ(circ_w[layer, q, 1], wires=q)
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qnode, (n_layers, n_qubits, 2), (n_qubits + 1,)


def predict(qnode, circ_w, head_w, X):
    """Scalar one-step forecasts for a batch X (n, seq_len): tanh(head . <Z> + b)."""
    readouts = pnp.stack(qnode(pnp.array(X, requires_grad=False), circ_w), axis=-1)
    return pnp.tanh(readouts @ head_w[:-1] + head_w[-1])


def mse_loss(qnode, circ_w, head_w, X, y):
    return pnp.mean((predict(qnode, circ_w, head_w, X) - pnp.array(y, requires_grad=False)) ** 2)


def nmse(qnode, circ_w, head_w, X, y) -> float:
    """Normalized MSE = MSE / var(y) — the NARMA-standard forecasting error (lower is better)."""
    pred = np.asarray(predict(qnode, circ_w, head_w, X))
    y = np.asarray(y)
    return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def init_weights(circ_shape, head_shape, seed: int = 42):
    rng = np.random.default_rng(seed)
    circ = pnp.array(0.1 * rng.standard_normal(circ_shape), requires_grad=True)
    head = pnp.array(0.1 * rng.standard_normal(head_shape), requires_grad=True)
    return circ, head
