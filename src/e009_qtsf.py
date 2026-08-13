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


# --- ansatz building blocks (shared by the forecaster and its QFI state circuit) ---
# entangler: "ring" (n_qubits CNOTs, wrap-around) | "chain" (n_qubits-1, no wrap) | "none"
# encoding : "ry_rz" (both axes re-upload the datum) | "ry" (single-axis, half the encode gates)
_ENTANGLERS = ("ring", "chain", "none")
_ENCODINGS = ("ry_rz", "ry")


def _entangle(n_qubits: int, entangler: str) -> None:
    n_links = {"ring": n_qubits, "chain": max(n_qubits - 1, 0), "none": 0}[entangler]
    for q in range(n_links):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])


def _reupload(window, circ_w, n_qubits: int, n_layers: int, seq_len: int,
              entangler: str, encoding: str) -> None:
    """Recurrent data re-uploading: at each of `seq_len` steps encode the datum then apply the
    shared variational block; the statevector persists across steps (weight-shared in time).

    `window[..., step]` indexes the step-th datum for both a batch (n, L) and a single (L,)."""
    for step in range(seq_len):
        v = window[..., step]
        for q in range(n_qubits):
            qml.RY(((q + 1) / n_qubits) * v, wires=q)
            if encoding == "ry_rz":
                qml.RZ(((n_qubits - q) / n_qubits) * v, wires=q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(circ_w[layer, q, 0], wires=q)
                qml.RZ(circ_w[layer, q, 1], wires=q)
            _entangle(n_qubits, entangler)


def make_forecaster(n_qubits: int = 4, n_layers: int = 2, seq_len: int = 8,
                    diff_method: str = "backprop", entangler: str = "ring",
                    encoding: str = "ry_rz"):
    """Return (qnode, circ_shape, head_shape).

    qnode(window, circ_w) re-uploads a length-`seq_len` window through one shared block and
    returns the `n_qubits` Pauli-Z expectations. `entangler`/`encoding` select the ansatz variant
    for the gate-count ablation; the defaults reproduce the original ring / two-axis model.
    """
    assert entangler in _ENTANGLERS and encoding in _ENCODINGS
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def qnode(window, circ_w):
        _reupload(window, circ_w, n_qubits, n_layers, seq_len, entangler, encoding)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qnode, (n_layers, n_qubits, 2), (n_qubits + 1,)


def make_state_forecaster(n_qubits: int = 4, n_layers: int = 2, seq_len: int = 8,
                          entangler: str = "ring", encoding: str = "ry_rz"):
    """Same recurrent re-uploading state-prep as make_forecaster, single-Z readout.

    Used only for the Quantum Fisher Information (metric tensor) of the quantum weights — the
    QFI is a property of the prepared state, independent of the classical readout head. Takes the
    same `entangler`/`encoding` so the QFI matches whichever ansatz variant is being trained.
    """
    assert entangler in _ENTANGLERS and encoding in _ENCODINGS
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_qnode(window, circ_w):
        _reupload(window, circ_w, n_qubits, n_layers, seq_len, entangler, encoding)
        return qml.expval(qml.PauliZ(0))

    return state_qnode


def temporal_qfi_diag(state_qnode, circ_w, windows, n_samples: int = 16, seed: int = 0):
    """Diagonal QFI of the quantum weights, averaged over a subsample of input windows.

    Returns a flat vector of length circ_w.size (quantum weights only; the classical head has
    no quantum Fisher). QFI = 4 * diag(Fubini-Study metric).
    """
    p = int(np.asarray(circ_w).size)
    metric = qml.metric_tensor(state_qnode, approx="diag")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=min(n_samples, len(windows)), replace=False)
    acc = np.zeros(p)
    for i in idx:
        g = np.asarray(metric(pnp.array(windows[i], requires_grad=False), circ_w)).reshape(p, p)
        acc += 4.0 * np.diag(g)
    return acc / len(idx)


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
