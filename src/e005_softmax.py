"""Paper-faithful readout for the continual-learning classifier (arXiv:2607.16030 Eq. 7).

Two Pauli-Z readout qubits (0 and 1) give two class scores, converted to probabilities by
softmax; training uses binary cross-entropy. Same ansatz as src.qnn_pennylane (RY/RZ + CNOT),
so the prepared state — and therefore the QFI — is identical; only the measurement differs.
Labels are the project convention {-1, +1}, mapped to classes {0, 1}.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def make_softmax_qnode(n_qubits: int = 4, n_layers: int = 20, diff_method: str = "backprop"):
    """Return (qnode, weight_shape); qnode(features, weights) -> (score_0, score_1)."""
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
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return qnode, (n_layers, n_qubits, 2)


def _labels_to_classes(y: np.ndarray) -> np.ndarray:
    """Map {-1, +1} -> {0, 1}."""
    return ((np.asarray(y) + 1) // 2).astype(int)


def scores(qnode, weights, X):
    """Return the (N, 2) score matrix (differentiable when weights are trainable)."""
    s0, s1 = qnode(X, weights)
    return pnp.stack([s0, s1], axis=-1)


def _log_likelihood(qnode, weights, X, y_pm1):
    """Per-sample log p(y|x) under softmax(scores); shape (N,). One-hot for autodiff safety."""
    logits = scores(qnode, weights, X)
    log_norm = pnp.log(pnp.sum(pnp.exp(logits), axis=-1))
    onehot = pnp.array(np.eye(2)[_labels_to_classes(y_pm1)], requires_grad=False)
    picked = pnp.sum(logits * onehot, axis=-1)
    return picked - log_norm


def bce_loss(qnode, weights, X, y_pm1):
    """Mean binary cross-entropy over softmax(scores), labels in {-1, +1}."""
    return -pnp.mean(_log_likelihood(qnode, weights, X, y_pm1))


def accuracy(qnode, weights, X, y_pm1) -> float:
    logits = np.asarray(scores(qnode, weights, X))
    pred = np.argmax(logits, axis=-1)
    return float(np.mean(pred == _labels_to_classes(y_pm1)))


def classical_fisher_diag(qnode, weights, X, y_pm1) -> np.ndarray:
    """Empirical CFI diagonal from squared softmax log-likelihood gradients (Eq. 9)."""
    def loglik(W):
        return _log_likelihood(qnode, W, X, y_pm1)

    jac = np.asarray(qml.jacobian(loglik)(weights)).reshape(len(X), -1)
    return np.mean(jac**2, axis=0)
