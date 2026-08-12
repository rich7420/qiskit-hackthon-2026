"""Formulation tests for E005 (EWC/QEWC). Fast: no MNIST, no training."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.e005_consolidation import EWC, quantum_fisher_diag
from src.e005_softmax import accuracy, bce_loss, classical_fisher_diag, make_softmax_qnode


def _one_qubit_qnode():
    """Amplitude-embed a fixed |0>, then RY(w0), RZ(w1) on one qubit."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="backprop")
    def qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=[0], normalize=True)
        qml.RY(weights[0, 0, 0], wires=0)
        qml.RZ(weights[0, 0, 1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return qnode


def test_qfi_ry_on_zero_is_one():
    # Known result: QFI of RY(theta) acting on |0> equals 1 for any theta.
    qnode = _one_qubit_qnode()
    w = pnp.array(np.array([[[0.6, 0.3]]]), requires_grad=True)
    X = np.array([[1.0, 0.0]])  # |0>
    qfi = quantum_fisher_diag(qnode, w, X, n_samples=1, seed=0)
    assert qfi.shape == (2,)
    assert np.isclose(qfi[0], 1.0, atol=1e-6)          # first (RY) generator on |0>
    assert np.all(qfi >= -1e-9)                         # QFI diagonal is non-negative


def test_ewc_alpha_weighting():
    # Two anchors at theta*=0 with unit Fisher; at task k=3 alpha should be (1/2, 1).
    reg = EWC(lam=2.0)
    p = 4
    reg.consolidate(np.zeros(p), np.ones(p))   # task 1 anchor
    reg.consolidate(np.zeros(p), np.ones(p))   # task 2 anchor
    w = pnp.array(np.ones(p), requires_grad=True)
    # penalty = (lam/2) * [ alpha_1 * sum((w)^2) + alpha_2 * sum((w)^2) ]
    #         = (2/2) * [ 0.5 * p + 1.0 * p ] = 1.5 * p
    val = float(reg.penalty(w, task_index=3))
    assert np.isclose(val, 1.5 * p, atol=1e-9)
    # At k=2 only the first anchor exists with alpha=1: penalty = (2/2)*1*p = p
    reg2 = EWC(lam=2.0)
    reg2.consolidate(np.zeros(p), np.ones(p))
    assert np.isclose(float(reg2.penalty(w, task_index=2)), float(p), atol=1e-9)


def test_cfi_diag_shape_and_nonneg():
    qnode, wshape = make_softmax_qnode(n_qubits=3, n_layers=2)
    w = pnp.array(0.1 * np.random.default_rng(0).standard_normal(wshape), requires_grad=True)
    X = np.random.default_rng(1).random((8, 8))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = np.where(np.random.default_rng(2).random(8) > 0.5, 1, -1)
    cfi = classical_fisher_diag(qnode, w, X, y)
    assert cfi.shape == (int(np.prod(wshape)),)
    assert np.all(cfi >= 0.0)


def test_softmax_accuracy_and_bce_range():
    qnode, wshape = make_softmax_qnode(n_qubits=2, n_layers=1)
    w = pnp.array(np.zeros(wshape), requires_grad=True)
    X = np.eye(4)[:3]  # basis states
    y = np.array([-1, 1, -1])
    acc = accuracy(qnode, w, X, y)
    assert 0.0 <= acc <= 1.0
    assert float(bce_loss(qnode, w, X, y)) >= 0.0
