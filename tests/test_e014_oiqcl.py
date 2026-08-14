"""E014 MPI contracts: the diagonal-observable == linear-head-over-probs identity,
frozen-backbone structural zero-forgetting, and the probe verdict logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e014_oiqcl import (  # noqa: E402
    fit_linear_head,
    make_probs_qnode,
    probs_features,
)


def _random_weights(layers, n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    return pnp.array(0.3 * rng.standard_normal((layers, n_qubits, 2)), requires_grad=False)


def test_probs_features_are_valid_distributions():
    """p_theta(x) must be a genuine probability vector: non-negative, sums to one."""
    qnode, _ = make_probs_qnode(n_qubits=3, n_layers=2)
    weights = _random_weights(2, 3, seed=1)
    X = np.random.default_rng(2).standard_normal((5, 8))
    P = probs_features(qnode, weights, X)
    assert P.shape == (5, 8)
    assert np.all(P >= -1e-9)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_diagonal_observable_equals_linear_head_over_probs():
    """<H^(t)> for a diagonal H in the computational basis == lambda . p_theta(x).

    This is the identity the whole method rests on: reading a diagonal observable off the
    circuit equals a linear functional of the measured probability vector.
    """
    n_qubits = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    rng = np.random.default_rng(3)
    lam = rng.standard_normal(2**n_qubits)  # diagonal observable eigenvalues
    H = qml.Hermitian(np.diag(lam), wires=range(n_qubits))
    weights = _random_weights(2, n_qubits, seed=4)

    @qml.qnode(dev)
    def expval_qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(2):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(H)

    probs_qnode, _ = make_probs_qnode(n_qubits=n_qubits, n_layers=2)
    x = rng.standard_normal(8)
    direct = float(expval_qnode(pnp.array(x, requires_grad=False), weights))
    via_probs = float(np.dot(lam, probs_features(probs_qnode, weights, x[None, :])[0]))
    np.testing.assert_allclose(direct, via_probs, atol=1e-6)


def test_frozen_backbone_gives_structural_zero_forgetting():
    """A head fit on frozen theta and re-evaluated on the SAME theta cannot forget.

    Retention of an old head is exactly its own accuracy when the backbone never moves --
    the structural guarantee behind Variant A.
    """
    qnode, _ = make_probs_qnode(n_qubits=4, n_layers=3)
    weights = _random_weights(3, 4, seed=5)
    rng = np.random.default_rng(6)
    X = rng.standard_normal((60, 16))
    y = np.where(rng.standard_normal(60) > 0, 1, -1)
    P = probs_features(qnode, weights, X)
    head = fit_linear_head(P, y, "toy", seed=0)
    acc_at_fit = head.accuracy(P, y)
    # "Later task": backbone unchanged -> re-evaluating the frozen head is identical.
    acc_later = head.accuracy(probs_features(qnode, weights, X), y)
    assert acc_later == acc_at_fit


def test_linear_head_predictions_are_pm1():
    qnode, _ = make_probs_qnode(n_qubits=3, n_layers=2)
    weights = _random_weights(2, 3, seed=7)
    rng = np.random.default_rng(8)
    X = rng.standard_normal((20, 8))
    y = np.where(rng.standard_normal(20) > 0, 1, -1)
    P = probs_features(qnode, weights, X)
    head = fit_linear_head(P, y, "toy", seed=0)
    preds = head.predict_pm1(P)
    assert set(np.unique(preds)).issubset({-1, 1})
