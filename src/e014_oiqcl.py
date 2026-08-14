"""MPI: Measurement-based Parameter Isolation (e014).

Core idea (measurement-side continual learning): keep a *shared* variational circuit
rho_theta(x) fixed (or softly anchored) across tasks, and give every task its own
lightweight readout.  A per-task diagonal observable in a fixed basis U_b,

    H^(t) = U_b^dagger diag(lambda^(t)) U_b,

has expectation <H^(t)> = sum_k lambda_k^(t) p_k(x; theta), a linear functional of the
computational-basis probability vector p_theta(x) = |<k|U_b|psi_theta(x)>|^2.  For C
classes this is exactly a linear head W^(t) in R^{C x 2^n} over probs.  We fold U_b into
the circuit and read p_theta(x) with qml.probs over all wires, so the fixed basis is the
computational basis (U_b = I); the head is fit *classically* in seconds -- no quantum
gradients after the backbone is trained.

Honest framing (see EXPERIMENTS.md): one fixed basis shared by all tasks means the
task observables mutually commute -- this is a DANO-*inspired* commuting diagonal family,
i.e. a linear head over quantum basis probabilities, NOT full ANO/DANO expressivity.
The quantum content lives in p_theta(x); the head is a learned measurement functional.

This module deliberately keeps the backbone ansatz identical to ``src.e005_softmax`` and
``src.measqcl_model`` (AmplitudeEmbedding + RY/RZ + CNOT ladder) so that a theta trained
by any of those readouts can be reused verbatim here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LogisticRegression

from src.e005_softmax import (
    _labels_to_classes,
    bce_loss,
    make_softmax_qnode,
)


def make_probs_qnode(n_qubits: int = 4, n_layers: int = 20, diff_method: str = "backprop",
                     readout_wires=None):
    """Return (qnode, weight_shape); qnode(features, weights) -> p_theta(x) in R^{2^m}.

    Same ansatz as ``make_softmax_qnode`` so the prepared state is identical -- only the
    readout differs. ``readout_wires`` selects the measured subsystem (default: all n qubits,
    the full 2^n distribution); a smaller local readout (e.g. (0, 1)) gives a 2^m-dim
    probability vector -- a genuinely lightweight, hardware-cheap, barren-plateau-safe
    observable, at the cost of the circuit having to route task info onto those wires.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = tuple(range(n_qubits)) if readout_wires is None else tuple(int(w) for w in readout_wires)

    @qml.qnode(dev, diff_method=diff_method)
    def qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.probs(wires=wires)

    return qnode, (n_layers, n_qubits, 2)


def probs_features(probs_qnode, weights, X: np.ndarray) -> np.ndarray:
    """Return the (N, 2^n) frozen probability feature matrix p_theta(x) for a batch."""
    features = np.asarray(probs_qnode(pnp.array(X, requires_grad=False), weights), dtype=float)
    # PennyLane returns (2^n, N) for a batched probs qnode; transpose to (N, 2^n).
    if features.shape[0] != len(X):
        features = features.T
    if features.shape[0] != len(X):
        raise ValueError(f"unexpected probs shape {features.shape} for {len(X)} samples")
    return features


@dataclass(frozen=True)
class LinearHead:
    """A per-task diagonal-observable readout == linear classifier over probs.

    ``clf`` maps p_theta(x) in R^{2^n} -> class in {0, 1}.  Equivalently the rows of its
    weight matrix are the diagonal observable coefficients lambda^(t) (one per class).
    """

    clf: LogisticRegression
    task_name: str

    def predict_pm1(self, P: np.ndarray) -> np.ndarray:
        """Return {-1, +1} predictions from probability features P (N, 2^n)."""
        classes = self.clf.predict(P)
        return np.where(classes == 1, 1, -1)

    def accuracy(self, P: np.ndarray, y_pm1: np.ndarray) -> float:
        return float(np.mean(self.clf.predict(P) == _labels_to_classes(y_pm1)))


def fit_linear_head(
    P_train: np.ndarray,
    y_train_pm1: np.ndarray,
    task_name: str,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 0,
) -> LinearHead:
    """Fit a per-task linear head (= learnable diagonal observable) over frozen probs.

    This is the "trains in seconds, no quantum gradients" step: a classical logistic
    regression on the quantum probability features.
    """
    classes = _labels_to_classes(y_train_pm1)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(P_train, classes)
    return LinearHead(clf=clf, task_name=task_name)


def init_head_weights(n_probs: int, n_classes: int = 2, *, seed: int = 0):
    """Return trainable (W, b) for a linear head over probs: logits = probs @ W.T + b."""
    rng = np.random.default_rng(seed)
    W = pnp.array(0.01 * rng.standard_normal((n_classes, n_probs)), requires_grad=True)
    b = pnp.array(np.zeros(n_classes), requires_grad=True)
    return W, b


def _head_logits(probs, W, b):
    """logits (N, C) from probs (N, 2^n); == expectation of C diagonal observables + bias."""
    return probs @ W.T + b


def _softmax_ce(logits, y_pm1):
    """Mean softmax cross-entropy; labels {-1,+1} -> classes {0,1}."""
    classes = _labels_to_classes(np.asarray(y_pm1))
    log_norm = pnp.log(pnp.sum(pnp.exp(logits), axis=-1))
    onehot = pnp.array(np.eye(logits.shape[-1])[classes], requires_grad=False)
    picked = pnp.sum(logits * onehot, axis=-1)
    return -pnp.mean(picked - log_norm)


def _probs_batched(probs_qnode, weights, X):
    """Differentiable (N, 2^n) probability features for autograd training."""
    P = probs_qnode(X, weights)
    P = pnp.stack(P, axis=-1) if isinstance(P, (tuple, list)) else P
    # A batched probs qnode returns (2^n, N); transpose to (N, 2^n).
    if P.shape[0] != X.shape[0]:
        P = P.T
    return P


def head_accuracy_theta(probs_qnode, weights, W, b, X, y_pm1) -> float:
    """Test accuracy of head (W, b) on backbone ``weights`` (non-differentiable eval)."""
    P = probs_features(probs_qnode, weights, X)
    logits = np.asarray(P @ np.asarray(W).T + np.asarray(b))
    return float(np.mean(np.argmax(logits, axis=-1) == _labels_to_classes(y_pm1)))


def train_task_isolated_head(
    probs_qnode,
    weights,
    task,
    *,
    train_theta: bool,
    alpha: float = 0.0,
    anchor=None,
    lr: float = 0.02,
    epochs: int = 20,
    head_seed: int = 0,
):
    """Train one task's isolated head (and optionally the shared backbone).

    Implements mentor sec 29:  min_{theta, W_t} L_t + alpha * ||theta - anchor||^2,
    with earlier heads frozen (handled by the caller, which keeps them aside).

      * Variant A (frozen backbone):  train_theta=False -> only (W, b) move.
      * Variant B (free backbone):    train_theta=True,  alpha=0.
      * Variant C (soft-anchor):      train_theta=True,  alpha>0, anchor=theta_{t-1}.

    Returns (weights, W, b) after training.  ``weights`` is unchanged when train_theta
    is False (it is passed as a non-trainable constant into the cost).
    """
    n_probs = int(probs_features(probs_qnode, weights, task.X_train[:1]).shape[1])
    W, b = init_head_weights(n_probs, seed=head_seed)
    weights = pnp.array(np.asarray(weights), requires_grad=bool(train_theta))
    Xtr = pnp.array(task.X_train, requires_grad=False)
    ytr = np.asarray(task.y_train)
    anchor_arr = None if anchor is None else np.asarray(anchor)
    optimizer = qml.AdamOptimizer(lr)

    def cost(weights, W, b):
        logits = _head_logits(_probs_batched(probs_qnode, weights, Xtr), W, b)
        loss = _softmax_ce(logits, ytr)
        if train_theta and alpha > 0.0 and anchor_arr is not None:
            loss = loss + alpha * pnp.sum((weights - anchor_arr) ** 2)
        return loss

    for _ in range(epochs):
        weights, W, b = optimizer.step(cost, weights, W, b)
    return weights, W, b


def train_backbone(
    task,
    *,
    n_qubits: int = 4,
    n_layers: int = 20,
    lr: float = 0.02,
    epochs: int = 20,
    seed: int = 42,
    noise: dict | None = None,
    verbose: bool = False,
):
    """Train the shared VQC on one task via the paper-faithful softmax/BCE readout.

    Returns (weights, softmax_qnode, weight_shape).  The returned theta is what gets
    frozen (Variant A) or soft-anchored (Variant C) for the measurement-side heads.
    ``noise`` (a {bit,phase,depol,meas} dict) trains through the mixed-state noisy readout.
    """
    if noise is None:
        clf_qnode, weight_shape = make_softmax_qnode(n_qubits=n_qubits, n_layers=n_layers)
    else:
        from src.e014_noise import make_noisy_softmax_qnode
        clf_qnode, weight_shape = make_noisy_softmax_qnode(n_qubits=n_qubits, n_layers=n_layers,
                                                           noise=noise)
    weights = pnp.array(
        0.01 * np.random.default_rng(seed).standard_normal(weight_shape),
        requires_grad=True,
    )
    optimizer = qml.AdamOptimizer(lr)
    Xtr = pnp.array(task.X_train, requires_grad=False)
    ytr = pnp.array(task.y_train, requires_grad=False)

    def cost(W):
        return bce_loss(clf_qnode, W, Xtr, ytr)

    for epoch in range(epochs):
        weights = optimizer.step(cost, weights)
        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"    backbone[{task.name}] epoch {epoch + 1}/{epochs} "
                  f"loss={float(cost(weights)):.4f}", flush=True)
    return weights, clf_qnode, weight_shape
