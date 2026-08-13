"""OI-QCL for forecasting: measurement-side continual learning on time-series (e015).

Faithful port of e014's Observable-Isolated Quantum Continual Learning (``src.e014_oiqcl``)
from Task-IL *classification* to one-step-ahead *regression* on the e009 quantum/physics
forecasting series.

Same core idea (measurement-side CL): keep a *shared* recurrent data-reuploading circuit
rho_theta(x) fixed (Variant A) or softly anchored (Variant C) across tasks, and give every
task its own lightweight readout.  A per-task diagonal observable in the computational basis,

    H^(t) = diag(lambda^(t)),   <H^(t)>(x) = sum_k lambda_k^(t) p_k(x; theta) = w^(t) . p_theta(x),

is exactly a *linear* functional of the computational-basis probability vector
p_theta(x) = qml.probs over all wires.  For scalar forecasting the readout IS that
expectation, so the head is a plain classical linear regressor (Ridge) over probs -- it
"trains in seconds, no quantum gradients" once the shared backbone is fixed.

Old heads are frozen, so measurement-side forgetting is a structural zero; only backbone
drift (Variant B free-theta vs Variant C anchored-theta) can move an earlier task's error.
Task identity is known at test time (Task-IL): each task is read out with its own head.

The backbone ansatz is identical to ``src.e009_qtsf.make_forecaster`` so a theta trained by
the paper-faithful tanh/MSE forecaster can be reused verbatim -- only the readout differs
(full 2^n probabilities instead of n Pauli-Z expectations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import Ridge

from src.e009_qtsf import init_weights, make_forecaster, predict


def make_probs_forecaster(n_qubits: int = 4, n_layers: int = 2, seq_len: int = 8,
                          diff_method: str = "backprop"):
    """Return (qnode, circ_shape); qnode(window, circ_w) -> p_theta(x) in R^{2^n}.

    Identical recurrent re-uploading state-prep to ``make_forecaster`` (so the prepared
    state matches a backbone trained there), but reads the full computational-basis
    probability distribution instead of Pauli-Z expectations.
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
        return qml.probs(wires=range(n_qubits))

    return qnode, (n_layers, n_qubits, 2)


def _probs_batched(probs_qnode, circ_w, X):
    """Differentiable (N, 2^n) probability features for autograd backbone training."""
    P = probs_qnode(X, circ_w)
    P = pnp.stack(P, axis=-1) if isinstance(P, (tuple, list)) else P
    if P.shape[0] != X.shape[0]:  # a batched probs qnode returns (2^n, N)
        P = P.T
    return P


def probs_features(probs_qnode, circ_w, X: np.ndarray) -> np.ndarray:
    """Return the (N, 2^n) frozen probability feature matrix p_theta(x) for a batch."""
    P = np.asarray(probs_qnode(pnp.array(X, requires_grad=False), circ_w), dtype=float)
    if P.shape[0] != len(X):
        P = P.T
    if P.shape[0] != len(X):
        raise ValueError(f"unexpected probs shape {P.shape} for {len(X)} samples")
    return P


@dataclass(frozen=True)
class LinearHead:
    """A per-task diagonal-observable readout == linear regressor over probs.

    ``reg`` maps p_theta(x) in R^{2^n} -> scalar forecast.  Its coefficient vector is the
    diagonal observable lambda^(t) (plus an intercept), so the prediction is exactly the
    expectation <H^(t)>(x) of a per-task commuting diagonal observable.
    """

    reg: Ridge
    task_name: str

    def predict(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(self.reg.predict(P), dtype=float)

    def nmse(self, P: np.ndarray, y: np.ndarray) -> float:
        """Normalized MSE = MSE / var(y) -- the NARMA-standard forecasting error."""
        pred = self.predict(P)
        y = np.asarray(y, dtype=float)
        return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def fit_linear_head(P_train: np.ndarray, y_train: np.ndarray, task_name: str,
                    *, alpha: float = 1e-3) -> LinearHead:
    """Fit a per-task linear head (= learnable diagonal observable) over frozen probs.

    The "trains in seconds, no quantum gradients" step: a classical ridge regression on the
    quantum probability features.  ``alpha`` is a light L2 ridge for numerical stability.
    """
    reg = Ridge(alpha=alpha)
    reg.fit(P_train, np.asarray(y_train, dtype=float))
    return LinearHead(reg=reg, task_name=task_name)


def train_backbone_forecast(task, *, n_qubits: int = 4, n_layers: int = 2, seq_len: int = 8,
                            lr: float = 0.05, epochs: int = 40, seed: int = 42,
                            verbose: bool = False):
    """Train the shared recurrent forecaster on one task via the paper-faithful tanh/MSE head.

    Returns (circ_w, head_w, qnode).  The returned circ_w (theta) is what gets frozen
    (Variant A) or soft-anchored (Variant C) for the measurement-side heads.
    """
    qnode, cs, hs = make_forecaster(n_qubits=n_qubits, n_layers=n_layers, seq_len=seq_len)
    cw, hw = init_weights(cs, hs, seed=seed)
    opt = qml.AdamOptimizer(lr)
    Xtr = pnp.array(task.X_train, requires_grad=False)
    ytr = pnp.array(task.y_train, requires_grad=False)

    def cost(c, h):
        return pnp.mean((predict(qnode, c, h, Xtr) - ytr) ** 2)

    for epoch in range(epochs):
        (cw, hw), _ = opt.step_and_cost(cost, cw, hw)
        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"    backbone[{task.name}] epoch {epoch + 1}/{epochs} "
                  f"loss={float(cost(cw, hw)):.4f}", flush=True)
    return cw, hw, qnode


def train_isolated_variant(method: str, tasks, *, n_qubits: int = 4, n_layers: int = 2,
                           seq_len: int = 8, lr: float = 0.05, epochs: int = 40,
                           alpha: float = 5.0, seed: int = 42):
    """Train a full frozen/free/anchor OI-QCL run and return (probs_qnode, final_circ_w, heads).

    Shared helper for both the Task-IL comparison (``e015_oiqcl_forecast_compare``) and the
    task-agnostic router study (``e015_router``): the heads are each task's frozen linear head
    (fit at that task's backbone) and ``final_circ_w`` is the backbone after the last task.
    """
    probs_qnode, _ = make_probs_forecaster(n_qubits=n_qubits, n_layers=n_layers, seq_len=seq_len)
    circ_w = None
    heads: list[LinearHead] = []
    for phase, task in enumerate(tasks):
        if phase == 0:
            circ_w, _, _ = train_backbone_forecast(task, n_qubits=n_qubits, n_layers=n_layers,
                                                   seq_len=seq_len, lr=lr, epochs=epochs, seed=seed)
            P_tr = probs_features(probs_qnode, circ_w, task.X_train)
            heads.append(fit_linear_head(P_tr, task.y_train, task.name))
        else:
            train_theta = method != "frozen_head"
            use_alpha = alpha if method == "anchor_head" else 0.0
            anchor = np.asarray(circ_w) if method == "anchor_head" else None
            circ_w, head = train_task_isolated_head_forecast(
                probs_qnode, circ_w, task, train_theta=train_theta, alpha_anchor=use_alpha,
                anchor=anchor, lr=lr, epochs=epochs, head_seed=seed + phase)
            heads.append(head)
    return probs_qnode, circ_w, heads


def train_task_isolated_head_forecast(probs_qnode, circ_w, task, *, train_theta: bool,
                                      alpha_anchor: float = 0.0, anchor=None,
                                      lr: float = 0.05, epochs: int = 40, head_seed: int = 0,
                                      ridge_alpha: float = 1e-3):
    """Advance the shared backbone for one task, then fit its isolated linear head.

    Mirrors e014 sec 29:  min_{theta, w_t} MSE_t + alpha * ||theta - anchor||^2, with earlier
    heads frozen (kept aside by the caller).

      * Variant A (frozen backbone):  train_theta=False -> circ_w unchanged; only the head is fit.
      * Variant B (free backbone):    train_theta=True,  alpha_anchor=0.
      * Variant C (soft-anchor):      train_theta=True,  alpha_anchor>0, anchor=theta_{t-1}.

    The theta step uses a differentiable linear head over probs (closed-form-free autograd);
    the *returned* head is re-fit classically (Ridge) on the final theta for a converged,
    reproducible readout.  Returns (circ_w, LinearHead).
    """
    circ_w = pnp.array(np.asarray(circ_w), requires_grad=bool(train_theta))

    if train_theta:
        n_probs = 2 ** len(probs_qnode.device.wires)
        rng = np.random.default_rng(head_seed)
        W = pnp.array(0.01 * rng.standard_normal(n_probs), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
        Xtr = pnp.array(task.X_train, requires_grad=False)
        ytr = pnp.array(task.y_train, requires_grad=False)
        anchor_arr = None if anchor is None else np.asarray(anchor)
        opt = qml.AdamOptimizer(lr)

        def cost(circ_w, W, b):
            pred = _probs_batched(probs_qnode, circ_w, Xtr) @ W + b
            loss = pnp.mean((pred - ytr) ** 2)
            if alpha_anchor > 0.0 and anchor_arr is not None:
                loss = loss + alpha_anchor * pnp.sum((circ_w - anchor_arr) ** 2)
            return loss

        for _ in range(epochs):
            circ_w, W, b = opt.step(cost, circ_w, W, b)

    # Converged classical head on the final backbone (measurement side, no quantum gradients).
    P_tr = probs_features(probs_qnode, circ_w, task.X_train)
    head = fit_linear_head(P_tr, task.y_train, task.name, alpha=ridge_alpha)
    return circ_w, head
