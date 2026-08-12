"""Q-MemGuard — QFI trust-region continual learning (novel; this project, e007).

Instead of an always-on penalty (QEWC), Q-MemGuard predicts how much a candidate update
deforms each old task's quantum state, and intervenes only when the update leaves a
calibrated local stability region:

    R_{j}   = delta^T F_Q^(j) delta            (local quantum geometric displacement)
    A_{j}   = R_{j} / eps_j                     (normalized certificate ratio)
    safe    : all A_j <= 1  -> apply the raw candidate update (zero extra constraint)
    unsafe  : some A_j > 1  -> redirect inside the QFI trust region

With the diagonal QFI (tractable at many parameters), the trust-region solution has the
closed form delta_i = cand_i / (1 + lambda * F_i); lambda >= 0 is the dual variable found so
the binding constraint becomes active. (1/2)sqrt(R) is a conservative estimate of the Bures
state drift D_B; calibration of eps absorbs the diagonal-approximation constant.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def make_state_qnode(n_qubits: int = 4, n_layers: int = 20):
    """QNode returning the statevector (same ansatz as the classifier), for Bures distance."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return state_qnode


def partial_trace_keep(state, n_qubits: int, keep) -> np.ndarray:
    """Reduced density matrix on the `keep` qubits, tracing out the rest."""
    psi = np.asarray(state).reshape([2] * n_qubits)
    keep = list(keep)
    rest = [q for q in range(n_qubits) if q not in keep]
    psi = np.transpose(psi, keep + rest).reshape(2 ** len(keep), 2 ** len(rest))
    return psi @ psi.conj().T


def directional_qfi(state_qnode, theta, delta, X, keep=(0, 1), n_qubits: int = 4,
                    eps: float = 1e-3):
    """Directional QFI of the update `delta` at `theta`, averaged over inputs X.

    Returns (R_global, R_readout): R = delta^T F_Q delta for the full pure-state QFI and for
    the readout-reduced mixed-state QFI. Divide by ||delta||^2 for the Rayleigh quotient.
    """
    t0 = pnp.array(theta, requires_grad=False)
    tp = pnp.array(theta + eps * delta, requires_grad=False)
    tm = pnp.array(theta - eps * delta, requires_grad=False)
    Rg, Rr = [], []
    for x in X:
        xp = pnp.array(x, requires_grad=False)
        psi0 = np.asarray(state_qnode(xp, t0))
        dpsi = (np.asarray(state_qnode(xp, tp)) - np.asarray(state_qnode(xp, tm))) / (2 * eps)
        # global pure-state QFI along delta: 4(<dpsi|dpsi> - |<psi|dpsi>|^2)
        Rg.append(4.0 * (np.vdot(dpsi, dpsi).real - abs(np.vdot(psi0, dpsi)) ** 2))
        # readout mixed-state QFI along delta: 2 sum_ab |<a|D|b>|^2 / (la+lb)
        rho0 = partial_trace_keep(psi0, n_qubits, keep)
        rp = partial_trace_keep(np.asarray(state_qnode(xp, tp)), n_qubits, keep)
        rm = partial_trace_keep(np.asarray(state_qnode(xp, tm)), n_qubits, keep)
        D = (rp - rm) / (2 * eps)
        lam, U = np.linalg.eigh(rho0)
        Dab = U.conj().T @ D @ U
        q = 0.0
        for a in range(len(lam)):
            for b in range(len(lam)):
                s = lam[a] + lam[b]
                if s > 1e-10:
                    q += 2.0 * abs(Dab[a, b]) ** 2 / s
        Rr.append(q)
    return float(np.mean(Rg)), float(np.mean(Rr))


def mean_bures_distance(state_qnode, w1, w2, X) -> float:
    """Mean pure-state Bures distance over inputs X: D_B^2 = 2(1 - |<psi1|psi2>|)."""
    w1 = pnp.array(np.asarray(w1), requires_grad=False)
    w2 = pnp.array(np.asarray(w2), requires_grad=False)
    dists = []
    for x in X:
        xp = pnp.array(x, requires_grad=False)
        s1 = np.asarray(state_qnode(xp, w1))
        s2 = np.asarray(state_qnode(xp, w2))
        fid = np.abs(np.vdot(s1, s2))
        dists.append(np.sqrt(max(0.0, 2.0 * (1.0 - fid))))
    return float(np.mean(dists))


def mean_readout_bures(state_qnode, w1, w2, X, keep=(0, 1), n_qubits: int = 4) -> float:
    """Mean mixed-state Bures distance between the readout-reduced states over inputs X."""
    from scipy.linalg import sqrtm

    w1 = pnp.array(np.asarray(w1), requires_grad=False)
    w2 = pnp.array(np.asarray(w2), requires_grad=False)
    dists = []
    for x in X:
        xp = pnp.array(x, requires_grad=False)
        r1 = partial_trace_keep(np.asarray(state_qnode(xp, w1)), n_qubits, keep)
        r2 = partial_trace_keep(np.asarray(state_qnode(xp, w2)), n_qubits, keep)
        s = sqrtm(r1)
        fid = np.real(np.trace(sqrtm(s @ r2 @ s))) ** 2
        fid = min(max(float(fid), 0.0), 1.0)
        dists.append(np.sqrt(max(0.0, 2.0 * (1.0 - np.sqrt(fid)))))
    return float(np.mean(dists))


class QMemGuard:
    """Holds per-task (theta*, diagonal QFI, eps) anchors and redirects unsafe updates."""

    def __init__(self):
        self.anchors: list[dict] = []  # {"theta": flat, "fisher": flat, "eps": float, "name": str}

    def consolidate(self, theta_star, fisher_diag, eps: float, name: str = "") -> None:
        self.anchors.append({
            "theta": np.asarray(theta_star, dtype=float).ravel().copy(),
            "fisher": np.asarray(fisher_diag, dtype=float).ravel().copy(),
            "eps": float(eps), "name": name,
        })

    def certificate_ratios(self, delta_flat) -> list[float]:
        """A_j = R_j / eps_j for every past task at the candidate step delta."""
        d = np.asarray(delta_flat, dtype=float)
        return [float(np.sum(a["fisher"] * d**2) / a["eps"]) for a in self.anchors]

    def redirect(self, cand_flat, max_lambda: float = 1e6, tol: float = 1e-3):
        """Return (delta, activated, max_ratio_before). Safe -> candidate unchanged."""
        cand = np.asarray(cand_flat, dtype=float)
        if not self.anchors:
            return cand, False, 0.0
        ratios = self.certificate_ratios(cand)
        max_ratio = max(ratios)
        if max_ratio <= 1.0:
            return cand, False, max_ratio

        f_eff = np.sum([a["fisher"] for a in self.anchors], axis=0)  # shrink all directions

        def worst(lmbda: float) -> float:
            d = cand / (1.0 + lmbda * f_eff)
            return max(np.sum(a["fisher"] * d**2) / a["eps"] for a in self.anchors)

        lo, hi = 0.0, 1.0
        while worst(hi) > 1.0 and hi < max_lambda:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if worst(mid) > 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        delta = cand / (1.0 + hi * f_eff)
        return delta, True, max_ratio


def calibrate_epsilon(state_qnode, accuracy_fn, theta_star, fisher_diag, X_val, y_val,
                      scales=(0.01, 0.02, 0.04, 0.08, 0.16), n_dirs: int = 4,
                      acc_tol: float = 0.03, seed: int = 0) -> dict:
    """Calibrate eps at theta* by probing random perturbations against validation accuracy.

    eps = mean R at the largest perturbation scale whose validation-accuracy drop stays within
    acc_tol. Also returns (predicted (1/2)sqrt(R), actual Bures) pairs for the physics figure.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta_star, dtype=float)
    fisher = np.asarray(fisher_diag, dtype=float).ravel()
    acc0 = accuracy_fn(theta.reshape(theta_star.shape))
    bures_pairs, per_scale = [], []
    for s in scales:
        Rs, accs = [], []
        for _ in range(n_dirs):
            u = rng.standard_normal(theta.size)
            u /= np.linalg.norm(u)
            delta = (s * u).reshape(theta.shape)
            Rval = float(np.sum(fisher * (s * u) ** 2))
            Rs.append(Rval)
            accs.append(accuracy_fn(theta + delta))
            db = mean_bures_distance(state_qnode, theta, theta + delta, X_val[:8])
            bures_pairs.append((0.5 * np.sqrt(Rval), db))
        per_scale.append((s, float(np.mean(Rs)), float(np.mean(accs))))

    eps = per_scale[0][1]  # fallback: smallest-scale R
    for _s, meanR, meanacc in per_scale:
        if acc0 - meanacc <= acc_tol:
            eps = meanR
        else:
            break
    return {"eps": eps, "acc0": acc0, "per_scale": per_scale, "bures_pairs": bures_pairs}
