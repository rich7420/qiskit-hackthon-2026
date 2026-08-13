"""Noisy (mixed-state) readout for OI-QCL — separate from the noiseless simulator path.

A ``default.mixed`` version of the OI-QCL probability readout with a simple gate + readout
noise model, matching the config {'bit', 'phase', 'depol', 'meas'}:
  * depol  -- single-qubit DepolarizingChannel on every qubit after each layer (gate noise)
  * bit    -- BitFlip channel after each layer (off by default, = 0)
  * phase  -- PhaseFlip channel after each layer (off by default, = 0)
  * meas   -- BitFlip on each readout wire just before measurement (readout/measurement error)

The ansatz is identical to ``src.e014_oiqcl.make_probs_qnode`` so a theta trained noiselessly
can be evaluated here to study noisy deployment (train on the simulator, read out under noise).
"""

from __future__ import annotations

import pennylane as qml

DEFAULT_NOISE = {"bit": 0.0, "phase": 0.0, "depol": 0.01, "meas": 0.02}


def make_noisy_probs_qnode(n_qubits: int = 4, n_layers: int = 20, noise: dict | None = None,
                           readout_wires=None):
    """Return (qnode, weight_shape); mixed-state probs under the {bit,phase,depol,meas} model."""
    cfg = {**DEFAULT_NOISE, **(noise or {})}
    dev = qml.device("default.mixed", wires=n_qubits)
    wires = tuple(range(n_qubits)) if readout_wires is None else tuple(int(w) for w in readout_wires)

    @qml.qnode(dev, diff_method="backprop")
    def qnode(features, weights):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            for q in range(n_qubits):  # gate noise after each layer
                if cfg["depol"] > 0:
                    qml.DepolarizingChannel(cfg["depol"], wires=q)
                if cfg["bit"] > 0:
                    qml.BitFlip(cfg["bit"], wires=q)
                if cfg["phase"] > 0:
                    qml.PhaseFlip(cfg["phase"], wires=q)
        for w in wires:  # readout / measurement error
            if cfg["meas"] > 0:
                qml.BitFlip(cfg["meas"], wires=w)
        return qml.probs(wires=wires)

    return qnode, (n_layers, n_qubits, 2)
