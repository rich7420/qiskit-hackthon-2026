"""Render a schematic of the e009 recurrent data-reuploading quantum forecaster.

Shows a few time steps of the unrolled circuit so the recurrent structure is visible: at each
step the scalar input x_t is angle-encoded (RY/RZ), then the SAME trainable block (RY/RZ + CNOT
ring) is applied, with the quantum state persisting across steps. The real model uses seq_len=8
steps and 2 layers; here we draw 3 steps / 1 layer for readability.

Run:
    python scripts/e009_plot_circuit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "figures" / "e009_circuit.png"
N_QUBITS, SHOW_STEPS, SHOW_LAYERS = 4, 3, 1


def main() -> None:
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def qnode(window, circ_w):
        for step in range(SHOW_STEPS):
            v = window[step]
            for q in range(N_QUBITS):
                qml.RY(((q + 1) / N_QUBITS) * v, wires=q)
                qml.RZ(((N_QUBITS - q) / N_QUBITS) * v, wires=q)
            for layer in range(SHOW_LAYERS):
                for q in range(N_QUBITS):
                    qml.RY(circ_w[layer, q, 0], wires=q)
                    qml.RZ(circ_w[layer, q, 1], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    window = pnp.array(np.zeros(SHOW_STEPS), requires_grad=False)
    circ_w = pnp.array(np.zeros((SHOW_LAYERS, N_QUBITS, 2)), requires_grad=True)

    fig, ax = qml.draw_mpl(qnode, decimals=None, style="pennylane")(window, circ_w)
    fig.suptitle("E009 quantum temporal forecaster — recurrent data re-uploading\n"
                 "(3 of 8 time steps shown; encode $x_t$ → shared block $\\theta$ → CNOT ring, "
                 "state persists; then $\\langle Z\\rangle$ → tanh head)", fontsize=11)

    # annotate the repeating time-step structure
    ax.text(0.5, 1.02, "↓ each block = one time step (same θ, re-uploads next $x_t$) ↓",
            transform=ax.transAxes, ha="center", fontsize=9, color="dimgray")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
