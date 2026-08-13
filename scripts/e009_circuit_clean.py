"""Clean quantum-circuit diagram of the e009 recurrent data-reuploading forecaster.

Uses PennyLane's matplotlib drawer for a proper, paper-style circuit (like the QEWC paper's
ansatz figure). Shows 2 of the 8 time steps: each step angle-encodes x_t (RY/RZ) then applies
the shared variational block (RY/RZ theta + CNOT ring); the state persists across steps.

Run:
    python scripts/e009_circuit_clean.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "e009_circuit.png"
N_QUBITS, SHOW_STEPS, SHOW_LAYERS = 4, 2, 1


def main() -> None:
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def qnode(window, circ_w):
        for step in range(SHOW_STEPS):
            v = window[step]
            for q in range(N_QUBITS):                       # data re-uploading (encode x_t)
                qml.RY(((q + 1) / N_QUBITS) * v, wires=q)
                qml.RZ(((N_QUBITS - q) / N_QUBITS) * v, wires=q)
            for _ in range(SHOW_LAYERS):                    # shared variational block
                for q in range(N_QUBITS):
                    qml.RY(circ_w[q, 0], wires=q)
                    qml.RZ(circ_w[q, 1], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    window = pnp.array(np.zeros(SHOW_STEPS), requires_grad=False)
    circ_w = pnp.array(np.zeros((N_QUBITS, 2)), requires_grad=True)

    fig, ax = qml.draw_mpl(qnode, decimals=None, style="pennylane")(window, circ_w)
    fig.suptitle("Recurrent data re-uploading quantum forecaster (2 of 8 steps shown)\n"
                 "each step: encode $x_t$ (RY/RZ) → shared block $U(\\theta)$ (RY/RZ + CNOT ring); "
                 "state persists → $\\langle Z\\rangle$ → tanh head → $\\hat{y}$", fontsize=11)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
