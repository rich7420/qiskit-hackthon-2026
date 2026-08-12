"""Render the e002 amplitude-encoding QNN circuit (PennyLane).

Shows amplitude embedding as a block, then a representative 2 of the RyRz + CNOT ansatz
layers gate-level, then the <Z_0> measurement. Paper-style presentation figure.

Run:
    python scripts/plot_e002_circuit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qnn_pennylane import make_qnode  # noqa: E402

OUT = ROOT / "figures" / "e002_qnn_circuit.png"


def main() -> None:
    n_qubits, show_layers = 4, 2  # 2 representative layers (the real ansatz uses more)
    qnode, wshape = make_qnode(n_qubits=n_qubits, n_layers=show_layers)

    weights = np.zeros(wshape)
    features = np.zeros(2**n_qubits)
    features[0] = 1.0  # a valid normalized state just for drawing

    fig, _ = qml.draw_mpl(qnode, level="top", style="pennylane", decimals=None)(
        features, weights
    )
    fig.suptitle("Amplitude-encoding QNN (2 of L ansatz layers shown)", fontsize=11)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
