"""Render a publication-quality diagram of the e001 QNN circuit.

Mirrors the usual paper layout: the data encoding as a single labeled block, the
variational ansatz drawn gate-level (Ry rotations + CX entanglers), then measurement.
Uses Qiskit's LaTeX drawer (needs pdflatex + pdftocairo) for the clean quantikz look,
falling back to the matplotlib drawer if LaTeX isn't available.

Run:
    python scripts/plot_qnn_circuit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.visualization import circuit_drawer

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def build_display_circuit(n_qubits: int = 4, reps: int = 2) -> QuantumCircuit:
    """The e001 QNN as a presentation schematic: encoding block + gate-level ansatz."""
    # Encoding as one labeled block — its gate-level form is noisy and not the point.
    encoding = zz_feature_map(n_qubits, reps=1).to_gate(label="ZZ Feature Map  φ(x)")

    qc = QuantumCircuit(n_qubits, name="QNN")
    qc.append(encoding, range(n_qubits))
    qc.barrier()
    # Ansatz gate-level: Ry(θ) rotations + CX entanglers, the trainable part.
    qc.compose(real_amplitudes(n_qubits, reps=reps), inplace=True)
    qc.measure_all()
    return qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="use the optional LaTeX drawer")
    args = parser.parse_args()

    FIGURES.mkdir(exist_ok=True)
    qc = build_display_circuit()
    out = FIGURES / "e001_qnn_circuit.png"

    if args.latex:
        circuit_drawer(qc, output="latex", filename=str(out), scale=1.3)
        print(f"Rendered (LaTeX / quantikz) -> {out}")
        return

    fig = circuit_drawer(qc, output="mpl", style="iqp", scale=1.2, fold=-1)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Rendered (matplotlib) -> {out}")


if __name__ == "__main__":
    main()
