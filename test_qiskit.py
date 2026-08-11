"""Local environment smoke test.

Builds a Bell state and samples it on a local statevector simulator.
A healthy setup prints counts dominated by '00' and '11' (roughly 50/50).
"""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


def main() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=1024).result()

    print(result[0].data.meas.get_counts())
    print(qc.draw())


if __name__ == "__main__":
    main()
