"""Environment smoke tests.

Confirms the core stack imports and a Bell state behaves as expected.
Problem-specific formulation tests (QUBO objective, Hamiltonian ground state,
circuit-equivalence) should be added here once the problem is announced.
"""


def test_qiskit_imports():
    import qiskit

    assert qiskit.__version__


def test_bell_state_counts():
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = StatevectorSampler().run([qc], shots=2048).result()[0].data.meas.get_counts()

    # Bell state: only |00> and |11> should appear (no |01> or |10>).
    assert set(counts) <= {"00", "11"}
    assert counts.get("00", 0) + counts.get("11", 0) == 2048
