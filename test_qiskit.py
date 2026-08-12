"""Environment preflight check for the Qiskit Hackathon Taiwan 2026.

Runs a series of independent checks and prints a PASS / WARN / FAIL summary so
that, before the problem is announced, we know the whole stack works:

  1. Qiskit import + version
  2. Statevector simulator (Bell state)
  3. Aer simulator
  4. Transpilation to a basis gate set
  5. IBM Quantum Runtime connectivity (WARN if no account is configured)

Usage:
    python test_qiskit.py
"""

from __future__ import annotations


class Check:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append((name, status, detail))
        icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(status, "?")
        line = f"[{icon}] {name}: {status}"
        if detail:
            line += f" — {detail}"
        print(line)

    def ok(self) -> bool:
        return all(status != "FAIL" for _, status, _ in self.results)


def check_qiskit(c: Check) -> None:
    try:
        import qiskit

        c.record("qiskit import", "PASS", f"version {qiskit.__version__}")
    except Exception as exc:  # noqa: BLE001
        c.record("qiskit import", "FAIL", str(exc))


def check_statevector(c: Check) -> None:
    try:
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = StatevectorSampler().run([qc], shots=1024).result()
        counts = result[0].data.meas.get_counts()
        bell = counts.get("00", 0) + counts.get("11", 0)
        if bell >= 900:
            c.record("statevector simulator", "PASS", f"Bell counts {counts}")
        else:
            c.record("statevector simulator", "WARN", f"unexpected counts {counts}")
    except Exception as exc:  # noqa: BLE001
        c.record("statevector simulator", "FAIL", str(exc))


def check_aer(c: Check) -> None:
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        sim = AerSimulator()
        counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
        c.record("aer simulator", "PASS", f"counts {counts}")
    except Exception as exc:  # noqa: BLE001
        c.record("aer simulator", "FAIL", str(exc))


def check_transpile(c: Check) -> None:
    try:
        from qiskit import QuantumCircuit, transpile

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        tqc = transpile(
            qc,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3,
        )
        two_qubit = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
        c.record("transpilation", "PASS", f"depth {tqc.depth()}, 2Q gates {two_qubit}")
    except Exception as exc:  # noqa: BLE001
        c.record("transpilation", "FAIL", str(exc))


def check_ibm_runtime(c: Check) -> None:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:  # noqa: BLE001
        c.record("ibm runtime import", "FAIL", str(exc))
        return

    try:
        service = QiskitRuntimeService()
    except Exception as exc:  # noqa: BLE001
        c.record(
            "ibm runtime account",
            "WARN",
            f"no saved account ({exc}). Save with "
            "QiskitRuntimeService.save_account(token=..., instance=...)",
        )
        return

    try:
        backends = service.backends()
        names = ", ".join(b.name for b in backends[:3])
        c.record("ibm runtime account", "PASS", f"{len(backends)} backends ({names} ...)")
    except Exception as exc:  # noqa: BLE001
        c.record("ibm runtime account", "WARN", f"connected but query failed: {exc}")


def main() -> int:
    c = Check()
    print("=== Qiskit Hackathon 2026 preflight ===\n")
    check_qiskit(c)
    check_statevector(c)
    check_aer(c)
    check_transpile(c)
    check_ibm_runtime(c)

    passed = sum(1 for _, s, _ in c.results if s == "PASS")
    warned = sum(1 for _, s, _ in c.results if s == "WARN")
    failed = sum(1 for _, s, _ in c.results if s == "FAIL")
    print(f"\nSummary: {passed} pass, {warned} warn, {failed} fail")
    return 0 if c.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
