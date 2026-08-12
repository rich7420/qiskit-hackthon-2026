# Experiment Log

One entry per meaningful run. Keep it append-only. Copy the template below.

Goal: on the last night, anyone can answer "which backend / shots / seed produced
result 0.91?" without guessing.

---

## Template

```
## EXXX — <short title>

Commit:
Backend:            # e.g. AerSimulator, ibm_<name>
Qiskit version:

Configuration:
- p / layers =
- optimizer =
- shots =
- seed =
- optimization_level =
- seed_transpiler =

Circuit:
- qubits =
- depth =
- two-qubit gates =

Result:
- objective =
- approximation ratio =

IBM Job ID:         # hardware runs only

Notes:
- ...
```
