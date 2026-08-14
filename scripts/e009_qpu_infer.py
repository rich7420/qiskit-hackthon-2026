"""Step B — evaluate the sim-trained simplified models on real IBM hardware (inference only).

Rebuilds the SAME simplified ansatz in Qiskit (single-axis RY encoding, 1 variational layer, CNOT
chain; state persists across the 8 re-upload steps), evaluates each (method, seed) model on the
fixed test-window subset from Step A, and records the hardware NMSE per task.

  --validate : local circuit-equivalence check — Qiskit Statevector predictions must match the
               PennyLane noiseless predictions (no hardware, no API).
  --run      : transpile to the backend and submit via SamplerV2; compute hardware NMSE.

Run:
    python scripts/e009_qpu_infer.py --validate
    python scripts/e009_qpu_infer.py --run --backend ibm_marrakesh --shots 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

RESULTS = ROOT / "results"
N_QUBITS = 4


def build_circuit(window, circ_w, n_layers, entangler, encoding, measure=True):
    """The simplified forecaster ansatz as a Qiskit circuit (window + weights baked in as floats)."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS if measure else 0)
    for step in range(len(window)):
        v = float(window[step])
        for q in range(N_QUBITS):
            qc.ry(((q + 1) / N_QUBITS) * v, q)
            if encoding == "ry_rz":
                qc.rz(((N_QUBITS - q) / N_QUBITS) * v, q)
        for layer in range(n_layers):
            for q in range(N_QUBITS):
                qc.ry(float(circ_w[layer][q][0]), q)
                qc.rz(float(circ_w[layer][q][1]), q)
            links = {"ring": N_QUBITS, "chain": N_QUBITS - 1, "none": 0}[entangler]
            for q in range(links):
                qc.cx(q, (q + 1) % N_QUBITS)
    if measure:
        qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def z_from_counts(counts, shots):
    z = np.zeros(N_QUBITS)
    for key, c in counts.items():
        bits = key.replace(" ", "")
        for q in range(N_QUBITS):
            z[q] += c * (1.0 if bits[N_QUBITS - 1 - q] == "0" else -1.0)   # rightmost char = qubit 0
    return z / shots


def z_exact(window, circ_w, meta):
    qc = build_circuit(window, circ_w, meta["n_layers"], meta["ansatz"]["entangler"],
                       meta["ansatz"]["encoding"], measure=False)
    sv = Statevector(qc)
    out = np.zeros(N_QUBITS)
    for q in range(N_QUBITS):
        p = ["I"] * N_QUBITS
        p[N_QUBITS - 1 - q] = "Z"                       # Qiskit Pauli string: leftmost = highest qubit
        out[q] = sv.expectation_value(SparsePauliOp("".join(p))).real
    return out


def predict_from_z(z, head_w):
    head = np.asarray(head_w)
    return float(np.tanh(np.asarray(z) @ head[:-1] + head[-1]))


def nmse(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    return float(np.mean((pred - y) ** 2) / (np.var(y) + 1e-12))


def load_env(key):
    for line in open(ROOT.parent.parent / ".env") if (ROOT.parent.parent / ".env").exists() \
            else open(Path("/Users/rich/qiskit-hackthon-2026/.env")):
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def validate(meta):
    """Qiskit Statevector predictions must equal the PennyLane noiseless predictions."""
    sys.path.insert(0, str(ROOT))
    from src.e009_qtsf import make_forecaster, predict
    qn, _, _ = make_forecaster(n_qubits=4, n_layers=meta["n_layers"], seq_len=meta["seq_len"],
                               **meta["ansatz"])
    worst = 0.0
    for key, mdl in meta["models"].items():
        cw = np.array(mdl["circ_w"])
        for t in meta["tasks"]:
            X = np.array(meta["test_subset"][t]["X"])
            pl = np.asarray(predict(qn, cw, np.array(mdl["head_w"]), X))        # PennyLane
            qk = np.array([predict_from_z(z_exact(x, mdl["circ_w"], meta), mdl["head_w"]) for x in X])
            worst = max(worst, float(np.max(np.abs(pl - qk))))
    print(f"validate: max |PennyLane - Qiskit(statevector)| prediction diff = {worst:.2e}")
    assert worst < 1e-6, "Qiskit circuit does NOT match the PennyLane ansatz!"
    print("OK — Qiskit circuit reproduces the PennyLane forecaster exactly.")


def run_hardware(meta, backend_name, shots, output):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=load_env("IBM_QUANTUM_TOKEN"),
                                   instance=load_env("IBM_QUANTUM_INSTANCE"))
    backend = service.backend(backend_name)
    print(f"backend {backend.name}  ({backend.num_qubits} qubits)", flush=True)

    # build every (model, task, window) circuit; remember its index
    circuits, tags = [], []
    for key, mdl in meta["models"].items():
        for t in meta["tasks"]:
            for x in meta["test_subset"][t]["X"]:
                circuits.append(build_circuit(x, mdl["circ_w"], meta["n_layers"],
                                              meta["ansatz"]["entangler"], meta["ansatz"]["encoding"]))
                tags.append((key, t))
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(circuits)
    depth2q = max(c.depth(lambda i: i.operation.num_qubits == 2) for c in isa)
    print(f"{len(circuits)} circuits, transpiled 2q-depth<= {depth2q}; submitting {shots} shots...", flush=True)

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa, shots=shots)
    print(f"job id: {job.job_id()}  (queued; waiting)", flush=True)
    res = job.result()

    # collect predictions per (model, task)
    preds = {}
    for i, (key, t) in enumerate(tags):
        counts = res[i].data.c.get_counts()
        z = z_from_counts(counts, shots)
        preds.setdefault((key, t), []).append(predict_from_z(z, meta["models"][key]["head_w"]))
    hw = {}
    for key, mdl in meta["models"].items():
        hw[key] = {t: round(nmse(preds[(key, t)], meta["test_subset"][t]["y"]) , 4) for t in meta["tasks"]}
        print(f"  [{key}] hardware NMSE = {hw[key]}", flush=True)

    out = {"experiment": "e009_qpu_infer", "backend": backend.name, "shots": shots,
           "job_id": job.job_id(), "transpiled_2q_depth": int(depth2q),
           "windows_per_task": meta["windows_per_task"], "hardware_nmse": hw}
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=RESULTS / "e009_qpu_models.json")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--output", type=Path, default=RESULTS / "e009_qpu_hardware.json")
    args = ap.parse_args()

    meta = json.loads(args.models.read_text())
    if args.validate:
        validate(meta)
    if args.run:
        run_hardware(meta, args.backend, args.shots, args.output)
    if not (args.validate or args.run):
        print("nothing to do — pass --validate and/or --run")


if __name__ == "__main__":
    main()
