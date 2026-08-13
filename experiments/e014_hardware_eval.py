"""E014 hardware evaluation on an IBM QPU (separate hardware path).

OI-QCL is unusually hardware-friendly: the head is a classical linear map, deployment needs
no quantum gradients (just forward sampling), all task heads share ONE measurement basis
(computational), and the readout can be a few local qubits on a shallow circuit. So we:

  1. train theta on the noiseless simulator (PennyLane) for one task, Variant-A frozen,
  2. rebuild the *same* ansatz in Qiskit (StatePreparation + RY/RZ + CNOT ladder, theta baked in),
  3. sample the readout qubits -> empirical probs -> fit the classical head on Aer-sampled probs
     (so training and deployment use the identical counts->probs path; no bit-ordering footgun),
  4. run a small test subset on the chosen backend (Aer dry-run by default, or a real QPU),
     apply the frozen head, report accuracy + full provenance (backend, shots, job id).

Fill .env (IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE), then:
  python experiments/e014_hardware_eval.py --backend aer            # dry-run first
  python experiments/e014_hardware_eval.py --backend ibm_marrakesh  # real QPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual_data import load_two_tasks  # noqa: E402
from src.e014_oiqcl import _labels_to_classes, train_backbone  # noqa: E402
from src.phase_data import N_QUBITS, load_spt_atf  # noqa: E402

RESULTS = ROOT / "results"


def _load_env() -> dict[str, str]:
    """Find .env by walking up from here (worktree lives inside the main repo)."""
    for d in [ROOT, *ROOT.parents]:
        p = d / ".env"
        if p.exists():
            env = {}
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
            return env
    return {}


def _load_task(name, n_train, n_test, seed):
    if name == "mnist":
        t, _ = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
        return t
    if name == "fashion":
        _, t = load_two_tasks(n_features=2**N_QUBITS, n_train=n_train, n_test=n_test, seed=seed)
        return t
    if name == "spt":
        return load_spt_atf(n_train=n_train, n_test=n_test, n_qubits=N_QUBITS, seed=seed)
    raise ValueError(name)


def _build_circuit(x_state, weights, layers, readout):
    """Qiskit ISA-ready circuit: amplitude prep + RY/RZ + CNOT ladder, measure `readout` qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit

    n = N_QUBITS
    qc = QuantumCircuit(n)
    qc.prepare_state(list(map(float, x_state)), list(range(n)))  # amplitude embedding
    for layer in range(layers):
        for q in range(n):
            qc.ry(float(weights[layer, q, 0]), q)
            qc.rz(float(weights[layer, q, 1]), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    creg = ClassicalRegister(len(readout), "c")
    qc.add_register(creg)
    for i, q in enumerate(readout):
        qc.measure(q, creg[i])
    return qc


def _counts_to_probs(counts, m, shots):
    """Map a counts dict (bitstrings) to a length-2^m probability vector, consistent ordering."""
    p = np.zeros(2**m)
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2)] += c
    return p / max(shots, 1)


def _sample_probs(circuits, sampler, shots, creg_name="c"):
    """Run a batch of circuits through a SamplerV2, return (N, 2^m) empirical probs + job id."""
    job = sampler.run(circuits, shots=shots)
    res = job.result()
    m = circuits[0].cregs[0].size
    probs = np.array([_counts_to_probs(getattr(r.data, creg_name).get_counts(), m, shots)
                      for r in res])
    jid = getattr(job, "job_id", lambda: None)()
    return probs, jid


def run(*, task="mnist", layers=4, readout=(0, 1), epochs=20, n_head=150, n_test=24,
        shots=4096, backend_name="aer", seed=42) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    started = time.perf_counter()
    t = _load_task(task, n_train=n_head, n_test=max(n_test, 40), seed=seed)
    # 1) train theta on the noiseless simulator (PennyLane)
    weights, _, _ = train_backbone(t, n_qubits=N_QUBITS, n_layers=layers, epochs=epochs, seed=seed)
    weights = np.asarray(weights)

    # 2) build circuits for head-fit (train) and test subsets
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    _aer_dev = AerSimulator()

    def circuits_for(X):
        return [_build_circuit(x, weights, layers, readout) for x in X]

    def aer_ready(X):  # decompose state_preparation into basis gates for Aer
        return transpile(circuits_for(X), _aer_dev)

    aer = AerSamplerV2()
    # 3) fit the head on Aer-sampled probs (identical counts->probs path as deployment)
    Ptr, _ = _sample_probs(aer_ready(t.X_train[:n_head]), aer, shots)
    head = LogisticRegression(max_iter=2000, random_state=seed)
    head.fit(Ptr, _labels_to_classes(t.y_train[:n_head]))

    Xte, yte = t.X_test[:n_test], t.y_test[:n_test]

    # 4) deploy on the chosen backend
    if backend_name == "aer":
        Pte, job_id, backend_used = (*_sample_probs(aer_ready(Xte), aer, shots), "aer_simulator")
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        env = _load_env()
        svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=env["IBM_QUANTUM_TOKEN"],
                                   instance=env.get("IBM_QUANTUM_INSTANCE") or None)
        backend = svc.backend(backend_name)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa = [pm.run(qc) for qc in circuits_for(Xte)]
        sampler = SamplerV2(mode=backend)
        Pte, job_id = _sample_probs(isa, sampler, shots)
        backend_used = backend.name

    acc = float(np.mean(head.predict(Pte) == _labels_to_classes(yte)))
    # noiseless-simulator reference accuracy of the same head/theta path (Aer, more shots)
    return {
        "experiment": "e014_hardware_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("qiskit", "qiskit-ibm-runtime",
                                                             "qiskit-aer", "pennylane")}},
        "config": {"task": t.name, "layers": layers, "readout_qubits": list(readout),
                   "n_qubits": N_QUBITS, "n_head_fit": n_head, "n_test": n_test, "shots": shots,
                   "seed": seed, "variant": "OI-QCL frozen-A; sim training, QPU readout"},
        "backend": backend_used, "ibm_job_id": job_id,
        "test_accuracy": round(acc, 4), "n_test": int(len(yte)),
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def run_cl(*, layers=4, readout=(0, 1, 2, 3), epochs=20, n_head=250, n_test=48,
           shots=4096, backend_name="aer", seed=42) -> dict[str, Any]:
    """Complete OI-QCL frozen-A across all 3 tasks: shared theta trained on T1 (sim), each
    task's head fit + evaluated; report per-task noiseless-sim vs backend accuracy."""
    from sklearn.linear_model import LogisticRegression

    from src.e014_oiqcl import make_probs_qnode, probs_features

    started = time.perf_counter()
    tasks = [("task1", _load_task("mnist", n_head, max(n_test, 60), seed)),
             ("task2", _load_task("fashion", n_head, max(n_test, 60), seed)),
             ("task3", _load_task("spt", n_head, max(n_test, 60), seed))]
    # shared frozen backbone from Task 1 (noiseless simulator training)
    weights, _, _ = train_backbone(tasks[0][1], n_qubits=N_QUBITS, n_layers=layers,
                                   epochs=epochs, seed=seed)
    weights = np.asarray(weights)
    clean_qnode, _ = make_probs_qnode(n_qubits=N_QUBITS, n_layers=layers, readout_wires=readout)

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    aer_dev, aer = AerSimulator(), AerSamplerV2()

    def circuits_for(X):
        return [_build_circuit(x, weights, layers, readout) for x in X]

    backend = pm = sampler = None
    if backend_name != "aer":
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        env = _load_env()
        svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=env["IBM_QUANTUM_TOKEN"],
                                   instance=env.get("IBM_QUANTUM_INSTANCE") or None)
        backend = svc.backend(backend_name)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        sampler = SamplerV2(mode=backend)

    per_task, jobs = {}, {}
    for key, t in tasks:
        cls_tr = _labels_to_classes(t.y_train[:n_head])
        cls_te = _labels_to_classes(t.y_test[:n_test])
        # noiseless-sim reference (PennyLane exact probs)
        hsim = LogisticRegression(max_iter=2000, random_state=seed).fit(
            probs_features(clean_qnode, weights, t.X_train[:n_head]), cls_tr)
        sim_acc = float(np.mean(hsim.predict(
            probs_features(clean_qnode, weights, t.X_test[:n_test])) == cls_te))
        # hardware: fit head on Aer-sampled probs, evaluate on the chosen backend
        Ptr, _ = _sample_probs(transpile(circuits_for(t.X_train[:n_head]), aer_dev), aer, shots)
        hhw = LogisticRegression(max_iter=2000, random_state=seed).fit(Ptr, cls_tr)
        if backend_name == "aer":
            Pte, jid = _sample_probs(transpile(circuits_for(t.X_test[:n_test]), aer_dev), aer, shots)
        else:
            Pte, jid = _sample_probs([pm.run(qc) for qc in circuits_for(t.X_test[:n_test])],
                                     sampler, shots)
        hw_acc = float(np.mean(hhw.predict(Pte) == cls_te))
        per_task[key] = {"name": t.name, "sim_acc": round(sim_acc, 4),
                         "backend_acc": round(hw_acc, 4), "n_test": int(len(cls_te))}
        jobs[key] = jid
        print(f"  {key} ({t.name:16s}): sim={sim_acc:.3f}  {backend_name}={hw_acc:.3f}", flush=True)

    return {
        "experiment": "e014_hardware_eval_cl",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("qiskit", "qiskit-ibm-runtime",
                                                             "qiskit-aer", "pennylane")}},
        "config": {"tasks": [t.name for _, t in tasks], "layers": layers,
                   "readout_qubits": list(readout), "n_qubits": N_QUBITS, "n_head_fit": n_head,
                   "n_test": n_test, "shots": shots, "seed": seed,
                   "variant": "OI-QCL frozen-A (shared theta from T1); sim training, QPU readout"},
        "backend": backend_name if backend_name == "aer" else backend.name,
        "ibm_job_ids": jobs, "per_task": per_task,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-tasks", action="store_true", help="frozen-A across MNIST/Fashion/SPT")
    ap.add_argument("--task", choices=["mnist", "fashion", "spt"], default="mnist")
    ap.add_argument("--backend", default="aer", help="'aer' (dry-run) or an IBM backend name")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--readout", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-head", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.all_tasks:
        result = run_cl(layers=args.layers, readout=tuple(args.readout), epochs=args.epochs,
                        n_head=args.n_head, n_test=args.n_test, shots=args.shots,
                        backend_name=args.backend, seed=args.seed)
        out = args.output or RESULTS / f"e014_hardware_{args.backend}_alltasks.json"
        out = out if out.is_absolute() else ROOT / out
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nbackend={result['backend']}  shots={result['config']['shots']}")
        for k, v in result["per_task"].items():
            print(f"  {v['name']:16s}: sim={v['sim_acc']:.3f}  hw={v['backend_acc']:.3f}")
        print(f"wrote {out.relative_to(ROOT)}")
        return
    result = run(task=args.task, layers=args.layers, readout=tuple(args.readout),
                 epochs=args.epochs, n_head=args.n_head, n_test=args.n_test,
                 shots=args.shots, backend_name=args.backend, seed=args.seed)
    out = args.output or RESULTS / f"e014_hardware_{args.backend}_{args.task}.json"
    out = out if out.is_absolute() else ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\ntask={result['config']['task']}  backend={result['backend']}  "
          f"shots={result['config']['shots']}  n_test={result['n_test']}")
    print(f"QPU/sim test accuracy = {result['test_accuracy']:.3f}  (job {result['ibm_job_id']})")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
