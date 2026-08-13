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


def _build_softmax_circuit(x_state, weights, layers):
    """Same ansatz, but the QEWC/paper readout: measure qubits 0 and 1 (two Pauli-Z scores)."""
    from qiskit import ClassicalRegister, QuantumCircuit

    n = N_QUBITS
    qc = QuantumCircuit(n)
    qc.prepare_state(list(map(float, x_state)), list(range(n)))
    for layer in range(layers):
        for q in range(n):
            qc.ry(float(weights[layer, q, 0]), q)
            qc.rz(float(weights[layer, q, 1]), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    creg = ClassicalRegister(2, "c")
    qc.add_register(creg)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    return qc


def _counts_to_z(counts, shots):
    """<Z_0>, <Z_1> from 2-bit counts (key: ...c1 c0; last char = classical bit 0)."""
    z0 = z1 = 0.0
    for bits, c in counts.items():
        s = bits.replace(" ", "")
        z0 += c * (1 - 2 * int(s[-1]))
        z1 += c * (1 - 2 * int(s[-2]))
    return z0 / shots, z1 / shots


def _sample_counts(circuits, sampler, shots):
    job = sampler.run(circuits, shots=shots)
    res = job.result()
    counts = [r.data.c.get_counts() for r in res]
    return counts, getattr(job, "job_id", lambda: None)()


def _train_qewc(tasks, layers, epochs, lam_qewc, qfi_samples, seed, lr=0.05):
    import pennylane as qml
    from pennylane import numpy as pnp

    from src.e005_consolidation import EWC, quantum_fisher_diag
    from src.e005_softmax import bce_loss, make_softmax_qnode

    clf, shape = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    qfi, _ = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)
    reg = EWC(lam_qewc)
    w = pnp.array(0.01 * np.random.default_rng(seed).standard_normal(shape), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    for phase, t in enumerate(tasks):
        Xtr = pnp.array(t.X_train, requires_grad=False)
        ytr = pnp.array(t.y_train, requires_grad=False)

        def cost(W, Xtr=Xtr, ytr=ytr, phase=phase):
            return bce_loss(clf, W, Xtr, ytr) + reg.penalty(W.flatten(), phase + 1)

        for _ in range(epochs):
            w = opt.step(cost, w)
        if phase < len(tasks) - 1:
            f = quantum_fisher_diag(qfi, w, t.X_train, n_samples=qfi_samples, seed=seed)
            reg.consolidate(np.asarray(w).flatten(), f)
    return np.asarray(w)


def run_qewc_cl(*, layers=4, epochs=20, lam_qewc=0.8, qfi_samples=32, n_train=250, n_test=24,
                shots=4096, backend_name="aer", seed=42) -> dict[str, Any]:
    """QEWC (shared 2-Z softmax readout, QFI-consolidated theta) evaluated on hardware, per task."""
    from src.e005_softmax import accuracy as softmax_accuracy
    from src.e005_softmax import make_softmax_qnode

    started = time.perf_counter()
    tasks = [_load_task(n, n_train, max(n_test, 60), seed) for n in ("mnist", "fashion", "spt")]
    w = _train_qewc(tasks, layers, epochs, lam_qewc, qfi_samples, seed)  # final consolidated theta
    clf, _ = make_softmax_qnode(n_qubits=N_QUBITS, n_layers=layers)

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    aer_dev, aer = AerSimulator(), AerSamplerV2()
    pm = sampler = backend = None
    if backend_name != "aer":
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        env = _load_env()
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=env["IBM_QUANTUM_TOKEN"],
                                   instance=env.get("IBM_QUANTUM_INSTANCE") or None)
        backend = svc.backend(backend_name)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        sampler = SamplerV2(mode=backend)

    per_task, jobs = {}, {}
    keys = ("task1", "task2", "task3")
    for key, t in zip(keys, tasks):
        Xte, yte = t.X_test[:n_test], t.y_test[:n_test]
        sim_acc = float(softmax_accuracy(clf, w, Xte, yte))
        circs = [_build_softmax_circuit(x, w, layers) for x in Xte]
        if backend_name == "aer":
            counts, jid = _sample_counts(transpile(circs, aer_dev), aer, shots)
        else:
            counts, jid = _sample_counts([pm.run(c) for c in circs], sampler, shots)
        preds = np.array([0 if (lambda z: z[0] >= z[1])(_counts_to_z(c, shots)) else 1 for c in counts])
        hw_acc = float(np.mean(preds == _labels_to_classes(yte)))
        per_task[key] = {"name": t.name, "sim_acc": round(sim_acc, 4),
                         "backend_acc": round(hw_acc, 4), "n_test": int(len(yte))}
        jobs[key] = jid
        print(f"  {key} ({t.name:16s}): sim={sim_acc:.3f}  {backend_name}={hw_acc:.3f}", flush=True)

    return {
        "experiment": "e014_hardware_eval_qewc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "packages": {p: version(p) for p in ("qiskit", "qiskit-ibm-runtime",
                                                             "qiskit-aer", "pennylane")}},
        "config": {"method": "QEWC (shared 2-Z softmax, QFI-consolidated theta)", "layers": layers,
                   "n_qubits": N_QUBITS, "n_train": n_train, "n_test": n_test, "shots": shots,
                   "lam_qewc": lam_qewc, "qfi_samples": qfi_samples, "seed": seed},
        "backend": backend_name if backend_name == "aer" else backend.name,
        "ibm_job_ids": jobs, "per_task": per_task,
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
    ap.add_argument("--method", choices=["oiqcl", "qewc"], default="oiqcl",
                    help="with --all-tasks: OI-QCL frozen-A (default) or QEWC shared readout")
    ap.add_argument("--task", choices=["mnist", "fashion", "spt"], default="mnist")
    ap.add_argument("--backend", default="aer", help="'aer' (dry-run) or an IBM backend name")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--readout", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-head", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--qfi-samples", type=int, default=32, help="QEWC QFI mini-batch (--method qewc)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.all_tasks and args.method == "qewc":
        result = run_qewc_cl(layers=args.layers, epochs=args.epochs, n_train=args.n_head,
                             n_test=args.n_test, shots=args.shots, backend_name=args.backend,
                             qfi_samples=args.qfi_samples, seed=args.seed)
        out = args.output or RESULTS / f"e014_hardware_{args.backend}_qewc_alltasks.json"
        out = out if out.is_absolute() else ROOT / out
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\n[QEWC] backend={result['backend']}  shots={result['config']['shots']}")
        for k, v in result["per_task"].items():
            print(f"  {v['name']:16s}: sim={v['sim_acc']:.3f}  hw={v['backend_acc']:.3f}")
        print(f"wrote {out.relative_to(ROOT)}")
        return
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
