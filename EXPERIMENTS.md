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

---

## E001 — QNN on binary MNIST (local simulator smoke test)

Goal: confirm the quantum-ML stack runs locally end to end, not to chase accuracy.

Source hash:       a3afa1ff7dca4c12b538c7a722401bf3ef53ea45b9b8dfec8c26bf8870da80f4
Data split hash:   882a60d7a3b21e3f5d5a539b2dd7daccd93664eeeb2ce5e0e4e043d0cad6ec74
Backend:           StatevectorEstimator (exact, CPU local — 4 qubits, no shots)
Qiskit version:    qiskit 2.5.1, qiskit-machine-learning 0.9.0
Command:           `python experiments/e001_qnn_mnist.py --dataset mnist --n-train 150 --n-test 150 --maxiter 60 --ansatz-reps 2 --seed 42 --output results/e001_qnn_mnist_reference.json`

Configuration:
- task = MNIST 0 vs 1, PCA 784 -> 4 features, angle-scaled to [0, pi]
- feature map = zz_feature_map(4, reps=1); ansatz = real_amplitudes(4, reps=2), 12 weights
- optimizer = COBYLA(maxiter=60); seed = 42 (data + initial weights both seeded)

Circuit:
- qubits = 4; decomposed depth = 25; two-qubit gates = 18

Result:
- final objective = 0.6487 over 60 evaluations
- QNN test accuracy = 0.78  (train 0.82)
- classical baseline (LogisticRegression, same 4 features) = 0.9867
- train time = 18.129 s (runtime is informational and machine-dependent)

Finding: the stack works and the QNN learns (train/test climb ~0.63 -> ~0.80 over 60
COBYLA objective evaluations), but it plateaus well below a trivial classical baseline — MNIST 0-vs-1
on 4 PCA features is nearly linearly separable, so there is no quantum advantage to show
here. This is a green infrastructure check, not a result. Reproducible: same seed gives the
same curve. The curve is jagged because COBYLA is derivative-free (not gradient/epoch based).
The source/data hashes, package versions, weights, and full evaluation history are stored in
`results/e001_qnn_mnist_reference.json`.

Figures:
- `figures/e001_qnn_circuit.png` — circuit schematic (via `scripts/plot_qnn_circuit.py`)
- `figures/e001_training_curve.png` — train/test accuracy vs objective evaluation (via `scripts/plot_training_curve.py`)
