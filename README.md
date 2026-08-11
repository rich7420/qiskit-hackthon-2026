# Qiskit Hackathon Taiwan 2026

Team repository for the **NTU-IBM Quantum System 2026 User Conference & Qiskit Hackathon Taiwan 2026**.

We pick a problem on-site, analyze it with quantum computing / Qiskit, and present our results. Judging is based on the analysis approach, the depth of Qiskit usage, and the final report.

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register a Jupyter kernel for this project
python -m ipykernel install \
  --user \
  --name qiskit-hackathon-2026 \
  --display-name "Qiskit Hackathon 2026"

# 4. Verify the environment
python test_qiskit.py
```

`test_qiskit.py` runs a Bell-state circuit on a local simulator. Output should be mostly `00` and `11` counts (roughly 50/50), which confirms a working setup.

## Repository structure

```text
.
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── test_qiskit.py         # local environment smoke test
├── notebooks/             # exploratory notebooks (problem, baseline, quantum, hardware)
├── src/                   # reusable code (encoding, quantum, baseline, metrics)
├── experiments/           # runnable scripts (simulator / hardware)
├── results/               # experiment results (data, logs)
├── figures/               # plots for the presentation
└── slides/                # final presentation
```

## IBM Quantum

Access to IBM Quantum hardware goes through `qiskit-ibm-runtime`:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
```

Develop and debug on the local simulator (`qiskit-aer`), and reserve real QPU runs for final validation to avoid wasting queue time.

## Optional add-ons

Install these only if the chosen problem calls for them:

- `qiskit-optimization` — QUBO / QAOA / combinatorial optimization (Max-Cut, routing, scheduling, portfolio)
- `qiskit-machine-learning` — quantum machine learning
- `qiskit-nature` — chemistry / physics / VQE (molecules, ground state, electronic structure)
