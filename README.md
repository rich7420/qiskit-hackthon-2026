# Qiskit Hackathon Taiwan 2026

Team repository for the **NTU-IBM Quantum System 2026 User Conference & Qiskit Hackathon Taiwan 2026**.

We pick a problem on-site, analyze it with quantum computing / Qiskit, and present our results. Judging is based on the analysis approach, the depth of Qiskit usage, and the final report.

## Quick start

> Use **Python 3.11–3.13** (Qiskit 2.x supports 3.10+). Any OS works.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then, on any OS:

```bash
# Register a Jupyter kernel for this project
python -m ipykernel install --user \
  --name qiskit-hackathon-2026 \
  --display-name "Qiskit Hackathon 2026"

# Verify the environment
python test_qiskit.py
```

`test_qiskit.py` is a preflight check: it verifies the Qiskit import, the
statevector simulator (Bell state), Aer, transpilation, and IBM Quantum Runtime
connectivity, printing a PASS / WARN / FAIL summary. IBM connectivity is a WARN
(not a failure) until an account is configured.

## Repository structure

```text
.
├── README.md
├── CLAUDE.md              # shared project instructions for Claude Code
├── CONTRIBUTING.md
├── EXPERIMENTS.md         # append-only experiment log
├── requirements.txt
├── .env.example           # copy to .env for IBM credentials (git-ignored)
├── test_qiskit.py         # environment preflight check
├── notebooks/             # exploratory notebooks
├── src/                   # reusable code (encoding, quantum, baseline, metrics)
├── experiments/           # runnable experiment entrypoints (eNNN_*.py)
├── scripts/               # one-off utilities (data prep, figure export)
├── tests/                 # unit and regression tests
├── data/                  # datasets: raw/ and processed/ (only if needed)
├── results/               # machine-readable outputs (CSV / JSON)
├── figures/               # plots for the presentation
└── slides/                # final presentation
```

## IBM Quantum

Access to IBM Quantum hardware goes through `qiskit-ibm-runtime`:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
```

Develop and debug on the local simulator (`qiskit-aer`), and reserve real QPU runs for final validation to avoid wasting queue time.

## Troubleshooting

**macOS + Homebrew Python: `Symbol not found: _XML_SetAllocTrackerActivationThreshold`**

On macOS 26 (Tahoe), Homebrew's Python links `pyexpat` against the system
`libexpat`, which is missing a symbol the build expects. This breaks `venv`/`pip`.
Point the loader at Homebrew's newer expat (no system files are modified):

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:$DYLD_LIBRARY_PATH"
python3 -m venv .venv
```

To make it permanent for this project, append that `export` line to `.venv/bin/activate`.
This only affects macOS/Homebrew setups — Linux and Windows are unaffected.

## Optional add-ons

Install these only if the chosen problem calls for them:

- `qiskit-optimization` — QUBO / QAOA / combinatorial optimization (Max-Cut, routing, scheduling, portfolio)
- `qiskit-machine-learning` — quantum machine learning
- `qiskit-nature` — chemistry / physics / VQE (molecules, ground state, electronic structure)
