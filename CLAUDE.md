# Qiskit Hackathon Taiwan 2026

## Goal

Build a reproducible Qiskit-based research project for the hackathon.

## Priorities

1. Get an end-to-end baseline working first.
2. Always compare against a classical or Qiskit baseline.
3. Record every important experiment in `EXPERIMENTS.md`.
4. Prefer quantitative evidence over additional features.
5. Keep simulator and hardware paths clearly separated.

## Project layout

- `src/`: reusable library code (encoding, quantum, metrics, ...)
- `experiments/`: runnable experiment entrypoints (`eNNN_*.py`)
- `scripts/`: one-off utilities (data prep, figure export)
- `notebooks/`: exploratory work
- `tests/`: unit and regression tests (also verify formulations, not just "no crash")
- `results/`: machine-readable outputs (CSV / JSON)
- `figures/`: presentation-ready plots
- `slides/`: final presentation
- `data/`: input datasets (`raw/`, `processed/`) — only if the problem needs them

## Coding rules

- Python with type hints where reasonable.
- Reusable logic belongs in `src/`, not notebooks.
- Every experiment uses explicit seeds where applicable.
- Never commit IBM Quantum credentials or API keys (use `.env`, see `.env.example`).
- Store experiment results as JSON/CSV when possible.
- Record backend, shots, transpiler settings, and circuit metrics.

## Before claiming an improvement

Always compare:
- solution quality
- circuit depth
- two-qubit gate count
- runtime where relevant
- simulator vs. noisy/hardware results where relevant

## Environment

- Python 3.11–3.13 in a `.venv` (any OS).
- Install deps: `pip install -r requirements.txt`.
- Run the preflight check: `python test_qiskit.py`.
- macOS/Homebrew note: if `pyexpat` fails to load, see Troubleshooting in `README.md`.
