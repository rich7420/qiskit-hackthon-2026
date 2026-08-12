# Contributing

Working guidelines for the team during the hackathon.

## Working language

English is the working language of this repository. Write all file names, code,
comments, documentation, and commit messages in English.

## Git workflow

- **Solo (setup phase):** committing directly to `main` is fine.
- **Once teammates join:** stop pushing to `main`. Work on a feature branch and open
  a pull request: `git checkout -b <name>/<topic>`, then PR into `main`.
- Keep commits small and descriptive (imperative mood, e.g. `Add QAOA baseline for Max-Cut`).
- Have a teammate glance at each PR before merging.
- Pull `main` often to stay in sync — a 1.5-day event moves fast.

## Notebooks

- Clear cell outputs before committing to keep diffs small and avoid leaking data.
- Promote stable, reusable logic from notebooks into `src/` so others can import it.

## Code and results

- Reusable code lives in `src/`; runnable scripts live in `experiments/`.
- Save experiment outputs to `results/` and presentation plots to `figures/`.
- Never commit IBM Quantum tokens or other credentials (see `.gitignore`).

## Simulator vs. hardware

Develop and iterate on the local simulator (`qiskit-aer`). Use real QPU runs
(`qiskit-ibm-runtime`) only for final validation to avoid wasting queue time.
