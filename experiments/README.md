# Experiments

Runnable experiment entrypoints. Each file should reproduce a result on its own:

```bash
python experiments/e001_baseline.py
```

## Conventions

- Name files `eNNN_<short_name>.py` (e.g. `e002_ideal_qaoa.py`).
- Set explicit seeds so runs are reproducible.
- Write machine-readable output to `results/` (CSV / JSON), not stdout only.
- Log the run in `../EXPERIMENTS.md` (backend, shots, seed, circuit metrics).
- Keep plotting out of experiments — read `results/` from `scripts/` to build `figures/`.

Flow: `experiment -> results/ (CSV/JSON) -> scripts/plot -> figures/ -> slides/`
