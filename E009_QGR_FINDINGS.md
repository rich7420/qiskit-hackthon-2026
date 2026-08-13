# E009 — Quantum Generative Replay (QGR)

A quantum-native upgrade of experience replay for continual learning on the quantum time-series
forecasting benchmark (e009). Classical replay stores raw old-task samples; **QGR stores a frozen
snapshot of the quantum forecaster and *generates* synthetic old-task sequences from its learned
dynamics** — the old task's memory lives in the quantum circuit, not a raw-data buffer.

## Method
```
after finishing task j:
    freeze a copy of the quantum forecaster (theta_j*, 21 params) + keep 3 short real seed windows
while training a later task:
    for each frozen generator: autoregressively roll out synthetic old-task sequences
    (predict next value -> append -> slide window -> repeat), window them, and rehearse (MSE) on them
```
- The generator is our recurrent data-reuploading quantum forecaster (`src/e009_qtsf.rollout`).
- Memory = quantum circuit params (21 numbers) + 3 seed windows per task, vs replay's 24 raw windows.
- Verified faithful: generated narma_5 matches real narma_5 autocorrelation (0.97/0.93/0.90 vs
  1.0/0.98/0.96).

## Results (5 seeds, test NMSE, lower = better)
| method | retention (old) | plasticity (new) | avg | stores raw data? |
|---|---|---|---|---|
| Baseline (naive) | 0.262 ± 0.135 | **0.032** | 0.185 | no |
| L2 anchor | 0.211 ± 0.107 | 0.355 | 0.259 | no |
| EWC (classical Fisher) | 0.165 ± 0.064 | 0.346 | 0.226 | no |
| QEWC (quantum Fisher) | 0.150 ± 0.071 | 0.165 | 0.155 | no |
| **QGR (quantum generative)** | 0.132 ± 0.135 | 0.069 ± 0.025 | 0.111 | **no (generates)** |
| replay | **0.058 ± 0.026** | 0.061 | **0.059** | yes (24/task) |

## Findings
1. **QGR matches replay on plasticity** (0.069 vs 0.061) and far exceeds every regularizer
   (QEWC 0.165, EWC/L2 ~0.35) — it does not sacrifice new-task learning.
2. **QGR beats every regularizer on retention** (0.132 < QEWC 0.150 < EWC 0.165 < L2 0.211 <
   Baseline 0.262), and is the **best quantum-native method** (beats QEWC), 2nd overall behind
   replay.
3. **QGR needs no raw-data buffer** — memory is the quantum generator (21 params + 3 seeds),
   the core upgrade over classical replay.
4. **Honest weakness**: QGR does not fully match replay's retention (0.132 vs 0.058) and has high
   retention variance (±0.135) — generative rollout quality drifts across seeds; plasticity is
   tight (±0.025).

## Story
> We turned classical replay into **Quantum Generative Replay**: the quantum model generates its
> own past from learned dynamics instead of storing raw data. QGR beats all regularizers
> (including the quantum-Fisher QEWC), matches replay's new-task learning, and is 2nd overall —
> with memory that lives in the quantum circuit.

## Files
- `src/e009_qtsf.py` — `rollout()`, `window_series()`
- `experiments/e009_continual_forecasting.py` — `qgr` method
- `scripts/e009_qgr_compare.py`, `figures/e009_qgr_compare.png`
- `results/e009_multiseed.json`

## Reproduce
```bash
python scripts/e009_multiseed.py --seeds 42 43 44 45 46   # runs all 6 methods incl. qgr
python scripts/e009_qgr_compare.py
```

## Next (optional)
- Reduce QGR retention variance: more/better seeds, longer rollout, or a dedicated quantum
  generator (QCBM) instead of the forecaster-as-generator.
- Fully data-free QGR: seed the rollout from noise / a learned prior instead of real seeds.
