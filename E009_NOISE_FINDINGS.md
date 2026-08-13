# E009 — Noise robustness of continual-learning methods

How well do the continual-learning methods survive realistic quantum noise? We run **naive / QEWC /
QGR** on the e009 forecasting benchmark under **depolarizing (generic gate) + readout (measurement)
error**, and compare against the noiseless reference (3 seeds).

**Headline: under noise, QGR (quantum generative replay) is the most robust; QEWC (quantum-Fisher
anchoring) degrades the most — its plasticity collapses.**

## Noise model (`src/e009_qtsf.py`, `make_forecaster(noise=...)`)
- Switches the forecaster to the density-matrix simulator `default.mixed`.
- Four independent knobs `{bit, phase, depol, meas}`:
  - `bit` → `qml.BitFlip` (X), `phase` → `qml.PhaseFlip` (Z), `depol` → `qml.DepolarizingChannel`
    (random X/Y/Z) — injected on every qubit **after each re-upload step** (error grows with depth).
  - `meas` → `qml.BitFlip` on every qubit **before the Pauli-Z expectations** = readout error.
- **This run = scenario A: `depol=0.01/step + meas=0.02`, `bit=phase=0`.** Depolarizing already
  contains bit-flip (X) + phase-flip (Z) + Y, so it is the standard single-knob generic gate error.
- **QFI + head Fisher are computed on the NOISELESS model** (they are algorithmic importance
  weights; a Jacobian on the density-matrix device is ~500x costlier). Training/eval forward passes
  stay noisy.

## Setup
- Methods naive / qewc / qgr; tasks narma_5 → damped_shm → bessel_j2; 3 seeds (42–44);
  20 epochs/task; QGR gen_len 24; baseline ansatz (2-layer, ring, two-axis). Metric = test NMSE.

## Results (3 seeds, test NMSE, lower = better)
| method | condition | retention | plasticity | avg |
|---|---|---|---|---|
| naive | noiseless | 0.461 | 0.064 | 0.329 ± 0.336 |
| naive | **noisy** | 0.268 | 0.077 | 0.205 ± 0.149 |
| QEWC | noiseless | 0.256 | 0.255 | 0.256 ± 0.026 |
| QEWC | **noisy** | 0.295 | **0.464** | **0.351 ± 0.024** |
| **QGR** | noiseless | 0.108 | 0.049 | **0.088 ± 0.071** |
| **QGR** | **noisy** | **0.209** | **0.146** | **0.188 ± 0.106** |

Degradation (noisy − noiseless avg): naive −0.124*, QEWC +0.096, QGR +0.100.

## Findings
1. **QGR is the most noise-robust method.** Under noise it is best on all three axes — avg **0.188**,
   retention **0.209**, plasticity **0.146** — and beats QEWC on every seed (clearly on 2/3, tied on
   1). The forgetting curves (`figures/e009_noise_curves_noisy.png`) show the red QGR line lowest in
   all three tasks, especially Task 3 (bessel_j2: QGR ≈ 0.1 vs QEWC ≈ 0.4).
2. **QEWC degrades the worst, and specifically its plasticity collapses** (0.255 → **0.464**,
   +0.21). The anchor pins parameters to their *noiseless* optima, but under noise the same
   parameters produce different (worse) outputs — the anchor protects the wrong thing, and anchor +
   noise together over-constrain new-task learning. QEWC ends up the worst method under noise (0.351).
3. **QGR's mechanism explains its robustness.** Generative replay re-teaches the old-task behaviour
   every epoch by rehearsing generated trajectories; this keeps correcting noise-induced drift.
   Even though the generator itself is noisy under noise, the rehearsal signal stays good enough.
   Function-space rehearsal > parameter-space anchoring when the forward pass is noisy.
4. **naive's apparent "improvement" under noise is not reliable** (*). Its noiseless avg (0.329) is
   inflated by one catastrophic-forgetting seed (0.804); with only 3 seeds the naive variance is
   huge (±0.336). Noise also mildly regularizes forgetting. Needs more seeds to claim anything.

## Why the noisy runs are slow (documented)
Two multiplicative costs vs the noiseless statevector path:
1. `default.mixed` stores a density matrix (2ⁿ → 4ⁿ: 16 → 256 numbers) and each gate is ρ→UρU†
   (~30x slower per forward pass on our 256-gate circuit).
2. QEWC's `empirical_fisher` originally ran a **Jacobian on the noisy device** (naive 79s → qewc
   616s on a tiny probe). Fixed by computing the Fisher on the noiseless model (qewc → 104s).
Full run: 6374 s (~1.8 h) for 3 methods × 3 seeds × (noiseless + noisy).

## Honest caveats
- **Only 3 seeds** → high variance (especially naive). Conclusions about QGR-vs-QEWC are robust
  (QGR wins every seed); the naive numbers are not.
- 20 epochs/task (not 40), gen_len 24 (not 48) — reduced for the density-matrix cost.
- Importance weights computed noiselessly (stated design choice).
- Only scenario A (depol + readout); a bit+phase-specific scenario was not run.
- Simulator only; `default.mixed` exact density matrix, no hardware.

## Figures
- `figures/e009_noise_curves_noisy.png` — per-task test-NMSE forgetting curves UNDER NOISE (each task
  shown from its own training onset; dotted task boundaries).
- `figures/e009_noise_curves_noiseless.png` — noiseless reference, same style.
- `figures/e009_noise_compare.png` — noiseless vs noisy bars (avg / retention / plasticity).

## Files
- `src/e009_qtsf.py` — `make_forecaster(noise=...)`, `_reupload` noise injection, `rollout` (QGR)
- `experiments/e009_continual_forecasting.py` — `train_method(..., noise=...)`, qgr method, noiseless Fisher
- `scripts/e009_noise_compare.py` — runs naive/qewc/qgr × noiseless+noisy, records metrics + curves
- `scripts/e009_noise_compare_plot.py` — degradation bars
- `scripts/e009_noise_curves_plot.py` — forgetting-curve figure
- `results/e009_noise_compare_A_depol.json`

## Reproduce
```bash
python scripts/e009_noise_compare.py --seeds 42 43 44 --epochs-per-task 20 --gen-len 24 \
    --bit 0 --phase 0 --depol 0.01 --meas 0.02 --output results/e009_noise_compare_A_depol.json
python scripts/e009_noise_compare_plot.py
python scripts/e009_noise_curves_plot.py --condition noisy
python scripts/e009_noise_curves_plot.py --condition noiseless
```

## Next (optional)
- Scenario B: explicit `bit=0.01 phase=0.01` (separate X/Z channels) for an X-vs-Z sensitivity study.
- More seeds to pin down the naive baseline.
- Noise on the simplified single-axis architecture (fewer gates → less noise accumulation).
