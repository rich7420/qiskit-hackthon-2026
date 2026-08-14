# E009 — Real-QPU validation (ibm_marrakesh)

We ran the continual-learning forecaster on a **real 156-qubit IBM QPU** (`ibm_marrakesh`) and
compared against the simulator. Hardware does **inference only** — training on hardware is
infeasible (backprop needs a statevector; parameter-shift would need millions of noisy circuits and
the gradients would be swamped by noise). So we sim-train the models, then evaluate them on-device.

**Headline: QGR keeps its edge over QEWC on real hardware, and the simplified architecture
(transpiled 2-qubit depth 17) stays usable on-device.**

## Setup
- **Simplified ("aggressive") ansatz**: 1 variational layer, CNOT chain, single-axis RY encoding
  (24 CNOT / depth 41 logical; **transpiled 2q-depth 17 on marrakesh**). The gate-ablation
  hardware-friendly circuit — a deep 64-CNOT baseline would be noise-destroyed.
- Methods **QEWC (lam=0.1)** and **QGR (gen_len=24)**, sim-trained on narma_5 → damped_shm →
  bessel_j2, **3 seeds** (42–44). Evaluate the final model on **15 test windows/task**.
- Hardware: `ibm_marrakesh`, `SamplerV2`, **1024 shots**, no error mitigation (raw readout),
  270 circuits in one job (`d9uuo7d0vrcc73boiq40`). One Z-basis circuit per window gives all 4 ⟨Z⟩.
- **Circuit validated**: the Qiskit circuit reproduces the PennyLane forecaster to 1.4e-15
  (statevector); the counts→⟨Z⟩→prediction pipeline checked on Aer (shot-noise level).

## Results (avg NMSE over 3 tasks × 3 seeds, lower = better)
| method | sim-noiseless | sim-noisy (depol 0.01 + readout 0.02) | **QPU (real)** |
|---|---|---|---|
| **QGR** | 0.051 | 0.247 | **0.168** |
| QEWC | 0.114 | 0.246 | **0.412** |

(As R² = 1 − NMSE, this is the `figures/e009_qpu_compare.png` view — higher is better.)

## Findings
1. **QGR beats QEWC on real hardware** (avg NMSE 0.168 vs 0.412) — and on **every seed**
   (0.103/0.210/0.191 vs 0.222/0.667/0.346). The QGR robustness seen in the noise study carries
   over to the actual QPU. Per task the QGR hardware R² sits well above QEWC (bessel_j2: ≈0.78 vs
   0.49; damped_shm: ≈0.82 vs 0.51).
2. **Real hardware is gentler than our simulated noise for QGR**: QGR QPU (0.168) lands *between*
   sim-noiseless (0.051) and sim-noisy (0.247). Our depol=0.01/step model was pessimistic for this
   shallow 2q-depth-17 circuit — real marrakesh degraded the QGR model less than the sim predicted.
3. **QEWC degrades more on hardware than the sim predicted** (QPU 0.412 > sim-noisy 0.246), with
   one catastrophic seed (43: 0.667). The real device's error structure (coherent errors, crosstalk)
   hurts the QEWC-trained weights more than a plain depolarizing model captures.
4. **The simplified architecture is genuinely hardware-runnable** — transpiled 2q-depth 17 keeps the
   forecast meaningful on-device (narma_5 QGR R² ≈ 0.90). This is the gate-ablation payoff:
   fewer CNOTs → survives real hardware.

## Deep vs shallow on real hardware (baseline-arch control)
To prove the **depth** is the culprit (not the method), we ran the SAME QEWC/QGR models on the
original baseline architecture (2-layer, ring, two-axis; **transpiled 2q-depth 157 — 9× the
simplified 17**) on the same backend. R² = 1 − NMSE, mean over 3 seeds:

| task | method | sim-noiseless | simplified QPU (d17) | baseline QPU (d157) |
|---|---|---|---|---|
| narma_5 | QEWC | 0.87 | 0.76 | 0.33 |
| narma_5 | QGR | 0.96 | 0.90 | 0.36 |
| damped_shm | QEWC | 0.84 | 0.51 | 0.06 |
| damped_shm | QGR | 0.93 | 0.82 | 0.51 |
| bessel_j2 | QEWC | 0.95 | 0.49 | **−1.76** |
| bessel_j2 | QGR | 0.95 | 0.78 | **−1.10** |

- **The deep circuit collapses on hardware for BOTH methods** — bessel_j2 (the last/hardest task)
  goes to *negative* R² (worse than predicting the mean); narma_5/damped_shm fall far below the
  shallow run. It is the **circuit depth, not the method**: at 2q-depth 157 both QEWC and QGR are
  noise-destroyed; at depth 17 both survive and QGR leads. The gate-ablation payoff on-device — you
  must simplify the circuit to run continual learning on real hardware.
  Figure: `figures/e009_qpu_arch_compare.png`.

## Honest caveats
- **Inference only** — no continual *learning* happens on hardware (it can't; see above). This is a
  hardware **evaluation** of sim-trained CL models.
- 3 seeds, 15 windows/task → high variance (QEWC seed 43 is an outlier). The QGR-beats-QEWC ordering
  is robust (holds every seed); absolute numbers are noisy.
- Only 2 methods, single backend, **no error mitigation** (raw readout). The baseline-arch (deep)
  hardware control IS run (above); error mitigation, more seeds and a mitigated-QPU column are the
  natural next steps.

## Figures
- `figures/e009_datasets.png` — big-font intro to the three forecasting tasks (NARMA-5, damped SHM,
  Bessel J2): real series, held-out test tail shaded, one-line description + generating formula each.
- `figures/e009_r2_metric.png` — the accuracy metric definition card: R^2 = 1 - NMSE, with NMSE
  expanded to the coefficient-of-determination form.
- `figures/e009_arch.png` — big-font pitch schematic of the model: |0> -> U(x) encode -> V(theta)
  ansatz (recurrent, state persists) -> <Z> -> tanh head -> next-step forecast.
- `figures/e009_qgr_concept.png` — big-font pitch schematic of the method: freeze -> rollout ->
  rehearse, with L = MSE_new + MSE_replay (the low-text slide version of the mechanism).
- `figures/e009_qgr_flow.png` — QGR mechanism schematic (freeze -> rollout -> rehearse); the rollout
  panel is a *real* autoregressive rollout of the deployed `qgr:42` model (13 params, no stored data).
- `figures/e009_qpu_compare.png` — 3-panel per-task R² (sim-noiseless / sim-noisy / QPU) for QEWC & QGR.
- `figures/e009_qpu_arch_compare.png` — deep vs shallow on hardware (simplified d17 vs baseline d157).
- `figures/e009_forgetting_curves.png` — sim forgetting curves (R² view), QGR on the simplified arch (as deployed).
- `figures/e009_forgetting_curves_twoaxis.png` — same, QGR on the two-axis baseline (fair same-arch view).

## Files
- `scripts/e009_qpu_prepare.py` — sim-train models, save weights + sim refs (`--entangler/--encoding/--layers`)
- `scripts/e009_qpu_infer.py` — Qiskit circuit + `--validate` (equivalence) + `--run` (hardware, SamplerV2)
- `scripts/e009_qpu_compare_plot.py` — 3-panel sim-vs-QPU figure
- `scripts/e009_qpu_arch_compare_plot.py` — deep-vs-shallow hardware figure
- `scripts/e009_forgetting_curves.py` — 4-method sim forgetting curves (`--qgr-arch aggressive|baseline`)
- `scripts/e009_qgr_flow.py` — QGR mechanism figure (loads a trained model, does a real rollout)
- `scripts/e009_pitch_figs.py` — two big-font, low-text pitch figures (`--which arch|qgr|both`)
- `results/e009_qpu_{models,hardware}.json` (simplified), `results/e009_qpu_{models,hardware}_baseline.json` (deep)

## Reproduce
```bash
python scripts/e009_qpu_prepare.py --seeds 42 43 44 --methods qewc qgr --windows 15
python scripts/e009_qpu_infer.py --validate                              # local, no hardware
python scripts/e009_qpu_infer.py --run --backend ibm_marrakesh --shots 1024   # needs .env credentials + queue
python scripts/e009_qpu_compare_plot.py
```
