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

## Honest caveats
- **Inference only** — no continual *learning* happens on hardware (it can't; see above). This is a
  hardware **evaluation** of sim-trained CL models.
- 3 seeds, 15 windows/task → high variance (QEWC seed 43 is an outlier). The QGR-beats-QEWC ordering
  is robust (holds every seed); absolute numbers are noisy.
- Only 2 methods, simplified arch only, single backend, one job, **no error mitigation**.
- A baseline-arch (64-CNOT) hardware run (to show the deep circuit is noise-destroyed) and error
  mitigation are the natural next steps.

## Figures
- `figures/e009_qpu_compare.png` — 3-panel per-task R² (sim-noiseless / sim-noisy / QPU) for QEWC & QGR.
- `figures/e009_forgetting_curves.png` — companion sim forgetting curves (Baseline/EWC/QEWC/QGR).

## Files
- `scripts/e009_qpu_prepare.py` — sim-train simplified models, save weights + sim references
- `scripts/e009_qpu_infer.py` — Qiskit circuit + `--validate` (equivalence) + `--run` (hardware, SamplerV2)
- `scripts/e009_qpu_compare_plot.py` — the 3-panel sim-vs-QPU figure
- `scripts/e009_forgetting_curves.py` — 4-method sim forgetting curves
- `results/e009_qpu_models.json` (weights + sim refs), `results/e009_qpu_hardware.json` (QPU NMSE)

## Reproduce
```bash
python scripts/e009_qpu_prepare.py --seeds 42 43 44 --methods qewc qgr --windows 15
python scripts/e009_qpu_infer.py --validate                              # local, no hardware
python scripts/e009_qpu_infer.py --run --backend ibm_marrakesh --shots 1024   # needs .env credentials + queue
python scripts/e009_qpu_compare_plot.py
```
