# E009 — Continual learning on quantum time-series forecasting

Addresses the temporal topic directly: *use time-series data and a quantum/hybrid temporal
model for sequential learning; measure catastrophic forgetting; apply continual-learning methods
to balance retention and adaptation.* Regression counterpart to the classification study
(e005/e007), on the **provided competition datasets** (Peng & Chen, arXiv:2605.06734).

## Setup
- **Data:** the provided quantum/physics forecasting series (torch-free reimplementation,
  `src/e009_data.py`, Apache-2.0 attribution preserved). Task sequence
  **narma_5 → damped_shm → bessel_j2** — three distinct dynamics (nonlinear autoregression,
  damped oscillation, Bessel oscillation). One-step-ahead forecasting, windows of 8, scaled
  [-1,1], time-ordered 80/20 split.
- **Model:** a compact **recurrent data-reuploading quantum forecaster** (`src/e009_qtsf.py`):
  4 qubits, 2 shared RY/RZ+CNOT-ring blocks re-applied per time step (state persists), Z
  readouts → trainable linear+tanh head. 21 parameters, PennyLane backprop, exact statevector.
- **Metric:** test **NMSE** (MSE / var, NARMA-standard; lower is better). Retention = mean
  earlier-task NMSE at run end; plasticity = final-task NMSE; forgetting = NMSE increase on an
  earlier task from its phase end to the run end.
- **Methods:** naive (no CL), L2 anchor, EWC (empirical MSE/Gauss-Newton **classical** Fisher),
  QEWC (**quantum** Fisher / QFI on the temporal circuit), replay (balanced buffer of 24
  earlier-task samples). Adam(0.05), 40 epochs/task. **Fairness:** all Fisher importances are
  normalized to unit mean, so lam is comparable and only the *structure* (relative per-parameter
  weighting) differs between L2/EWC/QEWC. QEWC uses QFI on the 16 quantum weights and the same
  classical Fisher on the 5 classical head weights, isolating the quantum-vs-classical Fisher
  question on the qubits.

## Results (5 seeds, mean ± sample SD; lower NMSE better)
| method | retention (old NMSE) | plasticity (new NMSE) | avg final NMSE |
|---|---|---|---|
| naive | 0.262 ± 0.135 | **0.032 ± 0.013** | 0.185 |
| L2 anchor | 0.211 ± 0.107 | 0.355 ± 0.244 | 0.259 |
| EWC (classical Fisher) | 0.165 ± 0.064 | 0.346 ± 0.188 | 0.226 |
| QEWC (quantum Fisher) | 0.150 ± 0.071 | 0.165 ± 0.034 | 0.155 |
| **replay** | **0.058 ± 0.026** | 0.061 ± 0.043 | **0.059** |

Classical reference: a ridge AR model forecasts each series at NMSE ≈ 0 individually (these are
smooth deterministic signals) — so, as with MNIST, there is no quantum-advantage claim; the study
is about **forgetting and its mitigation** in a quantum temporal model.

## Findings
1. **Catastrophic forgetting is real.** Naive sequential training has the best new-task
   plasticity (0.032) but the worst earlier-task retention (0.262) — e.g. narma_5 NMSE rises from
   ~0.03 to ~0.27 once bessel_j2 training begins.
2. **Replay decisively balances both** — best retention (0.058), near-best plasticity (0.061),
   best average (0.059), tightest variance. It essentially removes forgetting while still learning
   the new task. Robust across all 5 seeds. Replay is the overall winner.
3. **Among the regularizers, QEWC > EWC > L2** on the stability–plasticity trade-off. With a
   matched anchoring budget (unit-mean Fisher), QEWC has the best retention (0.150) AND much better
   plasticity (0.165) than EWC (0.346) or L2 (0.355). Robust across seeds (plasticity SD 0.034).
4. **The quantum Fisher genuinely helps here — contradicting the classification study.** In
   e005/e007 (MNIST/Fashion/SPT) the QFI was isotropic and QEWC ≈ L2 (quantum geometry inert).
   On quantum *temporal forecasting* the quantum Fisher gives the best regularizer trade-off:
   **whether quantum geometry helps is task-dependent, not universal.** Caveat: mechanism not yet
   pinned down — the QFI eff-rank was ~15.6/16 at random init, but its structure at the trained
   theta* evidently matters; a frontier sweep + anisotropy analysis would firm up "QEWC > EWC > L2".

## Figures (5 methods, e005 style, mean ± SD bands over 5 seeds)
- `figures/e009_forgetting.png` — per-task **test** NMSE over the sequential run (naive spikes on
  old tasks; replay stays flat).
- `figures/e009_training.png` — per-task **train** NMSE (fit / plasticity), training-set curve.
- `figures/e009_compare.png` — retention vs plasticity scatter (both NMSE, lower-left = best) with
  error bars; replay sits in the ideal corner, QEWC is the best regularizer.

## Files
- `src/e009_data.py` (torch-free provided-dataset loader), `src/e009_qtsf.py` (quantum forecaster)
- `experiments/e009_continual_forecasting.py`, `scripts/e009_multiseed.py`, plot scripts
- `results/e009_continual_seed42.json`, `results/e009_multiseed.json`
- `tests/test_e009.py`

## Reproduce
```bash
python experiments/e009_continual_forecasting.py --tasks narma_5 damped_shm bessel_j2 --seed 42
python scripts/e009_multiseed.py --seeds 42 43 44 45 46
python scripts/e009_plot.py && python scripts/e009_plot_compare.py
```

## Not done / notes
- The two CUDA-Q datasets (jaynes_cummings, transmon) need their shipped CSVs or CUDA-Q + an
  NVIDIA GPU; the 5 pure-Python series cover the study. Could add them via the b200 box.
- No noise / hardware run; exact statevector only.
