# E009 — Gate-count ablation (how small can the quantum forecaster get?)

Can we cut the forecaster's gate cost — especially the **two-qubit (CNOT) count**, which dominates
hardware noise — without losing forecasting quality? We parameterize the ansatz along three levers,
measure the **real** per-forward-pass circuit cost with `qml.specs`, and the continual-learning
NMSE across 5 seeds under both `naive` (no protection) and `qewc` (quantum-Fisher anchoring).

Surprise result: the smallest circuits are not just cheaper — **dropping the two-axis data encoding
also *improves* continual-learning retention.** The over-expressive encoding was hurting us.

## Levers (`src/e009_qtsf.py`, backward-compatible defaults)
- **`n_layers`** — variational blocks per re-upload step (2 → 1).
- **`entangler`** — `ring` (4 CNOT/layer, wrap-around) | `chain` (3, no wrap) | `none` (0).
- **`encoding`** — `ry_rz` (two-axis: each datum re-uploaded via RY *and* RZ) | `ry` (single-axis).

## Circuit cost per forward pass (`qml.specs`, exact — locked by `tests/test_e009.py`)
| config | layers | entangler | encoding | gates | **CNOT** | depth | params |
|---|---|---|---|---|---|---|---|
| **baseline** (original) | 2 | ring | ry_rz | 256 | **64** | 112 | 21 |
| chain | 2 | chain | ry_rz | 240 | 48 | 81 | 21 |
| **enc_ry** | 2 | ring | ry | 224 | 64 | 104 | 21 |
| one_layer | 1 | ring | ry_rz | 160 | 32 | 64 | 13 |
| **aggressive** | 1 | chain | ry | 120 | **24** | 41 | 13 |
| minimal (floor) | 1 | none | ry | 96 | 0 | 24 | 13 |

Gate count `∝ seq_len × n_layers`; the recurrent re-upload applies each block `seq_len=8` times, so
per-step savings are amplified 8×.

## Results — 5 seeds (42–46), test NMSE, lower = better
**naive (no protection):**
| config | CNOT | retention (old) | plasticity (new) | avg |
|---|---|---|---|---|
| baseline | 64 | 0.262 ± 0.121 | 0.032 | 0.185 |
| chain | 48 | 0.256 ± 0.085 | 0.108 | 0.206 |
| **enc_ry** | 64 | **0.089 ± 0.042** | 0.027 | **0.068** |
| one_layer | 32 | 0.309 ± 0.113 | 0.073 | 0.230 |
| **aggressive** | **24** | 0.127 ± 0.071 | 0.033 | **0.096** |
| minimal | 0 | 0.633 ± 0.015 | 0.844 | 0.703 |

**qewc (quantum-Fisher anchoring, lam=5.0):**
| config | CNOT | retention (old) | plasticity (new) | avg |
|---|---|---|---|---|
| baseline | 64 | 0.150 ± 0.063 | 0.165 | 0.155 |
| chain | 48 | 0.176 ± 0.063 | 0.162 | 0.171 |
| **enc_ry** | 64 | **0.053 ± 0.036** | **0.105** | **0.070** |
| one_layer | 32 | 0.310 ± 0.134 | 0.895 | 0.505 |
| aggressive | 24 | 0.092 ± 0.044 | 0.296 | 0.160 |

(baseline reproduces the known e009 numbers exactly — naive avg 0.185, qewc avg 0.155 — confirming
the ablation pipeline.)

## QEWC lam retune — the fair best-vs-best (5 seeds)
The default lam=5.0 was tuned for the 21-param baseline and over-regularizes the 13-param
`aggressive`. Sweeping lam per config finds each one's own operating point (ret / plas / **avg**):

| lam | aggressive (24 CNOT) | baseline (64 CNOT) |
|---|---|---|
| 0.1 | 0.153 / 0.060 / **0.122** | 0.285 / 0.072 / 0.214 |
| 0.3 | 0.151 / 0.084 / 0.128 | 0.252 / 0.079 / 0.194 |
| 1.0 | 0.106 / 0.188 / 0.134 | 0.233 / 0.096 / 0.187 |
| 2.0 | 0.097 / 0.225 / 0.139 | 0.181 / 0.119 / 0.161 |
| 5.0 | 0.092 / 0.296 / 0.160 | 0.150 / 0.165 / **0.155** |

- **Best-vs-best: aggressive @ lam=0.1 (avg 0.122) beats baseline @ lam=5.0 (avg 0.155) — at 62%
  fewer CNOTs.** Tuned fairly, the 24-CNOT circuit wins outright; its retention-plasticity curve
  dominates baseline's across the entire sweep (`figures/e009_gate_lam_sweep.png`).
- **Opposite lam preferences reveal the mechanism.** aggressive wants *weak* anchoring (avg
  monotonically worse as lam grows), baseline wants *strong* anchoring (avg monotonically better).
  Single-axis encoding forgets less intrinsically → needs little protection → keeps plasticity;
  two-axis forgets more → needs heavy protection → sacrifices plasticity. Same story as Finding 1,
  now visible through the regularizer.

## Findings
1. **Single-axis encoding (`enc_ry`) is a strict win.** At **equal CNOT (64)** and **−32 gates**, it
   beats baseline on *every* metric under *both* methods: qewc avg **0.070 vs 0.155** (−55%),
   retention **0.053 vs 0.150** (−65%), and even plasticity (0.105 vs 0.165). Same under naive
   (0.068 vs 0.185). This is apples-to-apples (identical params=21, identical lam) so the gain is
   attributable to the encoding alone.
   - **Why:** data re-uploading turns each encoding gate into accessible Fourier frequencies
     (Schuld et al. 2021). Two-axis encoding widens the spectrum → sharper, more localized per-task
     functions → **more catastrophic interference** when a later task overwrites them. Single-axis
     is a smoother function class: same plasticity, far less forgetting. Encoding expressivity
     trades off against continual-learning stability.

2. **Entanglement is necessary.** The zero-CNOT floor (`minimal`) collapses (avg 0.703, plasticity
   0.844). CNOTs earn their keep — we can thin them, not remove them.

3. **`aggressive` (24 CNOT) wins when tuned fairly — not just cheaper.** −62% CNOT / −53% gates /
   −63% depth (24 CNOT, depth 41). Under **naive** it already beats baseline (avg 0.096 vs 0.185).
   Under **qewc at the default lam=5.0** it looked like a tie (avg 0.160) only because that lam,
   tuned for the 21-param model, **over-regularizes** the 13-param one (plasticity 0.296). The lam
   retune (above) fixes it: at lam=0.1, avg **0.122 — beating baseline's best (0.155) at 62% fewer
   CNOTs.** Not a capacity limit, a tuning artifact.

4. **Honest confounds.** lam=5.0 is fixed across configs — fair for the *encoding* comparison
   (enc_ry vs baseline, both 21 params) but **not** for the layer-reduced configs (13 params), whose
   QEWC results would shift under a per-config lam. The result is on **smooth, approximately
   deterministic** tasks (classical ridge ≈ NMSE 0); harder/higher-frequency dynamics may need the
   two-axis encoding for plasticity. No hardware run yet — CNOT/depth are logical counts.

## Recommendation
- **Adopt single-axis encoding as the default forecaster** (`encoding="ry"`). It is a free, strict
  improvement: fewer gates *and* better retention/plasticity, no downside on this benchmark. Best
  quality overall: `enc_ry` (qewc avg 0.070).
- **For maximum hardware thrift**, `aggressive` (24 CNOT, −62%) at **lam≈0.1** is the pick — tuned
  fairly it *beats* the 64-CNOT baseline (avg 0.122 vs 0.155), not just matches it.
- Under qewc the Pareto frontier is `aggressive (24 CNOT)` → `enc_ry (64 CNOT)`; **both are
  single-axis** — the two-axis baseline is dominated.

## QGR vs QEWC — method and architecture are separate levers (logged, 5 seeds)
**Cross-validation:** QEWC on the original architecture is *identical* in both source logs (the QGR
multiseed run and this ablation): ret 0.150 / plas 0.165 / avg 0.155 — so the two datasets sit on
one comparable axis. Ranking every logged combo by avg NMSE:

| combo | arch | retention | plasticity | avg | stores data? |
|---|---|---|---|---|---|
| replay | orig | 0.058 | 0.061 | **0.059** | yes |
| **QEWC + single-axis** | single-axis | 0.053 | 0.105 | **0.070** | no |
| **QGR** | orig | 0.122 | 0.093 | **0.113** | no |
| QEWC | orig | 0.150 | 0.165 | 0.155 | no |
| naive | orig | 0.262 | 0.032 | 0.185 | no |
| EWC | orig | 0.165 | 0.346 | 0.226 | no |
| QGR + single-axis | single-axis | — | — | *not measured* | no |

- **Method vs method, same architecture → QGR wins.** On the original 2-axis arch QGR (0.113) beats
  QEWC (0.155) on both retention *and* plasticity — the only fair same-arch comparison in the logs.
- **Best measured no-data combo → QEWC + single-axis (0.070)**, nearly matching the raw-data
  `replay` upper bound (0.059) — because it received the architecture upgrade QGR has not.
- **Architecture is the bigger lever here.** Single-axis moved QEWC by −0.085 avg, larger than the
  QGR-vs-QEWC *method* gap (−0.042). The two levers are orthogonal.
- **Open cell:** `QGR + single-axis` is in no log, so from records alone we cannot claim QGR beats
  QEWC+single-axis. Figure: `figures/e009_method_arch_compare.png`
  (`scripts/e009_method_arch_compare.py`, data `results/e009_method_arch_compare.json`).

## Files
- `src/e009_qtsf.py` — parameterized `make_forecaster`/`make_state_forecaster` (`entangler`, `encoding`)
- `experiments/e009_continual_forecasting.py` — `train_method(..., ansatz=...)`
- `scripts/e009_gate_ablation.py` — config sweep (records `qml.specs` cost + NMSE)
- `scripts/e009_gate_ablation_plot.py` — Pareto figure (NMSE vs CNOT, coloured by encoding)
- `scripts/e009_gate_lam_sweep.py` / `_plot.py` — per-config QEWC lam retune + tradeoff figure
- `scripts/e009_arch_curves.py` — per-task R² training curves (baseline / EWC / QEWC / simplified)
- `scripts/e009_method_arch_compare.py` — QGR-vs-QEWC-vs-architecture summary figure (logged data)
- `results/e009_gate_ablation_{naive,qewc}.json`, `results/e009_gate_lam_sweep.json`,
  `results/e009_arch_curves.json`, `results/e009_method_arch_compare.json`
- `figures/e009_gate_ablation_{naive,qewc}.png`, `figures/e009_gate_lam_sweep.png`,
  `figures/e009_arch_curves.png`, `figures/e009_method_arch_compare.png`
- `tests/test_e009.py` — gate-count regression tests (lock each config's gates/CNOT/params)

## Reproduce
```bash
python scripts/e009_gate_ablation.py --configs baseline chain enc_ry one_layer aggressive minimal \
    --seeds 42 43 44 45 46 --method naive --output results/e009_gate_ablation_naive.json
python scripts/e009_gate_ablation.py --configs baseline chain enc_ry one_layer aggressive \
    --seeds 42 43 44 45 46 --method qewc  --output results/e009_gate_ablation_qewc.json
python scripts/e009_gate_ablation_plot.py --input results/e009_gate_ablation_qewc.json \
    --output figures/e009_gate_ablation_qewc.png
python scripts/e009_gate_lam_sweep.py --configs aggressive baseline --lams 0.1 0.3 1.0 2.0 5.0
python scripts/e009_gate_lam_sweep_plot.py
```

## Next (optional)
- Confirm the encoding finding under **QGR** (our lead method) and **replay**.
- A `seq_len` sweep (8 → 6): the other linear gate lever (narma_5 needs ≥ 5).
- Adopt `encoding="ry"` as the e009 default and re-run the headline QGR/QEWC numbers.
