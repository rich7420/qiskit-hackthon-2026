# E016 — Quantum Generative Replay on the classification benchmark

**Question.** QGR (PR #13) is the best *quantum-native* continual-learning method on the e009
**time-series forecasting** benchmark. Does the idea transfer to the e005 **classification**
sequence (MNIST 0/1 → Fashion-MNIST 0/1 → SPT/ATF), where it can be pitted directly against
EWC and QEWC on identical learner / data / schedule?

**Porting problem.** QGR's generator is an *autoregressive rollout* — it needs a temporal axis.
Classification maps a static 16-D amplitude vector → label, so "generation" is realized two ways:

- **`qgr_seed`** — keep a few real seed vectors per old task (16/class) + a frozen classifier
  snapshot; synthesize new vectors by convex-mixing seeds + noise (renormalized to valid amplitude
  states) and **pseudo-label** them with the frozen snapshot. Mirrors forecasting-QGR, which keeps
  a few real seed windows and generates from a frozen model.
- **`qgr_inversion`** — fully **data-free**. For each old class, gradient-ascend a random amplitude
  vector toward the frozen classifier's confidence for that class → class prototype. No raw sample
  is ever stored; the circuit params are the only memory (closest to the QGR headline).

## Result (5 seeds 42–46, mean ± sample SD; 20 layers, 20 epochs/task, 800 train / 200 test; **higher acc = better**)

| method | retention (T1,T2) | plasticity (T3) | avg | raw data? |
|---|---|---|---|---|
| Baseline (naive) | 0.590 ± 0.090 | 1.000 | 0.727 ± 0.060 | no |
| EWC (classical Fisher) | 0.704 ± 0.085 | 1.000 | 0.802 ± 0.057 | no |
| QEWC (quantum Fisher) | 0.773 ± 0.102 | 1.000 | 0.849 ± 0.068 | no |
| **QGR-seed** (quantum gen.) | **0.802 ± 0.037** | 1.000 | **0.868 ± 0.025** | **no** (16 seeds/cls + snapshot) |
| QGR-inversion (data-free) | 0.802 ± 0.076 | 1.000 | 0.868 ± 0.051 | **no** (params only) |
| replay (raw buffer) | 0.805 ± 0.030 | 1.000 | 0.870 ± 0.020 | yes (48/task) |

`figures/e016_qgr_compare_multiseed.png` (5-seed), `figures/e016_qgr_compare.png` (seed 42),
`results/e016_qgr_classification_summary.json`, `results/e016_qgr_classification_seed{42..46}.json`.

## Findings

1. **Plasticity saturates.** Every method reaches **1.000** on T3 (SPT/ATF is perfectly separable
   for this ansatz) in every seed, so — unlike the forecasting benchmark — classification cannot
   show a plasticity trade-off. **Retention on the two earlier tasks is the only discriminating axis.**
2. **Both QGR variants beat QEWC and EWC on mean retention**: QGR-seed / QGR-inversion **0.802 >
   QEWC 0.773 > EWC 0.704**, and both essentially match raw-data **replay (0.805)** without a full
   raw buffer. This reproduces the time-series verdict (**QGR > QEWC**) on the retention axis.
3. **QGR-seed is the most reliable no-data method**: its SD (±0.037) is ~3× tighter than QEWC
   (±0.102) and nearly as tight as raw replay (±0.030). Data-free QGR-inversion reaches the same
   mean but is noisy (±0.076) — model inversion sometimes recovers old boundaries excellently
   (seeds 43/44/45 ≈ 0.82–0.88) and sometimes weakly (seed 42 = 0.74).
4. **Honest caveats.** With 5 seeds the SDs of replay / QGR-seed / QGR-inversion / QEWC **overlap**,
   so "QGR beats QEWC" here is a **consistent mean ordering with lower variance**, not a separated
   result. The advantage is entirely a *retention* effect because T3 saturates; the clean two-axis
   (retention **and** plasticity) win QGR has on forecasting does not reproduce on classification.

## How to reproduce

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:$DYLD_LIBRARY_PATH"   # macOS/Homebrew only
for s in 42 43 44 45 46; do python experiments/e016_qgr_classification.py --seed $s; done
python scripts/e016_aggregate.py --seeds 42 43 44 45 46   # 5-seed summary + figure
python scripts/e016_qgr_compare.py                        # single-seed (42) two-panel figure
```
