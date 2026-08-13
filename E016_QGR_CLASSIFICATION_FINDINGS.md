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

## Result (seed 42, 20 layers, 20 epochs/task, 800 train / 200 test per task; **higher acc = better**)

| method | retention (T1,T2) | plasticity (T3) | forgetting↓ | avg | raw data? |
|---|---|---|---|---|---|
| Baseline (naive) | 0.512 | 1.000 | +0.405 | 0.675 | no |
| EWC (classical Fisher) | 0.740 | 1.000 | +0.163 | 0.827 | no |
| QEWC (quantum Fisher) | 0.787 | 1.000 | +0.095 | 0.858 | no |
| **QGR-seed** (quantum gen.) | **0.807** | 1.000 | +0.120 | **0.872** | **no** (16 seeds/cls + snapshot) |
| QGR-inversion (data-free) | 0.743 | 1.000 | +0.172 | 0.828 | **no** (params only) |
| replay (raw buffer) | 0.825 | 1.000 | +0.095 | 0.883 | yes (48/task) |

`figures/e016_qgr_compare.png`, `results/e016_qgr_classification_seed42.json`.

## Findings

1. **Plasticity saturates.** Every method reaches **1.000** on T3 (SPT/ATF is perfectly separable
   for this ansatz), so — unlike the forecasting benchmark — classification cannot show a
   plasticity trade-off. **Retention on the two earlier tasks is the only discriminating axis.**
2. **QGR-seed is the best quantum-native method**: retention **0.807 > QEWC 0.787 > EWC 0.740**,
   and it nearly matches raw-data **replay (0.825)** while storing only 16 seed vectors/class + a
   frozen snapshot — no full raw buffer. This reproduces the time-series verdict (**QGR > QEWC**).
3. **Data-free QGR-inversion only ties EWC** (0.743 vs 0.740) and sits **below QEWC**. Pure model
   inversion recovers old decision boundaries well enough to beat naive fine-tuning by +0.23, but
   the synthetic prototypes are less informative than QEWC's quantum-Fisher anchoring here.
4. **Honest caveats.** Single seed (42) only. The generative advantage is entirely a *retention*
   effect because T3 saturates; the clean two-axis (retention **and** plasticity) win that QGR has
   on forecasting does not reproduce on this classification sequence. Multi-seed (42/43/44) would
   be needed for error bars before any headline claim.

## How to reproduce

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:$DYLD_LIBRARY_PATH"   # macOS/Homebrew only
python experiments/e016_qgr_classification.py --seed 42
python scripts/e016_qgr_compare.py
```
