# E007 — Directional QFI continual learning: multi-seed NO-GO

**Verdict:** on this benchmark, catastrophic forgetting is governed by **how far** the model
moves in parameter space (step magnitude), **not by where** it moves in quantum geometry.
The directional Q-MemGuard hypothesis is falsified and marked NO-GO. This is a complete,
reproducible result — frozen here before pivoting to the adaptive-control question (H2).

## Hypothesis (falsified)
Q-MemGuard's premise was that the Quantum Fisher Information identifies *dangerous directions*:
an update in a high-QFI (task-sensitive) direction should cause more forgetting, so a QFI
trust region could protect old tasks better than an always-on penalty (QEWC). Formally, with
`R = Δθ^T F_Q Δθ = ‖Δθ‖² · (v^T F_Q v)`, the claim was that the **directional** term
`Q_F = v^T F_Q v` carries forgetting signal beyond step magnitude.

## Three falsifying lines of evidence
1. **Global QFI is near-isotropic.** At a trained MNIST operating point the global QFI diagonal
   has effective rank **157.6 / 160** (max/min ≈ 2.1). An isotropic metric gives
   `R ≈ c·‖Δθ‖²`, so global-QFI risk is essentially rescaled step size. (CFI, the
   task-relevant Fisher, is more anisotropic — eff. rank ~96/160 — but see #2.)
2. **The directional coefficient is seed-dependent noise.** Across 5 seeds, regressing
   forgetting on `[‖Δθ‖², Q_F]`, the directional coefficient `β₂` has **mixed sign** for both
   global and readout QFI (2 positive / 3 negative), with small, high-variance incremental R²
   (mean ΔR² ≈ 0.06–0.09). No robust, consistent directional effect — for global **or** the
   readout-reduced-state QFI **or** CFI.
3. **Exact Bures distance does not beat Euclidean movement.** Using the actual finite-movement
   Bures state distance (not the local QFI approximation): Euclidean `‖Δθ‖` predicts forgetting
   with corr 0.90 (R² 0.81); global and readout Bures are **worse** (0.83, 0.81) and add only
   +2–3% R². This rules out "it was just local linearization."

A supporting sanity check (`e007_bures_check.py`): `(1/2)√R` tracks the actual Bures drift with
Pearson **0.994**, slope 1.03 — so `R` is a physically valid state-drift measure. The failure is
not that `R` is meaningless; it is that its *direction* carries no robust forgetting signal here.

## Consequence
Because global QFI ≈ isotropic, `Δθ^T F_Q Δθ ≤ ε` reduces to `‖Δθ‖² ≤ ε'` — so
**directional Q-MemGuard (global or readout) collapses to gradient clipping**, and by the same
token **QEWC ≈ norm / L2 anchoring** on this benchmark. This reframes the QFI-continual-learning
line: its benefit is movement control, not structured quantum directionality.

## Files
- Analysis code: `src/e007_qmemguard.py`, `experiments/e007_bures_check.py`,
  `experiments/e007_h1_earlywarning.py`, `experiments/e007_decisive.py`,
  `experiments/e007_bures_rescue.py`
- Aggregation: `scripts/e007_aggregate_nogo.py` → `results/e007_nogo_summary.json`,
  `results/e007_decisive_multiseed.csv`
- Raw: `results/e007_decisive_seed{42..46}.json`, `results/e007_bures_rescue.json`,
  `results/e007_bures.json`, `results/e007_h1.json`
- Figures: `figures/e007_nogo.png` (3-panel evidence), `figures/e007_bures.png` (R validation)

## Reproduce
```bash
for s in 42 43 44 45 46; do
  python experiments/e007_decisive.py --seed $s --output results/e007_decisive_seed$s.json
done
python experiments/e007_bures_rescue.py --seed 42
python experiments/e007_bures_check.py --seed 42
python scripts/e007_aggregate_nogo.py
python scripts/e007_plot_nogo.py
```

## Next (H2 — the question the evidence supports)
Since *how far* dominates *where*, the live question is: **do we need to constrain every
update?** Compare an **Adaptive Norm Trust Region** (constrain only when the step exceeds a
calibrated old-task budget) against always-on QEWC / L2, on a retention–plasticity Pareto plus
intervention fraction. QFI variants only get re-attached if adaptive-norm itself shows a
positive result (and are expected to matter, if at all, under noise).
