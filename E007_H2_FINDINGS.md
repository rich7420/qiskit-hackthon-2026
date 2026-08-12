# E007 H2 — Adaptive control vs always-on anchoring

**Question (after the directional NO-GO):** since forgetting is radial (step magnitude), not
directional, the live question is *when/how* to constrain — does event-triggered adaptive
control beat always-on consolidation (QEWC/L2)?

**Verdict:** No. Adaptive hard control is Pareto-dominated by soft anchoring. The surviving
positive result is mechanistic: on this benchmark **QFI-based consolidation acts as soft
norm anchoring in parameter space, not as directional quantum-geometric protection.**

## Findings (single-seed frontier + 5-seed confirmation of representative points)
1. **Per-step norm control does nothing.** Adaptive-v1 (per-step `‖Δθ‖² ≤ ε`) and gradient
   clipping never trigger (0% intervention) — no individual step is "too large." Forgetting is
   **cumulative** departure from old solutions, not isolated update spikes.
2. **Cumulative hard trust region works but over-constrains.** A ball `‖θ − θ*_j‖² ≤ ε_j`
   around each old optimum intervenes (~63% of steps) and gives the **lowest forgetting**
   (~0.03), but the ball around `θ*_1` blocks the *next* task from learning and the
   ball-intersection over-constrains the last task. Multi-seed: retention 0.697 ± 0.113,
   plasticity 0.968 ± 0.072 — **below QEWC on both axes.**
3. **Anchoring frontier dominates.** Sweeping strength, the L2 and QEWC frontiers overlap and
   reach retention ~0.83 at plasticity 1.0 (single seed); the adaptive frontier tops out at
   ~0.74 retention at plasticity 1.0 — strictly inside.
4. **QEWC ≈ L2 (mechanistic).** Because the global QFI is near-isotropic (`F_Q ≈ cI`, eff-rank
   158/160, from the directional NO-GO), the QEWC penalty `λ·Δθ^T F_Q Δθ ≈ λc·‖Δθ‖²` *is* an
   L2 anchor by construction. Empirically QEWC and L2 land in the same region; a clean
   frontier-equivalence claim would need per-seed frontier sweeps (the single-point multi-seed
   comparison uses an unmatched L2 λ and has overlapping error bars).

## Consolidated e007 thesis
- Directional QFI → falsified (multi-seed, `E007_DIRECTIONAL_NOGO.md`)
- Adaptive hard control → Pareto-dominated (multi-seed)
- **QFI continual learning works primarily as soft cumulative norm anchoring, not directional
  quantum-geometric protection.** The important design choice is how strongly and how softly to
  control cumulative drift from previous solutions — not which Fisher direction to protect.

## Files
- `src/e007_adaptive.py` (calibrated budget, cumulative trust region, clip), `experiments/e007_h2.py`
- `scripts/e007_h2_frontier.py`, `scripts/e007_h2_multiseed.py`, plot scripts
- `results/e007_h2_seed42.json`, `results/e007_h2_frontier.json`, `results/e007_h2_multiseed.json`
- `figures/e007_h2_frontier.png` (frontiers), `figures/e007_h2_multiseed.png` (5-seed, error bars)

## Reproduce
```bash
python experiments/e007_h2.py --seed 42
python scripts/e007_h2_frontier.py --seed 42 && python scripts/e007_plot_h2_frontier.py
python scripts/e007_h2_multiseed.py --seeds 42 43 44 45 46 && python scripts/e007_plot_h2_multiseed.py
```

## Not done (deliberate)
Per-seed frontier sweeps (to nail QEWC≡L2 rigorously) and a normalized-QFI ablation
(separating overall sensitivity scale from direction) — cheap follow-ups if the mechanistic
claim needs hardening for the writeup.
