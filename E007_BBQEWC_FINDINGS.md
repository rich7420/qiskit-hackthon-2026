# E007 — Quantum-state vs function-space trust regions (Bures-Budgeted QEWC)

Built on QEWC (PR #5) as a positive-method attempt: replace the always-on QEWC penalty with an
explicit per-task trust region, accept optimizer steps freely inside the budget, and scale them
to the boundary only when they would leave it (optimizer-aware, `src/e007_bbqewc.py`).

**Verdict:** hard trust regions (state *or* function) reach *higher* mean retention than soft
anchoring, but only by **unreliably sacrificing new-task learning** (plasticity 0.52 ± 0.34
across seeds — the final task often fails). Soft anchoring reliably learns the new task
(plasticity 1.0, SD 0) with strong retention (~0.79). For continual learning, where the new task
must actually be learned, **soft cumulative anchoring occupies the useful corner and the hard
trust region does not** — it is not a Pareto win for the hard constraint. The single-seed
CFI>QFI edge did **not** survive multi-seed (0.834 vs 0.845, within noise), so no robust
"function>state" claim.

## Physics (why function, not state)
Bounding QFI/Bures state drift is a **sufficient** forgetting safeguard:
`|Δ⟨O⟩| ≤ ‖O‖∞·‖Δρ‖₁ ≤ ‖O‖∞·2√(1−F) ∼ ‖O‖∞·D_B`. But it is **loose**: it guards the whole
quantum state, including degrees of freedom the readout can't see, so it over-constrains. The
**tight** object is the old-task predictive distribution: `KL(p_old(θ*)‖p_old(θ)) ≈ ½ δᵀF_C δ`,
i.e. the classical (readout) Fisher `F_C`. So the CFI trust region is physically aligned with
forgetting; the QFI trust region is representation preservation.

## Methods
`sequential` | `qewc` (soft global-QFI penalty) | `l2` (soft anchor) |
`tr_qfi` (QFI state trust region) | `tr_cfi` (CFI function trust region, primary).
All on MNIST → Fashion-MNIST → SPT/ATF, 4 qubits, RY/RZ+CNOT, Adam.

## Single-seed frontier (seed 42)
| method | retention @ plasticity 1.0 | high-retention point |
|---|---|---|
| L2 / QEWC (soft) | **0.83–0.84** | — |
| QFI state TR | ~0.50 | 0.865 @ plasticity **0.5** |
| CFI function TR | ~0.50 | **0.875** @ plasticity **0.5** |

- **Soft anchoring retains ~0.83 while *fully* learning the new task (plasticity 1.0).**
- **Both trust regions reach high retention only at plasticity ~0.5** — i.e. the quantum-native
  final task (SPT/ATF) is stuck near chance. `tr_qfi` collapses plasticity to ~0 for tight
  budgets (it freezes the very different task 3 out entirely).
- **CFI > QFI as predicted** (0.875 vs 0.865 at plasticity 0.5) — the function-space refinement
  helps, but both are dominated by soft anchoring in the useful high-plasticity region.

## Multi-seed confirmation (5 seeds, mean ± sample SD)
| method | retention | plasticity | intervention |
|---|---|---|---|
| sequential | 0.539 ± 0.063 | **1.000 ± 0.000** | 0% |
| QEWC (λ=0.8) | 0.786 ± 0.098 | **1.000 ± 0.000** | 0% |
| L2 (λ=0.2) | 0.723 ± 0.119 | **1.000 ± 0.000** | 0% |
| QFI-TR (B=0.5) | 0.845 ± 0.040 | 0.507 ± **0.341** | 62% |
| CFI-TR (B=0.005) | 0.834 ± 0.036 | 0.520 ± **0.319** | 61% |

Robust across seeds: soft anchoring keeps **plasticity exactly 1.0 (SD 0)** with retention ~0.79;
hard trust regions get higher retention (~0.84) but **plasticity 0.52 with SD ~0.33** — the new
task unreliably fails (per-seed plasticity spans ~0.17–1.0). So the trade-off — reliable
new-task learning vs. higher old-task retention — holds across seeds, and **soft anchoring is the
usable operating point.** CFI-TR ≈ QFI-TR here (function vs state indistinguishable at 5 seeds).

## Why hard trust regions fail here
Continual learning **must move** in parameter space to learn a new task. A hard trust region
around old optima blocks that motion; when the new task's optimum is far (SPT/ATF vs images),
the feasible region excludes any good new-task solution, so plasticity collapses. Soft anchoring
lets the model move — paying a graded cost — so it learns the new task *and* stays near old ones.

## Consolidated e007 thesis (unchanged, now with the positive-method attempt closed)
- Directional QFI → falsified (multi-seed)
- Adaptive / hard trust-region control (Euclidean, QFI-state, CFI-function) → Pareto-dominated
- **QFI continual learning works as soft cumulative anchoring, not directional or hard-constraint
  quantum-geometric protection.** The physically-correct object is the readout function (CFI),
  and even that is best handled *softly*, not as a hard trust region.

## Files
- `src/e007_bbqewc.py`, `experiments/e007_bbqewc.py`
- `scripts/e007_bbqewc_frontier.py`, `scripts/e007_bbqewc_multiseed.py`, plot scripts
- `results/e007_bbqewc_frontier.json`, `results/e007_bbqewc_multiseed.json`
- `figures/e007_bbqewc_frontier.png`, `figures/e007_bbqewc_multiseed.png`

## Reproduce
```bash
python scripts/e007_bbqewc_frontier.py --seed 42 && python scripts/e007_bbqewc_plot_frontier.py
python scripts/e007_bbqewc_multiseed.py --seeds 42 43 44 45 46 && python scripts/e007_bbqewc_plot_multiseed.py
```
