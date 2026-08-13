# E015 — OI-QCL (measurement-side continual learning) on quantum time-series forecasting

Ports **OI-QCL** (Observable-Isolated Quantum Continual Learning, the e014 Task-IL
*classification* method) to one-step-ahead **forecasting** on the e009 quantum/physics series
(`narma_5 → damped_shm → bessel_j2`). Same idea: keep a *shared* recurrent data-reuploading
backbone θ (frozen or soft-anchored across tasks) and give every task its own lightweight
readout — a per-task diagonal observable `⟨H^(t)⟩ = Σ_k λ_k^(t) p_k(x;θ) = w^(t)·p_θ(x)`,
i.e. a **linear (Ridge) head over the computational-basis probabilities** `qml.probs()`. For
scalar forecasting the readout *is* that expectation, so the head is fit classically in seconds
with no quantum gradients. Old heads are frozen ⇒ measurement-side forgetting is a structural
zero; only backbone drift can move an earlier task's error.

Backbone ansatz is identical to `src.e009_qtsf.make_forecaster`, so a θ trained by the
paper-faithful tanh/MSE forecaster is reused verbatim — only the readout differs.

## Methods

| method | description |
|---|---|
| `sequential` | e009 naive: shared tanh readout, θ continued (CF baseline) |
| `qewc` | e009 QFI-weighted EWC anchor on θ (strongest e009 regularizer) |
| `frozen_head` (A) | θ frozen after Task 1 + isolated per-task linear heads |
| `free_head` (B) | θ keeps training + isolated heads (representation-drift probe) |
| `anchor_head` (C) | soft-L2-anchored θ + isolated heads (**MAIN candidate**) |

## Results (seeds 42/43/44, test NMSE, lower better)

| method | retention | plasticity | forgetting | avg_final |
|---|---:|---:|---:|---:|
| Sequential (naive) | 0.273 | 0.032 | +0.195 | 0.193 |
| QEWC | 0.157 | 0.156 | −0.006 | 0.157 |
| **Frozen θ + heads (A)** | 0.092 | 0.030 | **+0.000** | 0.071 |
| Free θ + heads (B) | 2.866 | 0.018 | +2.823 | 1.917 |
| **Anchor θ + heads (C)** | **0.087** | 0.033 | −0.004 | **0.069** |

- **OI-QCL A/C beat QEWC** (the best e009 regularizer) on retention **and** plasticity — QEWC
  stalls on new tasks (plasticity 0.156), the same failure it shows in e014 classification.
- **B-vs-C decomposition** is even more dramatic than in classification: free-θ backbone drift
  makes retention explode (2.87), while the soft anchor (C) keeps it at 0.087. Representation
  drift is the *entire* danger; the measurement side never forgets.
- **Cross-domain confirmation:** measurement-side isolation transfers classification → regression.

Run: `python experiments/e015_oiqcl_forecast_compare.py --tasks narma_5 damped_shm bessel_j2 --seed 42`
Aggregate + figure: `python scripts/e015_plot_compare.py` → `figures/e015_oiqcl_forecast_compare.png`

## Remove the task id at test (task-agnostic router)

`experiments/e015_router.py` drops the Task-IL oracle: pool all tasks' test windows, hide the
task id, and route each window to a head — `centroid` (nearest per-task mean-probs, unsupervised)
or `learned` (LogisticRegression over `p_θ` trained on pooled-train task labels).

| method | oracle NMSE | learned router NMSE | router TIA |
|---|---:|---:|---:|
| Frozen (A) | 0.071 | 0.100 | 0.47 |
| **Anchor (C)** | 0.069 | **0.085** | 0.47 |
| Free (B) | 1.917 | 1.046 | 0.49 |

**Key nuance (differs from e014 classification, where the router recovered ~0.91 TIA):** here the
router TIA is only ~0.47 (3-task chance = 0.33) — the three series are too similar in `p_θ`, and
forecasting's time-ordered 80/20 split adds a train→test distribution shift, so the task id is
nearly undecodable. **Yet frozen/anchor NMSE stays near the oracle anyway**, because the per-task
linear heads are largely *interchangeable* under strong positive transfer: misrouting a window
still yields a decent forecast. Free-θ (B) is the exception — a drifted backbone makes heads
non-interchangeable, so misrouting inflates NMSE. → On this highly-transferable benchmark,
Task-IL is a convenience, not a hard requirement, *provided the backbone does not drift* (A/C).

Run: `python experiments/e015_router.py --seed 42`; aggregate `python scripts/e015_router_aggregate.py`
→ `figures/e015_router.png`.

## Per-task trajectory figure

`experiments/e015_trajectory.py` records each task's test NMSE at **every** gradient step of the
full sequential run (past tasks read with their frozen head at the current backbone so drift
surfaces as forgetting; each task's curve starts only at its training onset).
`scripts/e015_plot_trajectory.py` → `figures/e015_trajectory.png`: one panel per task, mean ± std
over seeds, shaded band = task being trained. Shows A/C staying flat post-training (zero/near-zero
forgetting) while B (free θ) blows up after each task boundary.

## Honest scope (no quantum-advantage claim)

This is a **mechanism** result: measurement-side isolation (frozen/anchored θ + per-task
observable) beats θ-protection (QEWC) **within the quantum model class**, and transfers from
classification to forecasting. It is **not** a quantum-advantage claim. As with e014's matched
classical control, a classical AR/ridge forecaster already fits these series at ~0 NMSE, and the
probability readout `p=|amp|²` discards phase — so a matched classical head would likely do at
least as well. Establishing quantum *value* needs a benchmark where a matched classical baseline
fails; that is the natural next step, not part of e015.
