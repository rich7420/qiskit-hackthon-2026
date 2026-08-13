# Experiment Log

One entry per meaningful run. Keep it append-only. Copy the template below.

Goal: on the last night, anyone can answer "which backend / shots / seed produced
result 0.91?" without guessing.

---

## Template

```
## EXXX — <short title>

Commit:
Backend:            # e.g. AerSimulator, ibm_<name>
Qiskit version:

Configuration:
- p / layers =
- optimizer =
- shots =
- seed =
- optimization_level =
- seed_transpiler =

Circuit:
- qubits =
- depth =
- two-qubit gates =

Result:
- objective =
- approximation ratio =

IBM Job ID:         # hardware runs only

Notes:
- ...
```

---

## E001 — QNN on binary MNIST (local simulator smoke test)

Goal: confirm the quantum-ML stack runs locally end to end, not to chase accuracy.

Source hash:       a3afa1ff7dca4c12b538c7a722401bf3ef53ea45b9b8dfec8c26bf8870da80f4
Data split hash:   882a60d7a3b21e3f5d5a539b2dd7daccd93664eeeb2ce5e0e4e043d0cad6ec74
Backend:           StatevectorEstimator (exact, CPU local — 4 qubits, no shots)
Qiskit version:    qiskit 2.5.1, qiskit-machine-learning 0.9.0
Command:           `python experiments/e001_qnn_mnist.py --dataset mnist --n-train 150 --n-test 150 --maxiter 60 --ansatz-reps 2 --seed 42 --output results/e001_qnn_mnist_reference.json`

Configuration:
- task = MNIST 0 vs 1, PCA 784 -> 4 features, angle-scaled to [0, pi]
- feature map = zz_feature_map(4, reps=1); ansatz = real_amplitudes(4, reps=2), 12 weights
- optimizer = COBYLA(maxiter=60); seed = 42 (data + initial weights both seeded)

Circuit:
- qubits = 4; decomposed depth = 25; two-qubit gates = 18

Result:
- final objective = 0.6487 over 60 evaluations
- QNN test accuracy = 0.78  (train 0.82)
- classical baseline (LogisticRegression, same 4 features) = 0.9867
- train time = 18.129 s (runtime is informational and machine-dependent)

Finding: the stack works and the QNN learns (train/test climb ~0.63 -> ~0.80 over 60
COBYLA objective evaluations), but it plateaus well below a trivial classical baseline — MNIST 0-vs-1
on 4 PCA features is nearly linearly separable, so there is no quantum advantage to show
here. This is a green infrastructure check, not a result. Reproducible: same seed gives the
same curve. The curve is jagged because COBYLA is derivative-free (not gradient/epoch based).
The source/data hashes, package versions, weights, and full evaluation history are stored in
`results/e001_qnn_mnist_reference.json`.

Figures:
- `figures/e001_qnn_circuit.png` — circuit schematic (via `scripts/plot_qnn_circuit.py`)
- `figures/e001_training_curve.png` — train/test accuracy vs objective evaluation (via `scripts/plot_training_curve.py`)

---

## E002 — amplitude-encoding QNN, gradient-trained (PennyLane)

Goal: test a paper-inspired amplitude-encoding classifier with exact-statevector gradients.
This is a single-seed engineering experiment, not a paper reproduction or a statistical
comparison. The closest reference is arXiv:2607.16030:
https://arxiv.org/html/2607.16030v1

Source hash:       cb0746cf66dc21276376935f0a34429c49f7af3be03d53baeaa73986a394776c
Data split hash:   263e2bef53d342880f60d4b28d74aedaaa91eda092e7d96a7873c8e414a07aec
Framework:         pennylane 0.45.1 (`default.qubit`, backprop, exact / no shots)
Command:           `python experiments/e002_amplitude_qnn.py --n-qubits 4 --layers 30 --epochs 75 --lr 0.01 --n-train 600 --n-validation 200 --n-test 200 --seed 42 --output results/e002_amplitude_qnn_reference.json`

Configuration (selected by validation only; test was held out during `scripts/tune_e002.py`):
- task = MNIST 0 vs 1; 600 train / 200 validation / 200 test
- preprocessing = train-only PCA -> 16 features, then per-sample L2 normalization
- ansatz = independent RY/RZ + nearest-neighbor CNOT, 30 layers, 240 weights
- output/loss = one `<Z_0>` expectation with {-1, +1} labels and mean squared error
- optimizer = full-batch Adam(lr=0.01), 75 epochs; seed = 42

Circuit:
- qubits = 4; logical depth = 122; two-qubit gates = 90

Result:
- QNN accuracy = 0.9833 train / 0.9900 validation / 0.9800 held-out test
- final training loss = 0.3260
- classical baseline (LogisticRegression, same 16 features) = 0.995
- train time = 24.933 s (runtime is informational and machine-dependent)

Finding: on this fixed split and seed, the QNN learns a high-accuracy classifier but remains
below the simple classical baseline (196/200 versus 199/200 correct). Validation-only tuning
selected 30 layers, lr=0.01, and epoch 75; the tuning artifact records zero test evaluations.
No optimizer-causality or quantum-advantage claim is made from this run.

Method note: arXiv:2607.16030 shares each qubit's RY/RZ angle, uses two Z readouts with
softmax/BCE, parameter-shift gradients, 20 training epochs, and six trials. E002 instead uses
independent RY/RZ angles, one Z readout with MSE, statevector backprop, and one seed. The
earlier arXiv:2108.02786 experiment is an 8-qubit continual-learning setup, so its accuracy is
not directly comparable: https://arxiv.org/abs/2108.02786

Framework note: Qiskit's `RawFeatureVector` / `ParameterizedInitialize` implementation cannot
be used with gradient-based optimizers; this is narrower than saying Qiskit cannot
differentiate amplitude encoding in general:
https://qiskit-community.github.io/qiskit-machine-learning/_modules/qiskit_machine_learning/circuit/library/raw_feature_vector.html

The package versions, weights, hashes, split sizes, circuit metrics, and full train/validation
history are stored in `results/e002_amplitude_qnn_reference.json`. Test accuracy is absent
from the per-epoch history because the test split is evaluated only after training.

Tuning artifact:
- `results/tune_e002.json` — all validation-only candidates, selected epoch, versions, hashes,
  seed, split sizes, and `test_evaluations = 0`

Figures:
- `figures/e002_qnn_circuit.png` — circuit schematic (via `scripts/plot_e002_circuit.py`)
- `figures/e002_training_curve.png` — train/validation accuracy vs epoch (via `scripts/plot_e002_curve.py`)

---

## E003 — sequential-training baseline: MNIST then Fashion-MNIST

Goal: establish the no-consolidation baseline for comparing training accuracy when one QNN
is trained on MNIST 0/1 first and then continued on Fashion-MNIST 0/1. This is a single-seed
engineering result, not a statistical reproduction or quantum-advantage claim.

Source hash:       96c529c300304f340cebe55715fb45e7e8fd644f6948f829b19894a060df4aec
Data split hash:   5445553de9ccdf87be4e545949f930e3fe9fd53dc44ed0e2edb9e7729cc3753a
Framework:         pennylane 0.45.1 (`default.qubit`, backprop, exact / no shots)
Command:           `python experiments/e003_continual_baseline.py --n-qubits 4 --layers 20 --lr 0.02 --epochs-per-task 40 --n-train 800 --n-test 200 --seed 42 --output results/e003_continual_baseline_reference.json`

Configuration:
- task order = MNIST 0/1 for epochs 1–40, then Fashion-MNIST 0/1 for epochs 41–80
- representation = PCA fit once on MNIST training data and applied unchanged to both tasks;
  each 16-feature vector is L2-normalized for amplitude encoding
- ansatz = independent RY/RZ + nearest-neighbor CNOT, 20 layers, 160 weights
- output/loss = one `<Z_0>` expectation with {-1, +1} labels and mean squared error
- optimizer = one full-batch Adam(lr=0.02) instance; weights and optimizer state both continue
  across the task boundary; no EWC or other consolidation; seed = 42

Circuit:
- qubits = 4; logical depth = 82; two-qubit gates = 60

Training-accuracy comparison:
- MNIST = 0.9625 after phase 1 -> 0.5800 after phase 2; forgetting = 0.3825
- Fashion-MNIST = 0.3600 before phase 2 -> 0.9500 after phase 2

Test-set diagnostic (fixed 40 epochs per task; not used for model selection):
- MNIST = 0.9600 at the boundary -> 0.6050 final
- Fashion-MNIST = 0.9550 final
- separate LogisticRegression baselines = 0.9950 MNIST / 0.9450 Fashion-MNIST
- QNN train time = 26.825 s (runtime is informational and machine-dependent)

Finding: sequential fine-tuning learns both tasks when they are active, but Fashion-MNIST
training reduces MNIST training accuracy by 0.3825. This is the intended catastrophic-
forgetting baseline; a later consolidation method should be compared against the same task
order, split, representation, epoch budget, and seed.

The reference JSON stores epoch 0 plus every post-update epoch, including train/test accuracy
for both tasks, active-task loss, initial/final weights, hashes, versions, circuit metrics, and
the explicit `optimizer_state_reset_at_boundary = false` contract.

Figures:
- `figures/e003_training_curve.png` — both training-accuracy curves across the two phases
  (via `scripts/plot_e003_training.py`)
- `figures/e003_continual_baseline.png` — train/test curves for each task, showing learning
  and forgetting (via `scripts/plot_e003_curve.py`)

---

## E004 — 3-task continual-learning baseline: MNIST -> Fashion-MNIST -> SPT/ATF

Goal: extend E003 with the quantum-native SPT/ATF task and measure training-accuracy
retention under naive sequential fine-tuning. This is a three-seed, paper-inspired engineering
baseline, not a reproduction or a quantum-advantage result. Reference:
https://arxiv.org/html/2607.16030v1

Source hash:       517acacfc01d2c1cda90e4fbae6f15f9cc19a07d1afc28002503d1a3f77c6e90
Data split hashes: seed 42 `ea2430213eb950aaf1fbc2bbbdd8f9bdbd500be94b328c90a27486917ab289c9`
                   seed 43 `22a8bd27e83c259f51407f4f0a6a59426e208e9628df7d91926d30b019cf573b`
                   seed 44 `374b55ddb4b8c649679341ee473b188e7e2cf4b49a6aeea9fbba734d40d70d21`
Framework:         pennylane 0.45.1 (`default.qubit`, backprop, exact / no shots)
Command:           `python scripts/run_e004_multiseed.py --seeds 42 43 44 --layers 20 --lr 0.02 --epochs-per-task 20 --n-train 800 --n-test 200`
Plot command:      `python scripts/plot_e004_curve.py --seeds 42 43 44`

Configuration:
- task order = MNIST 0/1, Fashion-MNIST 0/1, then SPT/ATF; 800 train / 200 fixed
  test samples per task and 20 full-batch epochs per task
- image representation = one PCA-16 transform fit on MNIST train and held fixed for both
  image tasks; each vector is L2-normalized for amplitude encoding
- phase representation = exact-diagonalization ground states of the four-spin open-boundary
  cluster-Ising Hamiltonian; SPT samples use `h in [0.0, 0.5]` and ATF samples use
  `h in [2.5, 3.0]`; the even-global-Z-parity representative is selected deterministically
  from the degenerate ground space and amplitude-encoded directly
- learner = 20 layers of independent RY/RZ rotations and nearest-neighbor CNOTs, 160
  weights, one `<Z_0>` readout, {-1,+1} labels, and MSE
- training = one Adam(lr=0.02) instance whose weights and optimizer state both continue
  across task boundaries; no EWC/QEWC; independently seeded data and weights for 42/43/44

Training-accuracy result (mean +/- sample standard deviation across three seeds):
- MNIST = 0.9204 +/- 0.0264 at its phase end -> 0.5280 +/- 0.0321 final;
  forgetting = 0.3925 +/- 0.0569
- Fashion-MNIST = 0.9296 +/- 0.0125 at its phase end -> 0.8029 +/- 0.1297 final;
  forgetting = 0.1267 +/- 0.1313
- SPT/ATF = 1.0000 +/- 0.0000 at its phase end and final

Fixed-test diagnostic (not used for parameter or epoch selection):
- final MNIST = 0.5317 +/- 0.0425
- final Fashion-MNIST = 0.8083 +/- 0.1665
- final SPT/ATF = 1.0000 +/- 0.0000

Finding: the baseline reliably learns each active task and exhibits catastrophic forgetting,
especially on MNIST. Fashion-MNIST retention varies substantially by seed, so the earlier
single-seed curve was not a credible estimate of its typical forgetting. Three seeds expose
that variance but remain a small sample; these values should be treated as an internal
baseline for later consolidation experiments, not as precise population estimates.

Paper-comparison limit: the reference uses 30 layers with 120 shared parameters, two Z
readouts with softmax/BCE, parameter-shift gradients, and six trials. E004 deliberately keeps
the E003 learner (20 layers, 160 independent parameters, one Z/MSE readout, backprop) and runs
three seeds. The reference reports final three-task accuracies of 0.632/0.498/1.000, whereas
our final training means are 0.528/0.803/1.000. These are not directly comparable metrics or
architectures, and E004 must not be described as reproducing the paper's curve.

Artifacts:
- `results/e004_continual_seed42.json`, `e004_continual_seed43.json`, and
  `e004_continual_seed44.json` store complete per-epoch train/test histories, initial/final
  weights, versions, hashes, circuit resources, and runtime
- `results/e004_continual_summary.json` stores the mean and sample SD curve, per-task
  aggregates, seed list, source hash, and per-seed data hashes
- `figures/e004_continual_seed42.png`, `e004_continual_seed43.png`, and
  `e004_continual_seed44.png` show the three training-accuracy runs
- `figures/e004_continual_mean.png` shows the three-seed mean with +/-1 sample-SD bands

---

## E005 — continual learning with consolidation: Baseline vs EWC (CFI) vs QEWC (QFI)

Goal: reproduce the core contribution of arXiv:2607.16030 on the e004 three-task sequence —
mitigate catastrophic forgetting with Elastic Weight Consolidation using the classical Fisher
(EWC) and the Quantum Fisher Information (QEWC). No public code exists for the paper; the
methods are implemented from its equations. Paper-inspired reproduction, 3 seeds.

Framework:         pennylane 0.45.1 (default.qubit, backprop, exact / no shots)
Command:           `python scripts/e005_run_multiseed.py --seeds 42 43 44 --qfi-samples 32`

Method (faithful to the paper's classifier):
- readout = two Pauli-Z on qubits 0,1 -> softmax (Eq. 7); loss = binary cross-entropy
- CFI = empirical Fisher from softmax log-likelihood gradients (Eq. 9)
- QFI diag = 4(<d_i psi|d_i psi> - |<psi|d_i psi>|^2) (Eq. 18) via Fubini-Study metric tensor,
  averaged over a data subsample; QFI is a state property so it is readout-independent
- penalty = (lambda/2) sum_j alpha_j^(k) sum_i F_i^(j) (theta_i - theta*_{j,i})^2 (Eq. 2/21)
- alpha_j^(k) = j/(k-1) for k>1 (Eq. 30); lambda_EWC = 30, lambda_QEWC = 0.8
- learner = 4 qubits, RY/RZ + CNOT (20 layers, 160 weights), Adam(lr=0.02), 20 epochs/task
- tasks = MNIST 0/1 -> Fashion-MNIST 0/1 -> SPT/ATF (same as e004); seeds 42/43/44

Result (mean final test accuracy on the two earlier tasks, mean +/- sample SD over 3 seeds):
- baseline = 0.627 +/- 0.103   (severe forgetting)
- EWC      = 0.755 +/- 0.033
- QEWC     = 0.826 +/- 0.045   (best retention, tightest variance)
- ordering QEWC > EWC > baseline reproduces the paper's headline; single-seed T1(MNIST) final:
  baseline 0.575, EWC 0.775, QEWC 0.895

Deviations from the paper (documented in the result JSON): independent RY/RZ (no weight
sharing), 20 layers/160 params vs the paper's shared 30 layers/120 params; Adam state continues
across task boundaries (paper unspecified). Noise experiments (depolarizing, mixed-state QFI,
retuned lambda_q) not implemented.

Files: src/e005_consolidation.py (CFI/QFI/EWC), src/e005_softmax.py (2-Z softmax readout),
experiments/e005_ewc_qewc.py, scripts/e005_run_multiseed.py, scripts/e005_plot_curve.py,
results/e005_seed{42,43,44}.json + results/e005_summary.json.

Figure: `figures/e005_ewc_qewc.png` (via `scripts/e005_plot_curve.py`) — three panels (per task)
comparing Baseline / EWC / QEWC with mean +/- 1 sample-SD bands, in the style of
`figures/e004_continual_mean.png`.

---

## E011 - schedule study: blocked vs interleaved (fixed epoch budget)

Question: E004/E005 forget earlier tasks because they train in *blocks* (T1x20, T2x20, T3x20).
If the per-task epoch budget is held fixed (still 20 each, 60 gradient steps total), does simply
*interleaving* the tasks change the picture? No consolidation (no EWC/QEWC) -- this isolates the
effect of the training schedule alone.

Method:
- blocked      = step order [T1]*20, [T2]*20, [T3]*20   (identical to the E005 baseline)
- interleaved  = step order [T1, T2, T3] repeated 20 times
- same initial weights, same learner (4 qubits, RY/RZ + CNOT, 20 layers/160 weights,
  Adam(lr=0.02)), same 20 epochs/task, same tasks (MNIST 0/1 -> Fashion-MNIST 0/1 -> SPT/ATF);
  test accuracy on all three tasks recorded after every gradient step. Seeds 42/43/44.

Result (mean final test accuracy over 3 seeds, mean +/- sample SD):
- earlier tasks (T1,T2):  blocked 0.627 +/- 0.103  ->  interleaved 0.849 +/- 0.051
- all tasks (T1-T3):      blocked 0.752 +/- 0.069  ->  interleaved 0.899 +/- 0.034
- interleaving recovers most of the forgetting for free (T1 0.540->0.748, T2 0.715->0.950
  single-seed means) at identical compute; both reach ~1.0 on T3. This is the classic
  continual-learning "interleaved ~ joint-training upper bound": the blocked schedule's
  forgetting is a property of the ordering, not the data. It is the reference EWC/QEWC (E005)
  aim to approach *without* revisiting earlier tasks -- interleaving assumes all tasks stay
  available, which the continual setting forbids.

Four-way comparison (schedule x consolidation), earlier-task (T1,T2) mean final acc, all with
identical params (4 qubits, 20 layers/160 weights, Adam lr=0.02, 20 epochs/task, n_train=800,
seeds 42/43/44 -- verified equal across the E005 and E011 summaries; E005 baseline == E011
blocked to 4 dp):
- blocked (no consol.)   0.627 +/- 0.103   (lower reference: naive schedule, forgets)
- QEWC blocked (E005)    0.826 +/- 0.045   (continual method: fixes forgetting WITHOUT revisiting)
- QEWC interleaved       0.719 +/- 0.008   (online QEWC on top of interleaving -- see below)
- interleaved            0.849 +/- 0.051   (upper bound: revisits every task each round)

qewc_interleaved = the interleaved step order PLUS online QEWC: a single running QFI-weighted
anchor, re-anchored to the current weights after every full T1->T2->T3 round with the diagonal
quantum Fisher (E005's QFI, estimated over a pooled subsample of all tasks, qfi_samples=16)
accumulated across rounds (Schwarz et al. online EWC). lambda_qewc=0.8, same 60-step budget.

Finding: adding consolidation on top of interleaving HURTS (0.719 < 0.849 interleaved), because
an interleaved stream has no forgetting left to prevent -- the penalty only trades away
plasticity for stability that is not needed. The loss is concentrated on T1 (mean final
0.748 interleaved -> 0.515 qewc_interleaved), the task most penalized for drifting from its
early, still-poor anchor; T3 stays ~1.0. It does collapse the seed variance (+/-0.008 vs
+/-0.051), i.e. it stabilises training at the cost of the ceiling. Takeaway for the slides:
QEWC's value is specifically in the *blocked* (continual) regime; when replay/interleaving is
available it is at best redundant and here mildly harmful.

Files: experiments/e011_interleaved.py, scripts/e011_run_multiseed.py,
scripts/e011_plot_curve.py, scripts/e011_plot_combined.py, tests/test_e011.py,
results/e011_seed{42,43,44}.json + results/e011_summary.json. Reuses src/e005_softmax.py,
src/e005_consolidation.py (QFI), src/qnn_pennylane.py, src/continual_data.py, src/phase_data.py
from the E005 PR. Figures (all three panels per task, mean +/- 1 sample-SD bands over seeds):
`figures/e011_interleaved.png` (blocked vs interleaved); `figures/e011_combined.png` (all four
arms overlaid); `figures/e011_training.png` (E009-style training curves, four arms, with a green
span marking each task's block in the blocked/sequential schedule). The combined/training figures
need results/e005_summary.json from scripts/e005_run_multiseed.py for the QEWC-blocked curve.

Caveat for the slides: only blocked vs interleaved differ *purely* in the order of the 60
gradient steps (same init, data, learner, per-task budget). qewc_interleaved additionally adds
the online QEWC penalty on top of the interleaved order -- it is not an order-only change.
