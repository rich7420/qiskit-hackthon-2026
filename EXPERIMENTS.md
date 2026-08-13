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

## E008 — MeasQCL: what should a quantum continual learner measure to remember?

Goal: test whether task-boundary measurements in complementary Pauli bases provide a better
EWC importance signal than the classifier's fixed readout. This first implementation is a
three-seed, exact-statevector engineering experiment. It does not yet estimate Fisher
information from finite shots or run on hardware.

Source hash:       b3901b55689d64b71b283f72d432468056e0ded9c0842bee2fbe83d3ca44b962
Data split hashes: seed 42 `6af663e6cfaf7f1d4afc923089f08924c06a7f3af7c698839d754f3a9c383404`
                   seed 43 `5d320685b06bceda86b35cea74089e9263edcd92ca24caef90be7a06d3b9a3b2`
                   seed 44 `971544ab3b43725e903660786c9afc2ccfb730279c22646ee35b9383a5d4af9f`
Framework:         pennylane 0.45.1 (`default.qubit`, backprop training; exact probabilities)
Run command:       `python scripts/run_e008_multiseed.py --seeds 42 43 44`
Plot command:      `python scripts/plot_e008_measqcl.py`
Tuning command:    `python scripts/tune_e008_train_only.py`

Configuration:
- task order = MNIST 0/1 then Fashion-MNIST 0/1; 400 train / 200 fixed test samples
  per task; one PCA-64 representation fit only on MNIST training data and held fixed
- classifier = six-qubit amplitude encoding, 10 independent RY/RZ + CNOT-chain layers
  (120 parameters), two Pauli-Z scores with softmax/BCE, Adam(lr=0.02), 40 epochs/task
- comparison = naive, empirical output-CFI EWC, joint-ZZ CFI EWC, uniform XX/YY/ZZ,
  measurement-optimized Fisher EWC (MOF-EWC), readout-subsystem QFI, and full-state QFI
- every method receives the identical Task-1 weights, Adam state, Task-2 examples, epoch
  budget, and `lambda=0.1`; every non-naive Fisher diagonal is normalized to mean one
- capacity and the shared lambda were selected on seed-42 training metrics only. The lambda
  sweep used the established output-CFI baseline, not MOF-EWC; test data were not evaluated
  during selection

Measurement and physics definition:
- prediction remains the original two-Z softmax measurement. Extra bases are used only at
  the Task-1 boundary and therefore do not change deployment inference cost
- each setting measures the joint outcomes of the two readout qubits. For a randomized
  measurement whose setting label is retained, the accessible Fisher is
  `F_acc(q) = sum_m q_m F_m`; `q_m` is also the asymptotic fraction of a fixed shot budget
- MOF-EWC chooses `q` by maximizing the task-agnostic diagonal coverage objective
  `sum_i log(epsilon + [F_acc(q)]_i)`. This discourages allocating everything to one already
  sensitive direction; it does not use Task-2 outcomes or optimize retention directly
- the exact reduced-state SLD QFI of the two readout qubits is included because it is the
  physically relevant measurement-independent upper bound for POVMs restricted to that
  subsystem. The checked hierarchy is `F_basis <= F_readout-QFI <= F_full-QFI` on every
  diagonal. The full QFI is a state-geometric reference, not a claim that one measurement
  can attain its multiparameter bound
- `ZZ/XX/YY` is deliberately a small, hardware-simple candidate set, not an
  informationally complete measurement family. Mixed Pauli products such as XZ and YX are
  left for the nine-basis extension

Final held-out test result (mean +/- sample standard deviation across seeds 42/43/44):

| Method | MNIST retention | Fashion adaptation | Forgetting | Mean final accuracy |
|---|---:|---:|---:|---:|
| Naive | 0.625 +/- 0.110 | 0.950 +/- 0.035 | 0.250 +/- 0.124 | 0.788 +/- 0.040 |
| Output CFI | 0.852 +/- 0.035 | 0.958 +/- 0.029 | 0.023 +/- 0.038 | 0.905 +/- 0.007 |
| Joint ZZ CFI | **0.865 +/- 0.040** | 0.952 +/- 0.023 | **0.010 +/- 0.039** | **0.908 +/- 0.012** |
| Uniform XYZ | 0.850 +/- 0.053 | 0.947 +/- 0.028 | 0.025 +/- 0.063 | 0.898 +/- 0.013 |
| MOF-EWC | 0.842 +/- 0.051 | 0.937 +/- 0.019 | 0.033 +/- 0.058 | 0.889 +/- 0.016 |
| Readout QFI | 0.862 +/- 0.046 | 0.945 +/- 0.022 | 0.013 +/- 0.053 | 0.903 +/- 0.012 |
| Full QFI | 0.840 +/- 0.044 | 0.943 +/- 0.021 | 0.035 +/- 0.053 | 0.892 +/- 0.013 |

Measurement diagnostic (mean +/- sample SD):
- MOF allocation = `q_ZZ=0.000`, `q_XX=0.922 +/- 0.078`,
  `q_YY=0.078 +/- 0.078`; these are exact-information allocations, with a 1,024-shot integer
  plan recorded only as a resource illustration
- cosine similarity to full QFI = 0.747 +/- 0.028 (ZZ), 0.909 +/- 0.008 (uniform),
  0.910 +/- 0.002 (MOF)
- readout-QFI trace coverage proxy = 0.186 +/- 0.010 (ZZ), 0.201 +/- 0.004 (uniform),
  0.212 +/- 0.003 (MOF)
- cosine similarity to the classifier output CFI moves in the other direction:
  0.887 +/- 0.026 (ZZ), 0.800 +/- 0.017 (uniform), 0.686 +/- 0.047 (MOF)

Finding: complementary measurements do reveal parameter sensitivity hidden from the fixed
readout. MOF-EWC obtains the highest accessible trace proxy and highest mean alignment
to full-state QFI, yet it does not improve retention on this task pair. Joint ZZ is the best
mean performer and readout-QFI is close behind. The physically useful conclusion is narrower
and more interesting than "more Fisher is better": state sensitivity and task-memory
relevance are different objectives. A measurement can cover quantum-state geometry while
protecting directions that do not control the old classifier, creating redundant protection
and reducing plasticity. A next optimizer should therefore balance accessible coverage with
old-task output or gradient alignment, evaluated on a validation protocol rather than tuned
against the held-out test set.

The descriptive correlations in the summary support only this diagnostic, not population
inference: the nine seed-method points are dependent and three seeds are too few for a
statistical claim. Exact boundary estimation is also expensive in the unoptimized reference
code: mean wall time is about 33 s for ZZ and 100 s for all three bases, versus 0.57 s for
autodifferentiated output CFI. These timings identify clear acceleration targets: vectorized
parameter shifts, shared shifted-state evaluations across bases, parallel anchors, and a
finite-shot estimator that measures only the allocated settings.

Claim boundaries:
- no finite-shot, noise, Aer, IBM hardware, quantum-advantage, or QFI-attainability claim
- no assertion that trace coverage is the fraction of globally attainable QFI
- results cover one two-task transition, three seeds, and a three-setting measurement family
- sample SD describes run-to-run spread; it is not a confidence interval

Artifacts:
- `results/e008_measqcl_seed{42,43,44}.json` — complete paired histories, weights, Adam-state
  hash, exact Fisher profiles, measurement allocations, information-hierarchy checks,
  resource counts, package versions, source/data hashes, and runtime
- `results/e008_measqcl_summary.json` — three-seed means/sample SDs and paired geometry-memory
  diagnostics
- `results/e008_train_only_tuning.json` — capacity and output-CFI lambda scans with zero test
  evaluations, selection rules, candidates, hashes, and train metrics
- `figures/e008_measqcl_curves.png` — Task-2 learning and Task-1 retention only; the common
  Task-1 phase is intentionally omitted
- `figures/e008_measqcl_stability_plasticity.png` — paired seed points and the final frontier
- `figures/e008_measqcl_measurement_geometry.png` — Fisher heatmap, optimized basis allocation,
  and geometry diagnostics

Primary references:
- Hsu et al., QEWC / measurement-dependent CFI and state-geometric QFI:
  https://arxiv.org/html/2607.16030v1
- EWC-DR / importance-estimation and redundant-protection motivation:
  https://arxiv.org/html/2603.18596v1

Cross-experiment overview:
- `results/e004_e008_comparison.json` and `figures/e004_e008_comparison.png` place the
  E004–E008 stability, plasticity, and forgetting summaries in one scope-aware figure.
- Values must be compared within each experiment block. E006 uses balanced accuracy and a
  different temporal benchmark; E008 has two tasks; the other blocks have three. E007's
  selected operating points are exploratory because frontier selection and adaptive budget
  calibration used test accuracy. E005 is marked unauthenticated because its local historical
  artifacts were not committed and their source hash does not match the final PR source.
- The common forgetting panel uses phase-end minus final held-out score for every block. For
  E006 this is recomputed per seed from the committed raw test-balanced-accuracy histories; it
  is intentionally not the E006 summary's max-boundary forgetting headline.

---

## E010 — PhysMeas-QCL: task-relevant and locality-resolved measurement design

Goal: follow the E008 negative result rather than hiding it. E008 showed that maximizing
task-agnostic accessible Fisher coverage selected mostly XX, moved closer to QFI, but retained
less old-task accuracy than fixed ZZ. E010 asks which measurable sensitivity is relevant to
the old task, how stable its selection is under finite shots, and whether phase memory depends
monotonically on output-observable locality.

Core method:
- EWC-DR reversed-logit importance is computed on the old-task anchors. For the proposed
  method it is used only as a relevance weight in
  `sum_i w_i log(epsilon + sum_m q_m F_mi)`; the EWC penalty itself remains the accessible
  measurement CFI. This isolates measurement selection from changing the regularizer
- every Fisher used in the penalty is normalized to mean one, so comparisons test profile
  shape rather than raw Fisher mass. Prediction remains the original Z-readout classifier
- the allocation solver rescales each nonzero parameter column (which adds only a constant
  to the zero-epsilon log objective), records its solver and certified Frank-Wolfe dual gap,
  and fails if the declared numerical tolerance is not reached

Paired MNIST -> Fashion result (held-out test, mean +/- sample SD, seeds 42/43/44):

| Method | MNIST retention | Fashion adaptation | Mean final accuracy |
|---|---:|---:|---:|
| Task-agnostic MOF (E008) | 0.842 +/- 0.051 | 0.937 +/- 0.019 | 0.889 +/- 0.016 |
| Output CFI | 0.852 +/- 0.035 | 0.958 +/- 0.029 | 0.905 +/- 0.007 |
| EWC-DR | 0.853 +/- 0.036 | 0.958 +/- 0.025 | 0.906 +/- 0.006 |
| Task-relevant MOF | 0.860 +/- 0.044 | 0.955 +/- 0.026 | 0.908 +/- 0.011 |
| Joint ZZ | **0.865 +/- 0.040** | 0.952 +/- 0.023 | **0.908 +/- 0.012** |

Task relevance changes the three-basis allocation from E008's task-agnostic
`ZZ/XX/YY = 0.000/0.922/0.078` to `0.878/0.112/0.010` on average. It repairs most of the
task-agnostic MOF loss and reaches the same practical frontier as fixed ZZ, but does not
establish a win over fixed ZZ with only three seeds. EWC-DR and vanilla output CFI are also
nearly identical here; reversed logits avoid confidence collapse in the estimator, but their
normalized profiles are highly aligned on this task.

Finite-shot selection audit:
- exact base and +/-pi/2 probability tables are independently sampled with multinomial
  noise. Each budget has 20 repetitions per seed and a Jeffreys 0.5 outcome pseudocount;
  first-order squared-derivative sampling noise is subtracted non-negatively
- 64/256/1024 shots are **per probability circuit**, not total shots. Counting 32 anchors,
  120 parameters, both shifts, and all three candidate bases gives 1,480,704 / 5,922,816 /
  23,691,264 pilot shots per seed and repetition
- selected accessible-profile cosine to exact is 0.9671 +/- 0.0051, 0.9949 +/- 0.0005,
  and 0.9991 +/- 0.0001 across the three budgets. This validates selection stability in
  simulation; production-shot allocation, shot-based training, noise, and hardware are not
  performed
- this budget covers measurement-specific CFI only. Task relevance is still computed by
  exact simulator autodifferentiation; its hardware estimation cost is not included, so the
  audit is conditional on exact relevance and is not yet a complete QPU resource estimate

Phase-first locality experiment:
- task order is SPT/ATF -> MNIST -> Fashion, so later tasks can actually erase phase memory.
  The four-qubit, three-layer classifier is the smallest candidate reaching >=0.99 phase
  training accuracy in the checked seed-42 train-only capacity scan. The shared lambda=0.1
  is inherited from E008 and is not tuned on E010 phase results
- for a Pauli observable P with binary +/-1 outcomes, the exact CFI is
  `(d<P>/dtheta)^2 / (1-<P>^2)`. Families include classifier readout, all one-local Pauli
  observables, same-axis nearest-neighbour two-local observables, the cluster-Ising XZX/YY
  Hamiltonian terms, and the weight-four stabilizer-product correlation XYYX. Pauli weight, support
  diameter, number of observables, and compatible product-basis settings are stored
- to isolate phase memory, every branch keeps one fixed phase-boundary anchor through both
  later image tasks; it does not add a second MNIST anchor. MNIST final accuracy is therefore
  a plasticity/interference diagnostic, not a claim of full three-task consolidation
- these are locality labels on the learned **output** state. The directed CNOT ansatz can
  enlarge their Heisenberg pullback, so E010 does not claim equality with locality in the
  input ground state or a thermodynamic phase transition

Final phase-task retention after both image tasks (mean +/- sample SD):
- naive = 0.500 +/- 0.000
- output CFI = 0.707 +/- 0.261; readout Pauli = 0.700 +/- 0.265
- one-local = 0.817 +/- 0.275; two-local = 0.667 +/- 0.289
- Hamiltonian-aligned = 0.833 +/- 0.289; XYYX weight-four = 0.707 +/- 0.257
- task-relevant all-Pauli = 0.788 +/- 0.259; full QFI comparator = **0.927 +/- 0.127**

The phase result is strongly seed-dependent. Seed 42 favored XYYX while seed 44 favored
two-local/Hamiltonian observables; therefore the data reject a simple "more nonlocal is
always better" story. The defensible conclusion is that physics-informed accessible
measurements recover part of the QFI retention benefit, observable choice is task- and
trajectory-dependent, and QFI has the highest mean phase retention among these comparators.
Three seeds and large sample SDs are not enough for statistical or universal locality claims.

Artifacts:
- `results/e010_physmeas_seed{42,43,44}.json` and `e010_physmeas_summary.json`
- `results/e010_finite_shot_seed{42,43,44}.json` and `e010_finite_shot_summary.json`
- `results/e010_phase_train_only_tuning.json`,
  `e010_phase_locality_seed{42,43,44}.json`, and `e010_phase_locality_summary.json`
- `figures/e010_physmeas_main.png` and `figures/e010_finite_shot.png`

Claim boundaries: exact statevector training only; no hardware, noise, finite-shot retention,
QFI attainability, thermodynamic-limit, statistical-significance, or quantum-advantage claim.

## E013 — Learnable measurement Fisher consolidation

Goal: test whether a continuous ensemble of physically implementable product-measurement
bases can recover old-task parameter sensitivity that fixed ZZ/XX/YY misses. The classifier,
two-Z prediction readout, Task-1 trajectory, Adam state, anchors, shared lambda, and data split
remain paired to E008/E010. Only the task-boundary Fisher measurement changes.

Method:
- each local observable has fixed spectrum `{-1,+1}` and is parameterized by a unit Bloch
  axis `n`, so `H=n_x X+n_y Y+n_z Z`. Joint bitstring probabilities from three two-qubit
  product settings define the measurement CFI; the setting ID remains part of the outcome
- the 32 anchor reduced density matrices at the Task-1 optimum and at every `+/-pi/2`
  classifier-parameter shift are cached once. This costs
  `32 * (1 + 2 * 120) = 7,712` exact circuit configurations per seed. All continuous-axis
  objective evaluations are classical projector contractions and add zero quantum circuits
- axes use seam-free local tangent coordinates with sphere retraction. Analytic automatic
  differentiation through the cached density/projector contractions drives bounded local
  trust-region solves; reaching a chart boundary causes a rebase, never convergence. Every
  solve is rejected unless its coordinate-free projected Bloch-sphere gradient is at most
  `1e-3`. When allocation is learned, axis steps alternate with the certified E010 convex
  simplex solve. Initial settings are near Z/X/Y; task-relevant variants use the paired E010
  EWC-DR relevance with a prespecified floor
- the information-only ablation gives every classifier parameter equal relevance. Every
  resulting accessible Fisher is normalized to mean importance one before the shared EWC
  penalty, and test history is recorded but not used by the prespecified measurement design

Paired MNIST -> Fashion result (held-out test, mean +/- sample SD, seeds 42/43/44):

| Method | MNIST retention | Fashion adaptation | Mean final accuracy |
|---|---:|---:|---:|
| Task-agnostic Alloc-XYZ (E008) | 0.842 +/- 0.051 | 0.937 +/- 0.019 | 0.889 +/- 0.016 |
| Task-relevant Alloc-XYZ (E010) | 0.860 +/- 0.044 | 0.955 +/- 0.026 | 0.908 +/- 0.011 |
| Joint ZZ | **0.865 +/- 0.040** | 0.952 +/- 0.023 | 0.908 +/- 0.012 |
| LearnBasis, information + allocation | 0.845 +/- 0.044 | 0.942 +/- 0.023 | 0.893 +/- 0.010 |
| LearnBasis, task relevance + uniform | 0.862 +/- 0.043 | **0.955 +/- 0.026** | **0.908 +/- 0.008** |
| LearnBasis, task relevance + allocation | 0.860 +/- 0.044 | **0.955 +/- 0.026** | 0.908 +/- 0.009 |
| Readout QFI | 0.862 +/- 0.046 | 0.945 +/- 0.022 | 0.903 +/- 0.012 |

Mechanistic result:
- information-only basis learning finds complementary, mostly equatorial measurements:
  allocation-weighted axis power is `X^2/Y^2/Z^2 = 0.657/0.264/0.079`. It reaches full-QFI
  cosine `0.9127 +/- 0.0018` and physically matched readout-QFI cosine
  `0.9819 +/- 0.0066`, but retains only `0.845 +/- 0.044`
- task-relevant basis learning rotates back toward the classifier measurement:
  `X^2/Y^2/Z^2 = 0.114/0.046/0.840`, and its output-CFI cosine rises to
  `0.892 +/- 0.020`, while readout-QFI cosine is `0.893 +/- 0.038`. Multiple settings often
  become antipodally equivalent, so learning q adds no aggregate performance over uniform
  allocation on this task pair
- every learned individual-basis CFI and accessible mixture passes the numerical diagonal
  hierarchy `F_measurement <= F_readout-QFI`; full-state QFI remains a secondary global
  reference, not the physically matched upper bound for these two-readout-qubit measurements
- task-relevant LearnBasis repairs the information-only loss and matches the E010/Joint-ZZ
  frontier. The uniform variant is `-0.0033 +/- 0.0284` retention relative to paired Joint ZZ;
  learning allocation changes that to `-0.0050 +/- 0.0265`. Three seeds
  do not establish superiority. The supported conclusion is that task relevance, not access
  to more QFI-like state geometry, determines useful consolidation here; a continuous basis
  is unnecessary for this Z-aligned image transition

Artifacts:
- `results/e013_learnable_measqcl_seed{42,43,44}.json`
- `results/e013_learnable_measqcl_summary.json`
- `results/e013_figure_provenance.json`
- `figures/e013_learnable_measqcl.png`

Phase-first full-output extension:
- the old task is the four-qubit SPT/ATF classifier from E010, followed by MNIST and Fashion.
  The exact E010 phase boundary, optimizer state, anchors, data split, lambda, and future-task
  training are replayed and verified. The only branch-specific quantity is the phase-boundary
  Fisher profile
- each of three product settings learns one Bloch axis on **all four output qubits** and keeps
  all 16 joint outcomes. Although each setting is a locally rotated product measurement, the
  joint bitstring distribution is correlation-sensitive. It is therefore strictly richer
  than measuring a single Z-aligned readout expectation, without claiming an entangled POVM
- full-state QFI is the physically matched measurement-independent diagonal upper bound for
  this all-output domain. Every fixed/learned basis and accessible mixture passes the checked
  numerical hierarchy `F_C <= F_Q`. The density cache costs
  `32 * (1 + 2 * 24) = 1,568` exact circuit configurations per seed; analytic-gradient axis
  optimization reuses it and adds no quantum circuits

Held-out phase-first result (mean +/- sample SD, seeds 42/43/44):

| Method | Phase retention | Mean final accuracy |
|---|---:|---:|
| Joint ZZZZ | 0.903 +/- 0.163 | 0.759 +/- 0.052 |
| Uniform joint Z/X/Y | 0.915 +/- 0.147 | 0.768 +/- 0.050 |
| LearnBasis, information + allocation | **0.940 +/- 0.104** | **0.777 +/- 0.030** |
| LearnBasis, task relevance + uniform | 0.927 +/- 0.127 | 0.772 +/- 0.045 |
| LearnBasis, task relevance + allocation | 0.933 +/- 0.115 | 0.769 +/- 0.043 |
| Full-state QEWC | 0.927 +/- 0.127 | 0.773 +/- 0.044 |

The paired information-learned-minus-QEWC difference is `+0.0133 +/- 0.0231` phase
retention and `+0.0039 +/- 0.0142` mean final accuracy. The task-relevant learned+allocation
difference is only `+0.0067 +/- 0.0115` retention and `-0.0044 +/- 0.0010` average accuracy.
This is a promising measurement diagnostic, not a
superiority claim: seeds 43/44 have a ceiling (Joint ZZZZ retention >=0.995), while seed 42
contains the identifiable forgetting event. On seed 42, information LearnBasis+Alloc raises
phase retention from `0.715` (Joint ZZZZ), `0.745` (Uniform XYZ), and `0.780` (QEWC) to
`0.820`; task-relevant LearnBasis+Alloc also reaches `0.800`.

The learned ensembles are genuinely non-Z. Allocation-weighted axis power is
`X^2/Y^2/Z^2 = 0.252/0.340/0.408` for the information objective and
`0.022/0.353/0.626` for task relevance plus allocation. The latter has lower cosine to full
QFI (`0.9662 +/- 0.0071`) than the information-only estimator (`0.9788 +/- 0.0026`) and also
slightly lower mean retention. The phase extension therefore does not support a general claim
that the chosen EWC-DR task weighting improves basis learning; it supports the narrower result
that non-Z, correlation-sensitive accessible measurements can recover part of QFI's memory.

Additional phase artifacts:
- `results/e013_phase_learnable_seed{42,43,44}.json`
- `results/e013_phase_learnable_summary.json`
- `results/e013_phase_figure_provenance.json`
- `figures/e013_phase_learnable.png`
- `results/e013_phase_training_figure_provenance.json`
- `figures/e013_phase_training.png` — per-task training accuracy across the full sequential
  trajectory, with three-seed mean/sample-SD bands and active-task shading

Claim boundaries: exact product projective measurements only—two readout qubits for the image
experiment and all four output qubits for the phase extension. Neither is an arbitrary or
entangled POVM, a finite-shot/hardware basis-optimization result, evidence of QFI attainability,
a statistical-significance claim, or quantum advantage. Continuous axes are calibrated and
diagnosed on the same 32 train anchors; no held-out test leaks into the optimizer, but this
first exact study does not measure anchor-level calibration overfitting.
