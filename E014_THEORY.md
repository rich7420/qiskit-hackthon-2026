# E014 — Theory of Observable-Isolated Quantum Continual Learning (OI-QCL)

A self-contained mathematical and physical account of measurement-side continual learning.

**Thesis.** In a variational quantum classifier, task adaptation is conventionally realized by
moving the unitary $U(\theta)$, which in the Heisenberg picture *drags a single effective
observable* $H_{\mathrm{eff}}(\theta)=U^\dagger H_0 U$ through observable space. Sequential
tasks overwrite this one observable — that overwrite *is* catastrophic forgetting. OI-QCL holds
the state preparation fixed (or softly anchored) and gives each task its **own** observable.
Because a task's observable is a diagonal operator, its expectation is a *linear functional of
the measured probability vector*, i.e. a lightweight classical head. Old heads are frozen, so
forgetting on the measurement side is a structural zero; a soft anchor keeps the shared
representation from drifting.

Contents: 1 setup · 2 Heisenberg picture · 3 readout identity · 4 quantum feature map /
kernel view · 5 commuting family · 6 training & gradients · 7 CL formulation · 8 zero-forgetting
· 9 forgetting decomposition · 10 stability–plasticity · 11 feature sufficiency · 12 Fisher
information · 13 shot noise & sample complexity · 14 memory & hardware · 15 relation to prior
QCL · 16 metrics · 17 summary.

---

## 1. Setup and notation

Let $\mathcal H=(\mathbb C^2)^{\otimes n}$, $N=2^n$, computational basis $\{|k\rangle\}_{k=0}^{N-1}$.
For input $x\in\mathbb R^d$ the classifier prepares

$$
|\psi_\theta(x)\rangle=U(\theta)\,V(x)\,|0\rangle^{\otimes n},
\qquad
\rho_\theta(x)=|\psi_\theta(x)\rangle\langle\psi_\theta(x)| .
$$

**Encoding.** $V(x)$ is a fixed amplitude embedding: for a normalized feature vector
$\tilde x\in\mathbb R^{N}$, $V(x)|0\rangle=\sum_{k}\tilde x_k|k\rangle$. It carries no trainable
parameters — all learning is in $U(\theta)$ and the readout.

**Ansatz.** $U(\theta)=\prod_{\ell=1}^{L} \Big[\big(\textstyle\prod_{q=0}^{n-2}\mathrm{CNOT}_{q,q+1}\big)
\big(\textstyle\prod_{q=0}^{n-1} R_Z(\theta_{\ell,q,1})R_Y(\theta_{\ell,q,0})\big)\Big]$,
with $\theta\in\mathbb R^{L\times n\times2}$ (here $n=4$). Each $R_P(\phi)=e^{-i\phi P/2}$.

**Readout.** A Hermitian observable $H$ gives the scalar score
$f(x)=\mathrm{Tr}[H\rho_\theta(x)]=\langle\psi_\theta(x)|H|\psi_\theta(x)\rangle$.

Naming: $\rho_\theta(x)$ = **shared representation**; $H$ = **readout / measurement**.

---

## 2. Where forgetting lives — the Heisenberg picture

With a fixed reference observable $H_0$,

$$
f(x)=\langle\psi_x|U^\dagger(\theta)H_0U(\theta)|\psi_x\rangle
=\langle\psi_x|H_{\mathrm{eff}}(\theta)|\psi_x\rangle,\qquad
H_{\mathrm{eff}}(\theta):=U^\dagger(\theta)H_0U(\theta),
$$

where $|\psi_x\rangle=V(x)|0\rangle$. Training the circuit **moves the effective observable**
through the space of Hermitian operators while $H_0$ and the encoding are fixed. Sequentially,

$$
\theta_1\to\theta_2\to\cdots\to\theta_T
\;\Longrightarrow\;
H_{\mathrm{eff}}^{(1)}\to H_{\mathrm{eff}}^{(2)}\to\cdots\to H_{\mathrm{eff}}^{(T)} ,
$$

so all tasks share **one, continually overwritten** effective observable; forgetting is the
misalignment of $H_{\mathrm{eff}}^{(T)}$ with what an earlier task required.

**OI-QCL** fixes state preparation and attaches a task-specific observable:

$$
\boxed{\;\rho_\theta(x)\ \text{shared},\qquad H^{(1)},\dots,H^{(T)}\ \text{task-specific},\qquad
f_t(x)=\mathrm{Tr}[H^{(t)}\rho_\theta(x)]\;}.
$$

---

## 3. The readout identity: diagonal observable $=$ linear head over probabilities

Take each task observable diagonal in a fixed basis $U_b$ (a DANO-style form):

$$
H^{(t)}=U_b^\dagger\mathrm{diag}(\lambda^{(t)})U_b,\qquad \lambda^{(t)}\in\mathbb R^N .
$$

Define the **measured probability vector**

$$
p_k(x;\theta)=\big|\langle k|U_b|\psi_\theta(x)\rangle\big|^2
=\langle k|U_b\,\rho_\theta(x)\,U_b^\dagger|k\rangle,\qquad
p_\theta(x)=(p_0,\dots,p_{N-1})\in\Delta^{N-1},
$$

a point on the probability simplex ($p_k\ge0$, $\sum_kp_k=1$).

**Proposition 1 (readout identity).**
$\displaystyle \big\langle H^{(t)}\big\rangle=\sum_k\lambda_k^{(t)}p_k(x;\theta)=\lambda^{(t)}\!\cdot p_\theta(x).$

*Proof.* By cyclicity, $\mathrm{Tr}[U_b^\dagger\mathrm{diag}(\lambda)U_b\rho]
=\mathrm{Tr}[\mathrm{diag}(\lambda)U_b\rho U_b^\dagger]
=\sum_k\lambda_k\langle k|U_b\rho U_b^\dagger|k\rangle=\sum_k\lambda_kp_k.$ $\blacksquare$

For $C$ classes, stack $C$ diagonal observables into $W^{(t)}\in\mathbb R^{C\times N}$ (row $c$ is
$\lambda^{(t,c)}$) to obtain class scores

$$
\boxed{\,z^{(t)}(x)=W^{(t)}p_\theta(x)\,},\qquad
\hat y_t(x)=\arg\max_c\,[\,z^{(t)}(x)\,]_c .
$$

Folding $U_b$ into the ansatz ($U\mapsto U_bU$) makes $U_b=\mathbb I$ and $p_\theta(x)=\texttt{probs()}$
the computational-basis distribution. **A task-specific observable is exactly a linear head over
the quantum probability vector.**

*Subsystem variant.* Reading only $m<n$ wires uses the reduced state
$\rho^{S}_\theta(x)=\mathrm{Tr}_{\bar S}\rho_\theta(x)$ and gives the marginal
$p^{S}_k=\mathrm{Tr}[(|k\rangle\langle k|_S\otimes\mathbb I_{\bar S})\rho_\theta]$; the same
identity holds with $N\to2^m$. The 2-wire marginal is our "few-observable" baseline.

---

## 4. Quantum feature map / kernel view — "is it just a classical head?"

Define the **quantum feature map** $\Phi_\theta:x\mapsto p_\theta(x)\in\Delta^{N-1}\subset\mathbb R^N$.
Then $z^{(t)}(x)=W^{(t)}\Phi_\theta(x)$ is a **linear classifier in feature space**. Two remarks
answer the standard objection:

1. **Yes, the head is classical and linear** — deliberately. Optimizing $W^{(t)}$ is a convex
   logistic regression, no quantum gradient. That is the source of the "trains in seconds" and
   "structurally no forgetting" properties.
2. **No, the method is not classical** — the *features* are genuinely quantum:
   $\Phi_\theta(x)=\big(|\langle k|U(\theta)V(x)|0\rangle|^2\big)_k$ is the output distribution of
   an $n$-qubit circuit, generally $\#\mathrm P$-hard to sample classically for expressive $U$.
   The associated kernel is
   $\;\kappa_\theta(x,x')=\langle\Phi_\theta(x),\Phi_\theta(x')\rangle=\sum_k p_k(x)p_k(x')$,
   a similarity of measurement distributions. The scientific question is thus sharp and testable:
   *does the full distribution $p_\theta$ (a $2^n$-dim quantum feature) carry more reusable,
   forgetting-resistant task information than the conventional few-observable readout?* — measured
   by the head-only gain over the marginal in §11.

---

## 5. Commuting family — an honest limitation

**Proposition 2.** With one shared basis $U_b$, $\big[H^{(t)},H^{(s)}\big]=0$ for all $t,s$.

*Proof.* $H^{(t)}=U_b^\dagger D_tU_b$, $D_t$ diagonal; $H^{(t)}H^{(s)}=U_b^\dagger D_tD_sU_b
=U_b^\dagger D_sD_tU_b=H^{(s)}H^{(t)}$. $\blacksquare$

The observables span the **abelian algebra** generated by projectors
$P_k=U_b^\dagger|k\rangle\langle k|U_b$. This is *DANO-inspired* (a linear head over basis
probabilities), **not** full adaptive-non-local-observable expressivity, which would need a
task-dependent basis $U_b^{(t)}$ and yield generally non-commuting $H^{(t)}$. Consequences:
a single measurement basis suffices for all tasks (hardware win, §14), but the readouts cannot
resolve information that a task-dependent basis change would expose.

---

## 6. Training and gradients

**Losses.** With softmax $\sigma$ and one-hot label $y$, the per-example loss is cross-entropy
$\ell(x,y)=-\sum_c y_c\log[\sigma(z^{(t)}(x))]_c$. (Task 1's backbone also uses a two-$Z$ softmax
readout; both are special cases of a diagonal readout.)

**Quantum gradient of the backbone.** Each score is a trigonometric polynomial in $\theta$; the
parameter-shift rule gives exact derivatives,

$$
\frac{\partial}{\partial\theta_r}\mathrm{Tr}[H\rho_\theta(x)]
=\tfrac12\Big(\mathrm{Tr}[H\rho_{\theta+\frac\pi2 e_r}(x)]-\mathrm{Tr}[H\rho_{\theta-\frac\pi2 e_r}(x)]\Big).
$$

**No quantum gradient for the head.** Since $z^{(t)}=W^{(t)}p_\theta$ is *linear in $W^{(t)}$*,
$\partial \ell/\partial W^{(t)}=(\sigma(z)-y)\,p_\theta(x)^\top$ needs only the (already measured)
probabilities. Training a head is convex logistic regression on fixed quantum features. In the
ACC/BWT table we fit each frozen head to convergence (`LogisticRegression`); in the trajectory
figure we train it by gradient epoch-by-epoch for a like-for-like learning curve against the
baselines.

**Joint Task-1 step.** Task 1 solves $\min_{\theta,W_1}L_1(W_1p_\theta)$, updating $\theta$ by
parameter-shift/backprop and $W_1$ by the linear gradient above.

---

## 7. Continual-learning formulation (Task-IL)

Tasks $t=1,\dots,T$ with losses $L_t$. **Task-Incremental Learning**: the id $t$ is revealed at
test, so head $H^{(t)}$ is selected (van de Ven et al.'s taxonomy; same assumption as classical
multi-head CL). This is distinct from *class-incremental* (must infer $t$) and *task-agnostic* CL.
The per-task program:

$$
\boxed{\;
\min_{\theta,\,W_t}\;L_t(W_tp_\theta)\;+\;\alpha\,\|\theta-\theta_{t-1}\|^2
\quad\text{s.t.}\quad W_1,\dots,W_{t-1}\ \text{frozen}\;}
$$

| Variant | $\theta$ after Task 1 | $\alpha$ | role |
|---|---|---|---|
| **A frozen** | held at $\theta_1^\*$ | — | structural zero-forgetting reference |
| **B free** | trained freely | $0$ | isolates representation drift |
| **C anchor** | soft-L2 anchored | $>0$ | **main**: retention + plasticity |

---

## 8. Structural zero-forgetting

Let $R_{ij}$ = test accuracy on task $j$ after training through task $i$ ($i\ge j$), and
$A(W,\theta)$ = accuracy of head $W$ on backbone $\theta$.

**Proposition 3 (Variant A).** If $\theta\equiv\theta_1^\*$ and each $W_j$ is frozen once task $j$
is learned, then $R_{ij}=A(W_j,\theta_1^\*)=R_{jj}$ for all $i\ge j$.

*Proof.* The task-$j$ prediction $\arg\max_c[W_jp_{\theta_1^\*}(x)]_c$ depends only on
$(\theta_1^\*,W_j)$, both constant for $i\ge j$; hence $R_{ij}$ is independent of $i$. $\blacksquare$

So an earlier task's accuracy **cannot** change: no later task touches $\theta$ or $W_j$. This is
an identity (exact preservation), stronger than the approximate preservation of EWC/QEWC.

---

## 9. Forgetting decomposition — representation drift vs measurement overwrite

Total forgetting $F_j=R_{jj}-R_{Tj}$; backward transfer
$\mathrm{BWT}=\frac1{T-1}\sum_{j<T}(R_{Tj}-R_{jj})=-\overline F$.

For **isolated heads** (B/C), $W_j$ is frozen but $\theta$ may drift $\theta_j\to\theta_T$, so
$R_{ij}=A(W_j,\theta_i)$ and

$$
\boxed{\,F_j=\underbrace{A(W_j,\theta_j)-A(W_j,\theta_T)}_{\text{representation drift}}\,}.
$$

The **measurement-overwrite** term is zero because $W_j$ never updates. For **shared-readout**
methods (naive/EWC/QEWC) *both* $\theta$ and the single head $W$ change, so

$$
F_j^{\text{shared}}=\underbrace{[A(W_j,\theta_j)-A(W_j,\theta_T)]}_{\text{representation drift}}
+\underbrace{[A(W_j,\theta_T)-A(W_T,\theta_T)]}_{\text{measurement overwrite}} .
$$

OI-QCL removes the second term entirely and controls the first with the anchor (§10). Empirically
(3 seeds): free-θ (B) has large $|\mathrm{BWT}|\approx0.26$ (pure representation drift) while
anchor (C) has $|\mathrm{BWT}|\approx0.004$ and frozen (A) $=0$.

---

## 10. Stability–plasticity as constrained optimization

Write learning task $t$ as maximize new-task fit subject to a drift budget $\delta$:

$$
\max_{\theta,W_t}\ -L_t(W_tp_\theta)\quad\text{s.t.}\quad \|\theta-\theta_{t-1}\|\le\delta .
$$

The Lagrangian is the anchor objective of §7 with multiplier $\alpha(\delta)$. If accuracy is
$L$-Lipschitz in $\theta$ along the path, drift bounds forgetting:

$$
F_j\le L\,\|\theta_T-\theta_j\|\le L\sum_{s>j}\|\theta_s-\theta_{s-1}\|,\qquad
\|\theta_s-\theta_{s-1}\|=O(1/\alpha).
$$

- $\delta\to0$ ($\alpha\to\infty$): Variant A, $F_j=0$, minimal plasticity.
- $\delta\to\infty$ ($\alpha=0$): Variant B, maximal plasticity, largest drift.
- intermediate: Variant C — the useful regime. Our earlier BB-QEWC negative result (hard
  trust-regions kill plasticity) is exactly the $\delta\to0$ failure; OI-QCL avoids it by moving
  task memory *off* $\theta$ so a small $\delta$ no longer starves new tasks (their capacity lives
  in $W_t$, unconstrained).

---

## 11. Feature sufficiency (the GO/NO-GO probe)

Freeze $\theta_1^\*$ and, for each task $j$, fit a linear head on the frozen features. The probe
accuracy is

$$
A_j^{\mathrm{probe}}=\max_{W}\ \Pr_{(x,y)\sim\mathcal D_j}\big[\arg\max_c(Wp_{\theta_1^\*}(x))_c=y\big].
$$

$A_j^{\mathrm{probe}}$ measures **linear separability of task $j$ in the frozen quantum feature
space** — an upper bound on what any Variant-A head can achieve. High $A_{j>1}^{\mathrm{probe}}$
$\Rightarrow$ a frozen backbone transfers (GO); low $\Rightarrow$ later tasks need representation
adaptation (Variant C). The **head-only gain**
$A_j^{\mathrm{probe}}(\text{full }2^n)-A_j^{\mathrm{probe}}(\text{2-wire marginal})$ quantifies the
extra reusable information in the full distribution over the conventional few-observable readout
(§4's testable question). Measured: later-task mean $A^{\mathrm{probe}}=0.958$ → GO.

---

## 12. Fisher information (diagnostic, not a claim)

For measurement $\{P_k\}$ and outcome law $p_\theta$, the **classical Fisher information** is
$F_C(\theta)_{ab}=\sum_k \frac{1}{p_k}\partial_a p_k\,\partial_b p_k$, and the **empirical CFI**
used in e005 is the mean-square score, $\widehat{F_C}=\mathbb E[(\nabla_\theta\log p_\theta)^2]$.
The Braunstein–Caves bound gives

$$
F_C(\theta)\ \preceq\ F_Q(\theta)=4\big(\langle\partial_a\psi|\partial_b\psi\rangle
-\langle\partial_a\psi|\psi\rangle\langle\psi|\partial_b\psi\rangle\big),
$$

with equality only for a state-dependent optimal measurement. **QEWC regularizes with $F_Q$** (a
state property, measurement-independent) to protect $\theta$; **OI-QCL instead chooses the
readout functional** that realizes $F_C$ and moves memory there. The two act on opposite sides of
$F_C\preceq F_Q$. We use this only for intuition; the reported diagnostics are linear-probe
accuracy and head-only gain, *not* a CFI/QFI ratio (which is ill-posed once $\theta$ is frozen and
the "outcome distribution" is the post-processed $z=Wp$).

---

## 13. Shot noise and sample complexity

On hardware $p_k$ is estimated from $S$ computational-basis shots as $\hat p_k=n_k/S$
(multinomial). For a bounded diagonal observable ($|\lambda_k^{(t)}|\le\Lambda$),

$$
\widehat{\langle H^{(t)}\rangle}=\sum_k\lambda_k^{(t)}\hat p_k,\qquad
\mathrm{Var}\big[\widehat{\langle H^{(t)}\rangle}\big]
=\frac1S\Big(\sum_k(\lambda_k^{(t)})^2p_k-\langle H^{(t)}\rangle^2\Big)\le\frac{\Lambda^2}{S}.
$$

So $S=O(\Lambda^2/\varepsilon^2)$ shots give $\varepsilon$-accurate readouts. Crucially the **same**
shot batch estimates $\hat p$ once and yields *every* task's readout by reweighting — the shot
cost does not grow with the number of tasks (§14).

---

## 14. Memory and hardware

**Memory.** Each task stores $W^{(t)}\in\mathbb R^{C\times N}$, so
$M(T)=O(T\,C\,2^n)$. Diagonal restriction reduces a generic Hermitian from $4^n$ to $2^n$
parameters but remains exponential in $n$ — fine at $n=4$, motivating structured/local/sparse
diagonal observables at scale.

**One basis, many observables.** All $H^{(t)}$ commute and are diagonal in the same basis, so a
single set of bitstring samples estimates $\hat p$ and all $\widehat{\langle H^{(t)}\rangle}$ by
reweighting — no per-task measurement circuit, unlike non-commuting readouts requiring separate
bases. A concrete hardware advantage of the commuting-family restriction of §5.

---

## 15. Relation to prior quantum CL (mechanism comparison)

| Method | intervenes on | mechanism | forgetting guarantee |
|---|---|---|---|
| Naive sequential | $\theta$ | none | none |
| EWC / **QEWC** | $\theta$ | $\min L_t+\sum_j\alpha_j (\theta-\theta_j^\*)^\top\!\mathrm{diag}(F_j)(\theta-\theta_j^\*)$; $F=F_C$ (EWC) or $F_Q$ (QEWC) | approximate (soft) |
| Quantum GEM | $\theta$ | project $\nabla L_t$ to not increase old-task loss | approximate (projection) |
| e008/e010/e013 | $\theta$ | QEWC-style with a *measurement-derived* Fisher (smarter importance) | approximate (soft) |
| **OI-QCL (ours)** | **measurement** | isolate $W^{(t)}$; share/soft-anchor $\theta$ | **structural on measurement side (Prop 3)**; residual = representation drift only |

All prior methods keep one readout and protect the *parameters*; OI-QCL keeps the representation
and isolates the *readout*. This is parameter isolation moved from circuit space to measurement
space, and it composes with a soft $\theta$-anchor (which our QFI-isotropy result shows behaves
like L2, so we use L2 for the backbone and cite QEWC as the equivalent baseline).

---

## 16. Metrics

For the lower-triangular accuracy matrix $R\in[0,1]^{T\times T}$,

$$
\mathrm{ACC}=\frac1T\sum_{j=1}^{T}R_{Tj},\qquad
\mathrm{BWT}=\frac1{T-1}\sum_{j=1}^{T-1}\big(R_{Tj}-R_{jj}\big).
$$

ACC = final average accuracy; $\mathrm{BWT}\le0$ = forgetting ($=0$ for Variant A by Prop 3).

---

## 17. One-line summary

$$
\textbf{Don't rewrite the quantum representation for every new task — give each task its own way to measure it:}\quad
z^{(t)}(x)=W^{(t)}p_\theta(x).
$$

Forgetting that lived in the drift of $H_{\mathrm{eff}}(\theta)$ is factorized into a *shared*
representation $\rho_\theta$ and *isolated* measurement functionals $\{W^{(t)}\}$: the measurement
side cannot forget (Prop 3), and a soft anchor bounds the drift of the shared side (§10).
