# MPI — 2-minute pitch (English)

**Measurement-based Parameter Isolation: where should a quantum model's memory live?**

---

**[0:00–0:20] Hook + problem**
> "Quantum models *forget* — train them on a new task and they lose the old one. Every method
> today — EWC, QEWC — protects the **circuit parameters θ**. We asked one question: *what if
> we're protecting the wrong thing?*"
> 🖼️ **`e014_circuit.png`**

**[0:20–0:50] The idea**
> "Our method, MPI, flips it: **keep the quantum circuit as a shared representation, and give
> every task its own way to measure it** — a learnable observable, which is mathematically just a
> **classical linear head on the measured probabilities**. Old readouts are frozen, so on the
> measurement side an old task is *structurally impossible to overwrite*."
> 🖼️ **`e014_circuit.png`** — point: shared circuit → three per-task heads

**[0:50–1:20] It works (simulator)**
> "On three tasks — MNIST, Fashion, and a quantum-phase task — **MPI keeps 0.96 with near-zero
> forgetting**. QEWC gets 0.82, because clinging to θ *kills the ability to learn new tasks*."
> 🖼️ **`e014_trajectory.png`** — point: our lines stay flat after learning, the baselines decay

**[1:20–1:50] The killer: real hardware**
> "We ran it on IBM's real quantum computer, ibm_marrakesh. **MPI loses no accuracy — 0.93.**
> QEWC **collapses to 0.58 — the old task below random chance** — because its quantum-Fisher
> geometry breaks under real noise. On hardware our lead **widens to +0.35**. And adding a new task
> on the device needs only **measurement plus a classical fit — no quantum retraining at all**."
> 🖼️ **`e014_hardware_compare.png`** — the strongest slide, hold on it

**[1:50–2:10] Honest close**
> "We're honest: on these easy tasks a classical model also wins — so this is **not a quantum-
> advantage claim, it's a mechanism result** about *where* a quantum model's memory should live.
> But that mechanism is real, it's theory-grounded, and it's **the only one that survives real
> hardware**."

---

## Show these 3 figures, in order
1. **`e014_circuit.png`** — the method (shared circuit + per-task measurement heads)
2. **`e014_trajectory.png`** — it works (retention vs forgetting)
3. **`e014_hardware_compare.png`** — the killer (real QPU: MPI 0.93 vs QEWC 0.58)

**Backup (only if asked):** `e014_fair_compare.png` (honest classical control),
`e014_noise_compare.png` (noise robustness).

## One-liner (if you only get one sentence)
> **"Don't rewrite the quantum circuit for every new task — give each task its own way to measure
> it. On real quantum hardware, that beats parameter-protection into the ground."**
