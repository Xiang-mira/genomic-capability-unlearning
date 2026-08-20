# Next-Steps Methods — Unlearning

**Status of prior work:** No-Go on RMU / GD / null-space projection. This document defines what to try next, derived from the final probing + LoRA-FT confound results.

---

## 1. The finding that constrains the entire method space

Final excess over the k-mer(3–6) baseline, measured on **random** and **composition-cluster-disjoint** splits, for the frozen linear probe vs LoRA fine-tuning:

### AUROC excess (random → cluster-disjoint)

| Task | frozen probe | LoRA-FT | Verdict |
|------|--------------|---------|---------|
| Host_Tropism | −0.038 → −0.054 | **+0.056 → +0.051** | **GENUINE** |
| Pathogenicity | −0.090 → −0.074 | **+0.074 → +0.047** | **GENUINE** |
| Transmissibility | −0.067 → −0.050 | +0.024 → +0.017 | CONFOUNDED |

### MCC excess (random → cluster-disjoint)

| Task | frozen probe | LoRA-FT | Verdict |
|------|--------------|---------|---------|
| Host_Tropism | −0.075 → −0.125 | **+0.135 → +0.133** | **GENUINE** |
| Pathogenicity | −0.156 → −0.101 | **+0.233 → +0.136** | **GENUINE** |
| Transmissibility | −0.147 → −0.084 | +0.066 → −0.007 | CONFOUNDED |

**Reading:**
- **The frozen representation is below k-mer on every task, on both splits.** The capability is not linearly present in activations — it lives in the *weights*, only expressed after fine-tuning (Category B).
- **LoRA-FT keeps a real edge over k-mer on the cluster-disjoint split for Host_Tropism and Pathogenicity** (+0.05 AUROC, +0.13 MCC). This survives the composition-overlap control → **genuine, generalizable weight-space capability.**
- **Transmissibility collapses** (AUROC +0.017, MCC −0.007 on disjoint) → composition-confounded. **Drop it as a forget target;** keep it only as a negative control.

### The single constraint this generates

> The capability is in the weights, and next-token cross-entropy is near-identical across forget and retain viral sequences. So **any method that acts on the frozen representation, or on next-token loss, acts on something that does not contain the capability.** A viable method must first *elicit* the capability (express it through a task readout or fine-tuning) before it can damage it.

Everything below follows from that.

---

## 2. What this rules out — and why

| Method | Why it fails here |
|--------|-------------------|
| **RMU** | Misdirects L5–L9 *activations*. Frozen activations are below k-mer → nothing capability-specific to misdirect. Only disrupts composition read-out, which a fresh probe recovers. |
| **Circuit breaking** | Same locus (activations); same failure. |
| **Next-token-CE gradient ascent / GD** | Forget and retain viral sequences have near-identical next-token statistics → the gradient means "be worse at nucleotides in general" → uniform backbone damage, not targeted erasure. This is exactly what we observed. |
| **Vanilla LAT** (free δ in residual stream) | If the capability is not linearly in activations, there may be **no bounded activation perturbation that elicits it** → the inner adversary finds nothing → method is inert. |
| **Null-space / orthogonal-to-probe projection** | Activation-space read-out disruption. On confounded signal it's provably futile (composition is re-injected by token embeddings at every layer); on genuine capability it hasn't been shown to work and, being activation-space, inherits the elicitation problem. |

**Corollary:** the confounded tasks (Transmissibility, and the earlier Coronaviridae/BVBRC_CoV work) are dead for *all* localized methods — you cannot unlearn an input-level composition property with a localized edit. Only the two GENUINE tasks (HT, Path) are valid targets.

---

## 3. Direction 1 — TAR (Tamper-Resistant Training)

TAR is the natural fit: its outer loop optimizes **post-attack** performance with a **freshly-trained** readout, which is exactly the fresh-probe gate that killed all prior methods. The problem is the phase-1 initializer, because every standard initializer acts on the wrong locus.

### Phase 1 — initializer options

**Option A — no phase 1 (recommended first).**
Run the meta-loop directly from base weights, with the "no-attack" case **K=0 included in the adversary distribution** so that "pre-attack capability is low" falls out of the same objective. Cleanest: no mismatched initializer, no inherited retain damage. Costs more outer steps.

**Option B — classification-head gradient ascent (recommended second; the conceptually important one).**
This repairs the exact flaw that killed GD. Attach a classification head, fine-tune briefly so the capability is *expressed*, then gradient-ascend on **classification CE over the host-tropism / pathogenicity labels** with a retain constraint. Now the gradient points at the capability, not at generic next-token fluency. Same method name as the GD you tried, completely different objective (classification-CE vs next-token-CE).
- **Catch:** it optimizes against one specific head → on its own it reproduces the fixed/fresh-probe failure (attacker trains their own head and recovers). That is acceptable *as a TAR initializer*, because the outer loop measures post-attack performance with a fresh head. **Do not report it as a standalone method.**

**Option C — LoRA-subspace de-priming.**
Fine-tune an attack LoRA to solve the forget task, extract the learned update `B·A`, then move base weights *against* that subspace. A one-shot first-order approximation of what the meta-loop does iteratively.

**Option D — anti-fine-tune displacement.**
`θ_init = θ₀ − η(θ_ft − θ₀)`. Crude, one step, but a non-trivial starting point for cheap.

### Phase 2 — meta-loop configuration

- **Attack distribution (this drives generalization more than any other knob):** LoRA-FT with LR ∈ {1e-5, 5e-5, 1e-4}, rank ∈ {8, 16, 32}, varied target layers, occasional full FT.
- **Tamper-resistance loss:** `(AUROC(θ′) − AUROC_kmer)²` on the **cluster-disjoint** split — *not* "maximize post-attack loss" (unbounded, degenerate). Optimum sits at **k-mer parity**, and it is identical to the evaluation metric.
- **Cost control:** restrict meta-trained parameters to **layers 5–9**; use a first-order approximation (Reptile / FOMAML) rather than second-order meta-gradients.

---

## 4. Direction 2 — LAT (Latent Adversarial Training)

- **Vanilla LAT (skip):** free δ in the residual stream. Given the frozen-probe result, likely no bounded activation perturbation elicits the capability → inner loop inert.
- **LoRA-shaped LAT (worth trying):** constrain the inner adversary to `δ = B·A·x` (a few steps of a small adapter) rather than a free ε-ball vector. The outer loop trains base weights to hold **k-mer parity under the worst-case low-rank perturbation** — the same functional form as the real attack, and far cheaper than TAR (min-max with a few inner steps, no meta-gradient through an optimizer trajectory).
  - **Release note:** merge any LoRA-shaped defense into base weights before release, or an attacker just discards the adapter.
  - **Provenance:** LoRA-shaped LAT is a synthesis of the LAT and LoRA-attack ideas, **not something confirmed in the literature** — check whether it exists under another name before claiming novelty.
- **Per-layer ε scaling (mandatory for any LAT variant):** scale ε per layer as a fraction of activation norm. Evo's L2 norms jump from ~257 at layer 10 to ~1.8M at layer 11 — a fixed ε is meaningless across layers.

---

## 5. Direction 3 — LoRA-subspace targeting (standalone)

Same mechanism as Phase-1 Option C, but reported on its own arm: fine-tune the attack LoRA, extract `B·A`, then damage / pre-consume that subspace in base weights. It is the **weight-space analog of your Phase-1 localization** — cheap, mechanistically direct, and reportable by itself. Worth one run.

---

## 6. Shared evaluation protocol & success criteria

All arms are judged the same way — this is non-negotiable and is what makes the results interpretable:

1. **Targets:** Host_Tropism and Pathogenicity only. Transmissibility = confounded negative control.
2. **Metric:** excess over k-mer(3–6), **both AUROC and MCC**, on the **cluster-disjoint** split. k-mer baseline must be strong (C-grid to 100, full features) — an underfit baseline inflates excess.
3. **Forget success = k-mer parity under attack:** after the strongest attack in the distribution, post-attack excess ≤ 0 (model no better than k-mer). Report the fresh-head/fresh-probe number, never the fixed-head one.
4. **Retain gate:** GUE ΔAUROC ≥ −0.05 across all 7 tasks, retain-PPL within 10%. Retain set must span everything evaluated (viral non-target **+ GUE**), not just viral genomes.
5. **Controls:** beat a random-layer control by a margin; label any result where a fresh probe still separates > 0.60 as `readout_disruption`, never `erasure`.
6. **Split validity check:** before trusting any disjoint split, confirm the k-mer/NN baseline actually *drops* and the test set stays label-balanced (Host_Tropism's first cluster split failed this and had to be regenerated).

---

## 7. Recommended order

1. **TAR Option A** (no phase 1, K=0 in adversary dist) — cleanest, no mismatched initializer.
2. **TAR Option B** (classification-CE gradient ascent initializer) — cheap, and the classification-CE vs next-token-CE insight is worth having regardless.
3. **LoRA-shaped LAT** — cheapest min-max, matched to the real threat model.
4. **LoRA-subspace targeting** (standalone arm) — one run, mechanistically direct.
5. Options C/D as ablations feeding into 1–2.

---

## 8. Open risks

- **Relearning is the real adversary, not the fixed probe.** Your earlier GD dropped the fixed probe to 0.57 while a fresh probe recovered to 0.9959. Every method here must be scored post-attack with a fresh head; "raise the cost of relearning" is the objective, not "zero a probe."
- **Two tasks, small excess.** The genuine excess is ~+0.05 AUROC / ~+0.13 MCC. It is real (survives the disjoint control) but small — retain damage tolerances must be tight or the "success" is just backbone degradation.
- **Transmissibility must not be reported as a target.** Its LoRA-FT excess collapses on the disjoint split; any "unlearning" there is uninterpretable.
- **LoRA-shaped LAT novelty** unverified — literature check before any claim.
