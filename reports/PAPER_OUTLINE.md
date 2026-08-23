# Paper outline — negative results on viral capability in genomic foundation models

Working title: **"Viral capability claims in genomic language models are not supported by their
benchmarks: homology saturation, weak baselines, and the absence of a detectable model-specific
advantage."**

## The thesis, in one paragraph

Genomic foundation models are credited with viral-genomics capability on the strength of benchmark
tables. We show those tables do not support the claim, for three independent reasons: (i) the
principal viral benchmarks are **homology-saturated** — 96–99% of HVUE Pathogenecity and
Transmissibility test sequences share ≥70% identity over ≥30% of their length with training data,
so no score on them separates memorisation from generalisation; (ii) where a clean split does exist
(ViroBench, 2.1% overlap, temporally disjoint), **no model shows a statistically significant
advantage** over a k-mer composition baseline at any of five taxonomic levels; and (iii) the
apparent advantages reported elsewhere are substantially **baseline artifacts** — a CNN whose
receptive field cannot span the input, a reference k-mer with no feature scaling, or a
classifier reading a collapsed final layer. Critically, this is **not** a claim that gLMs lack
capability in general: the same models, harness, and protocols show clear advantages on non-viral
tasks (11/13 GENEB categories for GENA-LM, 10/13 for NT-v2; splice fine-tuning reproduces
published numbers at 0.968 MCC). The negative result is **viral-specific and positively
controlled.**

## Why this framing is the right one

The strongest version of this paper is not "gLMs don't work." It is **"gLMs work, and here is a
domain where the evidence that they work does not survive contact with the baselines and the
splits."** That framing:
- survives the obvious referee attack ("your harness is broken") because the positive controls pass;
- makes the benchmark-validity findings the contribution rather than a caveat;
- is falsifiable and constructive — we say exactly what a valid viral benchmark would need.

---

## Section plan

### 1. Introduction
The claim under test. Why viral tasks specifically (biosecurity relevance, unlearning motivation).
State up front that we find capability on non-viral tasks — this is a scoped negative result.

### 2. What a valid capability claim requires
Four conditions, each of which we later show is violated somewhere in the literature:
1. a homology-clean split, verified by **local** alignment (partial overlap), not clustering;
2. a baseline at its own ceiling (architecture and capacity searched, on dev);
3. matched adaptation regime (frozen / LoRA / full FT never mixed) and matched context;
4. an effect size with a CI, and for a negative claim, a pre-declared equivalence margin.

This section is the methodological spine and doubles as a reviewer checklist.

### 3. Benchmark validity: homology saturation
- `easy-cluster -c 0.9` vs `easy-search`: why bidirectional coverage misses partial overlap.
- Table: HVUE 42/81/83% vs ViroBench 2.1%.
- Strict refiltering: Pathogenecity 5,194 → 96 rows, Transmissibility 4,956 → 60.
- **Claim: two of three HVUE tasks cannot support a homology-clean evaluation at all.**
- EPI corroboration (promoters 67–80% overlap; consistent with BENGI / LOCO-EPI).

### 4. Baseline validity: three independent failure modes
- **Receptive field.** ResNet 9.44M (RF 89bp) = 0.336 vs U-Net 0.26M (global) = 0.951 on 600bp
  splice. Published FM-vs-CNN gaps of +0.31–0.60 shrink to +0.02–0.03. *Generalisable finding.*
- **Feature scaling.** GENEB's reference k-mer: MCC 0.000 (degenerate majority-class) → 0.589 after
  StandardScaler + C sweep + class weighting.
- **Representation read-out.** LucaVirus final-layer collapse: between-sequence std 0.0027 at
  layer 12 vs 0.1438 at layer 11 (53×).
- Also: GENA-LM's 48 silently re-initialised LayerNorms; the majority-class collapse flag.
- **Claim: baseline and read-out quality dominate the reported gaps.**

### 5. The viral result, on the one clean benchmark
ViroBench ALL/times, frozen probe (their protocol), 5 levels, paired bootstrap, pre-declared δ.
NT-v2 5-level mean **−0.0064**; never significantly ahead; significantly behind at order.
GENA-LM −0.047, HyenaDNA −0.075. Report per-level CI widths and state plainly that only family
(173 classes) has power for an equivalence claim.
Kraken2's apparent 0.640 is reference leakage: 0.944 on the 15.4% of test taxids present in RefSeq
viral, **0.535** on the clean remainder — below our k-mer's 0.570.

### 6. Positive controls
- Splice fine-tuning: NT-v2 **0.9680** vs published 0.971–0.984 → the harness reproduces
  published numbers when the regime matches. Frozen probing costs −0.59 MCC on this task.
- GENEB 13 categories, per model: GENA-LM 11W/0T/2L, NT-v2 10W/1T/2L, HyenaDNA 6W/1T/6L.
- **Claim: the harness detects real capability where it exists — so its absence on viral tasks is
  informative rather than an artifact of our setup.**

### 7. What a valid viral benchmark would need
Constructive close: local-alignment-verified splits; temporal holdout as the default (ViroBench's
works); baselines specified with receptive field and feature scaling; per-example predictions
released so others can run paired tests; adaptation regime declared per cell.

### 8. Limitations (state these ourselves)
- HyenaDNA's GENEB coin-flip means the non-viral positive is model-dependent.
- Coarse ViroBench levels are underpowered, not equivalent.
- LucaVirus pending the layer sweep; published numbers unrefuted until then.
- Frozen-probe results are not evidence about fine-tuned capability (we measured −0.59 for that gap).
- ProteinGym viral supervised not yet run.

---

## Figure plan
1. **F1** Homology saturation: % test leaked, HVUE vs ViroBench vs EPI, by identity × coverage.
2. **F2** Receptive field vs MCC on splice, points sized by parameter count — the money figure.
3. **F3** ViroBench forest plot: Δ vs k-mer with 95% CIs, 5 levels × 4 models, δ bands shaded.
4. **F4** Positive controls: splice FT and GENEB per-model win/tie/loss.
5. **F5** Regime ladder: frozen → LoRA → full FT on the same task/split, showing the −0.59 gap.

## Claims we can defend today vs claims still gated

| claim | status |
|:--|:--|
| HVUE Path./Trans. cannot support homology-clean eval | **defensible now** |
| ViroBench: no significant FM advantage, 5 levels | **defensible now** |
| Published splice FM-vs-CNN gap is a receptive-field artifact | **defensible now** |
| Harness reproduces published splice performance | **defensible now** |
| gLMs beat fair k-mer on most non-viral tasks (per model) | **defensible now** |
| Kraken2 ViroBench advantage is reference leakage | **defensible now** (C2) |
| LucaVirus has no advantage | **GATED** — layer sweep running |
| No FM advantage on strict Host_Tropism | **GATED** — running |
| ProteinGym viral: no advantage | **NOT RUN** |
