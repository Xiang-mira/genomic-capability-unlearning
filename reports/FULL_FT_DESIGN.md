# Full fine-tuning experiment design — task selection, disjoint sets, and prior work

**Premise: GENEB probing results are DONE and will not be re-run.** They are the input to this
design, not an output of it. Full FT costs ~50× a frozen probe, so the whole point of this document
is to spend it on the cells where probing left the answer ambiguous.

---

## 1. Why full FT is the regime that decides the paper

Probing and fine-tuning give different answers, and the literature already says so. [DART-Eval
(NeurIPS 2024 D&B)](https://arxiv.org/abs/2412.05430) reports that *ab initio* supervised models
**"performed comparably to the best fine-tuned DNALMs and substantially outperformed all probed
DNALMs."* Our own splice measurement is the same shape and larger:

| regime | splice_all MCC | vs our dev-selected CNN 0.9528 |
|:--|--:|--:|
| frozen probe, best of 3 models | 0.3636 | **−0.589** |
| full FT, NT-v2 (3 seeds) | 0.9674 ± 0.0025 | **+0.0146** |
| full FT, HyenaDNA | 0.8498 | −0.103 |
| full FT, GENA-LM | 0.8294 | −0.123 |

So a probe-only null is **not** evidence about capability — it is evidence about the probe. Any claim
of the form "model X lacks capability Y" has to be made in the FT regime or scoped explicitly to
probing. This is the single strongest reason to run FT at all, and it is why our GENEB probing table
cannot carry a capability conclusion on its own.

## 2. Task selection — criteria declared before seeing any FT result

Full FT is spent only where probing was ambiguous or where the probe is a priori the wrong instrument.

**Include a task if it meets S1 and at least one of S2–S4:**

- **S1 (feasibility).** `n_train ≥ 5,000`, so FT is not data-starved.
- **S2 (ambiguous under probing).** Best probe within ±0.05 of `max(k-mer, CNN)`. These are the cells
  where the verdict could flip either way.
- **S3 (probe-hostile by construction).** Positionally-structured tasks where the read-out, not the
  representation, is plausibly the bottleneck — splice, TFBS, promoters, 4mC. Justification: our
  receptive-field result shows position-dependent tasks punish a pooled linear read-out hardest.
- **S4 (contested in the literature).** A published FM-vs-CNN gap exists that we can test directly.

**Exclude:**
- **k-mer-saturated tasks** (baseline ≥ 0.95, or species/taxonomy where composition solves it). No
  headroom to measure. ViroBench family is the extreme case — alignment nearest-hit gets 0.7383
  macro-F1 and 0.9915 accuracy on aligned genomes, so *nothing* learned is going to be the story.
- **Homology-saturated tasks.** HVUE Pathogenecity (96 of 5,194 clean rows) and Transmissibility
  (60 of 4,956) — no FT result on them can be interpreted.
- **>2 tasks per GENEB category**, to avoid a category dominating the aggregate.

**Budget:** 8–12 tasks × 3 models × 3 seeds, dev-selected LR from {1e-5, 3e-5, 1e-4} with linear
warmup. Each cell must be accompanied by the **CNN ladder** on the same split — this is the gap that
currently invalidates the GENEB comparison (no CNN was ever run there).

### Provisional slate (to be finalised from the probing table by the S-criteria)

| task | source | criteria | why |
|:--|:--|:--|:--|
| splice_all / acceptors / donors | NT | S1,S3,S4 | done; the anchor. Keep as the calibration cell |
| GUE human_tf_0 | GENEB | S1,S2,S3 | probe 0.576 vs fair k-mer 0.537 — inside the ambiguity band |
| GUE mouse_0 | GENEB | S1,S2 | probe 0.378 *below* k-mer 0.437 — probe-limited or capability-limited? |
| NT enhancers | GENEB | S1,S2,S3 | probe 0.396 vs k-mer 0.425 — same question |
| iDHS-EL DNase_I | GENEB | S1,S2,S4 | fair k-mer 0.589 vs probe 0.593; also where GENEB's own k-mer was degenerate |
| deep4mc A.thaliana 4mC | GENEB | S1,S3 | position-dependent methylation; probe spread 0.071–0.331 is suspicious |
| NT H3 (histone) | GENEB | S1,S2 | probes 0.662–0.705 vs k-mer 0.590 — modest, worth confirming under FT |
| ensembl_regulatory | GENEB | S1,S3 | largest probe gain (+0.266); test whether FT holds it up |
| DART-Eval Task 1 | DART | S1,S4 | direct comparison against a published ab initio baseline (ChromBPNet) |
| long-range expression | Borzoi-style | S1,S3,S4 | the missing third positive anchor; blocked on data access |

Deliberately **excluded**: human_or_worm and coding_vs_intergenomic (k-mer already 0.81/0.73 with
large probe gains — the verdict is not in doubt); phage_fragments (probe +0.250, unambiguous);
lncrna g_max (n_train likely below S1 — verify).

## 3. Disjoint-set framing

The split determines what an FT number means. We use four schemes and state which claim each licenses.

| scheme | construction | licenses | our instance |
|:--|:--|:--|:--|
| **random** | i.i.d. shuffle | in-distribution fit only. Never an OOD or capability claim. | GUE official splits (disjointness unverified) |
| **positional / chromosome-disjoint** | hold out whole chromosomes | generalisation to unseen loci within a genome | NT splice — verified 0% exact and revcomp overlap |
| **temporal** | train ≤ T, test ≥ T′ | generalisation to future/emergent sequences; the right scheme for biosecurity | ViroBench `times` (train ≤2017-10-21, test ≥2020-02-03), 2.2% near-duplicates |
| **homology-disjoint** | remove test items sharing sequence with train above a threshold | generalisation beyond memorised homologues | our `strict` splits; HVUE Host_Tropism 8,390 → 3,391 |
| **taxonomic / group-disjoint** | hold out whole genera/families/species | generalisation to unseen clades — the strongest test | ViroBench `genus` — **built, never run with a gLM** |

### The measurement rule that matters most

**Homology holdout must use local alignment (`easy-search`), not clustering (`easy-cluster`).**
`easy-cluster -c 0.9` demands 90% *bidirectional* coverage and is therefore blind to a test sequence
sharing half its length with a training sequence at high identity. Measured consequence:

| split | leaked at ≥90% id / ≥50% cov |
|:--|--:|
| HVUE Pathogenecity "identity-disjoint" | **80.5%** |
| HVUE Transmissibility "identity-disjoint" | **83.2%** |
| HVUE Host_Tropism "identity-disjoint" | 42.2% |
| ViroBench `times` | **2.1%** |

Two distinct questions must be reported separately, because conflating them is how "clean" wrongly
comes to imply "hard":
- **(a) duplicate leakage** — is the test item substantially a copy of a training item? ViroBench: 2.2%.
- **(b) homology reachability** — does it share *any* detectable conserved region? ViroBench: 85%.

Both numbers are correct. ViroBench is clean on (a) and saturated on (b), which is precisely why an
alignment baseline beats every foundation model there while the split remains legitimate.

### Dev sets must match the test scheme
Currently violated: splice and ViroBench carve dev as a random 15% of train while test is
chromosome-disjoint / temporal. Selection therefore optimises an easier distribution than we report
on. HVUE already uses `GroupShuffleSplit` on the homology cluster id; extend that pattern.

## 4. Prior work — what is already established, and where we add

**Split construction and leakage.** [Whalen, Schreiber, Noble & Pollard, *Nature Reviews Genetics*
23:169–181 (2022)](https://www.nature.com/articles/s41576-021-00434-9) is the canonical treatment of
ML pitfalls in genomics, including how data structure biases performance evaluation and how
preprocessing train and test together leaks information. [Guiding questions to avoid data leakage in
biological ML (*Nature Methods*, 2024)](https://www.nature.com/articles/s41592-024-02362-y) gives a
checklist framing, and [DataSAIL (*Nature Communications*, 2025)](https://www.nature.com/articles/s41467-025-58606-8)
provides tooling for similarity-aware splitting. [Kapoor & Narayanan on leakage and the
reproducibility crisis](https://www.sciencedirect.com/science/article/pii/S2666389923001599) is the
general-ML framing.

**gLM evaluation against supervised baselines.** [DART-Eval](https://arxiv.org/abs/2412.05430)
(NeurIPS 2024) is the closest prior work to our Track A: it evaluates zero-shot, probed and
fine-tuned regimes against *ab initio* CNNs on regulatory DNA and finds ab initio models comparable
to the best fine-tuned DNALMs and **substantially better than all probed DNALMs**.
[BEND](https://arxiv.org/abs/2311.12570) (ICLR 2024) finds gLM embeddings approach expert methods on
some tasks but capture only limited long-range information.

**Where we add something.** Existing work establishes *that* baselines are competitive and *that*
splits leak. It does not, as far as we can tell, isolate **why the baseline was weak**. Our three
mechanisms are the contribution:
1. **receptive field, not capacity** — ResNet 9.44M at RF 89bp scores 0.336 MCC on 600bp splice;
   U-Net 0.26M at global RF scores 0.951. 36× fewer parameters, **+0.62 MCC**.
2. **baseline fitting** — GENEB's own reference k-mer is degenerate on iDHS-EL (MCC 0.000 →
   0.589 refit with scaling, C sweep, class weighting).
3. **comparator class** — on taxonomy an alignment baseline is mandatory and beats every gLM by
   **+0.115**; nobody ran one.

Plus the tooling point: **`easy-cluster` vs `easy-search`** changes a split from "identity-disjoint"
to 80% leaked. DataSAIL and the *Nature Methods* checklist argue for similarity-aware splitting in
general; our number is the concrete cost of getting the coverage mode wrong.

**Verify before citing.** Exact venues, years and quantitative claims in this section should be
checked against the primary PDFs — we have read abstracts and search summaries, not the full papers,
for BEND and the two leakage-methods papers.

## 5. Deliverable per FT cell

Every cell reports: dev-selected LR; per-seed test scores with SD; the CNN-ladder number on the same
split; the frozen-probe number on the same split; effective context in bp for each method; split
scheme; the `collapsed_to_majority_class` flag; and per-example predictions for paired testing.
Regimes stay in separate columns — never averaged, never max'd.
