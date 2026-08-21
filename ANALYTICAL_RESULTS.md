# Analytical results — all benchmarks, viral and non-viral, both clusters

2026-08-21. Merges this cluster's runs with Vista's (`CLUSTER_HANDOFF_FROM_VISTA.md`,
`reports/hvue_real_data_verification.md`, `reports/virobench_full_spec_baselines.md`).
Baseline convention throughout: **`best(k-mer3-5, k-mer3-6, CNN)` evaluated at the model's own
effective context**, mean over seeds (never best-of-N).

---

## 0. Master summary — every benchmark tested

Baseline = `best(k-mer3-5, k-mer3-6, CNN)` at the model's own effective context, mean over seeds.

| domain | benchmark | split quality | cells | gLM vs best baseline | verdict |
|:--|:--|:--|:--|:--|:--|
| **viral** | HVUE ×3 tasks | MMseqs2 identity-disjoint, no gate | 18 (2 clusters) | 0 wins, 2 ties, 16 losses | **no capability** |
| viral | HVUE ×3 tasks | random | 9 | 1 tie (+0.0011), 8 losses | no capability |
| viral | HVUE ×3 tasks | composition-cluster (**invalid**: baseline-gated) | 9 | 6 wins | artifact of split selection |
| viral | ViroBench taxonomy, context ladder | temporal (verified) | 4 contexts | 0 wins, 1 tie | **no capability** |
| viral | ViroBench taxonomy, full spec | temporal | 5 levels | k-mer wins all by 0.19–0.36 | no capability |
| viral | ViroBench host, leave-one-family-out | family-disjoint | 4 | all lose, −0.10…−0.17, CIs exclude 0 | **no capability** |
| viral | GUE `virus_covid` | random, **11.9% leakage** | 3 | all lose (−0.043 best) | no capability |
| viral | GUE `virus_species_40` | random | 3 | 1 tie (+0.004), 2 losses | no capability |
| viral | Antibody escape, frozen | antibody-held-out | 20 folds | GBT wins **20/20** | **no capability** |
| viral | Antibody escape, LoRA matched *n* | antibody-held-out | 5 folds | GBT wins 4/5 | no capability |
| viral | ProteinGym viral, 25 assays | published leaderboard | 95 scorers | MSA holds top 15; best seq-only ranks 16 | no capability |
| **viral** | **ViroBench vs published LucaVirus** | temporal | 1 | **LucaVirus +0.079 over our k-mer** | **OPEN — head-to-head running** |
| **non-viral** | **NT splice sites ×3** | **chromosome-disjoint (verified)** | 3 | **gLMs +0.31 … +0.60 MCC** | **REAL CAPABILITY** |
| non-viral | NT benchmark, other 15 tasks | chromosome-disjoint | 15 | gLMs +0.04 … +0.12 MCC | modest capability |
| non-viral | GUE ×12 tasks | official, unverified | 12 | gLMs +0.003 … +0.21 MCC | modest, in-distribution |
| non-viral | EPI ×6 cell lines | random, **39–80% leakage** | 6 | **baseline wins 5/6** | benchmark unusable |
| non-viral | DART-Eval Task 1 | chromosome-disjoint | 1 | our CNN within 0.4% of published | harness validated |

**The single most important row is the splice-site one.** It proves the methodology detects large
genuine capability when present. Every viral row is a loss or a tie against the same baseline
convention — with one unresolved exception (LucaVirus).

---

## 1. HVUE core — two clusters, independently rebuilt splits, converge on zero wins

MMseqs2 90%-identity-disjoint hsd0, no baseline gate, 3 seeds per cell on both sides.

**Baselines reproduce across clusters** (independent rebuilds of the split):

| task | k-mer here | k-mer Vista | CNN here | CNN Vista |
|:--|--:|--:|--:|--:|
| Host_Tropism | 0.9194 | 0.9171 | 0.9482 | 0.9491 |
| Pathogenecity | 0.9479 | 0.9558 | 0.9667 | 0.9715 |
| Transmissibility | 0.9085 | 0.9224 | 0.9202 | 0.9340 |

Max disagreement 0.014 — the split construction is reproducible.

**gLM excess over `best(k-mer, CNN)`, 3 seeds:**

| task | model | here | Vista |
|:--|:--|--:|--:|
| Host_Tropism | NT-v2-500M | **+0.0012** | **+0.0003** |
| Host_Tropism | HyenaDNA | −0.0044 | −0.0119 |
| Host_Tropism | GENA-LM | −0.0111 | −0.0106 |
| Pathogenecity | NT-v2-500M | −0.0257 | −0.0283 |
| Pathogenecity | HyenaDNA | −0.0432 | −0.0164 |
| Pathogenecity | GENA-LM | −0.0173 | −0.0074 |
| Transmissibility | NT-v2-500M | −0.0508 | −0.0325 |
| Transmissibility | HyenaDNA | −0.0727 | −0.0373 |
| Transmissibility | GENA-LM | −0.0164 | −0.0177 |

**18 cells across two clusters: 0 wins, 2 ties (+0.0012, +0.0003), 16 losses.** Both "wins" are
NT-v2 on Host_Tropism and both are inside seed noise. Vista's framing is right — this is a tie,
not a win, and my earlier "8 losses, 1 win" should read **"8 losses, 1 tie."**

The CNN is the binding baseline in **9 of 9** cells (it beats the k-mer by +0.012 to +0.029
everywhere). Reporting k-mer alone would have shown 3 spurious gLM wins.

---

## 2. ViroBench context ladder — the k-mer's advantage is NOT a context artifact

DNA / `times` (temporal) / family, 44 classes. Baselines capped at each gLM's effective context.

| effective context | best k-mer | CNN | best gLM at that context | winner |
|--:|--:|--:|:--|:--|
| 3.1 kb | **0.8469** | 0.5623 | GENA-LM 0.6378 | **k-mer +0.209** |
| 6.1 kb | 0.8866 | 0.6451 | NT-v2 **0.8867** | **tie (+0.0001)** |
| 20 kb | **0.9297** | 0.6774 | HyenaDNA 0.8784 | **k-mer +0.051** |
| whole genome (43.5 kb) | **0.9527** | 0.6563–0.6858 | — | — |

**Correction to my own earlier claim:** I reported "NT-v2 +0.021 at 6.1 kb" using only k-mer3-5
(0.8662). k-mer3-6 at that cap reaches 0.8866, making it a **dead tie**, not a win. So the
matched-context ladder now shows **zero gLM wins at any context** — k-mer wins at 3.1 kb and
20 kb and ties at 6.1 kb.

This also settles the disagreement Vista raised and then retracted: their reading that gLMs show
capability the CNN lacks is correct *against the CNN* (HyenaDNA +0.236 at 20 kb) but the CNN is
not the binding baseline on taxonomy — the k-mer is, and it beats HyenaDNA at the same 20 kb.

Whole-genome run-to-run spread: k-mer3-6 0.9370–0.9527, CNN 0.6563–0.6858. Treat ±0.02 as the
resolution on this task.

---

## 3. ViroBench at full spec — and the one genuine open positive in the programme

Vista ran `--mod ALL` (46,651), `--min_count 1`, all 5 levels, `times` split — removing the two
caveats that made earlier numbers non-comparable to ViroBench's published figures.

| level | classes | k-mer3-6 (whole genome) | CNN (20 kb) |
|:--|--:|--:|--:|
| kingdom | 18 | 0.560 | 0.294 |
| phylum | 28 | 0.555 | 0.325 |
| class | 45 | 0.520 | 0.327 |
| order | 67 | 0.599 | 0.238 |
| family | 173 | **0.570** | 0.208 |

k-mer wins every level by 0.19–0.36. But placed against ViroBench's published numbers **on the
matching split** (their T-split, from the ViroBench paper's own table):

| method | T-split taxonomy macro-F1 | source |
|:--|--:|:--|
| BLAST | 0.412 | published |
| **our k-mer3-6** | **0.570** | this work |
| **LucaVirus** | **0.649** | published |

Two readings, both important:

1. **Our k-mer beats their alignment baseline by +0.158.** ViroBench's reported model advantage
   over BLAST (+0.237 on T-split) shrinks to **+0.079** once the baseline is a properly-fitted
   whole-genome k-mer rather than BLAST. Most of their headline gap is baseline weakness.
2. **LucaVirus still leads by +0.079.** This is the **only** credible case anywhere in this
   programme of a viral foundation model beating a well-fitted classical baseline on a genuinely
   disjoint split. It is the one live candidate for a qualified unlearning target.

**Three checks before treating it as established:**
- which taxonomic level their 64.91 refers to (ours is family; if theirs is kingdom, the
  comparison is invalid — kingdom is a far easier 18-class problem)
- whether their `ALL` set matches ours
- LucaVirus's effective context — if it reads whole genomes the comparison is fair; if it reads
  less, it is winning *despite* less context, which strengthens their claim

---

## 4. GUE viral — both tasks, and a leakage finding that undercuts one

| task | classes | best baseline | GENA-LM | HyenaDNA | NT-v2 | best gLM vs baseline |
|:--|--:|--:|--:|--:|--:|--:|
| `virus_covid` | 9 | **k-mer 0.7282** | 0.6662 | 0.4499 | 0.6850 | **−0.043** |
| `virus_species_40` | 25 | **k-mer 0.4407** | 0.3643 | 0.4329 | 0.4443 | **+0.004 (tie)** |

Both losses or ties. But Vista found that **`virus_covid`'s official split has 11.9% exact-duplicate
test sequences present verbatim in train** — not previously flagged. That inflates every method's
score on that task and makes it unusable as an OOD measurement without a dedup pass. It does not
change the ranking (all methods benefit), but the absolute numbers should not be quoted.

---

## 4b. NON-VIRAL results — the other half of the picture

These are not incidental. They are what makes the viral negatives interpretable: the same
baseline methodology, applied to non-viral benchmarks, produces **large gLM wins in one place,
small-to-zero gaps in most places, and baseline wins in one place.** A pipeline that only ever
returns "no advantage" would be uninformative; this one discriminates.

Baseline convention identical to the viral work: `best(k-mer3-5, k-mer3-6, mean-over-seeds CNN)`,
never best-of-N. Competitor numbers are published single point estimates.

### 4b.1 Nucleotide Transformer benchmark (18 tasks) — the clean positive control

`reports/positive_control_comparison.md`. Splice sites are chromosome-disjoint (0% overlap verified).

| task | our baseline MCC | best published gLM | gap |
|:--|--:|:--|--:|
| **Splice All** | 0.3731 | NTv2 0.9710 | **+0.5979** |
| **Splice Acceptor** | 0.6190 | GJ-B 0.9710 | **+0.3520** |
| **Splice Donor** | 0.6764 | GJ-B 0.9840 | **+0.3076** |
| H4K20me1 | 0.5693 | GJ-B 0.6920 | +0.1227 |
| H3K9ac | 0.4982 | GJ-B 0.6110 | +0.1128 |
| Enhancer Type | 0.4667 | NTv2 0.5760 | +0.1093 |
| H3K9me3 | 0.4184 | GJ-B 0.5200 | +0.1016 |
| H3K27ac | 0.4336 | GJ-B 0.5310 | +0.0974 |
| H3K4me1 | 0.4120 | GJ-B 0.5090 | +0.0970 |
| H3K36me3 | 0.5757 | GJ-B 0.6670 | +0.0913 |
| Promoter NoTATA | 0.7357 | NTv2 0.8230 | +0.0873 |
| Enhancer | 0.4933 | NTv2 0.5730 | +0.0797 |
| H3K27me3 | 0.5613 | GJ-B 0.6410 | +0.0797 |
| H3K4me2 | 0.5102 | GJ-B 0.5850 | +0.0748 |
| H2AFZ | 0.4696 | Hyena7M 0.5350 | +0.0654 |
| H3K4me3 | 0.5942 | DB2 0.6590 | +0.0648 |
| Promoter TATA | 0.9135 | GJ-B 0.9600 | +0.0465 |
| Promoter All | 0.7481 | NTv2 0.7880 | +0.0399 |

**Mean gap +0.1404.** The three splice-site tasks are the standout: **+0.31 to +0.60 MCC** on a
verified chromosome-disjoint split. This is a real, large, split-robust capability — the exact
thing that does not exist on any viral task.

### 4b.2 GUE benchmark (12 tasks) — mostly small gaps

| task | our baseline MCC | best published | gap |
|:--|--:|:--|--:|
| Promoter TATA | 0.6147 | GJ-B 0.8210 | +0.2063 |
| TF Human 4 | 0.4775 | DB2 0.6310 | +0.1535 |
| TF Human 1 | 0.6575 | DB2 0.7190 | +0.0615 |
| Promoter NoTATA | 0.9153 | GJ-B 0.9600 | +0.0447 |
| TF Human 5 | 0.7449 | DB2 0.7890 | +0.0441 |
| TF Human 2 | 0.6908 | GJ-B 0.7340 | +0.0432 |
| Core Prom. NoTATA | 0.6895 | GJ-B 0.7250 | +0.0355 |
| Core Prom. All | 0.6871 | DB2 0.7170 | +0.0299 |
| Splice All | 0.8714 | GJ-B 0.8900 | +0.0186 |
| Core Prom. TATA | 0.8104 | GJ-B 0.8190 | +0.0086 |
| Promoter All | 0.9134 | NTv2 0.9210 | +0.0076 |
| TF Human 3 | 0.6307 | DB2 0.6340 | +0.0033 |

**Mean gap +0.0547.** Across GUE + NT, **11 of 30 tasks** have the baseline within 0.05 MCC of
the best published gLM or beating it. GUE's disjointness is unverified, so treat these as
in-distribution.

### 4b.3 EPI enhancer–promoter interaction (6 cell lines) — the baseline WINS

| cell line | our baseline AUROC | EPIPDLF | EPINTLM | our margin vs best published |
|:--|--:|--:|--:|--:|
| IMR90 | **0.9879** | 0.936 | 0.909 | **+0.052** |
| HUVEC | **0.9829** | 0.935 | 0.935 | **+0.048** |
| K562 | **0.9670** | 0.943 | 0.947 | **+0.020** |
| GM12878 | **0.9655** | 0.939 | 0.949 | **+0.017** |
| HeLa-S3 | **0.9772** | 0.964 | 0.970 | **+0.007** |
| NHEK | 0.9886 | 0.993 | 0.985 | −0.004 (tie) |

**Our simple baseline beats both published pretrained-embedding methods on 5 of 6 cell lines.**
And the reason is measurable: MMseqs2 ≥90% identity clustering finds **enhancers 39–46% and
promoters 67–80% of test anchors leak into train** (`reports/mmseqs_leakage_check.csv`). Exact-string
matching alone would have shown only 42–62% promoter / 5.9–8.6% enhancer overlap — so the leakage
is worse than a naive duplicate check reveals. This independently confirms the published
BENGI/LOCO-EPI critique by direct measurement.

### 4b.4 Measured leakage across all benchmarks (MMseqs2 ≥90% identity)

| benchmark family | test sequences leaking into train |
|:--|--:|
| GUE promoter (core/300, all variants) | 1.4–5.6% |
| GUE TF human 0–4 | 2.1–3.4% |
| **GUE splice reconstructed** | **21.5%** |
| **GUE `virus_covid`** (exact duplicates) | **11.9%** |
| **EPI enhancers** | **39–46%** |
| **EPI promoters** | **67–80%** |

GUE promoter/TF splits are clean. GUE splice at 21.5% and `virus_covid` at 11.9% are compromised.
EPI is unusable as an OOD benchmark.

### 4b.5 DART-Eval Task 1 — harness reproduces published numbers

Ab-initio CNN, cCRE vs dinucleotide-shuffled negatives, chromosome-disjoint:

| metric | published | ours | diff |
|:--|--:|--:|--:|
| accuracy | 0.8460 ± 3.3e-4 | 0.8423 | **0.4%** |
| AUROC | 0.927 | 0.9264 | **0.06%** |

Verified against the arXiv LaTeX source directly.

### 4b.6 What the non-viral results establish

1. **The harness detects real capability when it exists** — +0.31 to +0.60 MCC on chromosome-disjoint
   splice sites. The viral negatives are therefore not a pipeline failure.
2. **The baseline is competitive far more often than the literature implies** — 11 of 30 GUE/NT tasks
   within 0.05 MCC, and EPI outright lost by the published methods.
3. **Split leakage is endemic, not a viral-specific problem.** EPI at 67–80% and GUE splice at 21.5%
   are the same failure mode as ViroBench's mislabelled "genus-disjoint" split and HVUE's
   baseline-gated composition split.
4. **Capability, where it exists, is concentrated.** Splice-site recognition is a local,
   motif-driven, position-specific task — exactly what a k-mer bag-of-features cannot represent
   and a sequence model can. That is the shape of a genuine gLM advantage, and no viral task
   we tested has it.

---

## 5. Harness validity — three independent confirmations

The negatives above are not a broken-pipeline artifact:

| check | result |
|:--|:--|
| **NTv3 splice sites**, chromosome-disjoint (0% overlap verified) | gLMs beat best classical baseline by **+0.30 to +0.60 MCC** — a large, clean, disjoint-split positive control |
| **DART-Eval Task 1** reproduction | ab-initio CNN within **0.4%** of the paper's published numbers (0.8423 vs 0.8460 acc; 0.9264 vs 0.927 AUROC) |
| **Cross-cluster baseline agreement** | identity-split k-mer and CNN reproduce to ≤0.014 across two independent rebuilds |

So when a gLM loses on a viral task here, the pipeline is capable of detecting a win — it detects
one on splice sites at +0.30–0.60 MCC.

---

## 6. Split-integrity findings (cumulative)

| split | problem | magnitude |
|:--|:--|:--|
| HVUE composition-cluster | built in k-mer5-PCA space **and** accepted only if k-mer loses ≥0.03 (`GATE=0.03`) | moves k-mer by **0.12–0.18** vs identity holdout |
| HVUE identity @90% after 99% dedup | too permissive — barely OOD | k-mer drops only 0.003–0.014 |
| ViroBench `genus` ("G-split") | **82–84% genus overlap**; only taxid is held out | it is a record-level holdout, not genus-disjoint |
| GUE `virus_covid` official | **11.9% exact test duplicates in train** | inflates all methods |
| GUE EPI (positive-control sweep) | **67–80% promoter leakage** train→test, MMseqs2-verified | corroborates BENGI/LOCO-EPI critique |
| ViroBench `times` | clean: zero date overlap, 1–2% species overlap | **the split to use** |

Identity-threshold sweep (90/70/50/30%) shows the k-mer barely degrades at any threshold
(0.9066–0.9209 on Host_Tropism), so composition-clustering is not a stricter OOD test — it is a
differently-biased one.

---

## 6b. What our leakage measurement actually captures — measured, not assumed

Prompted by the question "can't different viruses share parts and be wrongly called duplicates,
or share parts and be missed?" Both directions were tested empirically on HVUE Host_Tropism
(n=6,000, all sequences exactly 1000 bp — the same data the identity splits were built from).
All clustering used the production parameters: `--min-seq-id 0.90 -c 0.9`, default cov-mode
(bidirectional 90% coverage).

| test | question | result |
|:--|:--|:--|
| **1. label concordance** | are clustered pairs same-label, or incidental homology? | **95.2% of multi-member clusters are label-pure** (204 all-positive, 13 all-negative, 11 mixed) vs **50% chance**. The clusters are shared content *and* shared label — memorisable leakage, not neutral homology. |
| **2. reverse complement** | are revcomp duplicates caught? | **300/300 (100%)**. MMseqs2 searches both strands. Not a gap. |
| **3. partial overlap** | is a 50%-shifted window caught? | **0/150 (0%)**. Completely invisible to `-c 0.9`. **This is a real gap.** |
| **4. false positives** | does a shared 200 bp block (20% of length) falsely merge unrelated sequences? | **60/60 stayed distinct.** Bidirectional coverage prevents conserved-region false merging. |

**So the concern about false positives is answered: it does not happen.** A conserved gene shared
between two otherwise-different viruses will not merge them, because 90% coverage of *both*
sequences is required — and the 95% label purity confirms the clusters are not picking up
incidental homology.

**The false-negative direction is the real problem, and it is large.** Exhaustive
`mmseqs easy-search` (no coverage floor, `--search-type 3`) of test against train on our own
`identity_disjoint_hsd0` splits:

| task | test seqs with **no** ≥90%-id hit | ≥90% id over ≥50% of length | ≥90% id over ≥90% |
|:--|--:|--:|--:|
| Host_Tropism | 59.5% | **32.8%** | 1.2% |
| Transmissibility | 11.2% | **82.1%** | 23.1% |

Median coverage among test sequences that have a hit: 0.500 (Host_Tropism), 0.796
(Transmissibility).

**Our "identity-disjoint" splits are therefore not fully disjoint.** `-c 0.9` removes only the
1.2%/23.1% tail; a third of Host_Tropism and four fifths of Transmissibility test sequences retain
a ≥90%-identity match covering half their length in the training set. (The 23.1% at ≥90% coverage
that survived clustering also shows `easy-cluster` is a heuristic — exhaustive search finds pairs
its cascaded prefilter misses.)

### What this changes, and what it does not

**Does not change the model-vs-baseline conclusions.** Leakage inflates *absolute* scores for
every method that can exploit shared content — both the k-mer and the sequence models. There is no
mechanism by which partial train/test overlap would systematically favour a k-mer over a gLM; if
anything a sequence model is better placed to memorise a specific half-sequence than a
bag-of-k-mers is. So the *relative* ordering (0 wins / 2 ties / 16 losses) stands.

**Does change what we may call these results.** They are not clean OOD measurements. The
absolute numbers — k-mer 0.9085–0.9479, CNN 0.9202–0.9715 — are inflated by an unknown amount,
and Transmissibility in particular should not be described as out-of-distribution at all.

**It also means HVUE is not the venue for an OOD claim.** These are 1000 bp windows tiled from a
small number of viral genomes; near-duplicate structure at 50% overlap is intrinsic to the
dataset, not a fixable split artifact. A genuinely disjoint HVUE split would need whole-genome
grouping (hold out all windows from a given accession), which the released data may not support —
HVUE ships only `sequence,label` with no accession or coordinate metadata (independently confirmed
in `reports/unlearning_eval_protocol.md`).

### Recommended fix for future splits

Use `-c 0.3` (or `--cov-mode 1` with a low floor) when *measuring* leakage, and group by source
accession when *building* splits. Clustering at `-c 0.9` is the right tool for removing
near-identical duplicates and the wrong tool for certifying disjointness.

**Not yet measured:** the same test on ViroBench `times`, which we have been treating as the
gold-standard split. It is temporally separated with whole genomes, so the window-overlap failure
mode should not apply — but that is an assumption, not a measurement, and it should be checked
before the ViroBench conclusions are described as OOD either.

---

## 7. Bottom line

**Across 18 HVUE cells, 4 ViroBench contexts, 2 GUE tasks, ViroBench LOFO host prediction, and
the escape and ProteinGym arms — measured on two clusters with independently rebuilt splits —
biological foundation models do not beat `best(k-mer, CNN)` at matched context on any viral task
we control end-to-end.** Best case is a tie (+0.0003 to +0.0012).

**One exception is live and unresolved:** LucaVirus on ViroBench taxonomy, T-split, appears to
beat our properly-fitted k-mer by +0.079. Everything else in the programme is a loss or a tie.
If that survives the three checks in §3, it is the qualified target the unlearning project has
been looking for — and it would be the *only* one. If it does not survive, the viral modality is
closed.

**For the unlearning question specifically**, two facts are independent of the LucaVirus check:
frozen viral representations sit 0.02–0.25 AUROC *below* baseline (nothing localisable to excise),
and a 0.64M CNN trained from scratch matches or beats every gLM on 8 of 9 HVUE cells (the
capability is not model-gated, so removing it from open weights denies it to nobody).

---

## 8. Immediate next actions

| # | action | why | where |
|:--|:--|:--|:--|
| 1 | **Confirm ViroBench's taxonomic level** for the 47.67/75.88 and 41.22/64.91 figures | decides whether the one live positive is real | paper check, no compute |
| 2 | Run **LucaVirus** on our full-spec `times` split at family level | direct head-to-head, same data, same level | either cluster; weights on HF |
| 3 | **BLAST + Kraken2** baselines at full spec | Kraken2 is a k-mer classifier by design — the honest SOTA taxonomy comparator | Vista offered to take this |
| 4 | Dedup `virus_covid` and re-measure | 11.9% leakage makes current numbers unquotable | here |
| 5 | gLMs at ViroBench full spec (46,651 rows, 5 levels) | only the 6,042-row DNA/family subset has gLM numbers | ~7.7× cost, either cluster |
| 6 | **Supervised single-variant** (22 viral ProteinGym assays, position-disjoint) | the only untouched modality; leaderboard is all zero-shot | here — data is local |
| 7 | Escape completion (LASV/SARS2/H5 LoRA + full, matched *n*) | 3 of 4 antigens incomplete | here — data is local |
