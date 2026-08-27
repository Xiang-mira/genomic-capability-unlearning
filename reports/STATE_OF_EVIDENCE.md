# State of evidence — baselines vs gLMs/pLMs, positive and negative results

Analytical report. Every number is our own run unless marked **C2** (Vista cluster) or
**EXTERNAL/PUBLISHED**. Read `PROTOCOL.md` for the rules these runs obey and `PAPER_DESIGN.md` for
the validity grid.

---

## 1. The central analytical result

Across ~60 task-cells, four model families and five benchmark suites, **the choice of model is the
smallest lever we measured.** Method-of-evaluation choices dominate it by one to two orders of
magnitude.

| lever | measured spread | evidence |
|:--|--:|:--|
| **split / subset choice** | **0.37 macro-F1** | k-mer3-6 on ViroBench family: 0.9252 (DNA/genus) → 0.8667 (DNA/times) → 0.5566 (ALL/times) |
| **baseline architecture** | **0.66 MCC** | 13-cell CNN ladder on splice_all: 0.2909 (dilated 0.04M) → 0.9538 (U-Net 5.3M) |
| **adaptation regime** | **0.60 MCC** | NT-v2 on splice_all: frozen probe 0.3636 → full FT 0.9674 |
| **homology-audit tool** | **0.80 → 0.02** | HVUE Path. "leak-free" under `easy-cluster -c 0.9`; 80.5% leaked under `easy-search` |
| **comparator class** | **0.115 macro-F1** | ViroBench family: best gLM 0.6148 vs alignment nearest-hit 0.7383 |
| **read-out layer** | **0.02–0.04 macro-F1** | ViroBench family: L−1 → dev-best layer, per model |
| **model (fixed everything else)** | **0.32 macro-F1** | ViroBench family: NT-v2 0.611 → LucaVirus 0.295 |
| **best model vs best baseline** | **+0.01 to +0.015 MCC** | splice: NT-v2 0.9674 vs CNN ladder 0.9528 |

The last two rows are the point. The *entire* four-model spread on a fixed split is smaller than the
spread produced by changing the split. And the margin by which the best model beats the best
baseline is **40–60× smaller** than the artifacts we found in the evaluation method.

**This holds regardless of which side of the viral question one lands on**, and it is the finding
most likely to survive review.

---

## 2. What we have run

### 2.1 Models

| model | params | HVUE | ViroBench | GUE viral | splice | GENEB |
|:--|--:|:--|:--|:--|:--|:--|
| NT-v2-500M | 500M | ✓ | ✓ | ✓ | ✓ | ✓ **C2** |
| GENA-LM-bert-base-t2t | 111M | ✓ | ✓ | ✓ | ✓ | ✓ **C2** |
| HyenaDNA-medium-160k | 7M | ✓ | ✓ | ✓ | ✓ | ✓ **C2** |
| LucaVirus | 944M | — | ✓ (4 layers) | — | — | — |
| Evo-1-8k | 6.5B | LoRA only | — | — | — | — |
| **ESM-2 / ESM-1v (pLMs)** | — | — | — | — | — | **never run** |
| DNABERT-2 / Caduceus / Evo-2 / GenoJEPA | — | — | — | — | — | **never run** |

DNABERT-2 is blocked by a `config_class` conflict with transformers 4.48.

### 2.2 Baselines and comparators

| comparator | coverage |
|:--|:--|
| k-mer3-5 / k-mer3-6 → standardised LR, C on dev | broad, all benchmarks |
| **CNN architecture × capacity ladder** (dilated / U-Net / ResNet, 0.04M–9.4M, 13 cells) | **only 9 task-cells**: splice ×3, HVUE ×3, ViroBench ×3 levels |
| alignment nearest-hit (mmseqs, nucleotide) | ViroBench family only |
| BLASTn + Kraken2 | ViroBench **C2** |
| MSA / conservation | escape, variant effect |
| one-hot + biochemical (protein) | **never run** |

### 2.3 Regimes

frozen probe (layer swept only on ViroBench family) · LoRA (partial) · full FT (splice ×3, GUE viral ×2)

---

## 3. Positive results — where gLMs beat the baseline

**All non-viral.** These are what make a viral null informative rather than a harness failure.

### 3.1 NT splice, full fine-tune (our strongest controlled positive)

| task | kmer3-6 | CNN dev-selected | NT-v2 FT (3 seeds) | margin |
|:--|--:|--:|--:|--:|
| splice_all | 0.2782 | 0.9528 (dilated 9.33M) | **0.9674 ± 0.0025** | **+0.0146** |
| acceptors | 0.4131 | 0.9527 (dilated 9.33M) | **0.9660 ± 0.0014** | **+0.0133** |
| donors | 0.4764 | 0.9637 (U-Net 7.05M) | **0.9736 ± 0.0038** | **+0.0099** |

EXTERNAL/PUBLISHED comparators on these tasks: 0.971–0.984 (fine-tuned). **Our harness reproduces
published-level performance**, which is the reproduction claim and needs no tuning parity.

But: **only NT-v2 clears the baseline.** The other two lose to a 9.3M-parameter CNN:

| model | splice_all | acceptors | donors |
|:--|--:|--:|--:|
| HyenaDNA FT | 0.8472 (−0.106) | 0.8263 (−0.126) | 0.9009 (−0.063) |
| GENA-LM FT | 0.8341 (−0.119) | 0.7950 (−0.158) | 0.8158 (−0.148) |

### 3.2 GENEB, 13 categories, frozen probe **C2**

Per model against a fairly-refit k-mer (±0.005 tie band):

| model | wins | ties | losses | mean margin |
|:--|--:|--:|--:|--:|
| GENA-LM | 11 | 0 | 2 | **+0.083** |
| NT-v2 | 10 | 1 | 2 | **+0.078** |
| HyenaDNA | 6 | 1 | 6 | **−0.026** |

Survives per-model for 2 of 3 — so pretraining buys real headroom on non-viral tasks,
**model-dependently**. HyenaDNA is a coin flip.

### 3.3 HVUE Host_Tropism, strict homology-clean split (n=3,391)

| method | dev | test AUROC |
|:--|--:|--:|
| kmer3-6 | 0.9139 | 0.8411 |
| CNN dev-selected (2.53M) | 0.9470 | 0.8588 |
| **NT-v2 frozen probe** | 0.9217 | **0.8647** |
| GENA-LM | 0.8747 | 0.8229 |
| HyenaDNA | 0.7609 | 0.7145 |

**+0.0059** — the only clean viral positive anywhere in the programme, and it is tiny.

---

## 4. Negative results — where no advantage is detectable

**All viral.** Language: *no detectable model-specific advantage over the evaluated comparators
under this task/evaluation regime.*

### 4.1 ViroBench taxonomy — the load-bearing benchmark

Only viral benchmark with a verified-clean split (2.2% near-duplicates, two independent methods).

Family level, n=5,505, macro-F1, all comparators on identical examples:

| method | context | test | vs best baseline |
|:--|:--|--:|--:|
| **alignment nearest-hit** | whole genome | **0.7383** | — |
| k-mer3-5 | matched 32.7 kb | 0.6231 | — |
| NT-v2, dev-selected layer | 32.7 kb | 0.6148 | **−0.124 vs alignment** |
| k-mer3-6 | matched 32.7 kb | 0.5566 | |
| GENA-LM L−2 | 32.7 kb | 0.5450 | |
| HyenaDNA L−3 | 32.7 kb | 0.5385 | |
| LucaVirus L−2 | 32.7 kb | 0.2946 | |
| CNN ladder | 32.7 kb | 0.1959 | |

All five levels, NT-v2 vs whole-genome k-mer3-6, paired bootstrap with δ pre-declared:

| level | classes | Δ | 95% CI | verdict |
|:--|--:|--:|:--|:--|
| kingdom | 18 | +0.0509 | wide | underpowered |
| phylum | 28 | +0.0261 | [−0.061, +0.048] at L−1 | ns |
| class | 45 | −0.0039 | [−0.038, +0.028] | ns, equiv @ δ=0.05 |
| order | 67 | −0.0092 | [−0.068, −0.008] at L−1 | ns |
| family | 173 | +0.0373 | [+0.0013, +0.0528] at L−2 | **significant vs k-mer only** |

**Why the null is mechanistic, not an absence of evidence.** Alignment reaches **0.9636 macro-F1 /
0.9915 accuracy** on the 85% of test genomes it can align — over only **10–13% of query length** at
90–100% identity. That is a conserved gene, not a duplicated genome. **Viral family taxonomy is
determined by short conserved regions local alignment finds directly.** The split is clean *and* the
task is an alignment task; both true simultaneously.

### 4.2 GUE viral

| task | best baseline | NT-v2 | GENA-LM | HyenaDNA | CNN |
|:--|--:|--:|--:|--:|--:|
| virus_covid (deduped, n=8,050) | **k-mer3-6 0.7171** | 0.6764 | 0.6668 | 0.6460 | 0.6133 |
| virus_species_40 | k-mer 0.4407 | 0.4443 | 0.3643 | 0.4329 | — |

Removing 11.9% exact-duplicate test rows did **not** change the ranking.

### 4.3 HVUE Pathogenecity / Transmissibility — UNEVALUABLE

| task | test rows | homology-clean rows | dropped |
|:--|--:|--:|--:|
| Pathogenecity | 5,194 | **96** | 98.2% |
| Transmissibility | 4,956 | **60** | 98.8% |

Not a negative result — **these tasks cannot separate memorisation from generalisation for any
method**, including the original HVUE paper's positive claims.

---

## 5. Validity findings — the actual contribution

Prior work ([Whalen et al. 2022](https://www.nature.com/articles/s41576-021-00434-9),
[DART-Eval](https://arxiv.org/abs/2412.05430), [BEND](https://arxiv.org/abs/2311.12570)) establishes
*that* baselines are competitive and *that* splits leak. It does not isolate **why the baseline was
weak.** Six mechanisms, each measured:

| # | mechanism | magnitude |
|--:|:--|--:|
| 1 | **Receptive field, not capacity.** ResNet 9.44M @ RF 89bp = 0.336 MCC on 600bp splice; U-Net 0.26M @ global RF = 0.951 | **+0.62** at 36× fewer params |
| 2 | **Homology audit tool.** `easy-cluster -c 0.9` needs 90% *bidirectional* coverage; blind to a test seq sharing half its length (0/150 caught in direct test) | 0% → **80.5%** leaked |
| 3 | **Missing comparator class.** No alignment baseline on a taxonomy task | **+0.115** over every model |
| 4 | **Baseline fitting.** GENEB's shipped k-mer: no scaling, no C sweep, no class weighting → majority-class on iDHS-EL **C2** | 0.000 → **0.589** |
| 5 | **Read-out layer.** LucaVirus final LayerNorm collapses representation: between-sequence std 0.0027 (L12) vs 0.1438 (L11) | 53× |
| 6 | **Silent weight re-init.** `AutoModelForSequenceClassification` on GENA-LM discards all 48 pretrained LayerNorms (pre-LN ckpt, post-LN HF class) | exact 0.0000 at every LR |

Plus two benchmark-level findings: **GENEB specifies no embedding layer at all** (pooling
"model-specific", unrecorded across 40 models — its own reported ranking instability may be partly a
read-out artifact), and the frozen-probe/full-FT gap on splice is **0.59 MCC**, consistent with
DART-Eval's independent finding that ab initio models beat *all* probed DNALMs.

---

## 6. Splits — what each licenses

| split | construction | licenses | status |
|:--|:--|:--|:--|
| composition-cluster, **gated** | kmer5→PCA→KMeans, accepted iff k-mer drops ≥0.03 | **nothing** | invalid by construction |
| identity-disjoint | `easy-cluster -c 0.9` | duplicate removal only | insufficient |
| **strict** | `easy-search`, drop test ≥70% id / ≥30% cov | homology-clean claims | only valid HVUE split |
| ViroBench `times` | train ≤2017-10-21 / test ≥2020-02-03 | temporal generalisation | clean (2.2%) |
| ViroBench `genus` | whole genera withheld | unseen-clade generalisation | **built, never run with a gLM** |
| NT splice | chromosome-disjoint, 0% exact/revcomp | positional generalisation | clean, verified |
| GUE / GENEB official | unverified | **in-distribution only** | must not be called OOD |

---

## 7. What needs to be run

### Tier 1 — decides whether the paper holds
1. **CNN ladder on the 13 GENEB tasks.** Every non-viral positive is k-mer-anchored. Blocked on the
   **C2** re-run: their probes used untuned `C=1.0` and selected C on macro-F1 while reporting MCC, so
   the table is not protocol-matched (11/13, not 13/13). Provenance now closed — GENEB task #5 is
   byte-identical to our splice file (SHA-256 reproduced).
2. **Sweep the CNN's learning rate.** The ladder used a single LR (1e-3) while the FMs got 3–5.
   **NT-v2's +0.0146 is inside plausible baseline-tuning noise.** ~2 GPU-h. Until done, the splice
   *reproduction* claim stands but the *margin* claim does not.
3. **Second positive-control anchor** (long-range expression / Borzoi-style). With splice at +0.015,
   one small-margin anchor cannot carry calibration. Blocked on data access.
4. **Reconcile the alignment baseline.** Our mmseqs nearest-hit 0.7383 vs **C2** BLASTn 0.4235 (full)
   / 0.6190 (clean subset). A 0.32 spread between two nearest-hit aligners makes our central viral
   claim aligner-dependent. Requested real BLASTn on our exact n=5,505 set with per-example preds.

### Tier 2 — completes the analysis
5. **Layer sweep for every model on every probe** (currently ViroBench family only).
6. **pLMs — ProteinGym viral supervised, 22 assays**, with a one-hot + biochemical control. The only
   untouched modality; the 95-scorer leaderboard is entirely zero-shot, so supervised adaptation is
   genuinely untested.
7. **Alignment baseline at the other 4 ViroBench levels**, with per-example predictions.
8. **ViroBench `genus` split** — zero gLM numbers on the harder generalisation test.
9. **Matched-context reruns** at the other 4 ViroBench levels.
10. **`easy-search` re-audit of every benchmark.** EPI's 67–80% promoter overlap is a lower bound.
11. **Group-disjoint dev** for splice and ViroBench (currently random 15% while test is
    chromosome-disjoint / temporal — selection optimises an easier distribution than we report on).
12. **Full-FT subset**, 8–12 tasks × 3 models × 3 seeds, per `FULL_FT_DESIGN.md` criteria.

### Tier 3 — what reviewers will ask for, and why it is last
13. **More models** — DNABERT-2, Caduceus, Evo-2, and especially
    [GenoJEPA](https://www.biorxiv.org/content/10.64898/2026.04.02.716255v1.full) /
    [JEPA-DNA](https://arxiv.org/pdf/2602.17162). Their claimed gains (+4.8% on splice for JEPA-DNA)
    are *smaller than our read-out-layer effect* and far smaller than the baseline-architecture
    effect, so they are unresolvable without our controls. The high-value version is not "add a ninth
    model to the table" but **run one or two of them through the full harness** — ladder-selected CNN,
    layer sweep, matched context, audited split. If their gains survive, that is a genuine positive
    control. Adding models to an under-baselined comparison multiplies the error.

---

## 8. Claim status

| claim | status |
|:--|:--|
| HVUE Path./Trans. cannot support homology-clean evaluation | **defensible** |
| Baseline receptive field, not capacity, binds on positional tasks | **defensible** |
| Harness reproduces published splice numbers (0.9674 vs 0.971–0.984) | **defensible** |
| Viral family taxonomy is an alignment task | **defensible** (pending aligner reconciliation) |
| Deduplication does not change the GUE viral ranking | **defensible** |
| gLMs beat a fair k-mer on most non-viral GENEB tasks, per model | **defensible** **C2** |
| The composition-gated split manufactures the gLM win | **defensible** |
| **NT-v2 beats the best baseline on splice by +0.015** | **provisional** — CNN LR unswept |
| **No FM advantage on ViroBench** | **provisional** — aligner discrepancy unresolved |
| gLM wins on GENEB survive a CNN baseline | **unknown** — never tested |
| ~~Published splice gap is +0.31 to +0.60~~ | **withdrawn** — receptive-field artifact |
| ~~LucaVirus +0.079 advantage~~ | **withdrawn** — −0.28 across 4 layers |
| ~~Our k-mer is the ViroBench ceiling~~ | **withdrawn** — alignment +0.16 above |
| ~~HVUE identity-disjoint carries the safety claim~~ | **withdrawn** — 80.5% leaked |
| ~~NT-v2 significantly beats k-mer on ViroBench family~~ | **withdrawn** — context asymmetry + incomplete k-mer sweep |
