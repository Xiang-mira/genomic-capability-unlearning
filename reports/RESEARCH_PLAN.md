# Research plan — two tracks, one harness

We are running two overlapping programmes. They share the harness, the baseline ladder, the split
auditing and the measurement protocol; they differ in scope and in what a "result" means.

| | **Track A — benchmark & method audit** | **Track B — viral capability** |
|:--|:--|:--|
| question | across existing benchmarks, what does each *method class* actually achieve? | is there model-specific viral capability over the strongest comparator on a defensible split? |
| unit of result | a benchmark × method × regime cell | a verdict per viral task |
| benchmarks | GENEB (100), GUE (33), NT (18), EPI (6), DART-Eval | HVUE (3), ViroBench (5 levels × 2 splits), GUE viral (2), ProteinGym viral (22), escape |
| comparators | k-mer, CNN ladder | + alignment nearest-hit, Kraken2, MSA |
| contribution | methodological: published gaps are largely baseline artifacts | scoped negative result, positively controlled by Track A |
| status | **k-mer broad, CNN ladder on 6 tasks only, full FT on 3** | ViroBench + GUE viral settled; HVUE 2/3 unevaluable; ProteinGym untouched |

**The tracks are not independent.** Track B's negative result is only credible because Track A shows
the same harness detects real capability elsewhere. Conversely, Track A's headline — "baselines are
under-powered" — was discovered *because* Track B forced us to build a proper comparator.

---

## Track A — benchmark & method audit

### The question
For an existing benchmark task, what do you get from each method class, measured under one protocol?

    k-mer3-5 / k-mer3-6  ->  LR, standardised, C on dev, context declared
    CNN                  ->  architecture x capacity ladder (dilated / U-Net / ResNet, 0.04M-9.4M)
    gLM/pLM frozen probe ->  layer swept on dev, mean-pool, LR head
    gLM/pLM LoRA         ->  rank 16
    gLM/pLM full FT      ->  LR swept on dev, warmup, multi-seed  [expensive -> subset only]

### Why it is a contribution in its own right
We have three independent, mechanistically-explained instances of a published gap collapsing
against a properly-built baseline:

1. **Receptive field.** On 600bp splice, ResNet 9.44M (RF 89bp) = 0.336 MCC; U-Net 0.26M
   (global RF) = 0.951. 36x fewer parameters, +0.62 MCC. Published FM-vs-CNN splice gaps of
   +0.31-0.60 shrink to +0.02-0.03. Leakage verified zero.
2. **Feature scaling.** GENEB's shipped reference k-mer predicts the majority class on
   `iDHS-EL_DNase_I` (MCC 0.000). Refit fairly: 0.589.
3. **Read-out layer.** LucaVirus's final layer norm collapses its representation
   (between-sequence std 0.0027 at layer 12 vs 0.1438 at layer 11).

Plus one infrastructure class: `AutoModelForSequenceClassification` on GENA-LM silently discards
48 pretrained LayerNorms and the model collapses to majority class at every LR.

### THE CRITICAL OPEN GAP
**The GENEB sentinel is k-mer-anchored. No CNN baseline was ever run on any of those 13 tasks.**
On the one task where we have both, the CNN dominates:

| NT splice acceptors | MCC |
|:--|--:|
| GENEB fair k-mer | 0.387 |
| GENEB GENA-LM (the "+0.156 win") | 0.543 |
| GENEB best-of-40 published | 0.685 |
| **our CNN ladder, dev-selected 9.3M** | **0.9527** |

So the entire non-viral positive control currently rests on the weaker of our two baselines.
*Caveat:* GENEB runs frozen probes and our CNN trains end-to-end, so this is a regime mismatch and
NOT a clean "CNN beats GENA-LM" claim. The honest status is: **we do not know whether the GENEB
wins survive a CNN baseline, because nobody ran one.** This must be closed before any Track A
positive goes in a draft — if the wins don't survive, Track B's credibility goes with them.

### Full-FT subset selection (full FT is ~50x a frozen probe)
Do NOT full-FT everything. Select tasks on stated criteria, before seeing FT results:
- **ambiguous under probing** — frozen probe within +/-0.05 of max(k-mer, CNN)
- **positionally structured** — receptive field plausibly matters (splice, TFBS, promoters)
- **category spread** — no more than 2 tasks per GENEB category
- **size** — n_train >= 5k so FT is not data-starved
Target 8-12 tasks x 3 models x 3 seeds.

---

## Track B — viral capability

### Current state, per benchmark

| benchmark | strongest comparator | best gLM | verdict |
|:--|--:|--:|:--|
| ViroBench family (matched 32.7kb ctx) | **alignment 0.7383**, k-mer3-5 0.6231 | NT-v2 0.6148 | **no advantage**; alignment +0.115 over every model |
| ViroBench other 4 levels | k-mer | NT-v2 | no advantage; only family has power for equivalence |
| GUE virus_covid (deduped, n=8050) | k-mer3-6 0.7171 | NT-v2 0.6764 | no advantage (−0.041) |
| GUE virus_species_40 | k-mer 0.4407 | NT-v2 0.4443 | tie (+0.004) |
| HVUE Host_Tropism (strict, n=3391) | CNN ladder 0.8588 | NT-v2 0.8647 | **+0.0059** — the only clean viral positive, tiny |
| HVUE Pathogenecity / Transmissibility | — | — | **UNEVALUABLE**: 96/5194 and 60/4956 homology-clean rows |
| ProteinGym viral (22 assays) | — | — | **NOT RUN** |
| Antibody escape | MSA | — | no advantage (earlier) |

### Why ViroBench is the load-bearing benchmark
It is the only viral benchmark whose split we have verified clean: 2.1-2.2% of test genomes have a
train hit at >=90% identity over >=50% length, confirmed by two independent methods. HVUE is
homology-saturated; GUE viral had 11.9% exact duplicates (now deduped).

### And why its answer is interesting rather than merely negative
Alignment nearest-hit gets **macro-F1 0.7383 / accuracy 0.8431**, and **0.9636 / 0.9915** on the
85% of test genomes it can align at all — while aligning over only **10-13% of query length** at
90-100% identity. That is a conserved gene, not a duplicated genome. **Viral family taxonomy is
determined by short conserved regions that local alignment finds directly.** The split is clean AND
the task is an alignment task. Those are compatible, and together they explain why no foundation
model wins here.

---

## What each track still needs

**Track A (in priority order)**
1. **CNN ladder on the 13 GENEB sentinel tasks** — closes the critical gap above.
2. Frozen probe with **layer swept on dev** on those tasks (our fixed protocol, not `hidden_states[-1]`).
3. Full FT on the selected 8-12 subset, 3 seeds.
4. GENEB's remaining 87 tasks with the fair k-mer.
5. Report median-of-40 published alongside max-of-40.

**Track B**
1. **ProteinGym viral supervised (22 assays)** — only untouched modality; the leaderboard is
   entirely zero-shot, so supervised adaptation is genuinely untested.
2. Alignment baseline at the other 4 ViroBench levels + per-example predictions.
3. ViroBench `genus` split — withholds whole genera, a harder and more meaningful generalisation
   test than temporal. We have one k-mer point (DNA/genus/family 0.9301) and **zero gLM numbers**.
4. Matched-context reruns at the other 4 levels.

**Shared / protocol**
1. Group-disjoint dev for splice and ViroBench (currently a random 15% carve; test is
   chromosome-disjoint / temporal, so selection optimises an easier distribution).
2. Investigate the ViroBench train/test length shift (train median 2,316bp vs test 41,380bp).
3. Per-example predictions from every harness, not just the frozen probe.
