# Paper design — positive and negative controls across a validity grid

The design rests on populating a grid and choosing every task to land in a **known** quadrant. That
is what makes a null informative: a viral null only means "no capability" if the harness is shown to
detect capability elsewhere on comparably honest ground.

## The grid needs a third axis

The original design was 2×2 — {capability present / absent} × {split honest / leaky}. Our own
results forced a third, independent axis:

**{capability present / absent} × {split honest / leaky} × {baseline honest / under-powered}**

A "capability present" verdict can be manufactured three ways, and we have a measured instance of
each:

| failure mode | instance | magnitude |
|:--|:--|--:|
| leaky split | HVUE Pathogenecity: 80.5% of test rows have a train hit ≥90% id / ≥50% cov | 96 of 5,194 rows survive filtering |
| under-powered baseline | splice CNN receptive field 89bp vs 505bp vs global on 600bp input | **+0.62 MCC** |
| wrong comparator class | ViroBench taxonomy scored without an alignment baseline | **+0.115** above every model |

The third axis is where our distinctive contribution sits, and it is not optional: **a positive
control must be clean on all three axes**, or it cannot calibrate anything.

---

## ⚠ Three anchors have moved — read before reusing old numbers

**1. The splice positive control is NOT +0.30 to +0.64 MCC. It is +0.015.**
That gap was measured against a CNN whose receptive field (89–249bp) could not span a 600bp input.
Ladder result (13 cells, 0.04M–9.4M, dev-only selection):

| task | old baseline | **dev-selected baseline** | NT-v2 full FT (3 seeds) | real gap |
|:--|--:|--:|--:|--:|
| splice_all | 0.373 | **0.9528** | 0.9674 ± 0.0025 | **+0.0146** |
| splice_acceptors | 0.619 | **0.9527** | 0.9660 ± 0.0014 | **+0.0133** |
| splice_donors | 0.676 | **0.9637** | 0.9736 ± 0.0038 | **+0.0099** |

The control **still passes** — NT-v2 reproduces published 0.971–0.984 — so it remains the anchor.
But the claim changes from "gLMs beat the baseline by a landslide" to "gLMs beat the baseline
reproducibly by a small, tight margin, and 2 of 3 gLMs *lose* to the CNN." That is a weaker
calibration signal and it makes the third positive control (below) load-bearing rather than nice
to have.

**2. LucaVirus's +0.079 is gone.** Measured −0.279 to −0.294 across layers −1/−2/−3/−5, all
significant. The viral modality has no live exception. It is no longer "the open question that
decides whether the modality is closed" — it is closed.

**3. HVUE identity-disjoint does not carry the safety claim.** `easy-cluster -c 0.9` needs 90%
*bidirectional* coverage and is blind to partial overlap. Pathogenecity and Transmissibility retain
**96 / 5,194** and **60 / 4,956** homology-clean test rows. Only Host_Tropism survives (3,391 rows),
and there NT-v2 **wins** by +0.0059 over the dev-selected CNN. The "18 cells, 0 wins" tally was
computed on splits that cannot support the claim.

---

## Quadrant A — capability present, split honest, baseline honest
*Purpose: prove the harness detects capability. Everything else depends on these.*

| task | status | evidence |
|:--|:--|:--|
| **NT splice (3 tasks)** | **D** | NT-v2 FT 0.9674 ± 0.0025 vs published 0.971–0.984; baseline at its ladder ceiling; 0% exact/revcomp overlap verified |
| **DART-Eval Task 1** | **P** | CNN reproduced within 0.4% — needs formalising as an anchor with the ladder applied |
| **GENEB non-viral (13 categories)** | **! baseline gap** | GENA-LM 11/13, NT-v2 10/13 vs fair k-mer — but **no CNN baseline was ever run**; see action A1 |
| **Long-range expression / variant effect (Borzoi-style)** | **X** | **the required third anchor**; blocked on data access |

**This quadrant is the paper's weakest point, not its strongest.** One anchor with a +0.015 margin,
one partial, one with a missing baseline, one unrun. Calibration currently rests on splice alone.

## Quadrant B — capability may look present, split LEAKY
*Purpose: show inflation is real and measurable; corroborated by outside literature.*

| task | status | evidence |
|:--|:--|:--|
| **EPI, 6 cell lines** | **D** | our baseline beats published EPIPDLF/EPINTLM on 5/6 (+0.007…+0.052 AUROC); promoters **67–80%** and enhancers 39–46% train→test overlap; corroborates BENGI / LOCO-EPI |
| **GUE virus_covid (pre-dedup)** | **D** | 11.9% exact-duplicate test rows; deduped rerun does **not** change the ranking (k-mer 0.7171 > NT-v2 0.6764) |
| **HVUE Path. / Trans.** | **D** | 96/5,194 and 60/4,956 clean rows — promote from "leaky" to **unevaluable** |

Note the EPI number is measured with `easy-cluster`; per PROTOCOL rule 15 the `easy-search` figure
will be **higher**. Re-audit before quoting (action S1).

## Quadrant C — capability absent, split honest, baseline honest
*The viral core.*

| task | strongest comparator | best gLM | verdict |
|:--|--:|--:|:--|
| **ViroBench family** (matched 32.7kb ctx) | **alignment 0.7383**, k-mer3-5 0.6231 | NT-v2 0.6148 | no advantage; **alignment +0.115 over every model** |
| ViroBench kingdom/phylum/class/order | k-mer | NT-v2 | no advantage; only family has power for equivalence |
| GUE virus_covid (deduped, n=8,050) | k-mer3-6 0.7171 | NT-v2 0.6764 | −0.041 |
| GUE virus_species_40 | k-mer 0.4407 | NT-v2 0.4443 | +0.004 tie |
| HVUE Host_Tropism (strict, n=3,391) | CNN 0.8588 | NT-v2 **0.8647** | **+0.0059 — a small positive, report it** |
| ProteinGym viral (22 assays) | — | — | **X not run** |
| Antibody escape (3 antigens) | MSA | — | no advantage; 3 antigens incomplete |

**The ViroBench finding is stronger than "no advantage."** Alignment nearest-hit reaches 0.9915
accuracy on the 85% of test genomes it can align, over only **10–13% of query length** at 90–100%
identity — a conserved gene, not a duplicated genome. Near-duplicates are 2.2%, confirmed by two
independent methods. **Viral family taxonomy is determined by short conserved regions that local
alignment finds directly.** The split is clean *and* the task is an alignment task; both are true,
and together they explain the null mechanistically rather than by absence of evidence.

## Quadrant D — capability absent, split RIGGED against the baseline
*The sharpest single result in the paper.*

| task | status | evidence |
|:--|:--|:--|
| **HVUE composition-cluster-disjoint** | **D** | split built in kmer5-PCA space — the baseline's own feature space — and accepted only if the k-mer degrades ≥0.03 (`GATE = 0.03`). The MMseqs identity-disjoint alternative scored 0.9131, failed the gate, and was logged `VALIDITY mmseqs_identity_disjoint: INVALID`. gLMs "win" **only** on this split. |

## Quadrant E (new) — capability absent, baseline UNDER-POWERED
*The third axis. Same logical role as D, different mechanism.*

| instance | status | evidence |
|:--|:--|:--|
| splice vs weak-CNN baselines | **D** | published +0.31–0.60 → **+0.02–0.03** against a global-RF baseline |
| GENEB reference k-mer | **D** | iDHS-EL MCC **0.000** (majority class) → **0.589** refit with scaling + C sweep + class weighting |
| LucaVirus read-out layer | **D** | between-sequence std 0.0027 at layer 12 vs 0.1438 at layer 11 |
| GENA-LM weight loading | **D** | `AutoModelForSequenceClassification` discards 48 pretrained LayerNorms → exact 0.0000 at every LR |
| ViroBench context mismatch | **D** | frozen probe saw 38.8% of test bp vs unbounded k-mer; fixing it reversed +0.037 → −0.012 |

---

## Action items, ordered by how much they change the paper

### A — close Quadrant A (the calibration risk)
- **A1. CNN ladder on the 13 GENEB sentinel tasks.** *Highest value anywhere.* The non-viral positive
  control is k-mer-anchored; on the one overlapping task our CNN scores 0.9527 vs GENEB's fair k-mer
  0.387 and its best-of-40 published 0.685. If the GENEB wins don't survive a CNN, Quadrant A
  collapses to splice alone and Quadrant C loses its calibration.
  *Caveat to preserve:* GENEB probes frozen, our CNN trains end-to-end — regime mismatch, so the
  output is "do the wins survive," not "CNN beats GENA-LM."
- **A2. Long-range expression / variant effect anchor (Borzoi-style).** Unpark. With splice at +0.015
  this is no longer optional — one small-margin anchor cannot carry calibration. Decide the
  GCP/corral access question.
- **A3. Formalise DART-Eval Task 1** as an anchor: apply the ladder, report per-seed.
- **A4. Re-run GENEB frozen probes with the layer swept on dev.** We read `hidden_states[-1]`
  throughout; that under-read NT-v2 by +0.03 and GENA-LM by +0.02.

### B — complete Quadrant C
- **B1. ProteinGym viral supervised, 22 assays.** Only untouched modality. The 95-scorer leaderboard
  is entirely zero-shot, so supervised adaptation is genuinely untested — the most likely place a
  real viral positive appears. Must include a supervised one-hot + biochemical control.
- **B2. Alignment baseline at the other 4 ViroBench levels**, with per-example predictions.
- **B3. ViroBench `genus` split** — withholds whole genera; harder and more meaningful than temporal.
  One k-mer point (0.9301), **zero gLM numbers**.
- **B4. Matched-context reruns** at the other 4 levels.
- **B5. Escape completion** for the 3 incomplete antigens.

### S — shared / protocol debt
- **S1. Re-audit every benchmark with `easy-search`**, not `easy-cluster`. EPI's 67–80% is a lower
  bound.
- **S2. Group-disjoint dev for splice and ViroBench** (currently a random 15% carve while test is
  chromosome-disjoint / temporal).
- **S3. Explain the ViroBench train/test length shift** — train median 2,316bp vs test 41,380bp, 18×.
- **S4. Per-example predictions from every harness.** `virobench_baselines.py` still doesn't emit them.

### Dropped
- ~~Resolve LucaVirus~~ — closed, −0.28 across 4 layers.
- ~~Confirm ViroBench taxonomic level to validate LucaVirus~~ — moot.
- ~~BLAST/Kraken2 at full spec~~ — done on Cluster 2 and reproduced here via mmseqs nearest-hit.

---

## If we extend to the broad GENEB version

Stratify ~30 GENEB tasks across the quadrants, with the third axis enforced per task:

| bucket | n | purpose |
|:--|--:|:--|
| A — likely k-mer-trivial (species/taxonomy, some promoters) | ~10 | show the k-mer ceiling; these are where a "win" is cheapest to fake |
| B — genuinely hard positive controls (splice, long-range variant, expression) | ~6–8 | calibration; **each needs the CNN ladder**, not just a k-mer |
| C — known-leaky (EPI family) | ~4–6 | inflation showcase |
| D — viral / biosecurity (HVUE, GUE viral, ViroBench) | ~6–8 | the negative core |

Models: DNABERT-2, NT-v2, HyenaDNA, Caduceus, GENA-LM, Evo, Evo 2, ESM-2 — with **k-mer and the CNN
ladder as first-class rows, not footnotes.** That last point is the whole lesson of Quadrant E.

**Honest priority ordering.** The two items that most determine whether the paper is complete are now
**A1** (does the non-viral positive control survive a real baseline?) and **A2** (a second
independent anchor). Resolving LucaVirus was the old top priority and is done. Quadrant C is
substantially populated; Quadrant A is not, and it is the quadrant the whole design depends on.
