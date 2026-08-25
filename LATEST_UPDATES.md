Subject: Paper design as a validity grid — plus three anchors that moved, and the two runs that decide completeness

Hi — I've reframed everything around the controls grid, which is the right spine for this paper.
Docs on `viral-benchmark-continuation`:

- **`reports/PAPER_DESIGN.md`** — the grid, what's in each quadrant, and ordered action items. Start here.
- **`reports/PROTOCOL.md`** — 27 binding rules; each names the violation *in this project* that motivated it.
- **`reports/TESTED_MATRIX.md`** — benchmark × method × regime: done / partial / invalid, and claims withdrawn.
- `reports/RESEARCH_PLAN.md` — the operational view (two tracks sharing one harness).

## The grid needs a third axis

Design was {capability present / absent} × {split honest / leaky}. Our results forced a third:

**{capability present / absent} × {split honest / leaky} × {baseline honest / under-powered}**

A "capability present" verdict can be manufactured three ways, and we have a measured instance of each:

| failure mode | instance | magnitude |
|:--|:--|--:|
| leaky split | HVUE Pathogenecity: 80.5% of test rows have a train hit ≥90% id / ≥50% cov | 96 of 5,194 survive |
| under-powered baseline | splice CNN receptive field 89 → 505 → global on a 600bp input | **+0.62 MCC** |
| wrong comparator class | ViroBench taxonomy with no alignment baseline | **+0.115** over every model |

That third axis is our distinctive contribution, and it isn't optional: **a positive control has to
be clean on all three axes or it can't calibrate anything.**

## ⚠ Three anchors moved — please don't reuse the old numbers

**1. The splice positive control is +0.015, not +0.30–0.64.** That gap was measured against a CNN
whose receptive field (89–249bp) couldn't span a 600bp input. With the ladder (13 cells, 0.04M–9.4M,
dev-only selection):

| task | old baseline | dev-selected baseline | NT-v2 full FT (3 seeds) | real gap |
|:--|--:|--:|--:|--:|
| splice_all | 0.373 | **0.9528** | 0.9674 ± 0.0025 | **+0.0146** |
| acceptors | 0.619 | **0.9527** | 0.9660 ± 0.0014 | **+0.0133** |
| donors | 0.676 | **0.9637** | 0.9736 ± 0.0038 | **+0.0099** |

The control **still passes** — NT-v2 reproduces the published 0.971–0.984. But the claim changes from
"gLMs win by a landslide" to "gLMs win reproducibly by a small tight margin, and 2 of 3 gLMs *lose*
to the CNN." That's a weaker calibration signal, which is why the third anchor below is now
load-bearing rather than nice-to-have.

**2. LucaVirus's +0.079 is gone.** −0.279 to −0.294 across layers −1/−2/−3/−5, all significant. It
was the one live viral positive and the thing you'd flagged as deciding whether the modality is
closed. It's closed. (My earlier −0.29 was partly my own bug — a collapsed final layer — but fixing
the layer only recovered +0.014 of it.)

**3. HVUE identity-disjoint doesn't carry the safety claim.** `easy-cluster -c 0.9` needs 90%
*bidirectional* coverage and is blind to partial overlap. Pathogenecity and Transmissibility retain
**96/5,194** and **60/4,956** homology-clean test rows. Only Host_Tropism survives (3,391 rows) — and
there NT-v2 **wins** by +0.0059. So the "18 cells, 0 wins" tally was computed on splits that can't
support the claim; two of those tasks move from "negative result" to **unevaluable**.

## Where the quadrants actually stand

**Quadrant A (capability present, honest split, honest baseline) is the paper's weakest point, not
its strongest.** One anchor at +0.015 (splice), one partial (DART-Eval, reproduced within 0.4% but
not laddered), one with a **missing baseline** (GENEB — no CNN was ever run), one unrun (long-range
expression). Calibration currently rests on splice alone.

**Quadrant B (leaky splits)** is solid — EPI 5/6 with 67–80% promoter overlap corroborating
BENGI/LOCO-EPI; virus_covid's 11.9% duplicates (deduped rerun doesn't change the ranking). Note the
EPI figure used `easy-cluster`, so the honest `easy-search` number will be **higher**.

**Quadrant C (viral core)** is substantially populated and the ViroBench result is better than "no
advantage": alignment nearest-hit hits **0.9915 accuracy** on the 85% of test genomes it can align,
over only **10–13% of query length** at high identity — a conserved gene, not a duplicated genome.
Near-duplicates are 2.2%, confirmed twice. So **viral family taxonomy is determined by short
conserved regions local alignment finds directly.** The split is clean *and* the task is an alignment
task; both true, and together they explain the null mechanistically instead of by absence of evidence.

**Quadrant D (rigged split)** stands and is still the sharpest single result — gLMs "win" only on the
split built in kmer5-PCA space and gated to accept only when the k-mer drops ≥0.03.

## The two runs that decide whether the paper is complete

Your old top priority was resolving LucaVirus. That's done, so the ordering has changed:

**A1 — CNN ladder on the 13 GENEB sentinel tasks.** Highest value anywhere. The non-viral positive
control is k-mer-anchored; on the one overlapping task our CNN scores **0.9527** where GENEB's fair
k-mer scores 0.387 and its best-of-40 published scores 0.685. If those wins don't survive a CNN,
Quadrant A collapses to splice alone and Quadrant C loses its calibration.

```bash
cd scripts/common
python capacity_sweep.py --dataset <geneb> --task <task> --seeds 42 43 --epochs 30
```

You'll need to add a GENEB loader to its `load()` — splice and HVUE are the templates. Please also
re-run the GENEB frozen probes with the layer swept on dev; we read `hidden_states[-1]` throughout,
which under-read NT-v2 by +0.03 and GENA-LM by +0.02. **Keep the caveat in the output:** GENEB probes
frozen and our CNN trains end-to-end, so the deliverable is "do the wins survive," not "CNN beats
GENA-LM."

**A2 — the long-range expression / variant-effect anchor (Borzoi-style).** Unpark it. With splice at
+0.015, one small-margin anchor can't carry calibration, and long-range regulatory prediction is
where FMs beat local baselines for a mechanistic reason. The blocker is the GCP/corral access
decision — worth resolving now rather than later.

Then, in order: **B1** ProteinGym viral supervised (22 assays, the only untouched modality — the
whole 95-scorer leaderboard is zero-shot, so supervised adaptation is genuinely untested; include a
one-hot + biochemical control, it's the pLM equivalent of the CNN), **B3** ViroBench's `genus` split
(withholds whole genera, harder than temporal; we have one k-mer point at 0.9301 and zero gLM
numbers), and **S1** re-auditing every benchmark with `easy-search`.

## Layout

`scripts/common/` is the shared harness for both tracks — `capacity_sweep.py`,
`partial_overlap_audit.py`, `build_strict_splits.py`, `paired_bootstrap.py`.
`scripts/track_a_benchmarks/` and `scripts/track_b_viral/` hold the benchmark runners. Everything
resolves from `VB_*` env vars.

Two loading traps that cost me a day. `AutoModelForSequenceClassification` on GENA-LM silently
discards all 48 pretrained LayerNorms (pre-LN checkpoint, post-LN HF class) and the model then scores
exactly 0.0000 at every LR — use `AutoModel` plus your own head and keep
`assert_no_fresh_encoder_weights()`. And if you ever see an exact-zero metric, check for
re-initialised weights before concluding anything about capability.

Happy to talk through whether this is one paper with two legs or an audit paper followed by the
viral one. That's the genuinely open question — the evidence is in reasonable shape.
