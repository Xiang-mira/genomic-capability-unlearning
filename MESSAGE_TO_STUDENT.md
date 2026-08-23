Subject: Where the negative-results paper stands — what we've run, what's left, and what I'd like you to take

Hi — bringing you up to date on the viral-capability programme. The READMEs and older MDs in the
repo are stale in places, so treat this message as the current picture and the three docs below as
the detail. Everything is on `viral-benchmark-continuation`.

- `reports/PAPER_OUTLINE.md` — thesis, section plan, figures, and which claims are defensible today
  vs still gated
- `reports/CROSS_CLUSTER_SYNTHESIS.md` — both clusters reconciled, with the corrections
- `reports/BASELINE_CAPACITY_CEILING.md` — the baseline-strength work, which turned out to matter
  more than expected
- `ANALYTICAL_RESULTS.md` — the full per-task tables, viral and non-viral

---

## 1. The high-level picture

We set out to test whether any viral benchmark gives a genomic foundation model reproducible,
model-specific headroom over the strongest non-pretrained baseline on a defensible split. The answer
so far is no — but the interesting part is *why*, and that's shifted what the paper is about.

Two things now carry more weight than the headline negative:

**Benchmark validity.** The main viral benchmarks are homology-saturated. Measuring *partial*
overlap with MMseqs2 `easy-search` (local alignment) instead of `easy-cluster` — which needs 90%
bidirectional coverage and so misses a test sequence sharing half its length at high identity — two
of the three HVUE tasks retain almost no homology-independent test data once filtered. ViroBench's
temporal split, by contrast, is clean. So ViroBench should carry the viral result, not HVUE.

**Baseline validity.** Most published FM-vs-baseline gaps we checked shrink or vanish against a
properly-built baseline. Three distinct mechanisms, all measured: a CNN whose *receptive field*
can't span the input (the big one), a reference k-mer with no feature scaling, and a probe reading a
collapsed final layer. Numbers in `BASELINE_CAPACITY_CEILING.md`.

So the paper is heading toward: **gLMs do work — and here is a domain where the evidence that they
work doesn't survive contact with its baselines and its splits.** That framing is still under
discussion; the alternative is a narrower "benchmark audit" paper that leads with the validity
findings and treats the negative as a corollary. Worth talking through — I lean toward the first
because the positive controls make it defensible, but the second is easier to referee.

## 2. What we've run — tasks × methods

**Baselines** (the comparator side, which is where most of the work went):
- k-mer3-5 and k-mer3-6 frequency features → standardised logistic regression, C selected on dev
- supervised CNN — now a **13-cell architecture × capacity ladder** (dilated / U-Net / ResNet,
  0.04M–9.4M params), because a single CNN turned out to be an arbitrary point rather than a ceiling
- BLASTn nearest-hit and Kraken2 on ViroBench (alignment/taxonomic comparators)
- always report `max(k-mer, CNN)`, never the k-mer alone

**Models:** NT-v2-500M, GENA-LM-bert-base-t2t, HyenaDNA-medium-160k, LucaVirus, Evo-1-8k.
**Regimes:** frozen probe, LoRA, full fine-tune — kept in separate columns, never mixed, because on
splice the frozen-vs-FT gap alone is 0.59 MCC.

**Benchmarks covered:** HVUE (3 tasks × 3 split families), ViroBench taxonomy (5 levels), GUE viral
(2 tasks), GUE non-viral (12 tasks), NT benchmark (18 tasks incl. splice), EPI (6 cell lines),
DART-Eval task 1, GENEB (13-task sentinel), antibody escape, single-variant effect.

## 3. GUE results specifically

**Viral (2 tasks) — both losses or ties:**

| task | classes | best baseline | best gLM | gLM vs baseline |
|:--|--:|--:|--:|--:|
| `virus_covid` | 9 | k-mer 0.7282 | NT-v2 0.6850 | **−0.043** |
| `virus_species_40` | 25 | k-mer 0.4407 | NT-v2 0.4443 | **+0.004 (tie)** |

Important caveat on `virus_covid`: its official split has **11.9% exact-duplicate test sequences
present verbatim in train**. That inflates every method equally so the ranking holds, but the
absolute numbers shouldn't be quoted. I'm running the deduped version now (8,050 clean test rows,
k-mer + all three gLMs) so we can report a defensible number instead of dropping the task.

**Non-viral (12 tasks) — mostly small gaps, baseline competitive.** Mean gap to best-published
**+0.055**; across GUE + NT combined, **11 of 30 tasks** have our baseline within 0.05 MCC of the
best published gLM or beating it outright. Full table in `ANALYTICAL_RESULTS.md` §4b.2. Caveat we
state ourselves: GUE's split disjointness is unverified, so these are in-distribution comparisons.

## 4. Positive controls — what we have, and where we expect more

This is what stops the paper being "our harness is broken."

**Already in hand:**
- **NT splice, fine-tuned.** NT-v2 reaches **0.9680 MCC** against a published 0.971–0.984. The
  harness reproduces published numbers once the adaptation regime matches. Note that even fully
  fine-tuned, 2 of 3 gLMs *lose* to the properly-sized CNN here — so the control passes while the
  margin over a fair baseline stays small (+0.015).
- **GENEB, 13 categories, per model** against a fairly-tuned k-mer: GENA-LM 11 wins / 0 ties /
  2 losses, NT-v2 10/1/2, HyenaDNA 6/1/6. Two of three models win clearly, so pretraining does buy
  real headroom on non-viral tasks — model-dependently.

**Where I expect further positives to come from, in priority order:**
1. **ProteinGym viral supervised (22 assays).** The entire 95-scorer leaderboard is zero-shot;
   nobody has tested whether *supervised* adaptation on viral DMS beats MSA methods. This is the
   single most likely place a genuine viral positive shows up, and it's completely untouched.
2. **LucaVirus on ViroBench.** A virus-specific pretrained model on a viral benchmark is the most
   plausible source of a real viral advantage. Our earlier negative was our own bug (we read a
   collapsed final layer); a layer sweep is running and I'd treat their published numbers as
   unrefuted until it lands.
3. **GENEB's remaining 87 tasks.** The 13-task sentinel already wins most categories; the full run
   should broaden that and gives the paper a much stronger positive-control leg.
4. **Strict Host_Tropism.** The one HVUE task that survives homology filtering (3,391 clean rows).
   Running now — if a gap appears there it's our only clean viral positive.

## 5. What I'd like you to take

**(a) ProteinGym viral supervised — the highest-value open item.** 22 viral assays under
`/data/nvidia/proteingym/DMS_ProteinGym_substitutions/`. Protocol: ESM-2 650M/3B embeddings →
ridge/LR, position-disjoint `contiguous` and `modulo` splits, Spearman, against published
ESCOTT/GEMME/S3F_MSA **plus** a supervised one-hot + biochemical control (that control is the point
— it's the equivalent of the CNN on the DNA side). Needs a new script; `phase2/proteingym_*.py` has
the data-loading pieces to reuse.

**(b) GENEB remaining 87 tasks**, using `scripts/geneb/` — including the fair-k-mer refit, since
GENEB's shipped reference baseline is degenerate on at least one task.

**(c) Two protocol rules to apply to anything new**, both of which changed conclusions for us:
- Run `scripts/viral_benchmark/capacity_sweep.py` before quoting any baseline number. Selection is
  dev-only; it reports the dev-selected cell and the oracle-best separately, and for HVUE it carves
  a group-disjoint dev so architecture choice faces the same holdout as test.
- Audit every split with `easy-search` partial overlap, not `easy-cluster`. The script is
  `scratchpad/multimodel/partial_overlap_audit.py`. Our EPI numbers make me think the same tool
  difference is in play there and the real overlap is higher than we've reported.

Other scripts worth knowing: `virobench_frozen_probe.py` (has a `--layer` flag now — required for
LucaVirus), `splice_finetune.py` (LR sweep + warmup, and a guard that aborts if a load path silently
re-initialises pretrained weights — worth reusing, it caught a bug where GENA-LM was training with
48 randomly-initialised LayerNorms and scoring exactly 0.0000).

Running here now: LucaVirus layer sweep, strict Host_Tropism, deduped virus_covid, multi-seed splice
FT, GENA-LM LR extension.

Happy to talk through the paper framing whenever — that's the open question, not the evidence.
