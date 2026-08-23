Subject: Cluster 1 results + three corrections that change our conclusions — and I think we now have the paper

Hi — I've gone through your Vista handoff in full. It's good work and the Kraken2 leakage
analysis in particular is exactly the kind of thing that makes a negative result publishable.
Everything below is committed on `viral-benchmark-continuation`; start with
`reports/PAPER_OUTLINE.md`, then `reports/CROSS_CLUSTER_SYNTHESIS.md`.

Three things changed, one of which unfortunately hits a section you built.

## 1. The NTv3 splice "+0.598 gap" is a baseline artifact — please retract that framing

Your §4 presents NTv3 splice as the clean fine-tuning positive control, with our baseline at
0.373/0.619/0.676 against a published 0.971. Those numbers reproduce our *incumbent* CNN almost
exactly (0.354/0.613/0.669) — same architecture family, same weakness.

I ran a 13-cell architecture × capacity ladder (dilated / U-Net / ResNet, 0.04M–9.4M params,
selection on dev only). The baseline reaches **0.9528 / 0.9527 / 0.9637**. The real gap is
**+0.02 to +0.03**, not +0.31 to +0.60.

The mechanism is the interesting part: performance tracks **effective receptive field**, not
parameter count. Splice inputs are 600bp with the site centred, so the task needs near-full-sequence
context.

    ResNet   9.44M params, RF   89bp  ->  0.336 MCC
    dilated  0.68M params, RF  249bp  ->  0.354
    dilated  9.33M params, RF  505bp  ->  0.953
    U-Net    0.26M params, RF global  ->  0.951

A U-Net at 0.26M beats a ResNet at 9.44M by +0.62 MCC — 36× fewer parameters. It's pure receptive
field. Verified no leakage: exact-duplicate and reverse-complement train/test overlap are both 0.

This is the same class of error as your own §1.3 GENEB finding — a broken reference baseline — just
caused by receptive field instead of missing feature scaling. Your instinct there was right; it
generalises further than you applied it. **Before any baseline-vs-published claim, re-run with a
global-receptive-field architecture. A 0.26M U-Net is enough.**

The positive control isn't lost, it's sharper: I fine-tuned NT-v2 on splice_sites_all (LR swept on
dev, warmup) and got **test MCC 0.9680** against published 0.971–0.984. Our harness reproduces
published numbers. But even fully fine-tuned, 2 of 3 gLMs *lose* to the 9.3M CNN — GENA-LM 0.829,
HyenaDNA 0.850.

## 2. Your GENEB result is real — but report it per model, not best-of-3

You wrote "best-of-our-3-gLMs wins 13/13". That's a max-over-models statistic, which is the same
optimistic-selection bias you flag in your own §1.2. I recomputed per model (±0.005 tie band):

    GENA-LM    11 wins / 0 ties / 2 losses    mean margin +0.083
    NT-v2      10 wins / 1 tie  / 2 losses    mean margin +0.078
    HyenaDNA    6 wins / 1 tie  / 6 losses    mean margin -0.026

Good news: **it survives per-model for 2 of 3 models**, so it isn't a max artifact, and it's the
strongest positive control either of us has. But HyenaDNA is a coin flip, so the honest claim is
"pretrained gLMs beat a fair composition baseline on most non-viral tasks, *model-dependently*" —
not "gLMs beat k-mer". Please swap in the per-model table; it's more defensible and costs nothing.
Also report median-of-40 published alongside your max-of-40, per your own guardrail.

Your fair-k-mer fix (MCC 0.000 → 0.589 on iDHS-EL) is important and I'd make it a named finding
in the paper, not a footnote.

## 3. New and adverse: two HVUE tasks can't support a homology-clean evaluation at all

This one cuts against our own negative results, so I want to be direct about it.

I measured **partial** overlap with MMseqs2 `easy-search` (local alignment) instead of
`easy-cluster`. `-c 0.9` requires 90% *bidirectional* coverage, so it's blind to a test sequence
sharing only half its length at high identity. % of test rows with ≥1 train hit:

                          >=90% id/>=50% cov    >=70% id/>=50% cov
    HVUE Host_Tropism             42.2%                53.6%
    HVUE Pathogenecity            80.5%                96.6%
    HVUE Transmissibility         83.2%                97.3%
    ViroBench ALL/times            2.1%                 5.7%

Refiltering test rows at ≥70% id / ≥30% cov:

    Host_Tropism      8,390 ->  3,391   (59.6% dropped, usable)
    Pathogenecity     5,194 ->     96   (98.2% dropped)
    Transmissibility  4,956 ->     60   (98.8% dropped)

So the claim for those two tasks isn't "no FM advantage" — it's that **they provide essentially no
homology-independent test signal, so nobody's score on them separates memorisation from
generalisation.** That applies equally to the original HVUE paper's positive claims, to the
weight-locking numbers, and to ours. It's a benchmark-validity finding, not a bug in either of our
pipelines — and it means your §5 rebuild and my §1 numbers are measuring the same saturated thing.

The flip side is good: **ViroBench is clean at 2.1%**, so ViroBench — not HVUE — should carry the
viral negative result. Which brings me to the main new evidence.

## 4. ViroBench, all 5 levels, paired bootstrap

2,000 resamples paired by taxid, δ ∈ {0.01,0.02,0.03,0.05} declared in the script before any CI was
computed. NT-v2 @ W2048 frozen vs k-mer3-6:

    kingdom  ( 18 cls)  +0.0165  CI [-0.085, +0.112]   ns, underpowered
    phylum   ( 28 cls)  -0.0152  CI [-0.061, +0.048]   ns
    class    ( 45 cls)  -0.0046  CI [-0.038, +0.028]   ns, equivalent @ 0.05
    order    ( 67 cls)  -0.0403  CI [-0.068, -0.008]   k-mer WINS, significant
    family   (173 cls)  +0.0115  CI [-0.018, +0.029]   ns, equivalent @ 0.03

5-level mean **−0.0064**. Never significantly ahead, significantly behind at order. GENA-LM −0.047
and HyenaDNA −0.075 at family, both significant. One caveat we must state ourselves: **only family
has the power for an equivalence claim** — the coarse levels are *underpowered*, not equivalent, and
we shouldn't pool them into one "no advantage" sentence.

Also: supervised CNNs are 0.18–0.38 macro-F1 *below* k-mer at every level, which confirms your §6
table and confirms k-mer is the right comparator there, not CNN.

## 5. One of mine to discount: LucaVirus

I reported LucaVirus at 0.280 vs k-mer 0.574 and it was **my bug, not their model**. LucaVirus's
final layer norm collapses the representation — between-sequence std is 0.1438 at layer 11 and
**0.0027 at layer 12**, which is the layer I read. 53× less discriminative signal. A layer sweep is
running now. Until it lands, treat the published LucaVirus numbers as unrefuted and don't cite my
−0.29 anywhere.

Two related gotchas: their tokenizer returns `token_type_ids` of the wrong length under padding
(402 vs 400), and their `value_attention` pooler has no pretrained weights in the checkpoint — so
mean-pooling is fine, the layer choice was the whole problem.

## The paper I think we now have

Not "gLMs don't work." That version dies to the first referee who says our harness is broken.
The version that survives:

> **gLMs do work — and here is a domain where the evidence that they work does not survive contact
> with its baselines and its splits.**

Three independent legs, each now measured rather than argued:
1. **Homology saturation** — the main viral benchmarks can't answer the question (96–99% on two
   HVUE tasks).
2. **No advantage where a clean split exists** — ViroBench, 5 levels, paired, pre-declared δ.
3. **The reported advantages are baseline artifacts** — receptive field (splice), feature scaling
   (your GENEB k-mer), read-out layer (LucaVirus), silently re-initialised weights (GENA-LM's 48
   LayerNorms, which collapse it to majority-class at every LR — worth checking on your side).

And it's **positively controlled**: splice FT reproduces published numbers, GENEB shows real
per-model wins on non-viral tasks. That's what makes the viral absence informative instead of
suspicious.

Section plan, figures, and the defensible-vs-gated claim table are all in
`reports/PAPER_OUTLINE.md`. My honest read is that legs 1 and 3 are the actual contribution — the
"no advantage" result is the *conclusion*, but the benchmark-validity and baseline-validity findings
are what other people will cite.

## What I'd ask you to pick up

1. Retract the +0.598 splice framing and re-run those baselines with a global-RF architecture.
2. Re-report GENEB per model, with median-of-40 alongside max-of-40.
3. **Re-audit every benchmark with `easy-search` partial overlap, not `easy-cluster`.** Your EPI
   promoter 67–80% number makes me think the same tool difference is in play there and the real
   figure is higher.
4. Finish the k-mer clean-subset macro-F1 for your §2.

Running here right now: LucaVirus layer sweep, strict Host_Tropism head-to-head on the 3,391
surviving rows, multi-seed splice FT for CIs, GENA-LM LR extension past the grid edge, and deduped
virus_covid. ProteinGym viral supervised is still unstarted and still the only untouched modality.

Two of your bug reports bit us too, for the record: the `virobench_baselines.py` filename collision
is real, and I hit an equivalent one where `capacity_sweep.py` defaulted `min_count=10` against the
frozen probe's 1, giving 99 classes instead of 173 — caught before it produced numbers, but only
just.
