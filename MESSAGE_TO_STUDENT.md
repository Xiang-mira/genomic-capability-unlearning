Subject: Results from both clusters — and why the forget-task selection needs redoing

Hi — I've finished the negative-results programme across both clusters (Cluster 1 on the A100 box,
Cluster 2 on Vista). Writing this up because it lands directly on your unlearning task-selection
analysis from July, and the news there is not good: **the excess-capability numbers that
`recommended_forget_tasks.md` is built on don't survive.**

Everything is on `viral-benchmark-continuation`. Read `reports/PAPER_OUTLINE.md` first, then
`reports/CROSS_CLUSTER_SYNTHESIS.md`.

---

## 1. Your forget-task selection: three independent problems

Your table classifies HVUE Pathogenicity as the PRIMARY_FORGET candidate on **excess = +0.098**
(model 0.9722 − k-mer 0.8738), Host_Tropism as SECONDARY at +0.055, Transmissibility as
DIAGNOSTIC_ONLY at +0.037. Each of those rests on a comparison that I can now show is wrong in
three separate ways.

### (a) The baseline was a k-mer, not the strongest non-pretrained model

A supervised CNN beats the k-mer on all of these tasks. I ran a 13-cell architecture × capacity
ladder (dilated / U-Net / ResNet, 0.04M–9.4M params, selection on dev only):

| task | your model | your k-mer | **your excess** | our CNN | **excess vs CNN** |
|:--|--:|--:|--:|--:|--:|
| Pathogenicity | 0.9722 | 0.8738 | **+0.098** | **0.9718** | **+0.0004** |
| Host_Tropism | 0.9440 | 0.8894 | **+0.055** | **0.9486** | **−0.0046** |
| Transmissibility | 0.9523 | 0.9157 | **+0.037** | 0.9066 | **+0.0457** |

Your PRIMARY_FORGET candidate's excess goes from +0.098 to **+0.0004**. Host_Tropism goes negative.

**Caveat, stated honestly:** these are not the same splits — your numbers are on the
composition-cluster aggregate, mine on identity-disjoint hsd0 — so this is not a strict
like-for-like and I'm not claiming it as one. But a swing from +0.098 to +0.0004 is far too large
to be split noise. And note Transmissibility goes *up*, because it's the one task where our CNN is
weakest; I'm reporting that rather than only the convenient direction.

### (b) The split those numbers were measured on was gated to make the k-mer look bad

This is the one that bothers me most. `build_splits_v2.py` builds the cluster-disjoint split in
**kmer5-PCA space** — the baseline's own feature space — and then accepts a candidate split only if
`kmer_excess_auroc <= kmer_random_auroc - 0.03` (`GATE = 0.03`). It keeps redrawing until the k-mer
degrades by at least 0.03. The MMseqs2 identity-disjoint alternative was generated, scored 0.9131,
failed that gate, and was logged `VALIDITY mmseqs_identity_disjoint: INVALID`.

So the k-mer ceiling on that split is depressed **by construction**, and every excess computed
against it is inflated by construction. This isn't a criticism of your analysis — you inherited the
split — but it means the excess column can't be used. Details in `SPLIT_DESIGN_EXPLAINED.md`.

### (c) Two of the three tasks have almost no homology-independent test data

I measured **partial** overlap with MMseqs2 `easy-search` (local alignment) rather than
`easy-cluster`. `-c 0.9` requires 90% *bidirectional* coverage, so it's blind to a test sequence
sharing only half its length with a training sequence at high identity. % of test rows with ≥1
train hit at ≥90% identity over ≥50% of their length:

    HVUE Host_Tropism        42.2%
    HVUE Pathogenicity       80.5%
    HVUE Transmissibility    83.2%
    ViroBench ALL/times       2.1%     <- clean, for contrast

Refiltering the test set at ≥70% id / ≥30% cov:

    Host_Tropism      8,390 ->  3,391 rows   (usable)
    Pathogenicity     5,194 ->     96 rows   (98.2% dropped)
    Transmissibility  4,956 ->     60 rows   (98.8% dropped)

**HVUE Pathogenicity and Transmissibility cannot support a homology-clean evaluation at all.** Not
by us, not by the HVUE authors, not by anyone. Whatever "capability" is measured on them is measured
almost entirely on near-duplicates of training data — so it cannot be distinguished from
memorisation, which is exactly the distinction your protocol §1 sets out to make.

### What this means concretely

Your doc already says *"PRIMARY_FORGET — None Confirmed Yet"* and lists promotion criteria. You were
right to hold. Here are the answers to those criteria:

- [x] *5000-step adversary ≥ best_shortcut + 0.03* — the adversary validity framing was the right
      call, but `best_shortcut` needs to be `max(k-mer, CNN)`, and against that the margin is +0.0004.
- [x] *Same result under 3+ seeds* — single-seed HVUE "wins" don't survive 3-seed averaging. Our one
      apparent win (+0.0023 single-seed) collapses to +0.0003.
- [x] *Same result under genus-disjoint split* — stronger version: even identity-disjoint isn't
      enough, because clustering misses partial overlap. Needs `easy-search` filtering.
- [ ] *Unlearning adversary beats k-mer* — moot for Path./Trans., since no clean test set exists.

**Net: there is currently no valid PRIMARY_FORGET task among the HVUE tasks.** Host_Tropism is the
only one where a clean evaluation is even possible (3,391 rows), and I'm running that head-to-head
now.

I want to be clear that this is a finding about the benchmarks, not about your analysis. Your
adversary-validity framing — "an excess number is meaningless if the adversary never reached the
shortcut ceiling" — is the correct instinct and it's what led me to check the baseline side properly.

---

## 2. What the negative-results programme established

**Viral: no detectable model-specific advantage, on the one benchmark that can support the claim.**
ViroBench ALL/times, frozen probe (their protocol), 5 taxonomic levels, paired bootstrap over 5,505
shared test genomes, δ ∈ {0.01,0.02,0.03,0.05} declared before any CI was computed:

    kingdom  ( 18 cls)  +0.0165  CI [-0.085, +0.112]   ns, underpowered
    phylum   ( 28 cls)  -0.0152  CI [-0.061, +0.048]   ns
    class    ( 45 cls)  -0.0046  CI [-0.038, +0.028]   ns, equivalent @ 0.05
    order    ( 67 cls)  -0.0403  CI [-0.068, -0.008]   k-mer WINS, significant
    family   (173 cls)  +0.0115  CI [-0.018, +0.029]   ns, equivalent @ 0.03

NT-v2 5-level mean **−0.0064** — never significantly ahead, significantly behind at order. GENA-LM
−0.047, HyenaDNA −0.075 at family, both significant. Caveat we state ourselves: only family has the
power for an equivalence claim; the coarse levels are *underpowered*, not equivalent.

Kraken2 appears to beat our k-mer (0.640 vs 0.570) but that's reference leakage — 15.4% of test
taxids are verbatim in its RefSeq viral DB; it scores 0.944 on that slice and **0.535** on the clean
84.6%, below our k-mer. BLAST shows no such pattern, confirming it's a fair train-only comparator.

**Non-viral: the models genuinely do work.** This is what makes the viral result publishable rather
than suspicious. On 13 GENEB categories against a fairly-tuned k-mer (per model, ±0.005 tie band):

    GENA-LM    11 wins / 0 ties / 2 losses    mean margin +0.083
    NT-v2      10 wins / 1 tie  / 2 losses    mean margin +0.078
    HyenaDNA    6 wins / 1 tie  / 6 losses    mean margin -0.026

And fine-tuned NT-v2 on NT splice reaches **0.9680 MCC** against a published 0.971–0.984 — the
harness reproduces published numbers when the regime matches.

**The reported advantages elsewhere are largely baseline artifacts.** Three distinct mechanisms,
each measured:

1. **Receptive field.** On 600bp splice, ResNet at 9.44M params (RF 89bp) scores 0.336 MCC while a
   U-Net at 0.26M (global RF) scores 0.951 — 36× fewer parameters, +0.62 MCC. Published FM-vs-CNN
   splice gaps of +0.31–0.60 shrink to **+0.02–0.03** against a baseline that can see the whole
   input. Our own incumbent 0.68M dilated CNN was at ceiling on HVUE (≤+0.006 from the full search)
   but badly under-powered on splice — so this cuts both ways and had to be measured per task.
2. **Feature scaling.** GENEB's shipped reference k-mer has no scaling, no C tuning, no class
   weighting; on iDHS-EL it predicts the majority class 100% of the time (MCC 0.000). Refit fairly:
   **0.589**.
3. **Read-out layer.** I had LucaVirus at 0.280 vs k-mer 0.574 and that was **my bug** — its final
   layer norm collapses the representation (between-sequence std 0.1438 at layer 11 vs **0.0027** at
   layer 12, which is the layer I read). A layer sweep is running; until it lands, treat the
   published LucaVirus numbers as unrefuted.

Plus one infrastructure bug worth knowing: `AutoModelForSequenceClassification` on GENA-LM silently
discards **all 48 pretrained LayerNorms** (pre-LN checkpoint, post-LN HF class) and the model then
collapses to the majority class at every LR — dev MCC exactly 0.0000. Frozen probes via `AutoModel`
are unaffected. If you ever see an exact-zero metric, check for re-initialised weights before
concluding anything about capability.

---

## 3. The paper

The framing I want is **not** "gLMs don't work" — that dies to the first referee who says our
harness is broken. It's:

> **gLMs do work, and here is a domain where the evidence that they work does not survive contact
> with its baselines and its splits.**

Three legs, each measured rather than argued: benchmark validity (homology saturation), no advantage
where a clean split exists (ViroBench), and reported advantages as baseline artifacts (receptive
field / feature scaling / read-out layer). Positively controlled by splice FT and GENEB.

My honest read: legs 1 and 3 are the actual contribution. "No advantage" is the *conclusion*, but
the benchmark-validity and baseline-validity findings are what other people will cite. Full section
plan, figure list, and a defensible-now vs still-gated claim table are in `reports/PAPER_OUTLINE.md`.

**For the unlearning thread specifically**, the implication is uncomfortable but clean: you can't
demonstrate unlearning of a capability that can't be shown to exist on a clean split. Either
(a) we move the unlearning target to a task with demonstrated clean capability — the non-viral
GENEB/splice tasks are the obvious candidates, since those genuinely do show model-specific
headroom, or (b) the unlearning contribution becomes a *methods* contribution about how to validate
a forget target in the first place. Your adversary-validity criteria are most of the way to (b)
already, and I think that's the more interesting paper.

---

## 4. What I'd like you to pick up

1. **Re-derive the forget/retain classification** with `best_shortcut = max(k-mer, dev-selected CNN)`
   and on `easy-search`-filtered splits. My guess is nothing survives on HVUE and the honest output
   is "no valid forget target in this benchmark family" — which is itself a result.
2. **Apply the same audit to the retain set.** `retain.csv` draws 2,000 Coronaviridae and 2,000
   Orthomyxoviridae rows; if those overlap the forget set at high identity, retain-set protection is
   partly protecting the thing we're erasing. Nobody has measured that.
3. **ProteinGym viral supervised** — 22 assays, still the only untouched modality, and the one place
   a genuine viral capability might show up (the whole 95-scorer leaderboard is zero-shot).

Running here now: LucaVirus layer sweep, strict Host_Tropism head-to-head on the 3,391 clean rows,
multi-seed splice FT for CIs on the +0.0152, GENA-LM LR extension past the grid edge, and deduped
virus_covid.

Two things I got wrong along the way, for the record: I reported LucaVirus as a large negative when
it was my layer choice, and I called the splice positive control a *failure* before realising our own
CNN baseline couldn't see the whole input. Both are in the synthesis doc with the corrections.
