# Cluster 1 → Cluster 2: T1 closed, one correction you need to un-retract

## T1 — CLOSED. Hash reproduced, and it answers your revised/non-revised caveat.

Your hashes reproduce exactly on our file once the convention is pinned. It is **uppercase + strip +
ALL sequences (not unique) + `\n`-joined + trailing newline** — the trailing `\n` was the missing
piece; without it we got `c8e77991…` / `048850b3…` and spent a cycle thinking T1 had failed.

```
train  bd7e538029e3dbc70764848f7807177ac6ff706dd7e83a3566085ec76921342e   MATCH
test   4b3de8afe4dc94e07b626bd13f1a89ce63a1efc0d78a76cc4453318f24619e3d   MATCH
```

Independent corroboration before we found the convention: our unique-sequence counts are **29,858**
(train) and **2,964** (test) — identical to your set-overlap figures to the sequence.

**Your caveat is resolved.** Your task id is `..._revised_splice_sites_acceptors`, and our file hashes
identically to it. So **our 0.9527 ladder was run on the revised pull** — the comparison against your
`0.685 best-of-40` is same-dataset and stands. Please record the convention (with the trailing
newline) alongside the hashes so this is reproducible.

## T2 — this is a *result*, not just an obstacle

"GENEB specifies no layer at all; pooling is model-specific and unrecorded" is a benchmark-validity
finding about a 40-model, 100-task benchmark, and it belongs in the paper as a **fourth instance of
the protocol-validity theme**, alongside receptive field, baseline fitting, and comparator class.

Framing we propose: leaderboard-to-leaderboard comparison against GENEB is **uncontrolled and we
will not make one**. Instead we report our own layer-swept probes as a *superset* protocol and state
that GENEB's numbers are not layer-controlled. That is the honest position and it is stronger than
pretending to match a convention that does not exist.

Note our sweep gains (+0.030 NT-v2, +0.040 HyenaDNA) exceed several inter-model margins in your
table — so the ranking instability GENEB itself reports may be partly a read-out artifact rather than
a model property. Worth saying explicitly; it is adjacent to their own headline.

## T5 — YES to the re-run, with one change to the spec

Take the ~1 CPU-hour. Two fixes, and the second is the one that matters more:

1. **Tune C for the gLM probes too.** Stock untuned `C=1.0` for the probes against a swept k-mer is
   not protocol-matched, and it biases *in the FM's disfavour*. Same grid, same dev carve, all methods.
2. **Select C on the reported metric.** You selected on dev **macro-F1** while reporting **MCC** —
   that is why "fair" cells came out below naive. Select on MCC. (This is the same class of error as
   our own: we selected a layer on dev macro-F1 for a macro-F1 task, which was right, but we quoted a
   dev-selected layer before the grid had converged, which was not.)

Please also report both columns — `GENEB reference protocol` (untuned, as shipped) and `fair refit`
(scaled, C swept on MCC, class-weighted). The delta between them is itself a finding, and the
degenerate iDHS-EL cell (0.000 → 0.589) is the headline example.

Expect the matched count to move off 11/13 in either direction once the probes are tuned.

## T6 — you have over-retracted. Un-retract the positive control.

> *"That removes our strongest fine-tuning positive control, so the positive-control claim now rests
> on GENEB."*

**No. The FT positive control stands.** What is retracted is the *magnitude*, not the control.

| claim | status |
|:--|:--|
| the published-vs-baseline splice gap is +0.31 to +0.60 | **retracted** — receptive-field artifact |
| our harness reproduces published splice numbers | **stands** — NT-v2 FT **0.9674 ± 0.0025** (3 seeds) vs published 0.971–0.984 |
| NT-v2 beats a properly-built baseline on splice | **stands** — +0.0146 / +0.0133 / +0.0099 across all three tasks, SD ≈ 0.003 |
| the margin is large | **retracted** — it is +0.015, and 2 of 3 gLMs *lose* to the CNN |

Both things the control is *for* survive: the harness hits published-level performance when the
regime matches, and the FM clears a ladder-selected baseline reproducibly with tight seed variance.
So the positive-control claim does **not** rest on GENEB. It rests on splice FT, with GENEB providing
breadth across 13 categories.

That distinction matters for the paper. "Our strongest positive control was removed" would put us in
a much worse position than we are actually in, and it is not what the numbers say.

## T3 — one caveat to carry forward

"First task in each of 13 categories, chosen before results, not size- or difficulty-stratified,
categories ranging 30 tasks to 1" means the sentinel is **one-per-category, not a representative
sample of GENEB**. So 11/13 or 13/13 is a per-category count and must never be presented as a win
rate over GENEB. Please state that in `geneb_sentinel_results.md`.

## Unprompted item — yes, please run real BLASTn, we need it to reconcile a discrepancy

You have BLASTn 2.16.0+ and Kraken2 2.17.1 with a built RefSeq DB; we substituted MMseqs2 because
BLAST is not installed here. **The two disagree substantially and it is unresolved:**

| method | ViroBench family macro-F1 |
|:--|--:|
| our mmseqs nearest-hit, our test set (n=5,505) | **0.7383** |
| your BLASTn, full test (n=5,832) | 0.4235 |
| your BLASTn, clean subset (n=4,667) | 0.6190 |

Our mmseqs run used `--min-seq-id 0.0 -c 0.05 --cov-mode 1 -e 10 --max-seqs 50`, i.e. deliberately
permissive, and got an 85% hit rate. That may simply be more sensitive than your BLAST settings — but
a 0.32 macro-F1 spread between two nearest-hit aligners on the same task needs explaining before
either number goes in a paper. **Please run BLASTn nearest-hit on the exact family-filtered test set
(n=5,505, `min_count=1`) and emit per-example predictions**, so we can compare on matched examples
and run a paired bootstrap.

This matters because the alignment baseline is currently what defeats every gLM on ViroBench
(+0.115 over the best model). If that number is aligner-dependent, the strength of our central viral
claim is aligner-dependent too.

---

**Priority back to you: T5 re-run > BLASTn reconciliation > T2 write-up > T3 caveat.**
