Subject: Repo reorganised into two tracks — plus the one gap that decides whether the paper holds

Hi — I've restructured the repo around what turned out to be two overlapping programmes rather than
one. Everything is on `viral-benchmark-continuation`. Three docs carry the state; the old READMEs
were stale and I've replaced them.

- **`reports/RESEARCH_PLAN.md`** — the two tracks, current status, and what each still needs
- **`reports/PROTOCOL.md`** — 27 binding rules. Each one is there because violating it changed a
  conclusion in this project, and the violation is named next to the rule. Worth reading before you
  run anything, because most of them cost us a redo.
- **`reports/TESTED_MATRIX.md`** — benchmark × method × regime: what's done, partial, or invalid;
  which splits license which claims; and the list of claims we've withdrawn.

## The two tracks

**Track A — benchmark & method audit.** For existing benchmarks (GENEB, GUE, NT, EPI, DART-Eval),
what does each *method class* actually achieve under one protocol: k-mer, CNN (architecture ×
capacity swept), gLM/pLM frozen probe, LoRA, full FT? Contribution is methodological — published
FM-vs-baseline gaps are largely baseline artifacts.

**Track B — viral capability.** Is there model-specific viral capability over the *strongest*
comparator on a defensible split? HVUE, ViroBench, GUE viral, ProteinGym viral, escape. Adds
alignment and MSA comparators.

They share the harness, the baseline ladder, the split auditing and the protocol. And they depend on
each other: **Track B's negative result is only credible because Track A shows the same harness
detects real capability elsewhere.** That's the load-bearing relationship, and it's exactly what's
now at risk — see below.

## What Track A has established

Three independent instances of a published gap collapsing against a properly-built baseline, each
with a mechanism rather than a hand-wave:

1. **Receptive field, not capacity.** On 600bp splice, ResNet at 9.44M params (RF 89bp) scores
   0.336 MCC; a U-Net at 0.26M (global RF) scores 0.951. 36× fewer parameters, +0.62 MCC. Published
   splice gaps of +0.31–0.60 shrink to **+0.02–0.03**. No leakage — exact and revcomp overlap both 0.
2. **Feature scaling.** GENEB's shipped reference k-mer predicts the majority class on iDHS-EL
   (MCC 0.000); refit fairly it's 0.589. That was your find and it generalises further than we applied it.
3. **Read-out layer.** LucaVirus's final layer norm collapses its representation — between-sequence
   std 0.0027 at layer 12 vs 0.1438 at layer 11.

Plus the positive control: NT-v2 fine-tuned on NT splice hits **0.9674 ± 0.0025** (3 seeds) against
a published 0.971–0.984, so the harness reproduces published numbers when the regime matches.

## The gap that decides whether any of this holds

**No CNN baseline exists for any GENEB task.** The whole sentinel is k-mer-anchored. On the one task
where we have both:

| NT splice acceptors | MCC |
|:--|--:|
| GENEB fair k-mer | 0.387 |
| GENEB GENA-LM — the "+0.156 win" | 0.543 |
| GENEB best-of-40 published | 0.685 |
| **our CNN ladder, dev-selected 9.3M** | **0.9527** |

So the entire non-viral positive control rests on the weaker of our two baselines, and we've already
proven that baseline is weak on exactly this kind of task.

One caveat in the FM's favour, which I want to state rather than bury: GENEB runs **frozen probes**
and our CNN trains **end-to-end**, so that table mixes regimes — the thing PROTOCOL rule 12 forbids.
The honest status is not "the CNN beats GENA-LM." It's: **we don't know whether the GENEB wins
survive a CNN baseline, because nobody ran one.**

That's Track A's critical open item, and if the wins don't survive, Track B's credibility goes with
them. It's also the cheapest high-value run we have left.

## What Track B has settled

| benchmark | strongest comparator | best gLM | verdict |
|:--|--:|--:|:--|
| ViroBench family (matched 32.7kb ctx) | **alignment 0.7383**, k-mer3-5 0.6231 | NT-v2 0.6148 | no advantage |
| GUE virus_covid (deduped, n=8,050) | k-mer3-6 0.7171 | NT-v2 0.6764 | no advantage |
| GUE virus_species_40 | k-mer 0.4407 | NT-v2 0.4443 | tie |
| HVUE Host_Tropism (strict, n=3,391) | CNN 0.8588 | NT-v2 0.8647 | +0.0059 — only clean viral positive |
| HVUE Path. / Trans. | — | — | **unevaluable** |
| ProteinGym viral (22 assays) | — | — | **not run** |

Two things worth knowing. **HVUE Pathogenecity and Transmissibility retain 96 of 5,194 and 60 of
4,956 homology-clean test rows** — so no method's score on them separates memorisation from
generalisation, ours or anyone's. And **viral family taxonomy turns out to be an alignment task**:
nearest-hit alignment gets 0.9915 accuracy on the 85% of test genomes it can align, over only
10–13% of query length at high identity — a conserved gene, not a duplicated genome. The split is
genuinely clean (2.2% near-duplicates, confirmed twice) *and* alignment solves it. That's a much
more interesting statement than "the models lose."

## What I'd like you to take

**(1) CNN ladder on the 13 GENEB sentinel tasks — highest value, closes the critical gap.**

```bash
cd scripts/common
python capacity_sweep.py --dataset <geneb> --task <task> --seeds 42 43 --epochs 30
```

It sweeps 13 (architecture, capacity) cells from 0.04M to 9.4M across dilated / U-Net / ResNet,
selects on **dev only**, and reports the dev-selected cell and the oracle-best separately. You'll
need to add a GENEB loader to its `load()` — the splice and HVUE ones are the templates. Please also
re-run the frozen probes with the layer swept on dev (`--layer` on the probe scripts); we were
reading `hidden_states[-1]` for every model, which under-read NT-v2 by +0.03 and GENA-LM by +0.02.

**(2) ProteinGym viral supervised — Track B's only untouched modality.** 22 viral assays under
`/data/nvidia/proteingym/DMS_ProteinGym_substitutions/`. ESM-2 650M/3B embeddings → ridge, position-
disjoint `contiguous`/`modulo` splits, Spearman, against published ESCOTT/GEMME/S3F_MSA **plus a
supervised one-hot + biochemical control** — that control is the point, it's the pLM equivalent of
the CNN. The whole 95-scorer leaderboard is zero-shot, so supervised adaptation is genuinely
untested and it's the most likely place a real viral positive shows up.

**(3) If you have cycles: ViroBench's `genus` split.** It withholds whole genera — a harder and more
meaningful generalisation test than the temporal split. We have one k-mer point (0.9301) and **zero
gLM numbers**.

## Full-FT subset selection

Full FT is ~50× a frozen probe, so don't run it everywhere. Criteria, declared before seeing results:
frozen probe within ±0.05 of max(k-mer, CNN); positionally structured so RF plausibly matters; ≤2
tasks per GENEB category; n_train ≥ 5k. Target 8–12 tasks × 3 models × 3 seeds.

## Layout

`scripts/common/` is the shared harness — use it for both tracks. `scripts/track_a_benchmarks/` and
`scripts/track_b_viral/` hold the benchmark-specific runners and shim to `common/paths.py`.
Everything resolves from `VB_*` env vars; no absolute paths.

Two loading traps that cost me a day: `AutoModelForSequenceClassification` on GENA-LM silently
discards all 48 pretrained LayerNorms (pre-LN checkpoint vs post-LN HF class) and the model then
scores exactly 0.0000 at every LR — use `AutoModel` plus your own head, and keep
`assert_no_fresh_encoder_weights()` in the loop. And if you ever see an exact-zero metric, check for
re-initialised weights before concluding anything about capability.

Happy to talk through the paper framing — that's the genuinely open question, not the evidence.
