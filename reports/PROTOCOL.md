# Measurement protocol — binding on both tracks

Every rule here exists because violating it changed a conclusion in this project. The violation is
named next to each rule.

## Selection
1. **Never select on test.** Not a split, not a hyperparameter, not a layer, not a checkpoint, not
   an architecture. Report an oracle-over-test number only if labelled `oracle_*` and printed
   beside the dev-selected number.
   *Violated by:* `phase2/model_vs_kmer_cluster_disjoint.py` (C and layer chosen on test AUROC).
2. **Never build a split in a baseline's feature space, and never gate a split on baseline
   degradation.** *Violated by:* `build_splits_v2.py` — clusters in kmer5-PCA space, accepts only if
   the k-mer drops >= 0.03 (`GATE = 0.03`). Every excess measured against it is inflated by
   construction.
3. **Dev must match test's distribution.** If test is chromosome-disjoint, temporal or
   homology-disjoint, dev must be too. *Currently violated:* splice and ViroBench use a random 15%
   carve from train. Only HVUE uses `GroupShuffleSplit`.
4. **Sweep a selection grid until dev turns over.** *Violated three times in one session:* C at the
   grid edge, GENA-LM's LR at the grid edge, NT-v2's layer still improving at L-4 when a
   "dev-selected layer" was quoted.

## Baselines
5. **Report `max(k-mer, CNN)`, never the k-mer alone.** *Violated by:* the whole GENEB sentinel —
   k-mer-anchored, no CNN.
6. **Run the capacity/architecture ladder before quoting any baseline number.** A single CNN is an
   arbitrary point, not a ceiling. *Violated by:* every splice number before the ladder existed
   (0.68M incumbent 0.354 vs dev-selected 0.953).
7. **Declare the CNN's receptive field in bp**, not just its parameter count. RF, not capacity, is
   what binds on positionally-structured tasks.
8. **On taxonomy/homology tasks, an alignment baseline is mandatory.** *Violated by:* every
   ViroBench conclusion until alignment was run — it beats every model by +0.115.
9. **Sweep the k-mer variant too.** *Violated by:* ViroBench family, where only k-mer3-6 was run;
   k-mer3-5 turned out to be +0.067 better and reversed a positive result.

## Matched context
10. **State the effective context in bp for every method in a compared cell, and match it.**
    *Violated by:* the ViroBench frozen probe (`max_win=16 x 2048 = 32,768bp`, seeing 38.8% of test
    bp) vs the k-mer (unbounded, 100%). Fixing this reversed NT-v2's apparent +0.037 win to −0.012.
11. **Check the train/test length distributions.** *Currently unexplained:* ViroBench train median
    2,316bp vs test median 41,380bp — an 18x shift nobody has accounted for.

## Regimes
12. **Never mix frozen probe / LoRA / full FT in one column.** On splice the frozen-vs-FT gap alone
    is 0.59 MCC.
13. **Mark external numbers `EXTERNAL/PUBLISHED`, never `OUR RUN`**, and note their regime. Ours are
    supervised-from-scratch or probed; most published splice numbers are fine-tuned.
14. **Sweep the read-out layer on dev for every model, not just the one that looks bad.**
    *Violated by:* reading `hidden_states[-1]` for all models; L-2/L-3 is dev-better for NT-v2
    (+0.03) and GENA-LM (+0.02), worse for HyenaDNA. Investigating only the failing model is the
    asymmetry that manufactures a negative result.

## Splits and leakage
15. **Audit partial overlap with `easy-search`, not `easy-cluster`.** `-c 0.9` needs 90%
    *bidirectional* coverage and is blind to a test sequence sharing half its length at high
    identity. *This is how HVUE Pathogenecity (80.5% leaked) passed as "identity-disjoint".*
16. **Distinguish two different questions and report both:** (a) is the test genome substantially a
    copy of a train genome (duplicate leakage), (b) does it share any detectable conserved region
    (homology reachability). ViroBench is 2.2% on (a) and 85% on (b) — both correct, and conflating
    them makes "clean" wrongly imply "alignment-hard".

## Statistics
17. **Prefer effect sizes with CIs over significance tests.** Use a paired bootstrap over shared
    test examples, paired by example id.
18. **For a negative claim, pre-declare the equivalence margin δ before computing any CI**, and
    report per-level CI width. "Not significant" on an underpowered level means *underpowered*, not
    *equivalent* — on ViroBench only the family level (173 classes) has the power for an
    equivalence claim.
19. **Preserve per-seed results.** Never `max()` over seeds against a single published point.
20. **Multi-seed anything you intend to claim.** Single-seed HVUE "wins" of +0.0012/+0.0023 collapse
    to +0.0003 over 3 seeds.

## Engineering
21. **Assert no pretrained tensor is silently re-initialised.** Use
    `assert_no_fresh_encoder_weights()`; allow only the task head (`classifier`/`head`/`score`/`cls.`).
22. **Flag degenerate runs explicitly.** Write `collapsed_to_majority_class` into the result rather
    than letting an exact-0.0000 metric read as a capability measurement.
23. **Encode every varied argument in the output filename.** *Violated by:*
    `virobench_baselines.py` (omits `--kmer_cap`/`--cnn_len`, ladder runs overwrite each other) and
    the frozen probe before `--layer` was added to its tag.
24. **Emit per-example predictions from every harness.** Paired tests are impossible without them;
    `virobench_baselines.py` still does not.
25. **Never let a post-training exception discard a result.** *Violated by:* `hvue_glm.py`'s
    `KMER[(task, split)]` KeyError, which fires *after* training completes.

## Reporting language
26. Lack of superiority is **"no detectable model-specific advantage over the evaluated comparator
    under this task/evaluation regime"** — not "no capability".
27. If a new result contradicts the current narrative, surface it immediately. If a model genuinely
    wins, that is an important positive result, not something to explain away.
