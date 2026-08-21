# Handoff from Vista (8× GH200) — results, bugs found, what's still open

2026-08-21. Written in response to your "Split of work" doc. All paths below are on Vista
(`/work/10906/arisk/...`, `/scratch/10906/arisk/...`) unless noted; the code changes are in
`scripts/viral_benchmark/` and `scripts/positive_control/` on this branch, already committed to
the shared repo tree (not yet pushed — see note at the end).

---

## PART A — Confirms your Part A finding, independently

Before your doc arrived we'd already flagged (via a user question) that capping gLM context
below the CNN's — and calling the resulting gap "real capability" — was an incomplete comparison.
Your matched-context k-mer run closed it properly. `reports/virobench_glm_comparison.md` has been
corrected in place: reading #2 ("gLMs show real capability the CNN doesn't") is now marked
**retracted**, with your table cited directly. No disagreement between clusters here.

---

## PART B — What ran here today

### B1. HVUE core benchmark — found the real data, fixed two bugs, ran it fresh

We had marked HVUE "unverifiable from Vista" earlier in the session because `paths.py`'s hardcoded
defaults (`/home/nvidia/glm-locking/...`) don't resolve here. That was wrong — a local git clone
at `/work/10906/arisk/ls6/evo-locking/data/hvue/` has the actual data (Host_Tropism 47,194 rows,
Pathogenecity 134,066, Transmissibility 458,756, **every sequence exactly 1000bp** — matches your
context-audit table exactly).

Running `build_identity_splits.py` against it surfaced two real bugs in the checked-in script
(not something we introduced — check if you're hitting these too):
1. Missing `import paths as P` entirely (instant `NameError`) — every sibling script has the
   standard `sys.path.insert(...); import paths as P` shim; this one didn't.
2. `df.groupby("label", group_keys=False).apply(lambda g: g.sample(...))` silently drops the
   `label` column under pandas >=2.2 (`include_groups` default changed), causing
   `AttributeError: 'DataFrame' object has no attribute 'label'` two lines later. Rewrote as an
   explicit per-group loop.
3. (Not a bug, a trap) `hvue_glm.py`'s hardcoded `KMER[(task,split)]` dict has no entry for
   `identity_disjoint_hsd*` — running without `--kmer_json` raises `KeyError` right after training
   finishes, silently discarding the result. Built a flat adapter JSON from our own
   `build_identity_splits.py` output and passed `--kmer_json` explicitly.

**Rebuilt identity_disjoint_hsd0 from scratch, ran k-mer + CNN (3 seeds) + NT-v2/HyenaDNA (3
seeds) + GENA-LM (1 seed) on it.** Reproduces your numbers closely:

| task | our k-mer | HANDOFF's original | our CNN (mean, best LR) | HANDOFF's CNN |
|:--|--:|--:|--:|--:|
| Host_Tropism | 0.9171 | 0.9131 | 0.9491 | 0.9482 |
| Pathogenecity | 0.9558 | (gate-failed, no exact number) | 0.9715 | 0.9667 |
| Transmissibility | 0.9224 | (gate-failed, no exact number) | 0.9340 | 0.9202 |

gLM vs `best(k-mer,CNN)`, mean over seeds:

| task | NT-v2-500M | HyenaDNA | GENA-LM (n=1) |
|:--|--:|--:|--:|
| Host_Tropism | **+0.0003 (tie)** | -0.0119 | -0.0106 |
| Pathogenecity | -0.0283 | -0.0164 | -0.0074 |
| Transmissibility | -0.0325 | -0.0373 | -0.0177 |

**Important correction to both our scoreboards:** the single-seed NT-v2 Host_Tropism win we
initially got (+0.0023) closely matched your own single-seed "+0.0012" — but extending to 3 seeds
here drops it to +0.0003, a dead tie. If your "8 losses, 1 win of +0.0012" number is also
single-seed, it may not survive a 3-seed check either. Worth re-running with `--seeds 42 43 44`
if you haven't already — full report at `reports/hvue_real_data_verification.md`.

### B2. ViroBench full P3 spec (`--mod ALL`, `min_count=1`, all 5 levels, `times` split)

k-mer3-6 (whole genome) vs CNN (20kb), 3 seeds, `reports/virobench_full_spec_baselines.md`:

| level | classes | k-mer3-6 | CNN |
|:--|--:|--:|--:|
| kingdom | 18 | 0.560 | 0.294 |
| phylum | 28 | 0.555 | 0.325 |
| class | 45 | 0.520 | 0.327 |
| order | 67 | 0.599 | 0.238 |
| family | 173 | **0.570** | 0.208 |

k-mer wins every level, consistent with everything else on ViroBench. At family level our k-mer
(0.570) sits between ViroBench's published BLAST (0.477) and LucaVirus (0.759) — this is now a
legitimate cross-paper comparison (the min_count>=10/DNA-only filter that made it illegitimate is
gone), but we have **not** independently confirmed which taxonomic level ViroBench's headline
47.67/75.88 refers to — treat that specific comparison as tentative pending that check.

**Not done here (still open, on either cluster):** BLAST/Kraken2 baselines — not installed on
Vista (no aarch64 conda-forge/bioconda build attempted yet); gLM runs at the full-spec scale
(46,651 rows vs. the 6,042-row DNA/family subset already tested — ~7.7x cost, not attempted).

### B3. Positive-control sweep (GUE 12 + NTv3 18 + EPI 12 = 42 task-splits, `reports/positive_control_comparison.md` + `reports/epi_comparison.md`)

Not part of your task list but relevant context if you're citing "the harness isn't broken"
anywhere: NTv3 splice sites (chromosome-disjoint, verified 0% overlap) show a 0.30-0.60 MCC
gLM-vs-baseline gap — the one clean, large, disjoint-split-robust positive control found anywhere
in either cluster's work. Everything else in that sweep (27 other GUE/NTv3 tasks, all 6 EPI cell
lines) shows small-to-zero gaps, and EPI specifically shows our baseline *beating* published
pretrained-embedding methods (EPIPDLF/EPINTLM) on 5/6 cell lines — independently confirmed via
MMseqs2 that EPI's promoters leak 67-80% train-to-test, corroborating BENGI/LOCO-EPI's published
critique of that benchmark family.

Also independently measured, in case useful for your split-leakage tracking: GUE `virus_covid`'s
*official* split (used in the original viral-capability numbers) has 11.9% exact-duplicate test
sequences also verbatim in train — not previously flagged anywhere in HANDOFF/SPLIT_DESIGN docs.

### B4. DART-Eval Task 1 reproduction

Ab-initio CNN baseline (cCRE vs. dinucleotide-shuffled negatives, chromosome-disjoint) reproduced
within ~0.4% of the paper's own published numbers (verified against the arXiv LaTeX source
directly: 0.8460±3.3e-4 accuracy, 0.927 AUROC — ours: 0.8423, 0.9264). Fully public data, no
Synapse account needed. Not on your task list, just a harness-validation side quest — mentioning
in case it's useful precedent for a "does our pipeline reproduce known numbers" argument in the
writeup.

---

## PART C — Environment notes, in case you hit the same walls

- **mmseqs2**: not on conda-forge for linux-aarch64 by default; needed
  `pixi workspace channel add https://conda.anaconda.org/bioconda/` first, then
  `pixi add mmseqs2` resolved fine.
- **transformers version conflict**: the shared pixi env here (`biojepa-env`) has
  `huggingface-hub==1.28.0` pulled in by `datasets`, which conflicts with `transformers<5`
  (needed — NT-v2/HyenaDNA/GENA-LM's custom modeling code breaks under transformers 5.x, e.g.
  `ImportError: cannot import name 'find_pruneable_heads_and_indices'`). Built a **separate**
  pixi env (`/scratch/10906/arisk/virobench-glm-env`) with `transformers>=4.48,<5`,
  `huggingface_hub<1.0`, no `datasets` package, to avoid the conflict rather than risk breaking
  the shared env other work depends on.
- **NT-v2's custom `modeling_esm.py` doesn't support gradient checkpointing** (raises
  `does not support gradient checkpointing` even though the base class exposes the method) and
  uses eager (non-fused) attention — full fine-tune at 2048 tokens/bs=8 OOM'd a 96GB GPU. Capped
  to 1024 tokens/bs=2-16 depending on task. If your LoRA-at-2048 attempt (GPU 4) works, that's the
  better fix — full-FT at true context may still need an actual patch to the modeling code.
- **Per-node GPU topology**: this allocation is 8 nodes × 1 GPU each (Vista GH200), not 8 GPUs on
  one node — cost us a debugging cycle the first time (`cuda:1..7` don't exist locally). Dispatch
  via `srun --jobid=<job> --overlap --nodes=1 --ntasks=1 -w <node>` per job, not
  `CUDA_VISIBLE_DEVICES`.

---

## PART D — Suggested next split (avoid overlap)

Given B1-B4 above, and your Part B/C lists:
- **We can pick up:** BLAST/Kraken2 install + baselines for ViroBench full-spec (C1's remaining
  piece), assuming your cluster doesn't already have these tools available. Say so if you'd
  rather do it there.
- **Still yours per your own doc, not attempted here:** ProteinGym (C2), antibody escape (C3) —
  we don't have `/data/nvidia/proteingym` or `escape_screen` locally and haven't tried rebuilding
  either from public sources.
- **Worth a cross-check on your side:** re-run your HVUE identity-disjoint hsd0 win with 3 seeds
  if it's currently single-seed — see B1 above, ours flipped from a real-looking win to a tie.

---

## Note on repo state

The earlier positive-control work (`aggregate_positive_control.py`, `epi_baselines.py`,
`prepare_splits.py`/`prepare_epi_splits.py`, `mmseqs_leakage_check.py`, `virobench_glm.py`, and the
resulting `reports/*.md`) is already committed **and pushed** — commit `9bf3b42` matches
`origin/viral-benchmark-continuation` exactly (0 ahead / 0 behind). If you pull this branch you
already have that part. Everything else described in Parts A-D above from later in the session
(the two `build_identity_splits.py` bug fixes, `reports/hvue_real_data_verification.md`,
`reports/virobench_full_spec_baselines.md`, and this handoff doc itself) is **not yet
committed/pushed** — check with the user before assuming it's visible outside this filesystem.
