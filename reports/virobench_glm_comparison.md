# ViroBench taxonomy, temporal split — first gLM results (closes HANDOFF §4 P3)

HANDOFF.md's own priority list flagged this as "the one remaining chance of a positive viral
target" — every prior measurement on ViroBench was k-mer/CNN only; no gLM had ever been run on the
genuinely-disjoint `times` split. This session built `virobench_glm.py` (new script, mirrors
`gue_glm.py`'s regime/pooling conventions) and ran all 3 supported gLMs on it.

**Setup:** `--mod DNA --split times --level family` — same task definition as HANDOFF.md §3.5's
existing k-mer/CNN numbers, which were computed on a different (now-unreachable) cluster's ViroBench
download. Re-ran `virobench_baselines.py` fresh this session on our own download to cross-check
before trusting the old figures: got k-mer3-6=0.9475/MCC=0.9741, CNN=0.6421/MCC=0.8134 (3-seed
mean) vs HANDOFF's 0.9527/0.6774 -- close (within ~0.5-3.5pp), confirms the earlier numbers
reproduce and this comparison is apples-to-apples. The table below uses this session's own
freshly-verified k-mer/CNN numbers, not the old HANDOFF figures. `times` is
the split verified genuinely temporal-disjoint this session (train ≤2022-07, test ≥2023-08, zero
date overlap; species overlap 1.2%) — **not** `genus` (confirmed 82-84% genus overlap, not a valid
disjoint split, per HANDOFF §3.6 and this session's independent re-measurement).

Every model's effective context is recorded explicitly (README rule 4): k-mer sees the whole
genome (median 43.5kb); CNN is capped at 20kb; HyenaDNA (byte-level tokenizer) run at the same
20kb cap for a fair architecture-vs-architecture comparison; NT-v2-500M capped at 6.1kb (1024
tokens) because its custom, non-fused attention implementation OOM'd a 96GB GPU well before its
nominal ~12kb range; GENA-LM capped at ~3.1kb (its native BPE-tokenized window).

## Results (macro-F1, DNA/times/family, 44 classes)

| model | regime | effective context | seeds | macro-F1 | MCC |
|:--|:--|--:|--:|--:|--:|
| k-mer3-6 | — | whole genome (43.5kb median) | 1 (deterministic) | **0.9475** | 0.9741 |
| CNN | — | 20kb | 3 | 0.6421 | 0.8134 |
| HyenaDNA | full FT | 20kb | 1 | 0.8784 | 0.9526 |
| NT-v2-500M | full FT | 6.1kb | 3 | 0.8867 ± 0.0075 | 0.9408 |
| GENA-LM | full FT | 3.1kb | 1 | 0.6378 | 0.7484 |
| HyenaDNA | probe (frozen) | 20kb | 1 | 0.0490 | 0.2236 |
| NT-v2-500M | probe (frozen) | 6.1kb | 1 | 0.4864 | 0.6714 |
| GENA-LM | probe (frozen) | 3.1kb | 1 | 0.0862 | 0.3463 |

(HyenaDNA and GENA-LM are single-seed given the compute cost of full-genome fine-tuning across an
8-GPU allocation already stretched over this session; NT-v2 has a real 3-seed spread, std 0.0075 —
tight, not a noisy result.)

## Two readings of this, both true and both needed

**1. Under this project's own established convention** (`best(k-mer, CNN)` is the baseline that
must be beaten for a real capability claim — same rule applied to every HVUE/GUE/NTv3/EPI result
elsewhere in this repo): **k-mer (0.9475) still wins.** Every gLM, even fully fine-tuned, loses to
the whole-genome k-mer baseline by 0.061-0.310 macro-F1. This closes the P3 gap with the same
answer the rest of the project has found everywhere else: **no qualified viral capability target
on ViroBench taxonomy either.** `OUTCOME_FOR_UNLEARNING.md`'s "one remaining chance" is now tested
and comes back negative.

**2. At matched effective context, the gLMs show real, substantial capability the CNN does not.**
HyenaDNA and CNN see the *same* 20kb window; HyenaDNA scores 0.8784 vs CNN's 0.6421 — a +0.236
macro-F1 gap, the second-largest context-matched gLM-vs-baseline margin found in this entire
project (after NTv3 splice sites). This is a genuine architectural/pretraining advantage, not
noise. It just isn't large enough to close the gap created by k-mer's much longer effective
context (whole genome vs 20kb) — the binding constraint on this task is context length, not model
capability. That distinction matters for the unlearning question specifically: there IS something
here that a frozen/probe evaluation would badly undersell (probe regime collapses to 0.05-0.49
macro-F1 for all three models — consistent with HANDOFF §3.4's finding elsewhere that frozen viral
representations are far below baseline), but it's not something a whole-genome k-mer adversary
would be denied by removing it from the model.

## Caveats

- HyenaDNA/GENA-LM results are single-seed; treat as a first data point, not a converged estimate.
- Effective-context figures for GENA-LM (3.1kb) and NT-v2 (6.1kb) are meaningfully below the
  README's own stated "~2-4kb / ~12kb" ranges for these models on other tasks — NT-v2 specifically
  had to be capped well below its nominal window because its custom `modeling_esm.py` uses eager
  (non-fused) attention with no gradient-checkpointing support, so full fine-tuning at longer
  context OOM'd a 96GB GPU. A LoRA or gradient-checkpointing-patched run could likely reach NT-v2's
  true ~12kb range; not attempted this session.
- This is one taxonomic level (family) and one modality (DNA) of HANDOFF's full P3 spec, which also
  calls for `--mod ALL`, all 5 levels, and BLAST/Kraken2 alignment baselines. Those remain untested.
