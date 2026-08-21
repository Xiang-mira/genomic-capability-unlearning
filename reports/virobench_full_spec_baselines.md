# ViroBench taxonomy -- full P3 spec baselines (mod=ALL, min_count=1, all 5 levels)

HANDOFF.md flagged the earlier DNA-subset, min_count>=10-filtered numbers as explicitly
NOT comparable to ViroBench's own published cross-paper numbers (BLAST 47.67, LucaVirus 75.88):
"I filtered to families with >=10 train examples and present in test (69 of 152) and used DNA
(7,600) not ALL (46,651). That inflates macro-F1." This run removes both caveats: --mod ALL
(46,651 sequences, all nucleic-acid types), --min_count 1 (no class filter), all 5 taxonomic
levels, --split times (the verified temporal-disjoint split; genus is not genus-disjoint,
independently confirmed earlier this session at 83.65% genus overlap).

## Results (k-mer3-6 whole-genome vs 20kb-capped CNN, 3 seeds)

| level | classes | train | test | k-mer3-6 macro-F1 | CNN macro-F1 |
|:--|--:|--:|--:|--:|--:|
| kingdom | 18 | 46,629 | 5,823 | 0.560 | 0.294 |
| phylum | 28 | 46,591 | 5,799 | 0.555 | 0.325 |
| class | 45 | 46,383 | 5,776 | 0.520 | 0.327 |
| order | 67 | 46,259 | 5,742 | 0.599 | 0.238 |
| family | 173 | 46,389 | 5,505 | 0.570 | 0.208 |

k-mer wins at every level by a wide margin (0.19-0.36 macro-F1) -- consistent with the rest of
this project's finding that k-mer's whole-genome context advantage, not model capability, is
usually the deciding factor on ViroBench. No gLM has been run on this full-spec version yet
(only on the earlier DNA/family/times subset, see reports/virobench_glm_comparison.md).

## Comparison to ViroBench's own published numbers (tentative)

At family level, our k-mer (0.570) sits between ViroBench's published BLAST (0.477) and
LucaVirus (0.759). This is now a methodologically legitimate comparison (unlike the earlier
filtered run), but the exact taxonomic level ViroBench's headline 47.67/75.88 figures refer to
has not been independently confirmed against their paper -- treat this specific cross-paper
comparison as tentative until that is checked.

## Still missing from the full P3 spec
- BLAST and Kraken2 alignment baselines (ViroBench's own comparators) -- not installed on this
  cluster yet.
- gLM runs at mod=ALL / min_count=1 across the 5 levels -- only the earlier DNA/family/times
  subset has gLM numbers so far; full-spec gLM runs would be substantially more expensive
  (46,651 vs 6,042 training rows, ~7.7x) and were not attempted this session.
- mod=ALL genus split (secondary reference per the parallel cluster's own task split -- being
  run there, not duplicated here).
