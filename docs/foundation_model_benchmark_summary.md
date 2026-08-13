# Foundation-Model Benchmark Qualification Summary

This document is the concise scientific summary for the four completed
foundation-model qualification studies currently checked into this repository.

The qualification rule is strict:

> a benchmark only qualifies if the foundation model shows reproducible,
> model-specific predictive headroom over the strongest reasonable
> non-foundation baseline under the intended out-of-distribution evaluation.

All four completed studies are currently negative.

## Summary Table

| Benchmark | Model | Final Status | Strongest Conventional Comparator | Key Conclusion |
|:--|:--|:--|:--|:--|
| HVUE | Evo | `UNQUALIFIED` | full-sequence k-mer / composition controls | The earlier apparent advantage on host tropism and pathogenicity does not survive stronger sequence-composition controls. |
| ProteinGym | ESM2 | `UNQUALIFIED` | public evolutionary baselines (`VESPA`, `VESPAl`, `S2F_MSA`) | Position-held-out evaluation removes the apparent headroom, and fresh adaptation is unstable across seeds. |
| PHIStruct | SaProt | `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED` | BLASTp | SaProt improves over simple and structure-only baselines, but it does not show statistically reliable positive headroom over BLASTp. |
| EvoMIL | ESM-1b | `NO_QUALIFYING_HEADROOM` | `logistic_regression:aa_3mer_tfidf` | The AA 3-mer baseline beats all five ESM-1b seeds; there is no qualifying model-specific headroom. |

## Benchmark-by-Benchmark Notes

### HVUE / Evo

- Final status: `UNQUALIFIED`
- Canonical checked-in evaluation stack: the completed HVUE/GUE/ViroBench
  44-task downstream benchmark
- Best checked-in forgetting run: `lora_gd_full_ar3_s200`
- Best checked-in HVUE drop: `0.198063`
- Paired cost on GUE retain: `-0.144260`

Scientific interpretation:

The HVUE-related evidence in this repository does not support a clean,
isolated, Evo-specific viral capability once stronger sequence-composition
controls are considered. The benchmark remains useful as a downstream stress
test, but not as a clean foundation-model qualification win.

### ProteinGym / ESM2

- Final status: `UNQUALIFIED`
- Static candidate assays screened: `20`
- Pilot assays: `CCDB_ECOLI_Adkar_2012`, `GAL4_YEAST_Kitzman_2015`,
  `MET_HUMAN_Estevam_2023`
- Preliminary qualified pairs: `none`
- Assays advanced to LoRA: `CCDB_ECOLI_Adkar_2012`

Strongest public baselines by pilot:

- `CCDB_ECOLI_Adkar_2012`: `public_evolutionary:VESPAl`
- `GAL4_YEAST_Kitzman_2015`: `public_evolutionary:VESPA`
- `MET_HUMAN_Estevam_2023`: `public_evolutionary:S2F_MSA`

Key evidence:

- No candidate task-model pair passes the strong-baseline,
  position-held-out, and fresh-LoRA checks.
- Random-split gains do not transfer to the stricter position-held-out
  protocol.
- The only assay that advanced to LoRA remained unstable across three formal
  seeds with mean test excess `-0.218436`.

### PHIStruct / SaProt

- Final status: `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED`
- Dataset: `7627` RBPs from `3350` phages across `7` host genera
- Split sizes: train `7218`, validation `92`, test `317`
- SaProt test macro-F1: `0.454732`
- BLASTp test macro-F1: `0.475180`
- Observed delta (`SaProt - BLASTp`): `-0.020448`

Bootstrap evidence:

- Valid replicates: `10000`
- Invalid replicates: `5778`
- 95% CI: `[-0.113066, 0.071794]`
- `P(delta > 0) = 0.3594`
- `P(delta < 0) = 0.6406`

Scientific interpretation:

SaProt beats the simple and structure-only baselines in the original
qualification stack, but the stronger BLASTp reconstruction removes the claim
of reliable model-specific headroom.

### EvoMIL / ESM-1b

- Final status: `NO_QUALIFYING_HEADROOM`
- Expected viruses: `1768`
- Resolved viruses included in the formal task: `465`
- Excluded viruses: `1303`
- Split sizes: train `259`, validation `54`, test `51`
- Strongest non-foundation baseline:
  `logistic_regression:aa_3mer_tfidf`
- Baseline macro-F1: `0.841270`
- Best ESM-1b + MIL macro-F1: `0.782246`
- Observed delta (`model - baseline`): `-0.059024`

Bootstrap evidence:

- Valid replicates: `10000`
- Invalid replicates: `1613`
- 95% CI: `[-0.204111, 0.056453]`
- Positive seeds: `0 / 5`
- `P(delta > 0) = 0.1269`
- `P(delta < 0) = 0.8729`

Scientific interpretation:

The strongest amino-acid 3-mer baseline outperforms every checked-in ESM-1b
seed, so EvoMIL does not provide qualifying PLM-specific headroom.

## Shared Methodological Update

As of August 12, 2026, the repository standardizes the bootstrap sign
convention to:

`delta = model_metric - baseline_metric`

This is now enforced through a shared helper used by PHIStruct and EvoMIL, with
tests covering observed deltas, bootstrap summaries, confidence intervals, and
tail probabilities.

## What Is Not Concluded Here

The VPF-PLM controller is implemented and currently running separately, but it
is not part of the completed result set in this summary and should not be
reported as concluded.
