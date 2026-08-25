# GENEB Canonical Source Audit

Status: `CANONICAL_SOURCE_NOT_FULLY_RESOLVED_FROM_BRANCH`

Scope: repo-internal audit only. I did not download or substitute an external GENEB-like dataset.

## What The Branch Confirms

- A1 is the highest-priority Group 1 item: CNN ladder on the 13 GENEB sentinel tasks.
- GENEB has 100 total tasks; the sentinel subset has 13 tasks/categories.
- Existing GENEB evidence is k-mer anchored: fair k-mer is done, frozen probes are done for 3 models, but layer was not swept on dev.
- No CNN baseline was previously run on the 13 sentinel tasks.
- GENEB frozen probes are a different regime from our end-to-end CNN; the deliverable is "do the wins survive," not "CNN beats GENA-LM."
- GENEB reference k-mer was degenerate on `iDHS-EL_DNase_I` until refit fairly: MCC `0.000` to `0.589`.

## Explicit Task Names Found In Branch

These are mentioned in `reports/FULL_FT_DESIGN.md` as a provisional full-FT slate, not as the complete 13-task canonical sentinel manifest:

| task mention | source note |
|---|---|
| `GUE human_tf_0` | probe `0.576` vs fair k-mer `0.537` |
| `GUE mouse_0` | probe `0.378` vs k-mer `0.437` |
| `NT enhancers` | probe `0.396` vs k-mer `0.425` |
| `iDHS-EL DNase_I` | fair k-mer `0.589` vs probe `0.593` |
| `deep4mc A.thaliana 4mC` | position-dependent methylation; probe spread suspicious |
| `NT H3` | probes `0.662-0.705` vs k-mer `0.590` |
| `ensembl_regulatory` | largest probe gain `+0.266` |

This is insufficient to define the required 13-task A1 experiment.

## What Was Not Found

No branch file or git-history file provided all of the following:

- the exact 13 GENEB sentinel task IDs;
- the canonical GENEB dataset version/checksum;
- an official GENEB download/prepare script;
- an expected `VB_GENEB_DIR` path already used by Aris;
- a local GENEB data manifest;
- the published/fair-kmer/probe table keyed by all 13 task IDs.

Searched:

- `reports/PAPER_DESIGN.md`
- `reports/PROTOCOL.md`
- `reports/TESTED_MATRIX.md`
- `reports/RESEARCH_PLAN.md`
- `README.md`
- `LATEST_UPDATES.md`
- `CLUSTER_2_HANDOFF.md`
- `reports/FULL_FT_DESIGN.md`
- `reports/CROSS_CLUSTER_SYNTHESIS.md`
- `reports/PAPER_OUTLINE.md`
- tracked scripts under `scripts/`
- git history reachable from local refs

## Required Input To Unblock A1

Provide a GENEB data root and manifest in the formal compute environment:

```bash
export VB_GENEB_DIR=/path/to/canonical/geneb
export VB_GENEB_TASK_MANIFEST=$VB_GENEB_DIR/sentinel_tasks.csv
```

`sentinel_tasks.csv` must contain at least:

```text
task,benchmark,category,metric,source_version,split_source
```

For each task, the loader accepts either:

```text
$VB_GENEB_DIR/<task>/{train,dev,test}.parquet
$VB_GENEB_DIR/<task>/{train,dev,test}.csv
$VB_GENEB_DIR/<task>__{train,dev,test}.parquet
$VB_GENEB_DIR/<task>__{train,dev,test}.csv
$VB_GENEB_DIR/<task>.parquet  with split/partition column
$VB_GENEB_DIR/<task>.csv      with split/partition column
```

Each split must contain:

```text
sequence,label
```

Optional but recommended:

```text
id
```

## Next Gate

Do not submit the full 13-task array first.

First run one true-data smoke on the real compute environment:

```bash
scripts/orchestration/launch_group1.sh --smoke <one_task_from_manifest>
```

Only after this passes should the full 13-task dependency graph be enabled.
