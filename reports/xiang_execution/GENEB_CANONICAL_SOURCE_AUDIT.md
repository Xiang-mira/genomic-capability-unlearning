# GENEB Canonical Source Audit

Status: `SPECIFICATION_BLOCKED`

Scope: repo-internal audit only. I did not download or substitute an external GENEB-like dataset.

Conclusion: the current branch and its reachable git history do not uniquely define Aris's A1
canonical 13-task GENEB sentinel set or the canonical GENEB data/split source. Do not infer the
remaining tasks from the seven task mentions below.

## Audit Commands Run

Remote/history:

```bash
git remote -v
git branch -a
git ls-remote --heads origin
git ls-remote --heads https://github.com/Xiang-mira/genomic-capability-unlearning.git
git log --all --oneline --decorate -- reports/
git log --all -S"13 sentinel" -- .
git log --all -S"sentinel" -- .
git log --all -S"GENEB" -- reports scripts configs
git grep -n -i "sentinel" $(git rev-list --all)
git log --all --oneline --name-status -- scripts/geneb 'scripts/**/geneb*' '*geneb*' '*GENEB*'
git log --all -S"scripts/geneb" -- .
```

Result:

- Remote heads visible by SSH and HTTPS: only `main` and `viral-benchmark-continuation`.
- No other Aris remote branch is available from `origin`.
- No tracked or deleted `scripts/geneb/` directory was found in reachable history.
- No tracked or deleted GENEB manifest/config/result file containing the full 13-task set was found.

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

Additional GENEB task mentions found in the same source are explicitly **excluded** from the
provisional full-FT slate, so they must not be used to fill the missing A1 slots without Aris's
confirmation:

| task mention | source note |
|---|---|
| `human_or_worm` | excluded; k-mer already `0.81` with large probe gain |
| `coding_vs_intergenomic` | excluded; k-mer already `0.73` with large probe gain |
| `phage_fragments` | excluded; probe `+0.250`, unambiguous |
| `lncrna g_max` | excluded; `n_train` likely below S1, verify |

Candidate evidence table:

- `reports/xiang_execution/geneb_a1_candidate_task_evidence.csv`

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
- remote heads visible via SSH and HTTPS
- deleted/renamed files under `reports/`, `scripts/`, `configs`, `*geneb*`, and `*GENEB*`

## Specification Blocker

The blocker is now a missing experimental specification, not code:

```text
SPECIFICATION_BLOCKED: A1 cannot start until Aris identifies the exact 13 GENEB sentinel tasks and
the canonical data/split source to use.
```

Question to Aris:

```text
Which exact 13 GENEB sentinel tasks should be used for A1, and what canonical GENEB data release,
split files, label mappings, metrics, and local/shared path should define VB_GENEB_DIR and
VB_GENEB_TASK_MANIFEST?
```

Minimum answer required:

```text
1. The 13 task IDs exactly as they should be passed to capacity_sweep.py --task.
2. Dataset identity and version/checksum or release tag.
3. Train/dev/test split source and whether dev is official or must be derived.
4. Label mapping and canonical metric for each task.
5. Expected directory layout or existing shared filesystem path.
```

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
