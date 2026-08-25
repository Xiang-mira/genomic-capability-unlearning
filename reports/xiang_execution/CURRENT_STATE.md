# Current State

Status: `GROUP1_BLOCKED_BY_ACCESS`.

Aris source branch: `viral-benchmark-continuation`

Aris source commit: `2cf5c967ce6739026e7fabd2381b394b5add4b64`

Working branch: `xiang/viral-benchmark-continuation-a1`

The source branch was fetched from `origin` and verified before any code audit or edits:

```text
HEAD == origin/viral-benchmark-continuation == 2cf5c967ce6739026e7fabd2381b394b5add4b64
```

Protected pre-sync local state:

- tar backup: `/tmp/codex_project1_pre_sync_20260825T142004Z/untracked_files.tar.gz`
- status snapshot: `/tmp/codex_project1_pre_sync_20260825T142004Z/git_status_short.txt`
- git stash: `stash@{0}` with message `pre-sync before viral-benchmark-continuation 20260825T142004Z`
- nested untracked repositories moved under `/tmp/codex_project1_pre_sync_20260825T142004Z/moved_untracked_nested/`

Current local non-branch untracked item after checkout:

- `tools/vpf_local/`

Group 1 blockers:

- `VB_GENEB_DIR` data was not found locally.
- Complete canonical 13-task GENEB sentinel manifest was not found in branch docs or git history.
- `sbatch`/`squeue` are not available.
- `nvidia-smi` fails with a driver/library mismatch.

Code/interface work completed:

- `scripts/common/capacity_sweep.py` now accepts `--dataset geneb`.
- GENEB loader supports task split files under `VB_GENEB_DIR`.
- Capacity sweep outputs now include split checksum manifests and receptive-field metadata.
- Group 1 preflight/finalizer scripts are under `scripts/orchestration/`.
- GENEB canonical-source audit is recorded in `reports/xiang_execution/GENEB_CANONICAL_SOURCE_AUDIT.md`.

No formal experiment was launched.
