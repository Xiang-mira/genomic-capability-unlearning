# G1-A GENEB Results

Status: `BLOCKED_BY_ACCESS`

Progress completed:

- Added `--dataset geneb` support to `scripts/common/capacity_sweep.py`.
- Added `--dry_run` and `--max_cells` so the formal environment can run one true-data smoke before full arrays.
- Added split checksum manifest output for capacity sweeps.
- Added receptive-field metadata to capacity ladder rows.
- Added `scripts/orchestration/group1_status.py` and `launch_group1.sh` preflight.
- Audited branch docs/history for the GENEB canonical source and found the complete 13-task manifest is not recorded locally.

Formal experiments not launched:

- GENEB CNN ladder on 13 sentinel tasks.
- GENEB frozen probe layer sweep.
- Paired statistics/aggregation.

Blockers:

- No local GENEB data found under the default path or any searched project/data path.
- `VB_GENEB_DIR` is not set to a usable GENEB data root.
- `VB_GENEB_TASK_MANIFEST` / `sentinel_tasks.csv` for the full 13-task A1 set is not available.
- Slurm is unavailable.
- GPU is unavailable because `nvidia-smi` fails with driver/library mismatch.

Expected first command once data and GPU are available:

```bash
export VB_GENEB_DIR=/path/to/geneb
export VB_GENEB_TASK_MANIFEST=$VB_GENEB_DIR/sentinel_tasks.csv
scripts/orchestration/launch_group1.sh --smoke <one_task_from_manifest>
```

Only after that true-data smoke passes should full Group 1 submission be enabled.
