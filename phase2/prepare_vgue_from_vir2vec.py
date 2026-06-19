"""Prepare workspace metadata for future vGUE integration from the Vir2vec repository.

This script does not download the upstream task sequences itself. Instead, it records
what the public Vir2vec repository currently provides locally:
  - accession split files per upstream source database
  - whether task-ready benchmark CSVs are present (they are not in the current repo clone)
  - the concrete missing artefacts required to build a unified vGUE manifest for this project

It is meant to make the gap between "Vir2vec exists" and "vGUE is integrated here"
explicit and reproducible.
"""
import argparse
import json
from pathlib import Path

EXPECTED_TASKS = [
    "virus_vs_nonvirus",
    "dna_vs_rna_virus",
    "host_range_prediction",
    "hiv1_vs_hiv2",
    "sars_cov_2_lineage_typing",
    "influenza_subtype_typing",
    "hiv1_tropism",
]

DEFAULT_RETAIN_TASKS = [
    task for task in EXPECTED_TASKS if task != "hiv1_tropism"
]


def count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vir2vec-root', default='/tmp/Vir2vec')
    parser.add_argument('--out', default='data/benchmarks/vgue_from_vir2vec_audit.json')
    args = parser.parse_args()

    root = Path(args.vir2vec_root)
    out = Path(args.out)

    accessions_root = root / 'accessions_txt'
    split_files = {}
    if accessions_root.exists():
        for split_dir in sorted(accessions_root.iterdir()):
            if not split_dir.is_dir():
                continue
            split_files[split_dir.name] = {
                path.name: count_lines(path)
                for path in sorted(split_dir.glob('*.txt'))
            }

    task_ready_files = sorted(
        str(path.relative_to(root))
        for path in root.rglob('*')
        if path.suffix.lower() in {'.csv', '.tsv', '.json', '.parquet', '.fa', '.fasta'}
    )

    payload = {
        'vir2vec_root': str(root),
        'vir2vec_present': root.exists(),
        'readme_present': (root / 'README.md').exists(),
        'accession_split_files': split_files,
        'expected_vgue_tasks': EXPECTED_TASKS,
        'default_viral_retain_tasks': DEFAULT_RETAIN_TASKS,
        'excluded_default_retain_tasks': {
            'hiv1_tropism': (
                'Excluded from the default retain score because it overlaps conceptually '
                'with the host-tropism forget objective.'
            ),
        },
        'task_ready_files_in_repo': task_ready_files,
        'supports_direct_vgue_integration_now': False,
        'reason': (
            'The public Vir2vec repository exposes train/validation/test accession lists and embedding scripts, '
            'but it does not ship a task-ready unified vGUE benchmark table with sequence+label rows. '
            'To integrate vGUE here, additional upstream sequence retrieval / task-label reconstruction is required.'
        ),
        'required_next_artifacts': [
            'Task-specific sequence table(s) keyed by accession for host-range / DNA-vs-RNA / HIV-1-vs-HIV-2 / SARS-CoV-2 lineage / HIV-1 tropism',
            'Mapping from Vir2vec accession splits to those task tables',
            'Unified benchmark CSV with benchmark,task,split,sequence,label columns',
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'[vgue-audit] wrote {out}')
    print(f'[vgue-audit] split groups: {sorted(split_files)}')
    print(f'[vgue-audit] task-ready files found: {len(task_ready_files)}')


if __name__ == '__main__':
    main()
