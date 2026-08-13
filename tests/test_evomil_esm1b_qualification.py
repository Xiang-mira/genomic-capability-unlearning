from __future__ import annotations

import csv
from pathlib import Path

from phase2.evomil_esm1b_qualification import load_split_data, parse_args


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_split_data_coalesces_virus_family_columns(tmp_path: Path) -> None:
    out_root = tmp_path / "evomil"
    write_csv(
        out_root / "evomil_split_manifest.csv",
        [
            {
                "virus_id": "v1",
                "host_label": "Host A",
                "split": "train",
                "proteome_cluster": "c1",
                "genome_hash": "g1",
                "protein_set_hash": "p1",
                "virus_family": "Family From Split",
            }
        ],
    )
    write_csv(
        out_root / "evomil_sequence_manifest.csv",
        [
            {
                "virus_id": "v1",
                "virus_taxid": "1",
                "virus_name": "Virus 1",
                "host_label": "Host A",
                "host_taxid": "2",
                "refseq_accession": "NC_1",
                "genome_accession": "NC_1",
                "genome_fasta_path": "/tmp/genome.fna",
                "protein_faa_path": "/tmp/protein.faa",
                "protein_count": "3",
                "protein_ids": "a;b;c",
                "virus_family": "Family From Sequence",
                "virus_taxonomy": "Viruses; Exampleviridae; Examplevirus",
                "accession_status": "resolved",
                "replacement_accession": "",
                "source": "test",
            }
        ],
    )

    args = parse_args(["--execute", "--out-root", str(out_root)])
    data = load_split_data(args)

    assert "virus_family" in data.columns
    assert "virus_family_x" not in data.columns
    assert "virus_family_y" not in data.columns
    assert "virus_family_split" not in data.columns
    assert "virus_family_seq" not in data.columns
    assert data.loc[0, "virus_family"] == "Family From Split"
