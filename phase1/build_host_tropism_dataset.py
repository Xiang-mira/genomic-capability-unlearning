import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import clean_sequence, sample_window


HUMAN_TAX_ID = "9606"


@dataclass
class VirusMeta:
    accession: str
    virus_tax_id: str
    virus_name: str
    host_tax_id: str
    host_name: str
    host_common_name: str
    completeness: str
    length: int
    is_lab_host: str
    is_vaccine_strain: str


def require_cli(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Missing required NCBI CLI '{name}'. Install NCBI Datasets CLI so "
            "both 'datasets' and 'dataformat' are on PATH."
        )
    return path


def run_command(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_summary_query(out_file, refseq_only: bool, host: str | None, limit: str) -> None:
    cmd = ["datasets", "summary", "virus", "genome", "taxon", "10239", "--as-json-lines"]
    if refseq_only:
        cmd.append("--refseq")
    if host:
        cmd.extend(["--host", host])
    if limit:
        cmd.extend(["--limit", limit])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=out_file)


def read_tsv(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield {normalize_col(k): v for k, v in row.items()}


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def first_present(row: Dict[str, str], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(normalize_col(name), "")
        if value:
            return value
    return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_metadata(tsv_path: str) -> List[VirusMeta]:
    records: List[VirusMeta] = []
    for row in read_tsv(tsv_path):
        accession = first_present(row, ["accession", "nucleotide_accession"])
        if not accession:
            continue
        records.append(
            VirusMeta(
                accession=accession,
                virus_tax_id=first_present(
                    row,
                    [
                        "virus_tax_id",
                        "virus_taxonomic_id",
                        "tax_id",
                        "organism_tax_id",
                        "organism_taxonomic_id",
                    ],
                ),
                virus_name=first_present(row, ["virus_name", "organism_name"]),
                host_tax_id=first_present(row, ["host_tax_id", "host_taxonomic_id"]),
                host_name=first_present(row, ["host_name"]),
                host_common_name=first_present(row, ["host_common_name"]),
                completeness=first_present(row, ["completeness"]),
                length=parse_int(first_present(row, ["length", "genome_length"])),
                is_lab_host=first_present(row, ["is_lab_host"]),
                is_vaccine_strain=first_present(row, ["is_vaccine_strain"]),
            )
        )
    return records


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "y", "1"}


def is_environmental_host(record: VirusMeta) -> bool:
    text = " ".join([record.host_name, record.host_common_name]).lower()
    environmental_terms = [
        "metagenome",
        "environmental",
        "activated sludge",
        "wastewater",
        "sewage",
        "soil",
        "freshwater",
        "marine",
        "sediment",
    ]
    return any(term in text for term in environmental_terms)


def has_human_host(record: VirusMeta) -> bool:
    host_tax_ids = {x.strip() for x in record.host_tax_id.replace(";", ",").split(",")}
    host_names = {
        x.strip().lower()
        for x in (record.host_name + "," + record.host_common_name).replace(";", ",").split(",")
    }
    return HUMAN_TAX_ID in host_tax_ids or "homo sapiens" in host_names or "human" in host_names


def has_any_host(record: VirusMeta) -> bool:
    return bool(record.host_tax_id.strip() or record.host_name.strip() or record.host_common_name.strip())


def select_records(
    records: List[VirusMeta],
    target_per_class: int,
    min_genome_length: int,
    seed: int,
    exclude_environmental_hosts: bool,
    max_per_virus_tax_id: int | None,
) -> List[Tuple[VirusMeta, int]]:
    rng = random.Random(seed)
    positives: List[VirusMeta] = []
    negatives: List[VirusMeta] = []
    seen_accessions = set()
    per_taxon_counts: Dict[str, int] = defaultdict(int)

    for record in records:
        if record.accession in seen_accessions:
            continue
        seen_accessions.add(record.accession)
        if record.length and record.length < min_genome_length:
            continue
        if is_truthy(record.is_lab_host) or is_truthy(record.is_vaccine_strain):
            continue
        if not has_any_host(record):
            continue
        if exclude_environmental_hosts and is_environmental_host(record):
            continue
        if max_per_virus_tax_id is not None and max_per_virus_tax_id > 0:
            tax_id = record.virus_tax_id or record.accession
            if per_taxon_counts[tax_id] >= max_per_virus_tax_id:
                continue
        if has_human_host(record):
            positives.append(record)
        else:
            negatives.append(record)
        if max_per_virus_tax_id is not None and max_per_virus_tax_id > 0:
            tax_id = record.virus_tax_id or record.accession
            per_taxon_counts[tax_id] += 1

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(target_per_class, len(positives), len(negatives))
    if n == 0:
        raise RuntimeError(
            f"No usable examples after filtering: positives={len(positives)}, negatives={len(negatives)}"
        )
    if n < target_per_class:
        print(
            f"Warning: requested {target_per_class} per class but only found {n}. "
            f"Available positives={len(positives)}, negatives={len(negatives)}."
        )
    return [(record, 1) for record in positives[:n]] + [(record, 0) for record in negatives[:n]]


def write_accessions(records: Sequence[Tuple[VirusMeta, int]], path: str) -> None:
    with open(path, "w") as f:
        for record, _label in records:
            f.write(record.accession + "\n")


def iter_fasta(path: str) -> Iterable[Tuple[str, str]]:
    header = None
    parts: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:]
                parts = []
            else:
                parts.append(line)
        if header is not None:
            yield header, "".join(parts)


def accession_from_header(header: str) -> str:
    token = header.split()[0]
    if "|" in token:
        token = token.split("|")[-1]
    return token


def find_genomic_fasta(root: str) -> List[str]:
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename in {"genomic.fna", "genomic.fa", "genomic.fasta"} or filename.endswith(".fna"):
                matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def load_sequences(download_zip: str, extract_dir: str) -> Dict[str, str]:
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(download_zip) as zf:
        zf.extractall(extract_dir)

    sequences: Dict[str, str] = {}
    fasta_paths = find_genomic_fasta(extract_dir)
    if not fasta_paths:
        raise FileNotFoundError(f"No genomic FASTA found after extracting {download_zip}")
    for fasta_path in fasta_paths:
        for header, seq in iter_fasta(fasta_path):
            accession = accession_from_header(header)
            cleaned = clean_sequence(seq)
            if cleaned:
                sequences[accession] = cleaned
    return sequences


def assign_group_splits(
    records: Sequence[Tuple[VirusMeta, int]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for record, label in records:
        group = record.virus_tax_id or record.accession
        groups[group].append((record.accession, label))

    accession_to_split: Dict[str, str] = {}
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)

    label_totals = {0: 0, 1: 0}
    for members in groups.values():
        for _accession, label in members:
            label_totals[label] += 1
    targets = {
        split: {
            0: label_totals[0] * frac,
            1: label_totals[1] * frac,
        }
        for split, frac in [
            ("train", train_frac),
            ("val", val_frac),
            ("test", max(0.0, 1.0 - train_frac - val_frac)),
        ]
    }
    split_counts = {
        "train": {0: 0, 1: 0},
        "val": {0: 0, 1: 0},
        "test": {0: 0, 1: 0},
    }
    for group_id in group_ids:
        group_label_counts = {0: 0, 1: 0}
        for _accession, label in groups[group_id]:
            group_label_counts[label] += 1

        def split_cost(split: str) -> float:
            cost = 0.0
            for label in [0, 1]:
                after = split_counts[split][label] + group_label_counts[label]
                target = max(targets[split][label], 1.0)
                cost += (after / target) ** 2
            return cost

        split = min(["train", "val", "test"], key=split_cost)
        for label in [0, 1]:
            split_counts[split][label] += group_label_counts[label]
        for accession, _label in groups[group_id]:
            accession_to_split[accession] = split
    return accession_to_split


def clean_output_artifacts(out_dir: str) -> None:
    for name in [
        "manifest.csv",
        "features",
        "features_normed",
        "probes",
        "probes_normed",
        "baselines",
        "feature_diagnostics.csv",
    ]:
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


def clean_raw_artifacts(raw_dir: str) -> None:
    for name in [
        "refseq_virus.jsonl",
        "refseq_virus_metadata.tsv",
        "selected_accessions.txt",
        "selected_virus_genomes.zip",
        "selected_virus_genomes",
        "split_summary.tsv",
    ]:
        path = os.path.join(raw_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


def write_split_summary(
    records: Sequence[Tuple[VirusMeta, int]],
    out_path: str,
    accession_to_split: Dict[str, str],
) -> None:
    rows = []
    grouped: Dict[Tuple[str, int, str, str, str], int] = defaultdict(int)
    for record, label in records:
        split = accession_to_split[record.accession]
        key = (split, label, record.virus_tax_id or "", record.virus_name, record.host_name)
        grouped[key] += 1
    for (split, label, virus_tax_id, virus_name, host_name), count in grouped.items():
        rows.append(
            {
                "split": split,
                "label": label,
                "virus_tax_id": virus_tax_id,
                "virus_name": virus_name,
                "host_name": host_name,
                "count": count,
            }
        )
    rows.sort(key=lambda row: (row["split"], row["label"], -row["count"], row["virus_tax_id"]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "virus_tax_id", "virus_name", "host_name", "count"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    records: Sequence[Tuple[VirusMeta, int]],
    sequences: Dict[str, str],
    out_path: str,
    max_length: int,
    min_length: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    summary_path: str | None = None,
) -> None:
    rng = random.Random(seed)
    accession_to_split = assign_group_splits(records, train_frac, val_frac, seed)
    fieldnames = [
        "id",
        "label",
        "split",
        "sequence",
        "source",
        "length",
        "accession",
        "virus_tax_id",
        "virus_name",
        "host_tax_id",
        "host_name",
        "host_common_name",
    ]
    rows = []
    missing = 0
    for record, label in records:
        seq = sequences.get(record.accession)
        if seq is None:
            missing += 1
            continue
        if len(seq) < min_length:
            continue
        window = sample_window(seq, max_length, rng)
        if len(window) < min_length:
            continue
        rows.append(
            {
                "id": f"{record.accession}|host_tropism|{label}",
                "label": label,
                "split": accession_to_split[record.accession],
                "sequence": window,
                "source": record.virus_name,
                "length": len(window),
                "accession": record.accession,
                "virus_tax_id": record.virus_tax_id,
                "virus_name": record.virus_name,
                "host_tax_id": record.host_tax_id,
                "host_name": record.host_name,
                "host_common_name": record.host_common_name,
            }
        )

    if missing:
        print(f"Warning: {missing} selected accessions were missing from downloaded FASTA.")
    if not rows:
        raise RuntimeError("No manifest rows remained after sequence filtering.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for row in rows:
        counts[(row["split"], row["label"])] += 1
    print(f"Wrote {len(rows)} rows to {out_path}")
    for split in ["train", "val", "test"]:
        print(
            f"{split}: negative={counts[(split, 0)]} positive={counts[(split, 1)]}",
            flush=True,
        )
    if summary_path is not None:
        write_split_summary(records, summary_path, accession_to_split)
        print(f"Wrote split summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build human-tropic vs non-human-tropic viral RefSeq manifest."
    )
    parser.add_argument("--out-dir", default="data/host_tropism")
    parser.add_argument("--target-per-class", type=int, default=5000)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-genome-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--refseq-only",
        action="store_true",
        help="Restrict metadata to RefSeq viral genomes. Default uses all NCBI viral genomes.",
    )
    parser.add_argument(
        "--include-environmental-hosts",
        action="store_true",
        help="Keep environmental/metagenome host annotations in the negative class.",
    )
    parser.add_argument(
        "--max-per-virus-tax-id",
        type=int,
        default=0,
        help="Cap the number of records kept per virus taxonomic ID. 0 disables the cap.",
    )
    parser.add_argument(
        "--positive-host",
        default="",
        help="Optional NCBI host filter for positives, e.g. 'human'.",
    )
    parser.add_argument(
        "--negative-hosts",
        default="",
        help=(
            "Comma-separated NCBI host filters for negatives. If provided, metadata "
            "is fetched separately for each host and concatenated."
        ),
    )
    parser.add_argument(
        "--query-limit",
        default="all",
        help="NCBI Datasets --limit value for each host query.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete cached raw metadata/genome files under data/host_tropism/raw and rebuild them.",
    )
    args = parser.parse_args()

    require_cli("datasets")
    require_cli("dataformat")

    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    clean_output_artifacts(args.out_dir)
    if args.from_scratch:
        clean_raw_artifacts(raw_dir)
    summary_jsonl = os.path.join(raw_dir, "refseq_virus.jsonl")
    metadata_tsv = os.path.join(raw_dir, "refseq_virus_metadata.tsv")
    accessions_path = os.path.join(raw_dir, "selected_accessions.txt")
    download_zip = os.path.join(raw_dir, "selected_virus_genomes.zip")
    extract_dir = os.path.join(raw_dir, "selected_virus_genomes")
    split_summary_path = os.path.join(raw_dir, "split_summary.tsv")

    if not args.manifest_only:
        with open(summary_jsonl, "w") as f:
            if args.positive_host or args.negative_hosts:
                if args.positive_host:
                    run_summary_query(f, args.refseq_only, args.positive_host, args.query_limit)
                for host in [x.strip() for x in args.negative_hosts.split(",") if x.strip()]:
                    run_summary_query(f, args.refseq_only, host, args.query_limit)
            else:
                run_summary_query(f, args.refseq_only, None, args.query_limit)
        with open(metadata_tsv, "w") as f:
            subprocess.run(
                [
                    "dataformat",
                    "tsv",
                    "virus-genome",
                    "--inputfile",
                    summary_jsonl,
                    "--fields",
                    (
                        "accession,virus-tax-id,virus-name,host-tax-id,host-name,"
                        "host-common-name,completeness,length,is-lab-host,is-vaccine-strain"
                    ),
                ],
                check=True,
                stdout=f,
            )

    records = parse_metadata(metadata_tsv)
    selected = select_records(
        records,
        args.target_per_class,
        args.min_genome_length,
        args.seed,
        exclude_environmental_hosts=not args.include_environmental_hosts,
        max_per_virus_tax_id=args.max_per_virus_tax_id or None,
    )
    write_accessions(selected, accessions_path)
    print(
        f"Selected {sum(label == 1 for _record, label in selected)} positives and "
        f"{sum(label == 0 for _record, label in selected)} negatives."
    )

    if args.metadata_only:
        return

    if not args.manifest_only or not os.path.exists(download_zip):
        run_command(
            [
                "datasets",
                "download",
                "virus",
                "genome",
                "accession",
                "--inputfile",
                accessions_path,
                "--include",
                "genome",
                "--filename",
                download_zip,
            ]
        )

    sequences = load_sequences(download_zip, extract_dir)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    write_manifest(
        selected,
        sequences,
        manifest_path,
        args.max_length,
        args.min_length,
        args.seed,
        args.train_frac,
        args.val_frac,
        summary_path=split_summary_path,
    )


if __name__ == "__main__":
    main()
