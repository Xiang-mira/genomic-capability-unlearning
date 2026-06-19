import argparse
import concurrent.futures
import csv
import gzip
import json
import os
import random
import re
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm


TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
VIRAL_ASSEMBLY_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/viral/assembly_summary.txt"


@dataclass
class Taxonomy:
    parent: Dict[str, str]
    rank: Dict[str, str]
    name: Dict[str, str]
    name_to_taxids: Dict[str, List[str]]


@dataclass
class AssemblyRecord:
    accession: str
    taxid: str
    species_taxid: str
    organism_name: str
    assembly_level: str
    ftp_path: str


@dataclass
class SequenceRecord:
    accession: str
    taxid: str
    species_taxid: str
    family: str
    genus: str
    species: str
    scientific_name: str
    sequence_id: str
    sequence: str


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^ACGTN]", "", seq.upper())


def sample_windows(seq: str, max_length: int, n_windows: int, rng: random.Random) -> List[str]:
    if len(seq) <= max_length:
        return [seq]
    windows = []
    seen_starts = set()
    max_start = len(seq) - max_length
    for _ in range(n_windows):
        if len(seen_starts) > max_start:
            break
        start = rng.randint(0, max_start)
        while start in seen_starts and len(seen_starts) <= max_start:
            start = rng.randint(0, max_start)
        seen_starts.add(start)
        windows.append(seq[start : start + max_length])
    return windows


def gc_fraction(seq: str) -> float:
    length = max(len(seq), 1)
    return (seq.count("G") + seq.count("C")) / length


def download_file(
    url: str,
    out_path: str,
    timeout: int = 120,
    desc: str | None = None,
    retries: int = 3,
    quiet_cached: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        if not quiet_cached:
            print(f"[refseq] using cached {out_path}", flush=True)
        return
    tmp_path = out_path + ".part"
    label = desc or os.path.basename(out_path)
    for attempt in range(1, retries + 1):
        existing_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        mode = "ab" if existing_size else "wb"
        action = "resuming" if existing_size else "downloading"
        print(
            f"[refseq] {action} {label} from {url}"
            + (f" at byte {existing_size}" if existing_size else ""),
            flush=True,
        )
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if existing_size and response.status != 206:
                    existing_size = 0
                    mode = "wb"
                content_length = response.headers.get("Content-Length")
                total = int(content_length) + existing_size if content_length else None
                with open(tmp_path, mode) as out, tqdm(
                    total=total,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    desc=label,
                ) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        pbar.update(len(chunk))
            os.replace(tmp_path, out_path)
            return
        except Exception as exc:
            if attempt >= retries:
                raise
            print(f"[refseq] download failed for {label} ({exc}); retry {attempt}/{retries}", flush=True)
            time.sleep(min(30, 2**attempt))


def parse_dmp_line(line: str) -> List[str]:
    return [part.strip() for part in line.rstrip("\n").split("|")]


def ensure_taxdump(raw_dir: str) -> str:
    tax_dir = os.path.join(raw_dir, "taxdump")
    nodes_path = os.path.join(tax_dir, "nodes.dmp")
    names_path = os.path.join(tax_dir, "names.dmp")
    if os.path.exists(nodes_path) and os.path.exists(names_path):
        return tax_dir

    os.makedirs(tax_dir, exist_ok=True)
    archive_path = os.path.join(raw_dir, "taxdump.tar.gz")
    download_file(TAXDUMP_URL, archive_path, desc="NCBI taxdump.tar.gz")
    print("[refseq] extracting NCBI taxdump", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name in {"nodes.dmp", "names.dmp"}:
                tar.extract(member, tax_dir)
    return tax_dir


def load_taxonomy(raw_dir: str) -> Taxonomy:
    print("[refseq] loading NCBI taxonomy", flush=True)
    tax_dir = ensure_taxdump(raw_dir)
    parent: Dict[str, str] = {}
    rank: Dict[str, str] = {}
    name: Dict[str, str] = {}
    name_to_taxids: Dict[str, List[str]] = defaultdict(list)

    with open(os.path.join(tax_dir, "nodes.dmp")) as f:
        for line in f:
            parts = parse_dmp_line(line)
            if len(parts) >= 3:
                taxid, parent_taxid, node_rank = parts[:3]
                parent[taxid] = parent_taxid
                rank[taxid] = node_rank

    with open(os.path.join(tax_dir, "names.dmp")) as f:
        for line in f:
            parts = parse_dmp_line(line)
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid, scientific_name = parts[:2]
                name[taxid] = scientific_name
                name_to_taxids[scientific_name.lower()].append(taxid)

    return Taxonomy(parent=parent, rank=rank, name=name, name_to_taxids=dict(name_to_taxids))


def lineage(taxid: str, taxonomy: Taxonomy) -> List[str]:
    result = []
    current = taxid
    seen = set()
    while current and current not in seen:
        result.append(current)
        seen.add(current)
        next_taxid = taxonomy.parent.get(current)
        if not next_taxid or next_taxid == current:
            break
        current = next_taxid
    return result


def lineage_rank_taxid(taxid: str, taxonomy: Taxonomy, target_rank: str) -> str:
    for ancestor in lineage(taxid, taxonomy):
        if taxonomy.rank.get(ancestor) == target_rank:
            return ancestor
    return ""


def lineage_rank_name(taxid: str, taxonomy: Taxonomy, target_rank: str) -> str:
    rank_taxid = lineage_rank_taxid(taxid, taxonomy, target_rank)
    return taxonomy.name.get(rank_taxid, "") if rank_taxid else ""


def resolve_target_family_taxid(target_family: str, explicit_taxid: str, taxonomy: Taxonomy) -> str:
    if explicit_taxid:
        return explicit_taxid
    candidates = taxonomy.name_to_taxids.get(target_family.lower(), [])
    family_candidates = [taxid for taxid in candidates if taxonomy.rank.get(taxid) == "family"]
    if not family_candidates:
        raise RuntimeError(f"Could not resolve family taxid for {target_family!r}.")
    return family_candidates[0]


def is_descendant_of(taxid: str, ancestor_taxid: str, taxonomy: Taxonomy) -> bool:
    return ancestor_taxid in lineage(taxid, taxonomy)


def ensure_assembly_summary(raw_dir: str) -> str:
    path = os.path.join(raw_dir, "assembly_summary_refseq_viral.txt")
    download_file(VIRAL_ASSEMBLY_SUMMARY_URL, path, desc="NCBI RefSeq viral assembly summary")
    return path


def load_viral_assemblies(raw_dir: str) -> List[AssemblyRecord]:
    print("[refseq] loading RefSeq viral assembly summary", flush=True)
    path = ensure_assembly_summary(raw_dir)
    with open(path) as f:
        header = None
        records = []
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            if header is None:
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            ftp_path = row.get("ftp_path", "")
            if not ftp_path or ftp_path == "na":
                continue
            records.append(
                AssemblyRecord(
                    accession=normalize_text(row.get("assembly_accession")),
                    taxid=normalize_text(row.get("taxid")),
                    species_taxid=normalize_text(row.get("species_taxid")),
                    organism_name=normalize_text(row.get("organism_name")),
                    assembly_level=normalize_text(row.get("assembly_level")),
                    ftp_path=ftp_path,
                )
            )
        print(f"[refseq] loaded {len(records)} viral assemblies with FASTA paths", flush=True)
        return records


def assembly_fna_url(assembly: AssemblyRecord) -> str:
    base = assembly.ftp_path.rstrip("/")
    name = os.path.basename(base)
    return f"{base}/{name}_genomic.fna.gz"


def assembly_fna_path(raw_dir: str, assembly: AssemblyRecord) -> str:
    filename = os.path.basename(assembly_fna_url(assembly))
    return os.path.join(raw_dir, "assemblies", filename)


def iter_fasta_gz(path: str) -> Iterable[tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def sequence_id_from_header(header: str) -> str:
    return header.split()[0]


def collect_records(
    assemblies: Sequence[AssemblyRecord],
    taxonomy: Taxonomy,
    raw_dir: str,
    family_name: str,
    max_records: int,
    min_length: int,
    download: bool,
    desc: str,
    skip_download_errors: bool,
    download_workers: int,
) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    skipped_downloads = 0
    total = len(assemblies) if max_records <= 0 else None

    if download and download_workers > 1:
        download_limit = len(assemblies) if max_records <= 0 else min(len(assemblies), max_records * 3)
        download_assemblies = list(assemblies[:download_limit])
        print(
            f"[refseq] prefetching up to {len(download_assemblies)} {desc} FASTA files "
            f"with {download_workers} workers",
            flush=True,
        )

        def fetch(assembly: AssemblyRecord) -> tuple[str, bool, str]:
            try:
                download_file(
                    assembly_fna_url(assembly),
                    assembly_fna_path(raw_dir, assembly),
                    desc=f"{assembly.accession} genomic.fna.gz",
                    quiet_cached=True,
                )
                return assembly.accession, True, ""
            except Exception as exc:
                return assembly.accession, False, str(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {executor.submit(fetch, assembly): assembly for assembly in download_assemblies}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"download {desc}",
                unit="assembly",
            ):
                accession, ok, error = future.result()
                if not ok:
                    if not skip_download_errors:
                        raise RuntimeError(f"Download failed for {accession}: {error}")
                    skipped_downloads += 1
                    print(f"[refseq] warning: skipped {accession} after download failure: {error}", flush=True)

    progress = tqdm(assemblies, total=total, desc=desc, unit="assembly")
    for assembly in progress:
        path = assembly_fna_path(raw_dir, assembly)
        if download and not os.path.exists(path):
            try:
                download_file(assembly_fna_url(assembly), path, desc=f"{assembly.accession} genomic.fna.gz")
            except Exception as exc:
                if not skip_download_errors:
                    raise
                skipped_downloads += 1
                progress.set_postfix(records=len(records), skipped=skipped_downloads)
                print(
                    f"[refseq] warning: skipped {assembly.accession} after download failure: {exc}",
                    flush=True,
                )
                continue
        if not os.path.exists(path):
            continue
        family = lineage_rank_name(assembly.taxid, taxonomy, "family") or "unclassified"
        genus = lineage_rank_name(assembly.taxid, taxonomy, "genus")
        species = lineage_rank_name(assembly.taxid, taxonomy, "species")
        for header, seq in iter_fasta_gz(path):
            seq = clean_sequence(seq)
            if len(seq) < min_length:
                continue
            records.append(
                SequenceRecord(
                    accession=assembly.accession,
                    taxid=assembly.taxid,
                    species_taxid=assembly.species_taxid,
                    family=family,
                    genus=genus,
                    species=species,
                    scientific_name=assembly.organism_name,
                    sequence_id=sequence_id_from_header(header),
                    sequence=seq,
                )
            )
            progress.set_postfix(records=len(records), skipped=skipped_downloads)
            if max_records > 0 and len(records) >= max_records:
                progress.close()
                return records
    return records


def group_key(record: SequenceRecord, group_field: str) -> str:
    if group_field == "genus":
        return record.genus or record.taxid
    if group_field == "species":
        return record.species or record.species_taxid or record.taxid
    if group_field == "assembly":
        return record.accession
    raise ValueError(f"Unsupported group field: {group_field}")


def assign_group_splits(
    records: Sequence[SequenceRecord],
    group_field: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    groups: Dict[str, List[SequenceRecord]] = defaultdict(list)
    for record in records:
        groups[group_key(record, group_field)].append(record)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda key: len(groups[key]), reverse=True)

    total = sum(len(items) for items in groups.values())
    targets = {
        "train": total * train_frac,
        "val": total * val_frac,
        "test": total * max(0.0, 1.0 - train_frac - val_frac),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignment: Dict[str, str] = {}
    for key in group_ids:
        group_size = len(groups[key])

        def split_cost(split: str) -> float:
            target = max(targets[split], 1.0)
            return ((counts[split] + group_size) / target) ** 2

        split = min(["train", "val", "test"], key=split_cost)
        counts[split] += group_size
        assignment[key] = split

    if not any(split == "val" for split in assignment.values()) or not any(split == "test" for split in assignment.values()):
        raise RuntimeError(f"Could not produce non-empty val/test held-out {group_field} groups.")
    return assignment


def split_positive_records(
    records: Sequence[SequenceRecord],
    group_field: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[str, Dict[str, str]]:
    fields = ["species", "genus", "assembly"] if group_field == "auto" else [group_field]
    last_error = None
    for field in fields:
        try:
            return field, assign_group_splits(records, field, train_frac, val_frac, seed)
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not assign target splits: {last_error}")


def split_counts(records: Sequence[SequenceRecord], assignment: Dict[str, str], group_field: str) -> Counter:
    counts = Counter()
    for record in records:
        counts[assignment[group_key(record, group_field)]] += 1
    return counts


def select_negative_records(
    positive_rows: Sequence[dict],
    negative_records: Sequence[SequenceRecord],
    negative_per_positive: float,
    max_negative_family_per_split: int,
    length_bin_width: int,
    seed: int,
) -> List[tuple[str, SequenceRecord]]:
    rng = random.Random(seed)
    targets = Counter()
    for row in positive_rows:
        target = max(1, int(round(negative_per_positive)))
        key = (row["split"], row["na_type"], int(row["raw_length"]) // max(length_bin_width, 1))
        targets[key] += target

    by_bin: Dict[tuple, List[SequenceRecord]] = defaultdict(list)
    for record in negative_records:
        key = ("DNA", len(record.sequence) // max(length_bin_width, 1))
        by_bin[key].append(record)
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    selected: List[tuple[str, SequenceRecord]] = []
    used = set()
    family_counts = Counter()

    def can_use(split: str, record: SequenceRecord) -> bool:
        dedupe_key = (record.accession, record.sequence_id, record.sequence)
        family_key = (split, record.family)
        if dedupe_key in used:
            return False
        if max_negative_family_per_split > 0 and family_counts[family_key] >= max_negative_family_per_split:
            return False
        return True

    def add_record(split: str, record: SequenceRecord) -> None:
        dedupe_key = (record.accession, record.sequence_id, record.sequence)
        family_key = (split, record.family)
        selected.append((split, record))
        used.add(dedupe_key)
        family_counts[family_key] += 1

    for (split, na_type, length_bin), target_count in sorted(targets.items()):
        chosen = 0
        candidates = by_bin.get((na_type, length_bin), [])
        for record in candidates:
            if not can_use(split, record):
                continue
            add_record(split, record)
            chosen += 1
            if chosen >= target_count:
                break
        exact_chosen = chosen
        if chosen < target_count:
            fallback = sorted(
                negative_records,
                key=lambda record: abs((len(record.sequence) // max(length_bin_width, 1)) - length_bin),
            )
            for record in fallback:
                if not can_use(split, record):
                    continue
                add_record(split, record)
                chosen += 1
                if chosen >= target_count:
                    break
            if chosen < target_count:
                raise RuntimeError(
                    f"Negative matching failed for split={split} length_bin={length_bin}: "
                    f"need {target_count}, found {chosen}. Increase negative collection limits."
                )
            print(
                f"[refseq] warning: filled split={split} length_bin={length_bin} "
                f"with {target_count - exact_chosen} fallback negatives from nearby bins.",
                flush=True,
            )
    return selected


def make_row(
    record: SequenceRecord,
    label: int,
    split: str,
    target_family: str,
    window: str,
    window_index: int,
) -> dict:
    return {
        "id": f"{record.accession}|{record.sequence_id}|{target_family.lower()}|{label}|w{window_index}",
        "label": label,
        "split": split,
        "sequence": window,
        "source": record.scientific_name,
        "length": len(window),
        "accession": record.accession,
        "tax_id": record.taxid,
        "family": record.family,
        "genus": record.genus,
        "species": record.species,
        "na_type": "DNA",
        "raw_length": len(record.sequence),
    }


def rows_from_records(
    labeled_records: Sequence[tuple[int, str, SequenceRecord]],
    target_family: str,
    max_length: int,
    windows_per_sequence: int,
    rng: random.Random,
) -> List[dict]:
    rows = []
    seen_windows = set()
    for label, split, record in labeled_records:
        for window_index, window in enumerate(sample_windows(record.sequence, max_length, windows_per_sequence, rng)):
            if window in seen_windows:
                continue
            seen_windows.add(window)
            rows.append(make_row(record, label, split, target_family, window, window_index))
    return rows


def summarize_rows(rows: Sequence[dict]) -> dict:
    split_label = Counter((row["split"], int(row["label"])) for row in rows)
    family_counts = Counter((row["split"], row["family"], int(row["label"])) for row in rows)
    genus_counts = Counter((row["split"], row["genus"], int(row["label"])) for row in rows)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    split_stats = {}
    for split, split_rows in sorted(by_split.items()):
        lengths = [int(row["length"]) for row in split_rows]
        gcs = [gc_fraction(row["sequence"]) for row in split_rows]
        split_stats[split] = {
            "n_rows": len(split_rows),
            "mean_length": mean(lengths) if lengths else 0.0,
            "mean_gc": mean(gcs) if gcs else 0.0,
        }

    return {
        "split_label_counts": {f"{split}|{label}": count for (split, label), count in sorted(split_label.items())},
        "family_counts_top30": {
            f"{split}|{label}|{family}": count
            for (split, family, label), count in family_counts.most_common(30)
        },
        "genus_counts_top30": {
            f"{split}|{label}|{genus}": count
            for (split, genus, label), count in genus_counts.most_common(30)
        },
        "split_stats": split_stats,
    }


def validate_split_minimums(rows: Sequence[dict], min_val_per_label: int, min_test_per_label: int) -> None:
    counts = Counter((row["split"], int(row["label"])) for row in rows)
    requirements = {
        ("val", 0): min_val_per_label,
        ("val", 1): min_val_per_label,
        ("test", 0): min_test_per_label,
        ("test", 1): min_test_per_label,
    }
    failures = []
    for key, minimum in requirements.items():
        if minimum > 0 and counts[key] < minimum:
            failures.append(f"{key[0]}|{key[1]} has {counts[key]}, below {minimum}")
    if failures:
        raise RuntimeError("Split minimum validation failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a target viral-family manifest from NCBI RefSeq assemblies.")
    parser.add_argument("--out-dir", default="data/family_targets/coronaviridae")
    parser.add_argument("--raw-dir", default="data/refseq_family")
    parser.add_argument("--target-family", default="Coronaviridae")
    parser.add_argument("--target-family-taxid", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--windows-per-sequence", type=int, default=4)
    parser.add_argument("--max-positive-records", type=int, default=0, help="0 keeps all target-family records.")
    parser.add_argument("--negative-per-positive", type=float, default=1.0)
    parser.add_argument(
        "--negative-pool-multiplier",
        type=float,
        default=2.5,
        help="Collect this many candidate negative records per required negative window.",
    )
    parser.add_argument(
        "--max-negative-records",
        type=int,
        default=0,
        help="Optional hard cap on candidate negative sequence records; 0 uses the multiplier.",
    )
    parser.add_argument("--max-negative-family-per-split", type=int, default=500)
    parser.add_argument("--length-bin-width", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--min-val-per-label", type=int, default=0)
    parser.add_argument("--min-test-per-label", type=int, default=0)
    parser.add_argument("--target-group-field", choices=["auto", "genus", "species", "assembly"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true", help="Use already downloaded assembly FASTA files only.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.raw_dir, exist_ok=True)
    taxonomy = load_taxonomy(args.raw_dir)
    target_taxid = resolve_target_family_taxid(args.target_family, args.target_family_taxid, taxonomy)
    print(f"[refseq] target family {args.target_family} taxid={target_taxid}", flush=True)
    assemblies = load_viral_assemblies(args.raw_dir)

    print("[refseq] classifying viral assemblies by taxonomy lineage", flush=True)
    positive_assemblies = [
        assembly
        for assembly in assemblies
        if is_descendant_of(assembly.taxid, target_taxid, taxonomy)
    ]
    negative_assemblies = [
        assembly
        for assembly in assemblies
        if not is_descendant_of(assembly.taxid, target_taxid, taxonomy)
    ]
    rng.shuffle(positive_assemblies)
    rng.shuffle(negative_assemblies)
    print(
        f"[refseq] candidate assemblies: target={len(positive_assemblies)} "
        f"non_target={len(negative_assemblies)}",
        flush=True,
    )

    positives = collect_records(
        positive_assemblies,
        taxonomy,
        args.raw_dir,
        args.target_family,
        args.max_positive_records,
        args.min_length,
        download=not args.manifest_only,
        desc="target RefSeq assemblies",
        skip_download_errors=False,
        download_workers=args.download_workers,
    )
    if not positives:
        raise RuntimeError(f"No RefSeq records found for target family {args.target_family!r}.")
    print(f"[refseq] collected {len(positives)} target sequence records", flush=True)

    group_field, positive_assignment = split_positive_records(
        positives,
        group_field=args.target_group_field,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    positive_labeled = [
        (1, positive_assignment[group_key(record, group_field)], record)
        for record in positives
    ]
    positive_rows = rows_from_records(
        positive_labeled,
        args.target_family,
        args.max_length,
        args.windows_per_sequence,
        rng,
    )
    print(f"[refseq] generated {len(positive_rows)} target windows", flush=True)

    estimated_negative_windows = len(positive_rows) * max(args.negative_per_positive, 1.0)
    max_negative_records = args.max_negative_records or max(
        1,
        int(estimated_negative_windows * args.negative_pool_multiplier),
    )
    print(
        f"[refseq] collecting up to {max_negative_records} candidate negative records "
        f"for ~{int(estimated_negative_windows)} required negative windows",
        flush=True,
    )
    negatives = collect_records(
        negative_assemblies,
        taxonomy,
        args.raw_dir,
        args.target_family,
        max_negative_records,
        args.min_length,
        download=not args.manifest_only,
        desc="negative RefSeq assemblies",
        skip_download_errors=True,
        download_workers=args.download_workers,
    )
    print(f"[refseq] collected {len(negatives)} candidate negative sequence records", flush=True)
    selected_negatives = select_negative_records(
        positive_rows,
        negatives,
        args.negative_per_positive,
        args.max_negative_family_per_split,
        args.length_bin_width,
        args.seed + 1,
    )
    negative_labeled = [(0, split, record) for split, record in selected_negatives]
    negative_rows = rows_from_records(
        negative_labeled,
        args.target_family,
        args.max_length,
        1,
        rng,
    )
    print(f"[refseq] selected {len(negative_rows)} matched negative windows", flush=True)

    rows = positive_rows + negative_rows
    if not rows:
        raise RuntimeError("No rows remained after filtering/windowing.")
    validate_split_minimums(rows, args.min_val_per_label, args.min_test_per_label)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    fieldnames = [
        "id",
        "label",
        "split",
        "sequence",
        "source",
        "length",
        "accession",
        "tax_id",
        "family",
        "genus",
        "species",
        "na_type",
        "raw_length",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    summary.update(
        {
            "source": "NCBI RefSeq viral assemblies",
            "target_family": args.target_family,
            "target_family_taxid": target_taxid,
            "target_group_field": group_field,
            "positive_assembly_candidates": len(positive_assemblies),
            "negative_assembly_candidates": len(negative_assemblies),
            "positive_sequence_records": len(positives),
            "negative_sequence_records_collected": len(negatives),
            "windows_per_sequence": args.windows_per_sequence,
        }
    )
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(summary["split_label_counts"], indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
