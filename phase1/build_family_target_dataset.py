import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class TaxonomySequenceRecord:
    split: str
    taxid: str
    seq_id: str
    family: str
    genus: str
    species: str
    scientific_name: str
    na_type: str
    seq_len: int
    sequence: str


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^ACGTN]", "", seq.upper())


def sample_window(seq: str, max_length: int, rng: random.Random) -> str:
    if len(seq) <= max_length:
        return seq
    start = rng.randint(0, len(seq) - max_length)
    return seq[start : start + max_length]


def gc_fraction(seq: str) -> float:
    length = max(len(seq), 1)
    return (seq.count("G") + seq.count("C")) / length


def load_sequence_lists(path: str) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            grouped[str(obj["taxid"])] = [clean_sequence(seq) for seq in obj["sequences"]]
    return grouped


def iter_split_records(root: str, split: str) -> Iterable[TaxonomySequenceRecord]:
    csv_path = os.path.join(root, f"{split}.csv")
    seq_path = os.path.join(root, f"{split}_sequences.jsonl")
    sequences_by_taxid = load_sequence_lists(seq_path)
    per_taxid_index = defaultdict(int)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            taxid = normalize_text(row.get("taxid"))
            seqs = sequences_by_taxid.get(taxid)
            idx = per_taxid_index[taxid]
            per_taxid_index[taxid] += 1
            if not seqs or idx >= len(seqs):
                continue
            sequence = clean_sequence(seqs[idx])
            if not sequence:
                continue
            yield TaxonomySequenceRecord(
                split=split,
                taxid=taxid,
                seq_id=normalize_text(row.get("seq_id")),
                family=normalize_text(row.get("family")),
                genus=normalize_text(row.get("genus")),
                species=normalize_text(row.get("species")),
                scientific_name=normalize_text(row.get("scientific_name")),
                na_type=normalize_text(row.get("na_type")),
                seq_len=int(float(row.get("seq_len", len(sequence)) or len(sequence))),
                sequence=sequence,
            )


def load_records(root: str) -> List[TaxonomySequenceRecord]:
    records: List[TaxonomySequenceRecord] = []
    for split in ("train", "val", "test"):
        records.extend(iter_split_records(root, split))
    return records


def choose_length_bin(length: int, width: int) -> int:
    return int(length // max(width, 1))


def dedupe_records(records: Sequence[TaxonomySequenceRecord]) -> List[TaxonomySequenceRecord]:
    seen = set()
    deduped = []
    for record in records:
        if record.sequence in seen:
            continue
        seen.add(record.sequence)
        deduped.append(record)
    return deduped


def assign_group_splits(
    records: Sequence[TaxonomySequenceRecord],
    group_field: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    groups: Dict[str, List[TaxonomySequenceRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, group_field) or record.taxid].append(record)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)

    total = sum(len(items) for items in groups.values())
    targets = {
        "train": total * train_frac,
        "val": total * val_frac,
        "test": total * max(0.0, 1.0 - train_frac - val_frac),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignment: Dict[str, str] = {}

    for group_id in group_ids:
        group_size = len(groups[group_id])

        def split_cost(split: str) -> float:
            after = counts[split] + group_size
            target = max(targets[split], 1.0)
            return (after / target) ** 2

        split = min(["train", "val", "test"], key=split_cost)
        counts[split] += group_size
        assignment[group_id] = split

    if not any(split == "val" for split in assignment.values()) or not any(split == "test" for split in assignment.values()):
        raise RuntimeError(f"Could not produce non-empty val/test held-out {group_field} groups.")
    return assignment


def target_group_values(records: Sequence[TaxonomySequenceRecord], field: str) -> Dict[str, set]:
    result: Dict[str, set] = defaultdict(set)
    for record in records:
        result[record.split].add(getattr(record, field))
    return result


def check_target_heldout(records: Sequence[TaxonomySequenceRecord], field: str) -> None:
    groups = target_group_values(records, field)
    train = groups.get("train", set())
    leaked = (groups.get("val", set()) | groups.get("test", set())) & train
    if leaked:
        leaked_preview = ", ".join(sorted(x for x in leaked if x)[:10])
        raise RuntimeError(f"Target {field} leakage across splits: {leaked_preview}")


def select_negative_records(
    positive_records: Sequence[TaxonomySequenceRecord],
    negative_pool: Sequence[TaxonomySequenceRecord],
    seed: int,
    max_per_family_per_split: int,
    length_bin_width: int,
) -> List[TaxonomySequenceRecord]:
    rng = random.Random(seed)
    by_bin: Dict[tuple, List[TaxonomySequenceRecord]] = defaultdict(list)
    for record in negative_pool:
        key = (record.na_type, choose_length_bin(len(record.sequence), length_bin_width))
        by_bin[key].append(record)
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    positive_targets = Counter(
        (record.split, record.na_type, choose_length_bin(len(record.sequence), length_bin_width))
        for record in positive_records
    )
    family_counts: Dict[tuple, int] = defaultdict(int)
    selected: List[TaxonomySequenceRecord] = []
    used_ids = set()

    for key, target_count in sorted(positive_targets.items()):
        target_split, na_type, length_bin = key
        bucket = by_bin.get((na_type, length_bin), [])
        chosen_here = 0
        for record in bucket:
            family_key = (target_split, record.family)
            dedupe_key = (record.taxid, record.seq_id, record.sequence)
            if family_counts[family_key] >= max_per_family_per_split:
                continue
            if dedupe_key in used_ids:
                continue
            selected.append(
                TaxonomySequenceRecord(
                    split=target_split,
                    taxid=record.taxid,
                    seq_id=record.seq_id,
                    family=record.family,
                    genus=record.genus,
                    species=record.species,
                    scientific_name=record.scientific_name,
                    na_type=record.na_type,
                    seq_len=record.seq_len,
                    sequence=record.sequence,
                )
            )
            used_ids.add(dedupe_key)
            family_counts[family_key] += 1
            chosen_here += 1
            if chosen_here >= target_count:
                break

        if chosen_here < target_count:
            fallback = [
                record
                for record in negative_pool
                if (record.taxid, record.seq_id, record.sequence) not in used_ids
            ]
            rng.shuffle(fallback)
            for record in fallback:
                family_key = (target_split, record.family)
                if family_counts[family_key] >= max_per_family_per_split:
                    continue
                selected.append(
                    TaxonomySequenceRecord(
                        split=target_split,
                        taxid=record.taxid,
                        seq_id=record.seq_id,
                        family=record.family,
                        genus=record.genus,
                        species=record.species,
                        scientific_name=record.scientific_name,
                        na_type=record.na_type,
                        seq_len=record.seq_len,
                        sequence=record.sequence,
                    )
                )
                used_ids.add((record.taxid, record.seq_id, record.sequence))
                family_counts[family_key] += 1
                chosen_here += 1
                if chosen_here >= target_count:
                    break

        if chosen_here < target_count:
            raise RuntimeError(f"Negative matching failed for bucket={key}: need {target_count}, found {chosen_here}.")

    return selected


def summarize_rows(rows: Sequence[dict]) -> dict:
    split_label = Counter((row["split"], int(row["label"])) for row in rows)
    family_counts = Counter((row["split"], row["family"], int(row["label"])) for row in rows)
    genus_counts = Counter((row["split"], row["genus"], int(row["label"])) for row in rows)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    split_stats = {}
    for split, split_rows in by_split.items():
        gc_values = [gc_fraction(row["sequence"]) for row in split_rows]
        lengths = [len(row["sequence"]) for row in split_rows]
        split_stats[split] = {
            "n_rows": len(split_rows),
            "mean_length": mean(lengths) if lengths else math.nan,
            "mean_gc": mean(gc_values) if gc_values else math.nan,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Coronaviridae family-target manifest from local ViroBench taxonomy data.")
    parser.add_argument("--source-root", default="data/benchmarks/raw/virobench/ViroBench-CLS-Lite/ALL/taxon/genus")
    parser.add_argument("--out-dir", default="data/family_targets/coronaviridae")
    parser.add_argument("--target-family", default="Coronaviridae")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-target-per-split", type=int, default=0, help="0 keeps all target-family records.")
    parser.add_argument("--max-per-target-genus", type=int, default=50)
    parser.add_argument("--max-per-target-species", type=int, default=100)
    parser.add_argument("--max-negative-family-per-split", type=int, default=200)
    parser.add_argument("--length-bin-width", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--target-group-field", choices=["auto", "genus", "species"], default="auto")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = dedupe_records(load_records(args.source_root))
    positives = [record for record in records if record.family == args.target_family]
    negatives = [record for record in records if record.family and record.family != args.target_family]
    if not positives:
        raise RuntimeError(f"No records found for target family {args.target_family!r}.")

    target_genus_counts: Dict[tuple, int] = defaultdict(int)
    target_species_counts: Dict[tuple, int] = defaultdict(int)
    filtered_positives = []
    for record in positives:
        genus_key = (record.split, record.genus or record.taxid)
        species_key = (record.split, record.species or record.taxid)
        if target_genus_counts[genus_key] >= args.max_per_target_genus:
            continue
        if target_species_counts[species_key] >= args.max_per_target_species:
            continue
        filtered_positives.append(record)
        target_genus_counts[genus_key] += 1
        target_species_counts[species_key] += 1

    candidate_group_fields = ["genus", "species"] if args.target_group_field == "auto" else [args.target_group_field]
    target_group_field = None
    target_split_assignment = None
    last_error = None
    for candidate_group_field in candidate_group_fields:
        try:
            target_split_assignment = assign_group_splits(
                filtered_positives,
                group_field=candidate_group_field,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                seed=args.seed,
            )
            target_group_field = candidate_group_field
            break
        except RuntimeError as exc:
            last_error = exc
    if target_split_assignment is None or target_group_field is None:
        raise RuntimeError(f"Could not assign held-out target splits: {last_error}")
    filtered_positives = [
        TaxonomySequenceRecord(
            split=target_split_assignment[getattr(record, target_group_field) or record.taxid],
            taxid=record.taxid,
            seq_id=record.seq_id,
            family=record.family,
            genus=record.genus,
            species=record.species,
            scientific_name=record.scientific_name,
            na_type=record.na_type,
            seq_len=record.seq_len,
            sequence=record.sequence,
        )
        for record in filtered_positives
    ]

    if args.max_target_per_split > 0:
        capped = []
        by_split = defaultdict(list)
        for record in filtered_positives:
            by_split[record.split].append(record)
        for split, split_records in by_split.items():
            rng.shuffle(split_records)
            capped.extend(split_records[: args.max_target_per_split])
        filtered_positives = capped

    if target_group_field == "genus":
        check_target_heldout(filtered_positives, "genus")
    check_target_heldout(filtered_positives, "species")

    matched_negatives = select_negative_records(
        positive_records=filtered_positives,
        negative_pool=negatives,
        seed=args.seed,
        max_per_family_per_split=args.max_negative_family_per_split,
        length_bin_width=args.length_bin_width,
    )

    rows = []
    seen_sequences = set()
    for label, source_records in [(1, filtered_positives), (0, matched_negatives)]:
        for record in source_records:
            if len(record.sequence) < args.min_length:
                continue
            window = sample_window(record.sequence, args.max_length, rng)
            if len(window) < args.min_length or window in seen_sequences:
                continue
            seen_sequences.add(window)
            rows.append(
                {
                    "id": f"{record.taxid}|{record.seq_id}|{args.target_family.lower()}|{label}",
                    "label": label,
                    "split": record.split,
                    "sequence": window,
                    "source": record.scientific_name,
                    "length": len(window),
                    "accession": record.seq_id,
                    "tax_id": record.taxid,
                    "family": record.family,
                    "genus": record.genus,
                    "species": record.species,
                    "na_type": record.na_type,
                }
            )

    if not rows:
        raise RuntimeError("No rows remained after filtering/windowing.")

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
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    summary["target_family"] = args.target_family
    summary["target_group_field"] = target_group_field
    summary["heldout_target_genera_val_test"] = sorted(
        ({row["genus"] for row in rows if row["label"] == 1 and row["split"] in {"val", "test"}})
        - ({row["genus"] for row in rows if row["label"] == 1 and row["split"] == "train"})
    )
    summary["heldout_target_species_val_test"] = sorted(
        ({row["species"] for row in rows if row["label"] == 1 and row["split"] in {"val", "test"}})
        - ({row["species"] for row in rows if row["label"] == 1 and row["split"] == "train"})
    )
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(summary["split_label_counts"], indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
