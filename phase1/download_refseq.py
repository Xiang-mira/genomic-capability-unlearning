import argparse
import concurrent.futures
import os
import random
import re
import sys
import urllib.request
from typing import List, Tuple

from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import ManifestRecord, clean_sequence, iter_fasta_gz, sample_window, write_manifest


def list_fna_files(base_url: str) -> List[str]:
    with urllib.request.urlopen(base_url) as response:
        html = response.read().decode("utf-8")
    files = set(re.findall(r'href="([^"]+genomic\.fna\.gz)"', html))
    return sorted(files)


def list_local_fna_files(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    files = [f for f in os.listdir(base_dir) if f.endswith("genomic.fna.gz")]
    return sorted(files)


def download_file(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return
    tmp_path = out_path + ".part"
    existing_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) + existing_size if content_length else None
        mode = "ab" if existing_size > 0 else "wb"
        with open(tmp_path, mode) as f, tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(out_path),
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, out_path)


def download_all(pairs: List[Tuple[str, str]], workers: int) -> None:
    if workers <= 1:
        for url, out_path in pairs:
            download_file(url, out_path)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_file, url, out_path) for url, out_path in pairs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def collect_sequences(
    file_paths: List[str],
    label: int,
    target_count: int,
    min_length: int,
    max_length: int,
    rng: random.Random,
    source: str,
    require_full_length: bool,
) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    record_index = 0
    for path in file_paths:
        for header, seq in iter_fasta_gz(path):
            seq = clean_sequence(seq)
            if len(seq) < min_length:
                continue
            if require_full_length and len(seq) < max_length:
                continue
            seq = sample_window(seq, max_length, rng)
            if len(seq) < min_length:
                continue
            record_id = f"{source}-{label}-{record_index}"
            records.append(
                ManifestRecord(
                    record_id=record_id,
                    label=label,
                    split="",
                    sequence=seq,
                    source=header,
                    length=len(seq),
                )
            )
            record_index += 1
            if len(records) >= target_count:
                return records
    return records


def assign_splits(records: List[ManifestRecord], rng: random.Random, train_frac: float, val_frac: float) -> None:
    rng.shuffle(records)
    n_total = len(records)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    for idx, record in enumerate(records):
        if idx < n_train:
            record.split = "train"
        elif idx < n_train + n_val:
            record.split = "val"
        else:
            record.split = "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RefSeq data and build a viral vs non-viral manifest.")
    parser.add_argument("--out-dir", default="data/phase1", help="Output directory for raw data and manifest.")
    parser.add_argument("--target-per-class", type=int, default=10000, help="Number of sequences per class.")
    parser.add_argument("--min-length", type=int, default=200, help="Minimum sequence length after cleaning.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length (windowed).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--viral-max-files", type=int, default=4, help="Max viral files to download.")
    parser.add_argument("--nonviral-max-files", type=int, default=4, help="Max non-viral files to download.")
    parser.add_argument("--nonviral-group", default="bacteria", choices=["bacteria", "archaea", "fungi", "plant", "protozoa"], help="RefSeq group for non-viral.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--require-full-length",
        action="store_true",
        help="Only keep sequences with length >= max-length before windowing.",
    )
    parser.add_argument("--download-workers", type=int, default=4, help="Parallel download workers.")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw RefSeq files, do not build the manifest.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only build the manifest from existing raw files, do not download.",
    )
    args = parser.parse_args()

    if args.download_only and args.manifest_only:
        raise ValueError("--download-only and --manifest-only cannot be used together.")

    rng = random.Random(args.seed)
    out_dir = args.out_dir
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    viral_url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/"
    nonviral_url = f"https://ftp.ncbi.nlm.nih.gov/refseq/release/{args.nonviral_group}/"

    if args.manifest_only:
        viral_files = list_local_fna_files(os.path.join(raw_dir, "viral"))[: args.viral_max_files]
        nonviral_files = list_local_fna_files(
            os.path.join(raw_dir, args.nonviral_group)
        )[: args.nonviral_max_files]
    else:
        viral_files = list_fna_files(viral_url)[: args.viral_max_files]
        nonviral_files = list_fna_files(nonviral_url)[: args.nonviral_max_files]

    if not viral_files:
        raise RuntimeError(f"No viral genomic.fna.gz files found at {viral_url}")
    if not nonviral_files:
        raise RuntimeError(f"No non-viral genomic.fna.gz files found at {nonviral_url}")

    download_pairs: List[Tuple[str, str]] = []
    viral_paths = []
    for fname in viral_files:
        url = viral_url + fname
        out_path = os.path.join(raw_dir, "viral", fname)
        download_pairs.append((url, out_path))
        viral_paths.append(out_path)

    nonviral_paths = []
    for fname in nonviral_files:
        url = nonviral_url + fname
        out_path = os.path.join(raw_dir, args.nonviral_group, fname)
        download_pairs.append((url, out_path))
        nonviral_paths.append(out_path)

    if not args.manifest_only:
        download_all(download_pairs, args.download_workers)

    if args.download_only:
        return

    viral_records = collect_sequences(
        viral_paths,
        label=1,
        target_count=args.target_per_class,
        min_length=args.min_length,
        max_length=args.max_length,
        rng=rng,
        source="viral",
        require_full_length=args.require_full_length,
    )
    nonviral_records = collect_sequences(
        nonviral_paths,
        label=0,
        target_count=args.target_per_class,
        min_length=args.min_length,
        max_length=args.max_length,
        rng=rng,
        source=args.nonviral_group,
        require_full_length=args.require_full_length,
    )

    if len(viral_records) < args.target_per_class or len(nonviral_records) < args.target_per_class:
        print("Warning: did not reach target count for one or both classes.")

    all_records = viral_records + nonviral_records
    assign_splits(all_records, rng, args.train_frac, args.val_frac)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    write_manifest(manifest_path, all_records)
    print(f"Wrote manifest to {manifest_path} with {len(all_records)} records.")


if __name__ == "__main__":
    main()
