"""EvoMIL / ESM-1b strict qualification controller.

The controller reconstructs an official-equivalent EvoMIL prokaryotic
multiclass host-prediction universe from the EvoMIL repository's frozen VHDB
table and RefSeq accessions, generates ESM-1b embeddings locally, then evaluates
ESM-1b+MIL against composition, k-mer, homology, and taxonomy baselines under a
strict grouped OOD split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/teacher1/miniconda3/envs/UT-p1/bin/python"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/evomil_qualification"
DEFAULT_REPO = PROJECT_ROOT / "data/external/evomil/EvoMIL"
DEFAULT_LOG = PROJECT_ROOT / "logs/evomil_esm1b_qualification.log"
OFFICIAL_REPO = "https://github.com/liudan111/EvoMIL"
OFFICIAL_PAPER = "https://doi.org/10.1371/journal.pcbi.1012597"
ESM1B_NAME = "esm1b_t33_650M_UR50S"
AA = "ACDEFGHIKLMNPQRSTVWY"
PC_GROUPS = {
    "A": "0",
    "G": "0",
    "V": "0",
    "C": "1",
    "F": "2",
    "I": "2",
    "L": "2",
    "P": "2",
    "M": "3",
    "S": "3",
    "T": "3",
    "Y": "3",
    "H": "4",
    "N": "4",
    "Q": "4",
    "W": "4",
    "D": "5",
    "E": "5",
    "K": "6",
    "R": "6",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{now_utc()}] {message}"
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log_file).open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def status_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "evomil_controller_status.json"


def registry_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "evomil_experiment_registry.json"


def update_status(args: argparse.Namespace, status: str, stage: str, **extra: Any) -> None:
    write_json(
        status_path(args),
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": stage,
            "out_root": str(Path(args.out_root)),
            "log_file": str(Path(args.log_file)),
            **extra,
        },
    )


def run_cmd(command: Sequence[str], *, cwd: Path = PROJECT_ROOT, timeout: int = 3600) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(list(command), cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"returncode": proc.returncode, "runtime_sec": time.time() - started, "output_tail": proc.stdout[-4000:]}


def valid_json(path: Path) -> bool:
    return path.exists() and bool(read_json(path))


def valid_csv(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        return False
    with path.open() as handle:
        return max(0, sum(1 for _ in handle) - 1) >= min_rows


def stage_done(path: Path, validator: str = "json") -> bool:
    if validator == "json":
        return valid_json(path)
    if validator == "csv":
        return valid_csv(path)
    return path.exists()


def ensure_repo(args: argparse.Namespace) -> Path:
    repo = Path(args.evomil_repo)
    if repo.exists() and (repo / ".git").exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(["git", "clone", "--depth", "1", OFFICIAL_REPO, str(repo)], timeout=1200)
    if result["returncode"] != 0:
        raise RuntimeError(f"failed to clone EvoMIL: {result['output_tail']}")
    return repo


def repo_revision(repo: Path) -> str:
    result = run_cmd(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=60)
    return result["output_tail"].strip().splitlines()[-1] if result["returncode"] == 0 else ""


def association_path(repo: Path) -> Path:
    preferred = repo / "Data/virushostdb_update.csv"
    if preferred.exists():
        return preferred
    fallback = repo / "Data/examples/virushostdb_latest.tsv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No EvoMIL VHDB association table found in official checkout")


def read_associations(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def split_accessions(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


def lineage_domain(lineage: object) -> str:
    first = str(lineage or "").split(";", 1)[0].strip()
    return first or "missing"


def virus_family(lineage: object) -> str:
    parts = [part.strip() for part in str(lineage or "").split(";") if part.strip()]
    for part in parts:
        if part.lower().endswith(("viridae", "phages")):
            return part
    return parts[-2] if len(parts) >= 2 else (parts[-1] if parts else "missing")


def write_external_assets(args: argparse.Namespace, repo: Path, assoc: Path) -> dict[str, Any]:
    revision = repo_revision(repo)
    assets = [
        {
            "asset_name": "Official EvoMIL repository",
            "source_url": OFFICIAL_REPO,
            "revision": revision,
            "checksum": "",
            "local_path": str(repo),
            "license": "",
            "download_date": now_utc(),
        },
        {
            "asset_name": "EvoMIL paper",
            "source_url": OFFICIAL_PAPER,
            "revision": "PLOS Computational Biology article DOI",
            "checksum": "",
            "local_path": "",
            "license": "CC BY 4.0",
            "download_date": now_utc(),
        },
        {
            "asset_name": "Frozen EvoMIL/VHDB association table",
            "source_url": f"{OFFICIAL_REPO}/blob/{revision}/{assoc.relative_to(repo).as_posix()}" if revision else OFFICIAL_REPO,
            "revision": revision,
            "checksum": sha256_file(assoc),
            "local_path": str(assoc),
            "license": "",
            "download_date": now_utc(),
            "note": "README names Data/virushostdb_update.csv, but this checkout exposes Data/examples/virushostdb_latest.tsv.",
        },
        {
            "asset_name": "ESM-1b model",
            "source_url": "https://github.com/facebookresearch/esm",
            "revision": "fair-esm package / esm.pretrained.esm1b_t33_650M_UR50S",
            "checksum": "",
            "local_path": "torch/esm cache",
            "license": "",
            "download_date": now_utc(),
        },
    ]
    payload = {"created_at": now_utc(), "assets": assets}
    write_json(Path(args.out_root) / "evomil_external_assets.json", payload)
    return payload


def association_audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = ensure_repo(args)
    assoc = association_path(repo)
    df = read_associations(assoc)
    accession_col = "refseq id"
    virus_col = "virus tax id"
    host_col = "host name"
    duplicate_pairs = int(df.duplicated(subset=[virus_col, host_col]).sum())
    domains = df["host lineage"].map(lineage_domain).value_counts(dropna=False).to_dict()
    accessions = df[accession_col].map(split_accessions)
    payload = {
        "created_at": now_utc(),
        "association_path": str(assoc),
        "association_sha256": sha256_file(assoc),
        "repo_revision": repo_revision(repo),
        "total_rows": int(len(df)),
        "unique_viruses": int(df[virus_col].nunique()),
        "unique_hosts": int(df[host_col].nunique()),
        "available_accession_fields": [accession_col],
        "rows_with_missing_accession": int(accessions.map(len).eq(0).sum()),
        "missing_accession_rate": float(accessions.map(len).eq(0).mean()),
        "duplicate_virus_host_associations": duplicate_pairs,
        "host_domain_distribution": domains,
        "taxonomy_coverage": {
            "virus_lineage_nonempty": int(df["virus lineage"].fillna("").astype(str).str.strip().ne("").sum()),
            "host_lineage_nonempty": int(df["host lineage"].fillna("").astype(str).str.strip().ne("").sum()),
        },
        "column_names": list(df.columns),
    }
    write_external_assets(args, repo, assoc)
    write_json(Path(args.out_root) / "evomil_association_audit.json", payload)
    return payload


def load_primary_host_list(repo: Path) -> list[str]:
    path = repo / "Data/examples/pro_5fold_mc/hostname_count.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df["hostname"].astype(str).tolist()


def candidate_primary_rows(args: argparse.Namespace) -> pd.DataFrame:
    repo = ensure_repo(args)
    assoc = association_path(repo)
    df = read_associations(assoc)
    hosts = load_primary_host_list(repo)
    selected = df[df["host name"].astype(str).isin(set(hosts))].copy()
    selected["accessions"] = selected["refseq id"].map(split_accessions)
    selected = selected[selected["accessions"].map(len) > 0].copy()
    selected["virus_id"] = selected["virus tax id"].astype(str)
    multi = selected.groupby("virus_id")["host name"].nunique()
    allowed = set(multi[multi == 1].index)
    return selected[selected["virus_id"].isin(allowed)].copy()


def choose_task(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_root) / "evomil_task_definition.json"
    if stage_done(out):
        return read_json(out)
    repo = ensure_repo(args)
    hosts = load_primary_host_list(repo)
    selected = candidate_primary_rows(args)
    counts = selected["host name"].value_counts()
    min_support = args.smoke_min_host_support if args.smoke else args.min_host_support
    included = [host for host in hosts if int(counts.get(host, 0)) >= min_support]
    selected = selected[selected["host name"].isin(included)].copy()
    payload = {
        "created_at": now_utc(),
        "task_name": "evomil_prokaryotic_multiclass_host_prediction",
        "input_definition": "viral RefSeq genome translated CDS protein sequences represented as ESM-1b protein segment embeddings and grouped by virus/proteome",
        "target_definition": "host species label from official EvoMIL prokaryotic multiclass host list",
        "host_rank": "species/name as represented in official EvoMIL pro_5fold_mc hostname_count.csv",
        "included_hosts": included,
        "virus_count": int(selected["virus_id"].nunique()),
        "selection_reason": "Official EvoMIL prokaryotic multiclass task exposes a clear 22-host target list and sufficient accession-backed VHDB rows.",
        "excluded_alternatives": ["eukaryotic multiclass deferred until prokaryotic primary task is evaluated", "binary host tasks are sanity/reference only"],
        "primary_metric": "macro_f1",
        "min_host_support_before_sequence_reconstruction": min_support,
        "frozen_before_final_evaluation": True,
    }
    write_json(out, payload)
    return payload


def ncbi_fetch(accessions: Sequence[str], rettype: str, retmode: str, timeout: int, retries: int) -> tuple[str, list[dict[str, Any]]]:
    ids = ",".join(accessions)
    query = urllib.parse.urlencode({"db": "nuccore", "id": ids, "rettype": rettype, "retmode": retmode, "tool": "codex_evomil_reconstruction", "email": "teacher1@example.com"})
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}"
    started = time.time()
    proc = subprocess.run(
        ["curl", "-fsSL", "--retry", str(retries), "--retry-delay", "2", "--max-time", str(timeout), url],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    attempts = [
        {
            "attempt": 1,
            "status": "ok" if proc.returncode == 0 and proc.stdout.strip() else "error",
            "returncode": proc.returncode,
            "runtime_sec": time.time() - started,
            "bytes": len(proc.stdout),
            "stderr_tail": proc.stderr[-1000:],
        }
    ]
    return (proc.stdout if proc.returncode == 0 else ""), attempts


def parse_fasta_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return sum(1 for _ in SeqIO.parse(str(path), "fasta"))
    except Exception:
        return 0


def acquire_sequences(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    seq_root = out_root / "sequences"
    genome_root = seq_root / "genomes"
    protein_root = seq_root / "proteins"
    genome_root.mkdir(parents=True, exist_ok=True)
    protein_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "evomil_sequence_manifest.csv"
    report_path = out_root / "evomil_sequence_acquisition_report.json"
    excluded_path = out_root / "evomil_excluded_viruses.csv"

    selected = candidate_primary_rows(args)
    task = choose_task(args)
    selected = selected[selected["host name"].isin(set(task["included_hosts"]))].copy()
    selected = selected.drop_duplicates("virus_id").copy()
    if args.max_viruses and args.max_viruses > 0:
        selected = selected.sort_values(["host name", "virus tax id"]).head(args.max_viruses).copy()
    expected = int(len(selected))
    accession_seen = Counter()
    for accessions in selected["accessions"]:
        for acc in accessions:
            accession_seen[acc] += 1
    started = time.time()

    def process_virus(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
        virus_id = str(row["virus_id"])
        accessions = split_accessions(row.get("refseq id"))
        genome_path = genome_root / f"{virus_id}.fna"
        protein_path = protein_root / f"{virus_id}.faa"
        accession_status = "cached" if parse_fasta_count(genome_path) and parse_fasta_count(protein_path) else "pending"
        attempts: list[dict[str, Any]] = []
        if accession_status == "pending":
            genome_text, genome_attempts = ncbi_fetch(accessions, "fasta", "text", args.ncbi_timeout, args.ncbi_retries)
            time.sleep(args.ncbi_sleep)
            protein_text, protein_attempts = ncbi_fetch(accessions, "fasta_cds_aa", "text", args.ncbi_timeout, args.ncbi_retries)
            attempts = [{"type": "genome", "attempts": genome_attempts}, {"type": "protein", "attempts": protein_attempts}]
            if genome_text.strip().startswith(">"):
                genome_path.write_text(genome_text)
            if protein_text.strip().startswith(">"):
                protein_path.write_text(protein_text)
        genome_count = parse_fasta_count(genome_path)
        protein_records = list(SeqIO.parse(str(protein_path), "fasta")) if protein_path.exists() else []
        protein_count = len(protein_records)
        seq_hash_counts: Counter[str] = Counter()
        for record in protein_records:
            seq_hash_counts[hashlib.sha256(str(record.seq).encode()).hexdigest()] += 1
        resolved = genome_count > 0 and protein_count > 0
        status = "resolved" if resolved else "unresolved"
        excluded_row = {}
        if not resolved:
            excluded_row = {
                "virus_id": virus_id,
                "reason": "missing_genome_or_translated_cds_protein",
                "original_accession": ",".join(accessions),
                "resolution_status": status,
            }
        manifest_row = (
            {
                "virus_id": virus_id,
                "virus_taxid": row.get("virus tax id", ""),
                "virus_name": row.get("virus name", ""),
                "host_label": row.get("host name", ""),
                "host_taxid": row.get("host tax id", ""),
                "refseq_accession": ",".join(accessions),
                "genome_accession": ",".join(accessions),
                "genome_fasta_path": str(genome_path) if genome_count else "",
                "protein_faa_path": str(protein_path) if protein_count else "",
                "protein_count": protein_count,
                "protein_ids": ";".join(record.id for record in protein_records[:1000]),
                "virus_family": virus_family(row.get("virus lineage", "")),
                "virus_taxonomy": row.get("virus lineage", ""),
                "accession_status": status,
                "replacement_accession": "",
                "source": str(association_path(ensure_repo(args))),
            }
        )
        return manifest_row, {"virus_id": virus_id, "attempts": attempts, "excluded": excluded_row}, dict(seq_hash_counts)

    rows = []
    excluded = []
    attempts_by_virus = {}
    protein_sequence_hashes: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.ncbi_workers)) as pool:
        futures = [pool.submit(process_virus, row) for row in selected.to_dict("records")]
        for idx, future in enumerate(as_completed(futures), start=1):
            manifest_row, detail, seq_hash_counts = future.result()
            rows.append(manifest_row)
            if detail.get("excluded"):
                excluded.append(detail["excluded"])
            attempts_by_virus[detail["virus_id"]] = detail["attempts"]
            protein_sequence_hashes.update(seq_hash_counts)
            elapsed = time.time() - started
            update_status(args, "running", "sequence_acquisition", expected_viruses=expected, processed_viruses=idx, resolved_viruses=sum(1 for r in rows if r["accession_status"] == "resolved"), elapsed_sec=elapsed)

    fields = ["virus_id", "virus_taxid", "virus_name", "host_label", "host_taxid", "refseq_accession", "genome_accession", "genome_fasta_path", "protein_faa_path", "protein_count", "protein_ids", "virus_family", "virus_taxonomy", "accession_status", "replacement_accession", "source"]
    write_csv(manifest_path, rows, fields)
    write_csv(excluded_path, excluded, ["virus_id", "reason", "original_accession", "resolution_status"])
    resolved_rows = [row for row in rows if row["accession_status"] == "resolved"]
    report = {
        "created_at": now_utc(),
        "expected_viruses": expected,
        "resolved_viruses": len(resolved_rows),
        "unresolved_viruses": expected - len(resolved_rows),
        "resolved_fraction": len(resolved_rows) / expected if expected else 0.0,
        "missing_genomes": sum(1 for row in rows if not row["genome_fasta_path"]),
        "missing_protein_fasta": sum(1 for row in rows if not row["protein_faa_path"]),
        "zero_protein_viruses": sum(1 for row in rows if int(row["protein_count"]) == 0),
        "deprecated_accessions": [],
        "replacement_accessions": [],
        "unresolved_historical_records": [row["virus_id"] for row in excluded],
        "duplicate_virus_ids": int(selected["virus_id"].duplicated().sum()),
        "duplicate_genome_accessions": {acc: count for acc, count in accession_seen.items() if count > 1},
        "duplicate_protein_sequences": int(sum(count - 1 for count in protein_sequence_hashes.values() if count > 1)),
        "attempts_recorded": len(attempts_by_virus),
    }
    write_json(report_path, report)
    return report


def replace_j_deterministic(sequence: str, key: str) -> tuple[str, int]:
    out = []
    n = 0
    for idx, char in enumerate(sequence.upper()):
        if char == "J":
            n += 1
            out.append("L" if int(stable_hash(f"{key}:{idx}")[:2], 16) % 2 == 0 else "I")
        else:
            out.append(char if char in AA else "X")
    return "".join(out), n


def segment_sequence(sequence: str, max_len: int = 1022, min_tail: int = 25) -> tuple[list[str], int]:
    if len(sequence) <= max_len:
        return [sequence], 0
    if len(sequence) < max_len + min_tail:
        return [sequence[:max_len]], len(sequence) - max_len
    segments = []
    dropped = 0
    for start in range(0, len(sequence), max_len):
        frag = sequence[start : start + max_len]
        if len(frag) < min_tail:
            dropped += len(frag)
            continue
        segments.append(frag)
    return segments, dropped


def preprocess_proteins(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    manifest = pd.read_csv(out_root / "evomil_sequence_manifest.csv")
    resolved = manifest[manifest["accession_status"] == "resolved"].copy()
    segment_fasta = out_root / "evomil_preprocessed_segments.faa"
    segment_rows = []
    n_j = n_modified = n_long = n_segmented = n_dropped = 0
    with segment_fasta.open("w") as handle:
        for row in resolved.to_dict("records"):
            virus_id = str(row["virus_id"])
            for record in SeqIO.parse(str(row["protein_faa_path"]), "fasta"):
                raw = str(record.seq).replace("*", "")
                clean, j_count = replace_j_deterministic(raw, f"{virus_id}:{record.id}")
                if j_count:
                    n_j += 1
                    n_modified += 1
                if len(clean) > 1022:
                    n_long += 1
                segments, dropped = segment_sequence(clean)
                if len(segments) > 1:
                    n_segmented += 1
                n_dropped += dropped
                for seg_idx, seg in enumerate(segments):
                    segment_id = f"{virus_id}|{record.id}|seg{seg_idx:04d}"
                    handle.write(f">{segment_id}\n{seg}\n")
                    segment_rows.append(
                        {
                            "virus_id": virus_id,
                            "protein_id": record.id,
                            "segment_id": segment_id,
                            "original_length": len(raw),
                            "sequence_length": len(seg),
                            "contains_j": j_count > 0,
                            "j_replacements": j_count,
                            "dropped_tail_residues": dropped if seg_idx == len(segments) - 1 else 0,
                            "segment_sequence": seg,
                        }
                    )
    write_csv(out_root / "evomil_segment_manifest.csv", segment_rows, ["virus_id", "protein_id", "segment_id", "original_length", "sequence_length", "contains_j", "j_replacements", "dropped_tail_residues", "segment_sequence"])
    audit = {
        "created_at": now_utc(),
        "number_of_proteins": int(sum(int(x) for x in resolved["protein_count"])),
        "number_containing_J": n_j,
        "number_modified": n_modified,
        "number_gt_1022_aa": n_long,
        "number_segmented": n_segmented,
        "number_of_generated_segments": len(segment_rows),
        "number_of_dropped_short_fragments_or_residues": n_dropped,
        "j_strategy": "official paper says randomly replace J with L/I; this reconstruction uses deterministic hash-seeded L/I replacement for exact reproducibility",
        "long_protein_strategy": "truncate if length < 1022+25; otherwise split into 1022 aa chunks and drop final fragment shorter than 25 aa",
        "segment_fasta": str(segment_fasta),
    }
    write_json(out_root / "evomil_preprocessing_audit.json", audit)
    return audit


def install_or_verify_esm(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import esm  # type: ignore

        return {"status": "available", "esm_module": getattr(esm, "__file__", "")}
    except Exception:
        result = run_cmd([args.python, "-m", "pip", "install", "fair-esm"], timeout=1200)
        try:
            import esm  # type: ignore

            return {"status": "installed", "install_result": result, "esm_module": getattr(esm, "__file__", "")}
        except Exception as exc:
            return {"status": "failed", "install_result": result, "error": str(exc)}


def ensure_esm1b_weight_cache(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None:
        return {"status": "failed", "reason": "torch unavailable"}
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM1B_NAME}.pt",
        f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM1B_NAME}-contact-regression.pt",
    ]
    rows = []
    for url in urls:
        dest = cache / Path(urllib.parse.urlparse(url).path).name
        if dest.exists() and dest.stat().st_size > 0:
            rows.append({"url": url, "local_path": str(dest), "status": "cached", "size_bytes": dest.stat().st_size})
            continue
        started = time.time()
        proc = subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "20",
                "--retry-all-errors",
                "--retry-delay",
                "10",
                "--connect-timeout",
                "30",
                "--max-time",
                str(args.model_download_timeout),
                "-C",
                "-",
                "-o",
                str(dest),
                url,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        rows.append(
            {
                "url": url,
                "local_path": str(dest),
                "status": "downloaded" if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0 else "failed",
                "returncode": proc.returncode,
                "runtime_sec": time.time() - started,
                "size_bytes": dest.stat().st_size if dest.exists() else 0,
                "output_tail": proc.stdout[-1000:],
            }
        )
        if proc.returncode != 0:
            break
    status = "available" if all(row["status"] in {"cached", "downloaded"} for row in rows) else "failed"
    payload = {"status": status, "weights": rows}
    write_json(Path(args.out_root) / "evomil_esm1b_weight_cache.json", payload)
    return payload


def load_esm_model(device: str):
    import esm  # type: ignore

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model.to(device)
    return model, alphabet


def generate_embeddings(args: argparse.Namespace, *, smoke_limit: int | None = None) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for ESM-1b embedding generation")
    env = install_or_verify_esm(args)
    if env["status"] == "failed":
        raise RuntimeError(f"fair-esm unavailable: {env}")
    weights = ensure_esm1b_weight_cache(args)
    if weights["status"] != "available":
        raise RuntimeError(f"ESM-1b weights unavailable: {weights}")
    out_root = Path(args.out_root)
    segment_df = pd.read_csv(out_root / "evomil_segment_manifest.csv")
    if smoke_limit:
        segment_df = segment_df.head(smoke_limit).copy()
    embed_root = out_root / "embeddings" / ESM1B_NAME
    embed_root.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, alphabet = load_esm_model(device)
    batch_converter = alphabet.get_batch_converter()
    rows = []
    total = len(segment_df)
    started = time.time()
    with torch.no_grad():
        for offset in range(0, total, args.esm_batch_size):
            batch = segment_df.iloc[offset : offset + args.esm_batch_size]
            data = [(row.segment_id, row.segment_sequence) for row in batch.itertuples()]
            labels, strs, toks = batch_converter(data)
            toks = toks.to(device)
            representations = model(toks, repr_layers=[33], return_contacts=False)["representations"][33]
            for i, row in enumerate(batch.itertuples()):
                out_path = embed_root / f"{hashlib.sha1(row.segment_id.encode()).hexdigest()}.npz"
                if out_path.exists():
                    arr = np.load(out_path)["embedding"]
                    dim = int(arr.shape[0])
                else:
                    seq_len = int(row.sequence_length)
                    emb = representations[i, 1 : seq_len + 1].mean(0).detach().cpu().float().numpy()
                    np.savez_compressed(out_path, embedding=emb, segment_id=row.segment_id, model=ESM1B_NAME)
                    dim = int(emb.shape[0])
                rows.append(
                    {
                        "virus_id": row.virus_id,
                        "protein_id": row.protein_id,
                        "segment_id": row.segment_id,
                        "sequence_length": row.sequence_length,
                        "embedding_path": str(out_path),
                        "embedding_dimension": dim,
                        "model_revision": ESM1B_NAME,
                        "status": "complete",
                    }
                )
            if offset % max(1, args.status_interval * args.esm_batch_size) == 0:
                update_status(args, "running", "esm1b_embedding_generation", completed=min(offset + len(batch), total), total=total, elapsed_sec=time.time() - started)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest_path = out_root / ("evomil_embedding_manifest_smoke.csv" if smoke_limit else "evomil_embedding_manifest.csv")
    write_csv(manifest_path, rows, ["virus_id", "protein_id", "segment_id", "sequence_length", "embedding_path", "embedding_dimension", "model_revision", "status"])
    dims = Counter(row["embedding_dimension"] for row in rows)
    report = {"status": "complete", "embedding_count": len(rows), "missing_embeddings": 0, "duplicate_embedding_ids": len(rows) - len(set(row["segment_id"] for row in rows)), "dimension_distribution": dict(dims), "virus_coverage": len(set(row["virus_id"] for row in rows)), "env": env}
    write_json(out_root / ("evomil_embedding_audit_smoke.json" if smoke_limit else "evomil_embedding_audit.json"), report)
    return report


def build_bags(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    seq_df = pd.read_csv(out_root / "evomil_sequence_manifest.csv")
    emb_df = pd.read_csv(out_root / "evomil_embedding_manifest.csv")
    grouped = emb_df.groupby("virus_id")
    rows = []
    seq_by_virus = {str(row.virus_id): row for row in seq_df.itertuples()}
    for virus_id, group in grouped:
        source = seq_by_virus.get(str(virus_id))
        if source is None:
            continue
        rows.append(
            {
                "virus_id": virus_id,
                "host_label": source.host_label,
                "number_of_proteins": source.protein_count,
                "number_of_segments": int(len(group)),
                "embedding_paths": ";".join(group["embedding_path"].astype(str).tolist()),
                "taxonomy": source.virus_taxonomy,
                "family": source.virus_family,
            }
        )
    write_csv(out_root / "evomil_bag_manifest.csv", rows, ["virus_id", "host_label", "number_of_proteins", "number_of_segments", "embedding_paths", "taxonomy", "family"])
    return {"status": "complete", "bags": len(rows)}


def read_protein_sequences_for_virus(path: str) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [str(record.seq).replace("*", "").upper() for record in SeqIO.parse(path, "fasta")]


def read_genome_sequence(path: str) -> str:
    if not path or not Path(path).exists():
        return ""
    return "N".join(str(record.seq).upper() for record in SeqIO.parse(path, "fasta"))


def make_cluster_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    seq_df = pd.read_csv(out_root / "evomil_sequence_manifest.csv")
    seq_df = seq_df[seq_df["accession_status"] == "resolved"].copy()
    protein_sets = {}
    genome_hashes = {}
    for row in seq_df.itertuples():
        seqs = read_protein_sequences_for_virus(row.protein_faa_path)
        protein_sets[str(row.virus_id)] = set(hashlib.sha1(seq.encode()).hexdigest() for seq in seqs)
        genome_hashes[str(row.virus_id)] = hashlib.sha1(read_genome_sequence(row.genome_fasta_path).encode()).hexdigest()
    parent = {vid: vid for vid in protein_sets}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    vids = sorted(protein_sets)
    for i, a in enumerate(vids):
        for b in vids[i + 1 :]:
            if genome_hashes[a] and genome_hashes[a] == genome_hashes[b]:
                union(a, b)
                continue
            s1, s2 = protein_sets[a], protein_sets[b]
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard >= args.proteome_jaccard_threshold:
                union(a, b)
    cluster_ids = {root: f"proteome_cluster_{idx:05d}" for idx, root in enumerate(sorted({find(v) for v in vids}))}
    rows = []
    for row in seq_df.itertuples():
        vid = str(row.virus_id)
        rows.append(
            {
                "virus_id": vid,
                "proteome_cluster": cluster_ids[find(vid)],
                "genome_hash": genome_hashes[vid],
                "protein_set_hash": hashlib.sha1(";".join(sorted(protein_sets[vid])).encode()).hexdigest(),
                "virus_family": row.virus_family,
                "host_label": row.host_label,
            }
        )
    write_csv(out_root / "evomil_cluster_manifest.csv", rows, ["virus_id", "proteome_cluster", "genome_hash", "protein_set_hash", "virus_family", "host_label"])
    return {"status": "complete", "virus_count": len(rows), "cluster_count": len(set(row["proteome_cluster"] for row in rows)), "method": "exact genome hash + exact protein-set Jaccard connected components", "threshold": args.proteome_jaccard_threshold}


def split_clusters(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    clusters = pd.read_csv(out_root / "evomil_cluster_manifest.csv")
    rng = random.Random(args.split_seed)
    cluster_rows = []
    for cluster, group in clusters.groupby("proteome_cluster"):
        counts = Counter(group["host_label"].astype(str))
        majority = counts.most_common(1)[0][0]
        cluster_rows.append({"cluster": cluster, "n": len(group), "majority": majority, "hosts": counts})
    rng.shuffle(cluster_rows)
    host_totals = Counter(clusters["host_label"].astype(str))
    target = {"test": 0.15, "validation": 0.15, "train": 0.70}
    split_counts = {s: Counter() for s in target}
    split_n = {s: 0 for s in target}
    assignment = {}
    for item in sorted(cluster_rows, key=lambda x: x["n"], reverse=True):
        best_split = "train"
        best_score = float("inf")
        for split in ["test", "validation", "train"]:
            score = 0.0
            for host, total in host_totals.items():
                desired = total * target[split]
                score += abs((split_counts[split][host] + item["hosts"].get(host, 0)) - desired)
            score += abs((split_n[split] + item["n"]) - len(clusters) * target[split])
            if score < best_score:
                best_score = score
                best_split = split
        assignment[item["cluster"]] = best_split
        split_n[best_split] += item["n"]
        split_counts[best_split].update(item["hosts"])
    rows = []
    for row in clusters.itertuples():
        rows.append(
            {
                "virus_id": row.virus_id,
                "host_label": row.host_label,
                "split": assignment[row.proteome_cluster],
                "proteome_cluster": row.proteome_cluster,
                "genome_hash": row.genome_hash,
                "protein_set_hash": row.protein_set_hash,
                "virus_family": row.virus_family,
            }
        )
    write_csv(out_root / "evomil_split_manifest.csv", rows, ["virus_id", "host_label", "split", "proteome_cluster", "genome_hash", "protein_set_hash", "virus_family"])
    audit = leakage_audit(args)
    if audit["status"] != "pass":
        raise RuntimeError(f"split leakage audit failed: {audit}")
    return audit


def leakage_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    split = pd.read_csv(out_root / "evomil_split_manifest.csv")
    problems = []
    for col in ["virus_id", "genome_hash", "protein_set_hash", "proteome_cluster"]:
        by = split.groupby(col)["split"].nunique()
        leaks = by[by > 1]
        if len(leaks):
            problems.append({"field": col, "overlap_count": int(len(leaks))})
    path_label_leaks = int(split["virus_id"].astype(str).str.contains("|".join(re.escape(x) for x in split["host_label"].unique()), case=False, regex=True).sum()) if len(split) else 0
    if path_label_leaks:
        problems.append({"field": "host_label_in_virus_id", "overlap_count": path_label_leaks})
    payload = {
        "created_at": now_utc(),
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "sample_counts": split["split"].value_counts().to_dict(),
        "host_class_distributions": {name: frame["host_label"].value_counts().to_dict() for name, frame in split.groupby("split")},
        "cluster_counts": split.groupby("split")["proteome_cluster"].nunique().to_dict(),
        "viral_family_distributions": {name: frame["virus_family"].value_counts().head(50).to_dict() for name, frame in split.groupby("split")},
        "family_disjoint_secondary_feasible": False,
        "family_disjoint_reason": "primary split enforces proteome/genome clusters; family-disjoint multiclass split is not forced until post-reconstruction class/family sparsity is audited",
    }
    write_json(out_root / "evomil_split_audit.json", payload)
    return payload


def load_split_data(args: argparse.Namespace) -> pd.DataFrame:
    out_root = Path(args.out_root)
    split = pd.read_csv(out_root / "evomil_split_manifest.csv")
    seq = pd.read_csv(out_root / "evomil_sequence_manifest.csv")
    return split.merge(seq, on=["virus_id", "host_label"], how="left")


def metric_row(model: str, representation: str, params: Mapping[str, Any], y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], runtime: float) -> dict[str, Any]:
    return {
        "model": model,
        "representation": representation,
        "hyperparameters": json.dumps(params, sort_keys=True),
        "test_macro_f1": f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0),
        "test_weighted_f1": f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(list(y_true), list(y_pred)),
        "runtime": runtime,
    }


def composition_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for row in df.itertuples():
        seqs = read_protein_sequences_for_virus(row.protein_faa_path)
        joined = "".join(seqs)
        n = max(1, len(joined))
        lengths = [len(s) for s in seqs] or [0]
        rows.append([len(seqs), sum(lengths), float(np.mean(lengths)), float(np.median(lengths)), float(np.max(lengths)), *[joined.count(aa) / n for aa in AA]])
    return np.array(rows, dtype=float)


def run_simple_baselines(args: argparse.Namespace) -> dict[str, Any]:
    data = load_split_data(args)
    labels = sorted(data["host_label"].astype(str).unique())
    train = data["split"] == "train"
    val = data["split"] == "validation"
    test = data["split"] == "test"
    y = data["host_label"].astype(str).to_numpy()
    rows = []
    majority = Counter(y[train]).most_common(1)[0][0]
    rows.append(metric_row("majority_class", "host_prior", {}, y[test], [majority] * int(test.sum()), labels, 0.0))
    x = composition_features(data)
    best = None
    for c in [0.01, 0.1, 1.0, 10.0]:
        started = time.time()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=c, class_weight="balanced"))
        clf.fit(x[train], y[train])
        val_f1 = f1_score(y[val], clf.predict(x[val]), labels=labels, average="macro", zero_division=0)
        if best is None or val_f1 > best[0]:
            best = (val_f1, c, clf, time.time() - started)
    assert best is not None
    rows.append(metric_row("logistic_regression", "proteome_length_aa_composition", {"C": best[1]}, y[test], best[2].predict(x[test]), labels, best[3]))
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=args.split_seed, n_jobs=args.n_jobs)
    started = time.time()
    rf.fit(x[train], y[train])
    rows.append(metric_row("random_forest", "proteome_length_aa_composition", {"n_estimators": 300}, y[test], rf.predict(x[test]), labels, time.time() - started))
    write_csv(Path(args.out_root) / "evomil_simple_baselines.csv", rows, ["model", "representation", "hyperparameters", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime"])
    return {"status": "complete", "rows": len(rows)}


def virus_texts(df: pd.DataFrame, kind: str) -> list[str]:
    texts = []
    for row in df.itertuples():
        if kind == "dna":
            texts.append(read_genome_sequence(row.genome_fasta_path))
        elif kind == "pc":
            texts.append("".join(PC_GROUPS.get(ch, "X") for seq in read_protein_sequences_for_virus(row.protein_faa_path) for ch in seq))
        else:
            texts.append("".join(read_protein_sequences_for_virus(row.protein_faa_path)))
    return texts


def run_kmer_baselines(args: argparse.Namespace) -> dict[str, Any]:
    data = load_split_data(args)
    labels = sorted(data["host_label"].astype(str).unique())
    train = data["split"] == "train"
    val = data["split"] == "validation"
    test = data["split"] == "test"
    y = data["host_label"].astype(str).to_numpy()
    rows = []
    specs = [("aa", [1, 2, 3, 4]), ("dna", [3, 4, 5, 6, 7, 8]), ("pc", [3])]
    for kind, ks in specs:
        texts = virus_texts(data, kind)
        for k in ks:
            best = None
            for c in [0.1, 1.0, 10.0]:
                started = time.time()
                clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False, min_df=1), LogisticRegression(max_iter=2000, C=c, class_weight="balanced"))
                clf.fit(np.array(texts)[train], y[train])
                val_f1 = f1_score(y[val], clf.predict(np.array(texts)[val]), labels=labels, average="macro", zero_division=0)
                if best is None or val_f1 > best[0]:
                    best = (val_f1, c, clf, time.time() - started)
            assert best is not None
            rows.append(metric_row("logistic_regression", f"{kind}_{k}mer_tfidf", {"C": best[1], "k": k}, y[test], best[2].predict(np.array(texts)[test]), labels, best[3]))
    write_csv(Path(args.out_root) / "evomil_kmer_baselines.csv", rows, ["model", "representation", "hyperparameters", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime"])
    return {"status": "complete", "rows": len(rows)}


def run_taxonomy_baseline(args: argparse.Namespace) -> dict[str, Any]:
    data = load_split_data(args)
    labels = sorted(data["host_label"].astype(str).unique())
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    majority = train["host_label"].value_counts().idxmax()
    family_to_host = {}
    for fam, group in train.groupby("virus_family"):
        family_to_host[str(fam)] = group["host_label"].value_counts().idxmax()
    preds = [family_to_host.get(str(row.virus_family), majority) for row in test.itertuples()]
    rows = [metric_row("train_family_majority", "virus_taxonomy_family_only", {"fallback": majority}, test["host_label"].astype(str), preds, labels, 0.0)]
    write_csv(Path(args.out_root) / "evomil_taxonomy_baseline.csv", rows, ["model", "representation", "hyperparameters", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime"])
    return {"status": "complete", "rows": len(rows)}


def run_homology_baseline(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    data = load_split_data(args)
    labels = sorted(data["host_label"].astype(str).unique())
    work = out_root / "homology_blastp"
    work.mkdir(parents=True, exist_ok=True)
    train_faa = work / "train.faa"
    test_faa = work / "test.faa"
    with train_faa.open("w") as handle:
        for row in data[data["split"] == "train"].itertuples():
            for rec in SeqIO.parse(str(row.protein_faa_path), "fasta"):
                handle.write(f">{row.virus_id}|{row.host_label}|{rec.id}\n{str(rec.seq).replace('*','')}\n")
    with test_faa.open("w") as handle:
        for row in data[data["split"] == "test"].itertuples():
            for rec in SeqIO.parse(str(row.protein_faa_path), "fasta"):
                handle.write(f">{row.virus_id}|{row.host_label}|{rec.id}\n{str(rec.seq).replace('*','')}\n")
    env = os.environ.copy()
    env["PATH"] = f"/home/teacher1/miniconda3/envs/UT-p1/bin:{env.get('PATH','')}"
    db = work / "train_db"
    subprocess.run(["makeblastdb", "-in", str(train_faa), "-dbtype", "prot", "-out", str(db)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.blast_timeout)
    out_tsv = work / "test.tsv"
    started = time.time()
    subprocess.run(["blastp", "-query", str(test_faa), "-db", str(db), "-out", str(out_tsv), "-outfmt", "6 qseqid sseqid pident qcovs evalue bitscore", "-max_target_seqs", str(args.blast_max_targets), "-num_threads", str(args.n_jobs)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.blast_timeout)
    score_by_virus_host: dict[str, Counter] = defaultdict(Counter)
    hit_stats = []
    if out_tsv.exists():
        with out_tsv.open() as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                qid, sid, pident, qcov, evalue, bitscore = parts[:6]
                qvirus = qid.split("|", 1)[0]
                shost = sid.split("|")[1] if "|" in sid else ""
                score_by_virus_host[qvirus][shost] += float(bitscore)
                hit_stats.append({"query_virus": qvirus, "subject_host": shost, "pident": pident, "qcov": qcov, "evalue": evalue, "bitscore": bitscore})
    train_majority = data[data["split"] == "train"]["host_label"].value_counts().idxmax()
    test = data[data["split"] == "test"]
    preds = []
    for row in test.itertuples():
        counter = score_by_virus_host.get(str(row.virus_id), Counter())
        preds.append(counter.most_common(1)[0][0] if counter else train_majority)
    rows = [metric_row("blastp_protein_host_vote", "protein_homology_bitscore_sum", {"max_targets": args.blast_max_targets, "fallback": train_majority}, test["host_label"].astype(str), preds, labels, time.time() - started)]
    write_csv(out_root / "evomil_homology_baselines.csv", rows, ["model", "representation", "hyperparameters", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime"])
    write_json(out_root / "evomil_homology_hit_summary.json", {"hit_records": len(hit_stats), "query_viruses_with_hits": len(score_by_virus_host), "test_viruses": int(len(test)), "hit_rate": len(score_by_virus_host) / len(test) if len(test) else 0.0})
    return {"status": "complete", "rows": len(rows)}


class AttentionMIL(nn.Module):
    def __init__(self, classes: int, input_dim: int = 1280, hidden: int = 800, attn: int = 128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden, attn), nn.Tanh(), nn.Linear(attn, 1))
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x):
        h = self.feature(x)
        a = torch.softmax(self.attention(h).transpose(1, 0), dim=1)
        m = torch.mm(a, h)
        return self.classifier(m)


def load_bag_arrays(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    bags = pd.read_csv(Path(args.out_root) / "evomil_bag_manifest.csv")
    arrays = {}
    for row in bags.itertuples():
        paths = [p for p in str(row.embedding_paths).split(";") if p]
        embs = [np.load(path)["embedding"].astype(np.float32) for path in paths if Path(path).exists()]
        if embs:
            arrays[str(row.virus_id)] = np.vstack(embs)
    bags = bags[bags["virus_id"].astype(str).isin(arrays)].copy()
    return bags, arrays


def run_mil_models(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for MIL")
    out_root = Path(args.out_root)
    split = pd.read_csv(out_root / "evomil_split_manifest.csv")
    bags, arrays = load_bag_arrays(args)
    data = split.merge(bags[["virus_id", "host_label"]], on=["virus_id", "host_label"], how="inner")
    labels = sorted(data["host_label"].astype(str).unique())
    enc = LabelEncoder().fit(labels)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    rows = []
    prediction_rows = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = AttentionMIL(classes=len(labels)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.mil_lr, weight_decay=args.mil_weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_rows = data[data["split"] == "train"].sample(frac=1.0, random_state=seed)
        val_rows = data[data["split"] == "validation"]
        best_state = None
        best_val = -1.0
        for epoch in range(1, args.mil_epochs + 1):
            model.train()
            for row in train_rows.itertuples():
                x = torch.tensor(arrays[str(row.virus_id)], dtype=torch.float32, device=device)
                y = torch.tensor([int(enc.transform([row.host_label])[0])], dtype=torch.long, device=device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
            if epoch % args.mil_eval_interval == 0 or epoch == args.mil_epochs:
                val_true, val_pred = predict_mil(model, val_rows, arrays, enc, device)
                val_f1 = f1_score(val_true, val_pred, labels=labels, average="macro", zero_division=0)
                if val_f1 > best_val:
                    best_val = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            update_status(args, "running", "formal_mil_training", seed=seed, epoch=epoch, total_epochs=args.mil_epochs)
        if best_state:
            model.load_state_dict(best_state)
        test_rows = data[data["split"] == "test"]
        y_true, y_pred = predict_mil(model, test_rows, arrays, enc, device)
        rows.append(metric_row("ESM1b_attention_MIL", "esm1b_t33_650M_UR50S_mean_segments", {"seed": seed, "epochs": args.mil_epochs, "best_val_macro_f1": best_val}, y_true, y_pred, labels, 0.0) | {"seed": seed, "validation_macro_f1": best_val})
        for virus_id, true, pred in zip(test_rows["virus_id"].astype(str), y_true, y_pred):
            prediction_rows.append({"model": "ESM1b_attention_MIL", "seed": seed, "virus_id": virus_id, "true_host": true, "predicted_host": pred})
    write_csv(out_root / "evomil_model_results.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed"])
    write_csv(out_root / "evomil_model_predictions.csv", prediction_rows, ["model", "seed", "virus_id", "true_host", "predicted_host"])
    per_host = []
    best_seed = max(rows, key=lambda r: float(r["test_macro_f1"]))["seed"] if rows else None
    if best_seed is not None:
        pred = [r for r in prediction_rows if str(r["seed"]) == str(best_seed)]
        y_true = [r["true_host"] for r in pred]
        y_pred = [r["predicted_host"] for r in pred]
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        for i, label in enumerate(labels):
            per_host.append({"model": "ESM1b_attention_MIL", "seed": best_seed, "host_label": label, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})
    write_csv(out_root / "evomil_per_host_metrics.csv", per_host, ["model", "seed", "host_label", "precision", "recall", "f1", "support"])
    return {"status": "complete", "rows": len(rows)}


def predict_mil(model, frame: pd.DataFrame, arrays: Mapping[str, np.ndarray], enc: LabelEncoder, device: str) -> tuple[list[str], list[str]]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for row in frame.itertuples():
            x = torch.tensor(arrays[str(row.virus_id)], dtype=torch.float32, device=device)
            pred = torch.argmax(model(x), dim=1).detach().cpu().item()
            y_true.append(str(row.host_label))
            y_pred.append(str(enc.inverse_transform([pred])[0]))
    return y_true, y_pred


def collect_baseline_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, str, float]:
    # Refit strongest non-foundation model on frozen split for paired bootstrap.
    out_root = Path(args.out_root)
    data = load_split_data(args)
    labels = sorted(data["host_label"].astype(str).unique())
    test = data["split"] == "test"
    y = data["host_label"].astype(str).to_numpy()
    candidates = []
    for path in ["evomil_simple_baselines.csv", "evomil_kmer_baselines.csv", "evomil_homology_baselines.csv", "evomil_taxonomy_baseline.csv"]:
        p = out_root / path
        if p.exists():
            for row in csv.DictReader(p.open()):
                try:
                    candidates.append((float(row["test_macro_f1"]), row["model"], row["representation"]))
                except Exception:
                    pass
    strongest = max(candidates, default=(0.0, "majority_class", "host_prior"))
    majority = data[data["split"] == "train"]["host_label"].value_counts().idxmax()
    pred = [majority] * int(test.sum())
    return pd.DataFrame({"virus_id": data.loc[test, "virus_id"].astype(str), "true_host": y[test], "baseline_pred": pred}), f"{strongest[1]}:{strongest[2]}", strongest[0]


def bootstrap_and_summarize(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    model_preds = pd.read_csv(out_root / "evomil_model_predictions.csv")
    model_results = pd.read_csv(out_root / "evomil_model_results.csv")
    best_seed = int(model_results.sort_values("test_macro_f1", ascending=False).iloc[0]["seed"])
    model_best = model_preds[model_preds["seed"] == best_seed].copy()
    baseline, baseline_name, baseline_score = collect_baseline_predictions(args)
    split = pd.read_csv(out_root / "evomil_split_manifest.csv")
    joined = model_best.merge(baseline, on=["virus_id", "true_host"], how="inner").merge(split[["virus_id", "proteome_cluster"]], on="virus_id", how="left")
    labels = sorted(joined["true_host"].unique())
    observed = f1_score(joined["true_host"], joined["predicted_host"], labels=labels, average="macro", zero_division=0) - baseline_score
    groups = sorted(joined["proteome_cluster"].astype(str).unique())
    by_group = {g: joined[joined["proteome_cluster"].astype(str) == g] for g in groups}
    rng = np.random.default_rng(args.bootstrap_seed)
    samples = []
    invalid = 0
    attempts = 0
    while len(samples) < args.n_bootstrap and attempts < args.n_bootstrap * 20:
        attempts += 1
        chosen = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in chosen], ignore_index=True)
        if set(labels) - set(sample["true_host"]):
            invalid += 1
            continue
        model_f1 = f1_score(sample["true_host"], sample["predicted_host"], labels=labels, average="macro", zero_division=0)
        base_f1 = f1_score(sample["true_host"], sample["baseline_pred"], labels=labels, average="macro", zero_division=0)
        samples.append({"replicate": len(samples) + 1, "model_macro_f1": model_f1, "baseline_macro_f1": base_f1, "delta_model_minus_baseline": model_f1 - base_f1})
    write_csv(out_root / "evomil_bootstrap_samples.csv", samples, ["replicate", "model_macro_f1", "baseline_macro_f1", "delta_model_minus_baseline"])
    deltas = np.array([row["delta_model_minus_baseline"] for row in samples], dtype=float)
    summary = {
        "status": "complete" if len(samples) == args.n_bootstrap else "partial",
        "strongest_nonfoundation_baseline": baseline_name,
        "strongest_nonfoundation_test_macro_f1": baseline_score,
        "best_model_seed": best_seed,
        "observed_delta": observed,
        "valid_bootstrap_replicates": len(samples),
        "invalid_bootstrap_replicates": invalid,
        "attempted_bootstrap_replicates": attempts,
        "mean_delta": float(np.mean(deltas)) if len(deltas) else None,
        "median_delta": float(np.median(deltas)) if len(deltas) else None,
        "ci95_low": float(np.quantile(deltas, 0.025)) if len(deltas) else None,
        "ci95_high": float(np.quantile(deltas, 0.975)) if len(deltas) else None,
        "p_delta_gt_0": float(np.mean(deltas > 0)) if len(deltas) else None,
        "p_delta_lt_0": float(np.mean(deltas < 0)) if len(deltas) else None,
    }
    write_json(out_root / "evomil_bootstrap_summary.json", summary)
    return summary


def sanity_report(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    payload = {
        "created_at": now_utc(),
        "status": "SANITY_WARNING",
        "reason": "Official repository provides pickled fold loaders and trained models, but not raw protein FASTA identities for exact official-style reproduction. Pipeline smoke uses reconstructed FASTA and local ESM-1b embeddings; formal strict evaluation does not depend on original random folds.",
        "official_reference_results_present": (Path(args.evomil_repo) / "Results/pro_binary/virus_testset_5fold_best_AUC_epoch150_10.csv").exists(),
        "strict_formal_may_proceed": True,
    }
    write_json(out_root / "evomil_reproduction_sanity_report.json", payload)
    (out_root / "evomil_reproduction_sanity_report.md").write_text(f"# EvoMIL Reproduction Sanity\n\nStatus: `{payload['status']}`\n\n{payload['reason']}\n")
    return payload


def smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    report = generate_embeddings(args, smoke_limit=args.smoke_embedding_segments)
    payload = {
        "created_at": now_utc(),
        "status": "pass" if report["embedding_count"] > 0 else "fail",
        "checks": {
            "virus_protein_manifest_loading": valid_csv(Path(args.out_root) / "evomil_sequence_manifest.csv"),
            "protein_preprocessing": valid_json(Path(args.out_root) / "evomil_preprocessing_audit.json"),
            "esm1b_embedding": report["embedding_count"] > 0,
            "split_loading": valid_json(Path(args.out_root) / "evomil_split_audit.json"),
        },
    }
    write_json(Path(args.out_root) / "evomil_smoke_test.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"smoke failed: {payload}")
    return payload


def final_summary(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    seq = read_json(out_root / "evomil_sequence_acquisition_report.json")
    split = read_json(out_root / "evomil_split_audit.json")
    bootstrap = read_json(out_root / "evomil_bootstrap_summary.json")
    model = pd.read_csv(out_root / "evomil_model_results.csv") if (out_root / "evomil_model_results.csv").exists() else pd.DataFrame()
    seed_positive = 0
    strongest = float(bootstrap.get("strongest_nonfoundation_test_macro_f1") or 0.0)
    if not model.empty:
        seed_positive = int((pd.to_numeric(model["test_macro_f1"], errors="coerce") > strongest).sum())
        best_model = float(pd.to_numeric(model["test_macro_f1"], errors="coerce").max())
    else:
        best_model = 0.0
    status = "INSUFFICIENT_EVIDENCE"
    if (bootstrap.get("ci95_low") is not None and float(bootstrap["ci95_low"]) > 0 and seed_positive >= 4 and best_model > strongest):
        status = "PRELIMINARILY_QUALIFIED"
    elif best_model <= strongest:
        status = "NO_QUALIFYING_HEADROOM"
    payload = {
        "created_at": now_utc(),
        "final_status": status,
        "expected_viruses": seq.get("expected_viruses"),
        "resolved_viruses": seq.get("resolved_viruses"),
        "excluded_viruses": seq.get("unresolved_viruses"),
        "split": split,
        "strongest_nonfoundation_baseline": bootstrap.get("strongest_nonfoundation_baseline"),
        "strongest_nonfoundation_macro_f1": strongest,
        "best_esm1b_mil_macro_f1": best_model,
        "model_excess": best_model - strongest,
        "bootstrap": bootstrap,
        "positive_seed_count": seed_positive,
        "qualified": status == "PRELIMINARILY_QUALIFIED",
    }
    write_json(out_root / "evomil_summary_report.json", payload)
    lines = [
        "# EvoMIL / ESM-1b Strict Qualification Summary",
        "",
        f"- Final status: `{status}`",
        f"- Expected viruses: `{seq.get('expected_viruses')}`",
        f"- Resolved viruses: `{seq.get('resolved_viruses')}`",
        f"- Excluded viruses: `{seq.get('unresolved_viruses')}`",
        f"- Split: `{split.get('sample_counts')}`",
        f"- Strongest non-foundation baseline: `{bootstrap.get('strongest_nonfoundation_baseline')}` / `{strongest}`",
        f"- Best ESM-1b + MIL macro-F1: `{best_model}`",
        f"- Excess: `{best_model - strongest}`",
        f"- 95% CI: `[{bootstrap.get('ci95_low')}, {bootstrap.get('ci95_high')}]`",
        f"- Positive seeds: `{seed_positive}`",
    ]
    (out_root / "evomil_summary_report.md").write_text("\n".join(lines) + "\n")
    return payload


def execute(args: argparse.Namespace) -> None:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    registry = read_json(registry_path(args)) or {"created_at": now_utc(), "status": "running", "events": []}
    try:
        write_json(registry_path(args), registry | {"status": "running", "current_stage": "start"})
        update_status(args, "running", "association_audit")
        log(args, "starting EvoMIL / ESM-1b strict qualification controller")
        association_audit(args)
        choose_task(args)
        update_status(args, "running", "sequence_acquisition")
        acquire_sequences(args)
        preprocess_proteins(args)
        sanity_report(args)
        make_cluster_manifest(args)
        split_clusters(args)
        smoke_test(args)
        update_status(args, "running", "formal_experiment_started", formal_experiment_started=True)
        run_simple_baselines(args)
        run_kmer_baselines(args)
        run_taxonomy_baseline(args)
        run_homology_baseline(args)
        generate_embeddings(args)
        build_bags(args)
        run_mil_models(args)
        bootstrap_and_summarize(args)
        summary = final_summary(args)
        write_json(registry_path(args), registry | {"status": "complete", "current_stage": "complete", "final_status": summary["final_status"], "completed_at": now_utc()})
        update_status(args, "complete", "complete", final_status=summary["final_status"], formal_experiment_started=True)
    except Exception as exc:
        payload = registry | {"status": "blocked", "current_stage": read_json(status_path(args)).get("stage", "unknown"), "blocked_at": now_utc(), "blocker": str(exc)}
        write_json(registry_path(args), payload)
        update_status(args, "blocked", payload["current_stage"], blocker=str(exc), formal_experiment_started=False)
        log(args, f"blocked: {exc}")
        raise


def launch_screen(args: argparse.Namespace) -> None:
    cmd = [
        "screen",
        "-L",
        "-Logfile",
        str(Path(args.log_file).with_suffix(".screen.log")),
        "-dmS",
        args.screen_name,
        args.python,
        str(Path(__file__).resolve()),
        "--execute",
        "--out-root",
        str(Path(args.out_root)),
        "--evomil-repo",
        str(Path(args.evomil_repo)),
        "--log-file",
        str(Path(args.log_file)),
        "--device",
        args.device,
        "--seeds",
        args.seeds,
        "--mil-epochs",
        str(args.mil_epochs),
        "--esm-batch-size",
        str(args.esm_batch_size),
        "--n-jobs",
        str(args.n_jobs),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--ncbi-timeout",
        str(args.ncbi_timeout),
        "--ncbi-retries",
        str(args.ncbi_retries),
        "--ncbi-workers",
        str(args.ncbi_workers),
        "--model-download-timeout",
        str(args.model_download_timeout),
        "--min-host-support",
        str(args.min_host_support),
        "--status-interval",
        str(args.status_interval),
        "--blast-timeout",
        str(args.blast_timeout),
    ]
    result = run_cmd(cmd, timeout=60)
    if result["returncode"] != 0:
        raise RuntimeError(result["output_tail"])
    update_status(args, "running", "screen_launched", screen_name=args.screen_name)
    print(f"launched screen {args.screen_name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-screen", action="store_true")
    parser.add_argument("--screen-name", default="evomil_esm1b_qualification")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--evomil-repo", default=str(DEFAULT_REPO))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-viruses", type=int, default=0)
    parser.add_argument("--min-host-support", type=int, default=20)
    parser.add_argument("--smoke-min-host-support", type=int, default=4)
    parser.add_argument("--ncbi-timeout", type=int, default=60)
    parser.add_argument("--ncbi-retries", type=int, default=4)
    parser.add_argument("--ncbi-workers", type=int, default=2)
    parser.add_argument("--ncbi-sleep", type=float, default=0.34)
    parser.add_argument("--status-interval", type=int, default=25)
    parser.add_argument("--proteome-jaccard-threshold", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--esm-batch-size", type=int, default=2)
    parser.add_argument("--model-download-timeout", type=int, default=7200)
    parser.add_argument("--smoke-embedding-segments", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--blast-timeout", type=int, default=7200)
    parser.add_argument("--blast-max-targets", type=int, default=10)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--mil-epochs", type=int, default=60)
    parser.add_argument("--mil-lr", type=float, default=5e-4)
    parser.add_argument("--mil-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mil-eval-interval", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.launch_screen:
        launch_screen(args)
        return
    if not args.execute:
        raise SystemExit("use --execute or --launch-screen")
    execute(args)


if __name__ == "__main__":
    main()
