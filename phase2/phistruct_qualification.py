"""PHIStruct / SaProt benchmark qualification controller.

This workflow is intentionally conservative: it may run smoke/debug stages with
partial inputs, but formal qualification requires the frozen PHIStruct universe,
strict split audit, homology tools, Foldseek, and frozen SaProt/sequence-PLM
predictions. Missing mandatory evidence yields INSUFFICIENT_EVIDENCE rather
than a weakened benchmark.
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
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, file_sha256, stable_hash, write_metadata
from phase2.project_python import project_python


DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/phistruct_qualification"
DEFAULT_LOG = PROJECT_ROOT / "logs/phistruct_qualification.log"
DEFAULT_PYTHON = project_python()
DEFAULT_TOOL_ROOT = PROJECT_ROOT / "data/external/tools"
OFFICIAL_REPO = "https://github.com/bioinfodlsu/PHIStruct"
OFFICIAL_REPO_COMMIT = "77e5753c62d17b4f21cdbf9200008143aebf6551"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO*-]+$", re.I)
STATES = {"pending", "running", "complete", "valid", "invalid", "failed", "skipped", "blocked"}
FINAL_STATUSES = {
    "PRELIMINARILY_QUALIFIED",
    "STRUCTURE_BASELINE_SATURATED",
    "GENERAL_PLM_SIGNAL_ONLY",
    "HOMOLOGY_BASELINE_SATURATED",
    "SIMPLE_BASELINE_SATURATED",
    "NO_QUALIFYING_HEADROOM",
    "INSUFFICIENT_EVIDENCE",
}

ASSETS: list[dict[str, str]] = [
    {
        "asset_name": "PHIStruct repository",
        "asset_type": "source_code",
        "source_url": OFFICIAL_REPO,
        "version": "main",
        "revision": OFFICIAL_REPO_COMMIT,
        "local_path": "data/external/phistruct/PHIStruct",
        "license": "MIT",
        "notes": "Official PHIStruct repository.",
    },
    {
        "asset_name": "PHIStruct trained ESKAPEE model",
        "asset_type": "trained_model",
        "source_url": "https://drive.google.com/file/d/1hf2UDs0rt34_T6FaUc5nB7g_kQLWFoi_/view?usp=sharing",
        "version": "published",
        "revision": "1hf2UDs0rt34_T6FaUc5nB7g_kQLWFoi_",
        "local_path": "data/external/phistruct/model/phistruct_trained.joblib.gz",
        "license": "not_specified_in_readme",
        "notes": "Official trained model linked by PHIStruct README.",
    },
    {
        "asset_name": "PHIStruct ESKAPEE SaProt training CSV",
        "asset_type": "embedding_dataset",
        "source_url": "https://drive.google.com/file/d/17yxaoeCF8H_rBIGPibP9qJUlL_N7H8kt/view?usp=sharing",
        "version": "published",
        "revision": "17yxaoeCF8H_rBIGPibP9qJUlL_N7H8kt",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_eskapee_training.csv",
        "license": "not_specified_in_readme",
        "notes": "README sample/training dataset: id, host genus, 1280 SaProt embedding columns.",
    },
    {
        "asset_name": "PHIStruct consolidated metadata",
        "asset_type": "metadata_archive",
        "source_url": "https://drive.google.com/file/d/1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI/view?usp=sharing",
        "version": "published",
        "revision": "1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI",
        "local_path": "data/external/phistruct/experiments/data/inphared/consolidated.zip",
        "license": "not_specified_in_readme",
        "notes": "Expected to contain consolidated/rbp.csv and related metadata.",
    },
    {
        "asset_name": "PHIStruct FASTA archive",
        "asset_type": "sequence_archive",
        "source_url": "https://drive.google.com/file/d/1NMFR3JrrrCHLoCMQp2nia4dgtcXs5x05/view?usp=sharing",
        "version": "published",
        "revision": "1NMFR3JrrrCHLoCMQp2nia4dgtcXs5x05",
        "local_path": "data/external/phistruct/experiments/data/inphared/fasta.zip",
        "license": "not_specified_in_readme",
        "notes": "Expected to contain RBP FASTA files.",
    },
    {
        "asset_name": "PHIStruct predicted RBP PDB archive",
        "asset_type": "structure_archive",
        "source_url": "https://zenodo.org/records/11202338/files/rbp_dataset.zip?download=1",
        "version": "published",
        "revision": "10.5281/zenodo.11202338:v1",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/pdb.zip",
        "license": "not_specified_in_readme",
        "notes": "Official Zenodo DOI release of ColabFold-predicted RBP structures; README also references Google Drive ID 1ZPRdaHwsFOPksLbOyQerREG0gY0p4-AT.",
    },
    {
        "asset_name": "PHIStruct consolidated structural FASTA",
        "asset_type": "sequence_fasta",
        "source_url": "https://drive.google.com/file/d/1LTZte1f4lreQ5MXWeM-y2Mtp9z96pXS7/view?usp=sharing",
        "version": "published",
        "revision": "1LTZte1f4lreQ5MXWeM-y2Mtp9z96pXS7",
        "local_path": "data/external/phistruct/experiments/data/inphared/fasta/complete-struct.fasta",
        "license": "not_specified_in_readme",
        "notes": "Official consolidated FASTA used in PHIStruct classifier notebooks before CD-HIT clustering.",
    },
    {
        "asset_name": "PHIStruct SaProt embedding archive",
        "asset_type": "embedding_archive",
        "source_url": "https://drive.google.com/file/d/1l1r41Ze56tXQv_U_KShjECpdaoHffJ8d/view?usp=sharing",
        "version": "published",
        "revision": "1l1r41Ze56tXQv_U_KShjECpdaoHffJ8d",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_embeddings.zip",
        "license": "not_specified_in_readme",
        "notes": "Per-protein published SaProt embeddings.",
    },
    {
        "asset_name": "PHIStruct SaProt consolidated CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1rY65V6wKvfVzC0AENyERMHJIY0b432r6/view?usp=sharing",
        "version": "published",
        "revision": "1rY65V6wKvfVzC0AENyERMHJIY0b432r6",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Consolidated SaProt embedding CSV.",
    },
    {
        "asset_name": "PHIStruct SaProt low-confidence mask CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/15M25MbPMmfpk9rAy2I5Y3SlqC4Gi-EId/view?usp=sharing",
        "version": "published",
        "revision": "15M25MbPMmfpk9rAy2I5Y3SlqC4Gi-EId",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_mask_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Published SaProt low-confidence-masked consolidated CSV.",
    },
    {
        "asset_name": "PHIStruct SaProt sequence-mask CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1TTNlUVcaNaWHXMq4n962JTvFEfvGsbVj/view?usp=sharing",
        "version": "published",
        "revision": "1TTNlUVcaNaWHXMq4n962JTvFEfvGsbVj",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_seq_mask_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Published SaProt sequence-masked consolidated CSV.",
    },
    {
        "asset_name": "PHIStruct SaProt structure-mask CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1eeQphah4GVjxms8vutlt43HuEFmTUTug/view?usp=sharing",
        "version": "published",
        "revision": "1eeQphah4GVjxms8vutlt43HuEFmTUTug",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_struct_mask_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Published SaProt structure-masked consolidated CSV.",
    },
    {
        "asset_name": "PHIStruct ProtT5 consolidated CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1PLrfpkUd37G8jbYInWFoghlw_SGHogSV/view?usp=sharing",
        "version": "published",
        "revision": "1PLrfpkUd37G8jbYInWFoghlw_SGHogSV",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_prostt5_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Published sequence-PLM consolidated CSV.",
    },
    {
        "asset_name": "PHIStruct PST consolidated CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1VaAtVZOxgSWG2vy53AKw71teE_pjS6ST/view?usp=sharing",
        "version": "published",
        "revision": "1VaAtVZOxgSWG2vy53AKw71teE_pjS6ST",
        "local_path": "data/external/phistruct/experiments/data/inphared/structure/rbp_pst_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Published structure-aware comparator CSV.",
    },
    {
        "asset_name": "PHIStruct ESM-1b consolidated CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI/view?usp=sharing",
        "version": "published_consolidated_archive",
        "revision": "1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI",
        "local_path": "data/external/phistruct/experiments/data/inphared/consolidated/rbp_embeddings_esm1b_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Expected member of official consolidated archive; required for sequence-PLM comparison when available.",
    },
    {
        "asset_name": "PHIStruct ESM2 consolidated CSV",
        "asset_type": "embedding_csv",
        "source_url": "https://drive.google.com/file/d/1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI/view?usp=sharing",
        "version": "published_consolidated_archive",
        "revision": "1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI",
        "local_path": "data/external/phistruct/experiments/data/inphared/consolidated/rbp_embeddings_esm2_relaxed_r3.csv",
        "license": "not_specified_in_readme",
        "notes": "Expected member of official consolidated archive; required for sequence-PLM comparison when available.",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: object, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
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


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def append_log(args: argparse.Namespace, message: str) -> None:
    line = f"[{now_utc()}] {message}"
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log_file).open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def status_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "phistruct_status.json"


def registry_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "phistruct_registry.json"


def summary_json_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "summary_report.json"


def summary_md_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "summary_report.md"


def update_status(args: argparse.Namespace, status: str, stage: str, **extra: Any) -> None:
    if status not in STATES:
        raise ValueError(f"invalid status: {status}")
    write_json(
        status_path(args),
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": stage,
            "out_root": str(Path(args.out_root)),
            "registry_path": str(registry_path(args)),
            "log_file": str(Path(args.log_file)),
            **extra,
        },
    )


def load_registry(args: argparse.Namespace) -> dict[str, Any]:
    existing = read_json(registry_path(args))
    if existing:
        return existing
    return {
        "workflow": {
            "name": "phistruct_saprot_qualification",
            "created_at": now_utc(),
            "out_root": str(Path(args.out_root)),
            "formal": bool(args.formal),
        },
        "tasks": [],
    }


def save_registry(args: argparse.Namespace, registry: Mapping[str, Any]) -> None:
    payload = dict(registry)
    payload.setdefault("workflow", {})["updated_at"] = now_utc()
    write_json(registry_path(args), payload)


def update_task(args: argparse.Namespace, registry: dict[str, Any], task_id: str, stage: str, status: str, **extra: Any) -> None:
    tasks = registry.setdefault("tasks", [])
    row = next((item for item in tasks if item.get("task_id") == task_id), None)
    if row is None:
        row = {"task_id": task_id, "created_at": now_utc(), "pid": os.getpid()}
        tasks.append(row)
    row.update({"stage": stage, "status": status, "updated_at": now_utc(), **extra})
    if status == "running":
        row["started_at"] = now_utc()
    if status in {"complete", "valid", "invalid", "failed", "skipped", "blocked"}:
        row["completed_at"] = now_utc()
    save_registry(args, registry)
    update_status(args, status, stage, task_id=task_id, **extra)


def run_stage(
    args: argparse.Namespace,
    registry: dict[str, Any],
    task_id: str,
    stage: str,
    output: Path,
    func: Any,
    *,
    validator: Any | None = None,
    skippable: bool = True,
) -> Any:
    if args.resume and skippable and output.exists() and (validator is None or validator(output)):
        update_task(args, registry, task_id, stage, "skipped", output_path=str(output), reason="resume_output_valid")
        append_log(args, f"skip {stage}: {output}")
        return None
    update_task(args, registry, task_id, stage, "running", output_path=str(output))
    started = time.time()
    try:
        result = func()
        if validator and not validator(output):
            raise RuntimeError(f"validator failed for {output}")
        update_task(args, registry, task_id, stage, "complete", output_path=str(output), elapsed_sec=round(time.time() - started, 2))
        append_log(args, f"complete {stage}: {output}")
        return result
    except Exception as exc:
        update_task(
            args,
            registry,
            task_id,
            stage,
            "failed",
            output_path=str(output),
            elapsed_sec=round(time.time() - started, 2),
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        append_log(args, f"failed {stage}: {type(exc).__name__}: {exc}")
        raise


def json_ok(path: Path) -> bool:
    return bool(read_json(path))


def csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
            next(reader)
        except StopIteration:
            return False
        return True


def sha256_if_file(path: Path) -> str:
    if path.exists() and path.is_file():
        return file_sha256(path)
    return "missing"


def extract_google_id(url_or_id: str) -> str:
    text = str(url_or_id)
    match = re.search(r"/d/([^/]+)", text)
    if match:
        return match.group(1)
    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query:
        return query["id"][0]
    return text


def tool_search_path(args: argparse.Namespace | None = None) -> str:
    entries = [
        str(DEFAULT_TOOL_ROOT / "bin"),
        str(DEFAULT_TOOL_ROOT / "ncbi-blast/bin"),
        str(DEFAULT_TOOL_ROOT / "hmmer/bin"),
        str(DEFAULT_TOOL_ROOT / "foldseek/bin"),
        str(Path(DEFAULT_PYTHON).parent),
    ]
    if args is not None and getattr(args, "tool_root", ""):
        root = Path(args.tool_root)
        entries = [
            str(root / "bin"),
            str(root / "ncbi-blast/bin"),
            str(root / "hmmer/bin"),
            str(root / "foldseek/bin"),
            *entries,
        ]
    return os.pathsep.join([*entries, os.environ.get("PATH", "")])


def command_env(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = tool_search_path(args)
    return env


def validate_downloaded_asset(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing_or_empty"
    prefix = path.read_bytes()[:4096].lower()
    if b"<html" in prefix or b"google drive - virus scan warning" in prefix:
        return False, "html_interstitial"
    suffix = path.suffix.lower()
    if suffix == ".zip" and not zipfile.is_zipfile(path):
        try:
            subprocess.check_call(["tar", "-tzf", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        except Exception:
            return False, "invalid_zip_or_tar_gz"
        return True, "valid_tar_gz_mislabeled_zip"
    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        try:
            subprocess.check_call(["tar", "-tzf", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False, "invalid_tar_gz"
    if suffix == ".csv":
        try:
            with path.open(errors="ignore") as handle:
                sample = handle.read(4096)
            if "," not in sample and "\t" not in sample:
                return False, "csv_delimiter_not_detected"
        except Exception as exc:
            return False, f"csv_validation_failed:{type(exc).__name__}"
    return True, "valid"


def download_with_gdown(file_id: str, dest: Path, args: argparse.Namespace) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown_cmd = [args.python, "-m", "gdown", file_id, "-O", str(dest), "--continue"]
    last = ""
    for attempt in range(args.download_retries + 1):
        proc = subprocess.run(
            gdown_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=command_env(args),
            check=False,
            timeout=args.download_timeout_sec,
        )
        ok, reason = validate_downloaded_asset(dest)
        if proc.returncode == 0 and ok:
            return {"status": "downloaded", "method": "gdown", "path": str(dest), "size_bytes": dest.stat().st_size, "sha256": file_sha256(dest)}
        if dest.exists() and not ok and reason == "html_interstitial":
            dest.unlink()
        last = f"returncode={proc.returncode} validation={reason} tail={proc.stdout.splitlines()[-10:]}"
        time.sleep(min(30, 2 + attempt * 3))
    return {"status": "failed", "method": "gdown", "path": str(dest), "reason": last}


def download_google_drive_file(file_id: str, dest: Path, retries: int = 2, timeout_sec: int = 60) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}"
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    last_error = ""
    for attempt in range(retries + 1):
        try:
            cookie_jar: dict[str, str] = {}
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                for key, value in response.headers.items():
                    if key.lower() == "set-cookie":
                        for part in value.split(";"):
                            if part.strip().startswith("download_warning"):
                                name, val = part.strip().split("=", 1)
                                cookie_jar[name] = val
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
                sample = response.read(4096)
                if b"confirm=" in sample or "text/html" in content_type.lower():
                    token_match = re.search(rb"confirm=([0-9A-Za-z_]+)", sample)
                    token = token_match.group(1).decode("ascii") if token_match else next(iter(cookie_jar.values()), "")
                    if token:
                        final_url = f"{url}&confirm={urllib.parse.quote(token)}"
                        headers = {"User-Agent": "Mozilla/5.0"}
                        if cookie_jar:
                            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookie_jar.items())
                        req = urllib.request.Request(final_url, headers=headers)
                        with urllib.request.urlopen(req, timeout=timeout_sec) as confirmed:
                            with tmp.open("wb") as handle:
                                shutil.copyfileobj(confirmed, handle)
                    else:
                        tmp.write_bytes(sample + response.read())
                else:
                    with tmp.open("wb") as handle:
                        handle.write(sample)
                        shutil.copyfileobj(response, handle)
            if tmp.exists() and tmp.stat().st_size > 0:
                prefix = tmp.read_bytes()[:4096].lower()
                if b"<html" in prefix or b"google drive - virus scan warning" in prefix:
                    tmp.unlink()
                    last_error = "download returned an HTML interstitial instead of the requested asset"
                    continue
                ok, reason = validate_downloaded_asset(tmp)
                if not ok:
                    tmp.unlink()
                    last_error = reason
                    continue
                os.replace(tmp, dest)
                return {"status": "downloaded", "method": "urllib", "path": str(dest), "size_bytes": dest.stat().st_size, "sha256": file_sha256(dest)}
        except Exception as exc:  # noqa: PERF203
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 + attempt)
    if tmp.exists():
        tmp.unlink()
    return {"status": "failed", "path": str(dest), "reason": last_error}


def safe_extract_archive(path: Path, dest: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.exists():
        return {"status": "skipped", "reason": "missing", "path": str(path)}
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(dest)
            names = archive.namelist()
        return {"status": "extracted", "archive_type": "zip", "path": str(path), "dest": str(dest), "members": len(names)}
    result = run_logged(["tar", "-xzf", str(path), "-C", str(dest)], args, timeout=args.archive_extract_timeout_sec)
    if result["returncode"] == 0:
        members = sum(1 for _ in dest.rglob("*"))
        return {"status": "extracted", "archive_type": "tar_gz", "path": str(path), "dest": str(dest), "members": members}
    return {"status": "failed", "archive_type": "unknown", "path": str(path), "dest": str(dest), "extract": result}


def official_consolidated_equivalent(dest: Path) -> Path | None:
    root = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/consolidated/consolidated"
    mapping = {
        "rbp_saprot_relaxed_r3.csv": "rbp_embeddings_saprot_relaxed_r3.csv",
        "rbp_saprot_mask_relaxed_r3.csv": "rbp_embeddings_saprot_mask_relaxed_r3.csv",
        "rbp_saprot_seq_mask_relaxed_r3.csv": "rbp_embeddings_saprot_seq_mask_relaxed_r3.csv",
        "rbp_saprot_struct_mask_relaxed_r3.csv": "rbp_embeddings_saprot_struct_mask_relaxed_r3.csv",
        "rbp_pst_relaxed_r3.csv": "rbp_embeddings_pst_relaxed_r3.csv",
        "rbp_prostt5_relaxed_r3.csv": "rbp_embeddings_prostt5_relaxed_r3.csv",
    }
    candidate = root / mapping.get(dest.name, "")
    return candidate if candidate.exists() else None


def command_versions(commands: Sequence[str], args: argparse.Namespace | None = None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    env = command_env(args)
    for cmd in commands:
        path = shutil.which(cmd, path=env["PATH"])
        version = ""
        if path:
            for version_args in ([path, "--version"], [path, "-version"], [path, "version"]):
                try:
                    version = subprocess.check_output(version_args, text=True, stderr=subprocess.STDOUT, timeout=10, env=env).splitlines()[0]
                    break
                except Exception:
                    continue
        result[cmd] = {"path": path or "", "version": version}
    return result


def static_audit(args: argparse.Namespace) -> dict[str, Any]:
    commands = command_versions(["screen", "git", "conda", "foldseek", "blastp", "psiblast", "makeblastdb", "hmmsearch", "hmmbuild", "hhsearch"], args)
    python_checks: dict[str, str] = {}
    for module in ["numpy", "pandas", "sklearn", "Bio", "torch", "transformers", "joblib", "gdown"]:
        try:
            mod = __import__(module)
            python_checks[module] = str(getattr(mod, "__version__", "available"))
        except Exception as exc:
            python_checks[module] = f"missing: {type(exc).__name__}"
    free_gb = shutil.disk_usage(Path(args.out_root).parent).free / (1024**3)
    mandatory_missing = [name for name in ["foldseek", "blastp", "psiblast", "hmmsearch", "hmmbuild"] if not commands[name]["path"]]
    payload = {
        "created_at": now_utc(),
        "status": "valid" if not mandatory_missing else "blocked_missing_tools",
        "mandatory_missing_tools": mandatory_missing,
        "commands": commands,
        "tool_search_path": tool_search_path(args),
        "python": sys.version.replace("\n", " "),
        "python_modules": python_checks,
        "free_disk_gb": free_gb,
        "stop_on_low_disk_gb": args.stop_on_low_disk_gb,
        "official_repo": OFFICIAL_REPO,
        "official_repo_commit": OFFICIAL_REPO_COMMIT,
        "formal_can_qualify": not mandatory_missing,
    }
    write_json(Path(args.out_root) / "static_audit.json", payload)
    return payload


def run_logged(command: list[str], args: argparse.Namespace, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=command_env(args),
        check=False,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "log_tail": proc.stdout.splitlines()[-80:],
    }


def download_url(url: str, dest: Path, args: argparse.Namespace, timeout: int) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "--retry",
        str(max(1, args.download_retries)),
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout),
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    result = run_logged(command, args, timeout=timeout + 60)
    ok, reason = validate_downloaded_asset(dest)
    return {
        **result,
        "url": url,
        "path": str(dest),
        "status": "downloaded" if result["returncode"] == 0 and ok else "failed",
        "validation": reason,
        "sha256": file_sha256(dest) if ok else "",
        "size_bytes": dest.stat().st_size if dest.exists() else 0,
    }


def install_official_blast(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.tool_root)
    dest = root / "downloads/ncbi-blast-2.17.0+-x64-linux.tar.gz"
    url = "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz"
    ok, _ = validate_downloaded_asset(dest)
    download = {"status": "reused", "path": str(dest), "sha256": file_sha256(dest) if ok else ""}
    if not ok:
        download = download_url(url, dest, args, timeout=args.tool_download_timeout_sec)
    ok, reason = validate_downloaded_asset(dest)
    if not ok:
        return {"tool": "blast", "status": "failed", "download": download, "reason": reason}
    extract_dir = root / "tmp/blast_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    result = run_logged(["tar", "-xzf", str(dest), "-C", str(extract_dir)], args, timeout=300)
    candidates = sorted(extract_dir.glob("ncbi-blast-*"))
    target = root / "ncbi-blast"
    if candidates:
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(candidates[0]), str(target))
    return {"tool": "blast", "status": "complete" if (target / "bin/blastp").exists() else "failed", "download": download, "extract": result, "path": str(target / "bin")}


def install_official_foldseek(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.tool_root)
    dest = root / "downloads/foldseek-linux-avx2.tar.gz"
    urls = [
        "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz",
        "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz",
    ]
    ok, _ = validate_downloaded_asset(dest)
    download = {"status": "reused", "path": str(dest), "sha256": file_sha256(dest) if ok else ""}
    if not ok:
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        for url in urls:
            download = download_url(url, dest, args, timeout=args.tool_download_timeout_sec)
            ok, _ = validate_downloaded_asset(dest)
            if ok:
                break
    ok, reason = validate_downloaded_asset(dest)
    if not ok:
        return {"tool": "foldseek", "status": "failed", "download": download, "reason": reason}
    extract_dir = root / "tmp/foldseek_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    result = run_logged(["tar", "-xzf", str(dest), "-C", str(extract_dir)], args, timeout=300)
    candidates = sorted(extract_dir.glob("foldseek"))
    target = root / "foldseek"
    if candidates:
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(candidates[0]), str(target))
    return {"tool": "foldseek", "status": "complete" if (target / "bin/foldseek").exists() else "failed", "download": download, "extract": result, "path": str(target / "bin")}


def conda_install(args: argparse.Namespace, packages: Sequence[str], *, classic: bool) -> dict[str, Any]:
    conda = shutil.which("conda", path=command_env(args)["PATH"])
    if not conda:
        return {"status": "failed", "reason": "conda_missing", "packages": list(packages)}
    cmd = [
        conda,
        "install",
        "-y",
        "-n",
        args.conda_env,
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        "--override-channels",
        *packages,
    ]
    env = command_env(args)
    if classic:
        cmd.insert(2, "--solver")
        cmd.insert(3, "classic")
        env["CONDA_NO_PLUGINS"] = "true"
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False, timeout=args.conda_install_timeout_sec)
    return {
        "status": "complete" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "command": cmd,
        "packages": list(packages),
        "classic_solver": classic,
        "log_tail": proc.stdout.splitlines()[-80:],
    }


def install_missing_tools(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    missing = list(audit.get("mandatory_missing_tools", []))
    out = Path(args.out_root) / "tool_install_report.json"
    if not args.install_missing_tools or not missing:
        payload = {"created_at": now_utc(), "status": "skipped", "missing_tools_before": missing}
        write_json(out, payload)
        return payload
    attempts: list[dict[str, Any]] = []
    refreshed = static_audit(args)
    for pass_idx in range(max(1, args.dependency_retries)):
        if not refreshed.get("mandatory_missing_tools"):
            break
        attempts.append({"pass": pass_idx + 1, **conda_install(args, ["foldseek", "blast", "hmmer"], classic=False)})
        refreshed = static_audit(args)
        if refreshed.get("mandatory_missing_tools"):
            attempts.append({"pass": pass_idx + 1, **conda_install(args, ["foldseek", "blast", "hmmer"], classic=True)})
            refreshed = static_audit(args)
        if "blastp" in refreshed.get("mandatory_missing_tools", []) or "psiblast" in refreshed.get("mandatory_missing_tools", []):
            attempts.append({"pass": pass_idx + 1, **install_official_blast(args)})
            refreshed = static_audit(args)
        if "foldseek" in refreshed.get("mandatory_missing_tools", []):
            attempts.append({"pass": pass_idx + 1, **install_official_foldseek(args)})
            refreshed = static_audit(args)
        if "hmmsearch" in refreshed.get("mandatory_missing_tools", []) or "hmmbuild" in refreshed.get("mandatory_missing_tools", []):
            attempts.append({"pass": pass_idx + 1, **conda_install(args, ["hmmer"], classic=True)})
            refreshed = static_audit(args)
        if refreshed.get("mandatory_missing_tools") and pass_idx + 1 < max(1, args.dependency_retries):
            time.sleep(args.dependency_retry_delay_sec)
    refreshed = static_audit(args)
    payload = {
        "created_at": now_utc(),
        "status": "complete" if not refreshed.get("mandatory_missing_tools") else "failed",
        "missing_tools_before": missing,
        "missing_tools_after": refreshed.get("mandatory_missing_tools", []),
        "attempts": attempts,
    }
    write_json(out, payload)
    return payload


def asset_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for asset in ASSETS:
        local = PROJECT_ROOT / asset["local_path"]
        rows.append(
            {
                **asset,
                "local_path": str(local),
                "exists": local.exists(),
                "size_bytes": local.stat().st_size if local.exists() and local.is_file() else "",
                "checksum": sha256_if_file(local),
                "download_date": now_utc() if local.exists() else "",
            }
        )
    payload = {
        "created_at": now_utc(),
        "status": "registered",
        "assets": rows,
        "official_sources": {
            "repository": OFFICIAL_REPO,
            "repository_commit": OFFICIAL_REPO_COMMIT,
            "paper": "https://doi.org/10.1093/bioinformatics/btaf016",
            "structure_dataset_doi": "https://doi.org/10.5281/zenodo.11202338",
        },
    }
    write_json(Path(args.out_root) / "phistruct_external_assets.json", payload)
    return payload


def acquire_assets(args: argparse.Namespace) -> dict[str, Any]:
    actions = []
    for asset in ASSETS:
        dest = PROJECT_ROOT / asset["local_path"]
        if dest.exists():
            ok, reason = validate_downloaded_asset(dest) if dest.is_file() else (True, "directory")
            if ok:
                actions.append({"asset_name": asset["asset_name"], "status": "reused", "path": str(dest), "sha256": sha256_if_file(dest), "validation": reason})
                if dest.suffix.lower() == ".zip":
                    extract_dest = dest.with_suffix("")
                    if not extract_dest.exists() or not any(extract_dest.iterdir()):
                        actions.append({"asset_name": asset["asset_name"], **safe_extract_archive(dest, extract_dest, args)})
                continue
            if not args.auto_download_assets:
                actions.append({"asset_name": asset["asset_name"], "status": "invalid_existing", "path": str(dest), "sha256": sha256_if_file(dest), "validation": reason})
                continue
            dest.unlink()
        if asset["version"] == "published_consolidated_archive":
            actions.append({"asset_name": asset["asset_name"], "status": "expected_from_archive", "path": str(dest), "archive_revision": asset["revision"]})
            continue
        equivalent = official_consolidated_equivalent(dest)
        if equivalent is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(equivalent, dest)
            actions.append(
                {
                    "asset_name": asset["asset_name"],
                    "status": "linked_from_official_consolidated_archive",
                    "path": str(dest),
                    "source_path": str(equivalent),
                    "sha256": file_sha256(equivalent),
                    "source_revision": "1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI",
                }
            )
            continue
        if not args.auto_download_assets:
            actions.append({"asset_name": asset["asset_name"], "status": "missing_not_downloaded", "path": str(dest)})
            continue
        if asset["asset_type"] == "source_code":
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["git", "clone", "--depth", "1", asset["source_url"], str(dest)]
            try:
                proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=args.git_clone_timeout_sec)
            except subprocess.TimeoutExpired as exc:
                actions.append(
                    {
                        "asset_name": asset["asset_name"],
                        "status": "failed",
                        "path": str(dest),
                        "reason": f"git clone timed out after {args.git_clone_timeout_sec}s",
                        "log_tail": (exc.stdout or "").splitlines()[-20:] if isinstance(exc.stdout, str) else [],
                    }
                )
                continue
            if proc.returncode == 0:
                actions.append({"asset_name": asset["asset_name"], "status": "downloaded", "path": str(dest)})
            else:
                actions.append({"asset_name": asset["asset_name"], "status": "failed", "path": str(dest), "log_tail": proc.stdout.splitlines()[-20:]})
            continue
        if "drive.google.com" in asset["source_url"] or re.fullmatch(r"[A-Za-z0-9_-]{20,}", asset["revision"]):
            file_id = extract_google_id(asset["source_url"])
            result = download_with_gdown(file_id, dest, args)
            if result.get("status") != "downloaded":
                result = download_google_drive_file(file_id, dest, retries=args.download_retries, timeout_sec=args.download_timeout_sec)
        else:
            result = download_url(asset["source_url"], dest, args, timeout=args.download_timeout_sec)
        actions.append({"asset_name": asset["asset_name"], **result})
        if result["status"] == "downloaded" and dest.suffix.lower() == ".zip":
            actions.append({"asset_name": asset["asset_name"], **safe_extract_archive(dest, dest.with_suffix(""), args)})
    payload = {
        "created_at": now_utc(),
        "status": "complete" if all(row["status"] in {"reused", "downloaded", "extracted"} for row in actions if row.get("asset_name")) else "partial",
        "actions": actions,
    }
    write_json(Path(args.out_root) / "asset_acquisition_report.json", payload)
    asset_manifest(args)
    return payload


def normalize_key(value: Any) -> str:
    text = str(value or "").strip()
    text = Path(text).name
    text = re.sub(r"(_relaxed.*|\.pdb|\.pt|\.fasta|\.faa|\.fa)$", "", text, flags=re.I)
    return text


def detect_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for col in columns:
        clean = str(col).lower().replace("_", "").replace("-", "").replace(" ", "")
        for candidate in candidates:
            if candidate.lower().replace("_", "").replace("-", "").replace(" ", "") in clean:
                return str(col)
    return None


def read_fasta_sequences(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.exists():
        return result
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix.lower() in {".fa", ".faa", ".fasta", ".fna"}]
    for path in files:
        current = ""
        chunks: list[str] = []
        with path.open(errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current and chunks:
                        result[normalize_key(current)] = "".join(chunks).upper()
                    current = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
            if current and chunks:
                result[normalize_key(current)] = "".join(chunks).upper()
    return result


def find_files_by_id(root: Path, suffixes: Sequence[str]) -> dict[str, str]:
    result = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            result[normalize_key(path.name)] = str(path)
    return result


def read_embedding_csv_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, nrows=10000, header=None)
    except Exception:
        try:
            df = pd.read_csv(path, nrows=10000)
        except Exception:
            return set()
    first = df.iloc[:, 0].astype(str)
    return {normalize_key(v) for v in first if v and str(v).lower() != "nan"}


def load_phage_structure_metadata(root: Path) -> dict[str, dict[str, str]]:
    table = root / "structure/pdb/rbp_dataset/rbp_phage_host_table.csv"
    payload: dict[str, dict[str, str]] = {}
    if not table.exists():
        return payload
    try:
        df = pd.read_csv(table)
    except Exception:
        return payload
    protein_col = detect_column(df.columns, ["Protein ID", "protein_id", "rbp_id"])
    phage_col = detect_column(df.columns, ["Phage Accession", "phage_accession", "phage_id"])
    host_col = detect_column(df.columns, ["Host", "host_genus", "host"])
    if not protein_col:
        return payload
    for _, row in df.iterrows():
        rbp_id = normalize_key(row[protein_col])
        payload[rbp_id] = {
            "phage_id": normalize_key(row[phage_col]) if phage_col else rbp_id,
            "pdb_host": str(row[host_col]).strip().lower() if host_col else "",
        }
    return payload


def load_training_embedding_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"embedding dataset has too few columns: {path}")
    cols = ["rbp_id", "host_genus", *[f"emb_{i}" for i in range(df.shape[1] - 2)]]
    df.columns = cols
    return df


def build_dataset_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    external = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared"
    consolidated_roots = [
        external / "consolidated",
        external / "consolidated" / "consolidated",
        external / "consolidated.zip",
    ]
    metadata_candidates = []
    for root in consolidated_roots:
        if root.is_file():
            continue
        if root.exists():
            metadata_candidates.extend(root.rglob("*.csv"))
            metadata_candidates.extend(root.rglob("*.tsv"))
    training_csv = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_eskapee_training.csv"
    fasta = read_fasta_sequences(external / "fasta")
    structures = find_files_by_id(external / "structure", [".pdb"])
    phage_structure_meta = load_phage_structure_metadata(external)
    embedding_csv = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_relaxed_r3.csv"
    embedding_ids = read_embedding_csv_ids(embedding_csv)
    rows: list[dict[str, Any]] = []
    source = ""
    if training_csv.exists():
        df = load_training_embedding_dataset(training_csv)
        source = str(training_csv)
        for _, row in df.iterrows():
            rbp_id = normalize_key(row["rbp_id"])
            seq = fasta.get(rbp_id, "")
            rows.append(
                {
                    "rbp_id": rbp_id,
                    "sequence": seq,
                    "sequence_length": len(seq) if seq else "",
                    "phage_id": phage_structure_meta.get(rbp_id, {}).get("phage_id") or (rbp_id.split("|")[0] if "|" in rbp_id else rbp_id),
                    "host_genus": str(row["host_genus"]),
                    "structure_path": structures.get(rbp_id, ""),
                    "embedding_path": str(training_csv),
                    "source": "phistruct_training_csv",
                    "viral_family": "",
                    "viral_taxonomy": "",
                    "protein_family": "",
                }
            )
    else:
        for path in sorted(metadata_candidates):
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
            except Exception:
                continue
            host_col = detect_column(df.columns, ["host_genus", "host", "genus"])
            id_col = detect_column(df.columns, ["rbp_id", "protein_id", "accession", "protein_accession"])
            phage_col = detect_column(df.columns, ["phage_id", "phage", "genome_id", "accession"])
            seq_col = detect_column(df.columns, ["sequence", "aa_sequence", "protein_sequence"])
            if not host_col or not id_col:
                continue
            source = str(path)
            for _, row in df.iterrows():
                rbp_id = normalize_key(row[id_col])
                seq = str(row[seq_col]).upper() if seq_col and not pd.isna(row[seq_col]) else fasta.get(rbp_id, "")
                if seq and not AA_RE.match(seq):
                    seq = ""
                rows.append(
                    {
                        "rbp_id": rbp_id,
                        "sequence": seq,
                        "sequence_length": len(seq) if seq else "",
                        "phage_id": normalize_key(row[phage_col]) if phage_col else rbp_id,
                        "host_genus": str(row[host_col]),
                        "structure_path": structures.get(rbp_id, ""),
                        "embedding_path": str(embedding_csv) if rbp_id in embedding_ids else "",
                        "source": "phistruct_consolidated_metadata",
                        "viral_family": str(row.get("family", "")),
                        "viral_taxonomy": str(row.get("taxonomy", "")),
                        "protein_family": str(row.get("protein_family", "")),
                    }
                )
            if rows:
                break
    seen = set()
    unique = []
    for row in rows:
        key = row["rbp_id"]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    fields = [
        "rbp_id",
        "sequence",
        "sequence_length",
        "phage_id",
        "host_genus",
        "structure_path",
        "embedding_path",
        "source",
        "viral_family",
        "viral_taxonomy",
        "protein_family",
    ]
    manifest = out_root / "dataset_manifest.csv"
    write_csv(manifest, unique, fields)
    payload = {
        "created_at": now_utc(),
        "status": "valid" if unique else "NOT_AVAILABLE",
        "source": source,
        "dataset_manifest": str(manifest),
        "rbp_count": len(unique),
        "phage_count": len({r["phage_id"] for r in unique if r["phage_id"]}),
        "host_genus_count": len({r["host_genus"] for r in unique if r["host_genus"]}),
        "with_sequence": sum(bool(r["sequence"]) for r in unique),
        "with_structure": sum(bool(r["structure_path"]) for r in unique),
        "with_embedding": sum(bool(r["embedding_path"]) for r in unique),
    }
    write_json(out_root / "dataset_audit.json", payload)
    return payload


def sequence_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return matches / max(len(a), len(b))


def fallback_sequence_clusters(rows: Sequence[Mapping[str, Any]], threshold: float = 0.40) -> dict[str, str]:
    reps: list[tuple[str, str]] = []
    clusters: dict[str, str] = {}
    for row in rows:
        rbp_id = str(row["rbp_id"])
        seq = str(row.get("sequence", ""))
        assigned = ""
        for cid, rep in reps:
            if sequence_identity(seq, rep) >= threshold:
                assigned = cid
                break
        if not assigned:
            assigned = f"fallback_seq_cluster_{len(reps):05d}"
            reps.append((assigned, seq))
        clusters[rbp_id] = assigned
    return clusters


def official_cdhit_clusters(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, Any]] | None:
    candidates = [
        PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/fasta/fasta/complete-struct-40.fasta.clstr",
        PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/fasta/complete-struct-40.fasta.clstr",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    clusters: dict[str, str] = {}
    current = ""
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current = f"official_cdhit_seq_cluster_40_{line.split()[-1].zfill(5)}"
                continue
            match = re.search(r">([^\.]+(?:\.[0-9]+)?)\.\.\.", line)
            if current and match:
                clusters[normalize_key(match.group(1))] = current
    ids = [str(row["rbp_id"]) for row in rows]
    missing = [rbp_id for rbp_id in ids if rbp_id not in clusters]
    if missing:
        return None
    audit = {
        "method": "official_cdhit_complete_struct_40",
        "source_path": str(path),
        "threshold_percent_identity": 40,
        "cluster_count": len(set(clusters[rbp_id] for rbp_id in ids)),
        "assigned_rbps": len(ids),
        "sha256": file_sha256(path),
    }
    write_json(Path(args.out_root) / "sequence_cluster_audit.json", audit)
    return {rbp_id: clusters[rbp_id] for rbp_id in ids}, audit


def write_fasta(path: Path, rows: Sequence[Mapping[str, Any]], id_col: str = "rbp_id") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            seq = str(row.get("sequence", "")).replace("*", "").replace("-", "").upper()
            if not seq or not AA_RE.match(seq):
                continue
            handle.write(f">{row[id_col]}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")
            count += 1
    return count


def blast_sequence_clusters(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], threshold: float = 40.0) -> tuple[dict[str, str], dict[str, Any]]:
    env = command_env(args)
    blastp = shutil.which("blastp", path=env["PATH"])
    makeblastdb = shutil.which("makeblastdb", path=env["PATH"])
    if not blastp or not makeblastdb:
        raise RuntimeError("BLAST+ blastp/makeblastdb are required for the strict <=40% sequence-cluster split")
    work = Path(args.out_root) / "sequence_cluster_blast"
    fasta = work / "all_rbps.faa"
    db_prefix = work / "all_rbps_db"
    out_tsv = work / "all_vs_all.tsv"
    seq_rows = [row for row in rows if str(row.get("sequence", ""))]
    if len(seq_rows) != len(rows):
        raise RuntimeError(f"strict split requires every row to have a sequence: {len(seq_rows)}/{len(rows)} available")
    write_fasta(fasta, seq_rows)
    run_logged([makeblastdb, "-in", str(fasta), "-dbtype", "prot", "-out", str(db_prefix)], args, timeout=args.blast_timeout_sec)
    command = [
        blastp,
        "-query",
        str(fasta),
        "-db",
        str(db_prefix),
        "-out",
        str(out_tsv),
        "-outfmt",
        "6 qseqid sseqid pident qcovs evalue bitscore",
        "-max_target_seqs",
        str(args.cluster_max_targets),
        "-num_threads",
        str(max(1, args.n_jobs)),
    ]
    blast_result = run_logged(command, args, timeout=args.blast_timeout_sec)
    if blast_result["returncode"] != 0:
        raise RuntimeError(f"BLAST clustering failed: {blast_result['log_tail']}")
    ds = DisjointSet()
    ids = [str(row["rbp_id"]) for row in rows]
    for rbp_id in ids:
        ds.find(rbp_id)
    hit_count = 0
    edge_count = 0
    with out_tsv.open(errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            qid, sid, pident, qcovs = parts[:4]
            hit_count += 1
            if qid == sid:
                continue
            try:
                if float(pident) >= threshold and float(qcovs) >= args.cluster_min_qcov:
                    ds.union(qid, sid)
                    edge_count += 1
            except ValueError:
                continue
    roots = {rbp_id: ds.find(rbp_id) for rbp_id in ids}
    root_names = {root: f"blast_seq_cluster_40_{idx:05d}" for idx, root in enumerate(sorted(set(roots.values())))}
    clusters = {rbp_id: root_names[root] for rbp_id, root in roots.items()}
    audit = {
        "method": "blastp_connected_components",
        "threshold_percent_identity": threshold,
        "minimum_query_coverage": args.cluster_min_qcov,
        "hit_count": hit_count,
        "cluster_edge_count": edge_count,
        "cluster_count": len(set(clusters.values())),
        "fasta": str(fasta),
        "blast_output": str(out_tsv),
        "blast_command": command,
    }
    write_json(Path(args.out_root) / "sequence_cluster_audit.json", audit)
    return clusters, audit


def phage_and_cluster_disjoint_groups(rows: Sequence[Mapping[str, Any]], clusters: Mapping[str, str]) -> dict[str, str]:
    ds = DisjointSet()
    for row in rows:
        rbp_id = str(row["rbp_id"])
        cluster_node = f"cluster::{clusters[rbp_id]}"
        phage_node = f"phage::{row.get('phage_id', rbp_id)}"
        ds.union(rbp_id, cluster_node)
        ds.union(rbp_id, phage_node)
    roots = {str(row["rbp_id"]): ds.find(str(row["rbp_id"])) for row in rows}
    root_names = {root: f"split_group_{idx:05d}" for idx, root in enumerate(sorted(set(roots.values())))}
    return {rbp_id: root_names[root] for rbp_id, root in roots.items()}


def build_split(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.out_root) / "dataset_manifest.csv"
    df = pd.read_csv(dataset).fillna("")
    rows = df.to_dict("records")
    if not rows:
        raise RuntimeError("dataset manifest is empty")
    cluster_audit: dict[str, Any]
    if args.formal:
        official = official_cdhit_clusters(args, rows)
        if official is not None:
            clusters, cluster_audit = official
        else:
            clusters, cluster_audit = blast_sequence_clusters(args, rows)
    else:
        clusters = fallback_sequence_clusters(rows)
        cluster_audit = {"method": "fallback_pairwise_identity_proxy", "formal_allowed": False}
    split_groups = phage_and_cluster_disjoint_groups(rows, clusters)
    group_keys = [split_groups[row["rbp_id"]] for row in rows]
    labels = df["host_genus"].astype(str).to_numpy()
    indices = np.arange(len(df))
    train_idx: np.ndarray
    temp_idx: np.ndarray
    splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_fraction, random_state=args.split_seed)
    train_idx, temp_idx = next(splitter.split(indices, labels, groups=np.array(group_keys)))
    temp_groups = np.array(group_keys)[temp_idx]
    temp_labels = labels[temp_idx]
    val_fraction_of_temp = args.val_fraction / (args.val_fraction + args.test_fraction)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=val_fraction_of_temp, random_state=args.split_seed + 1)
    val_rel, test_rel = next(splitter2.split(temp_idx, temp_labels, groups=temp_groups))
    split = np.full(len(df), "train", dtype=object)
    split[temp_idx[val_rel]] = "validation"
    split[temp_idx[test_rel]] = "test"
    out_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["sequence_cluster_40"] = clusters[row["rbp_id"]]
        out["split_group"] = split_groups[row["rbp_id"]]
        out["split"] = split[idx]
        out_rows.append(out)
    split_path = Path(args.out_root) / "split_manifest.csv"
    write_csv(split_path, out_rows, [*df.columns, "sequence_cluster_40", "split_group", "split"])
    audit = split_audit(out_rows)
    audit.update({"created_at": now_utc(), "split_manifest": str(split_path), "split_seed": args.split_seed, "sequence_cluster_audit": cluster_audit})
    write_json(Path(args.out_root) / "split_audit.json", audit)
    return audit


def split_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    overlap = {}
    for col in ["rbp_id", "sequence", "sequence_cluster_40", "phage_id", "structure_path"]:
        sets = {name: {str(r.get(col, "")) for r in group if str(r.get(col, ""))} for name, group in by_split.items()}
        overlap[col] = {
            "train_validation": len(sets.get("train", set()) & sets.get("validation", set())),
            "train_test": len(sets.get("train", set()) & sets.get("test", set())),
            "validation_test": len(sets.get("validation", set()) & sets.get("test", set())),
        }
    per_genus = []
    all_genera = sorted({str(r.get("host_genus", "")) for r in rows})
    for genus in all_genera:
        row = {"host_genus": genus}
        for split_name in ["train", "validation", "test"]:
            count = sum(1 for r in by_split.get(split_name, []) if str(r.get("host_genus")) == genus)
            total = max(1, len(by_split.get(split_name, [])))
            row[f"{split_name}_count"] = count
            row[f"{split_name}_proportion"] = count / total
        per_genus.append(row)
    leakage_errors = sum(sum(v.values()) for v in overlap.values())
    cluster_names = [str(r.get("sequence_cluster_40", "")) for r in rows]
    if any(name.startswith("official_cdhit_seq_cluster_40_") for name in cluster_names):
        method = "official_cdhit_complete_struct_40"
    elif any(name.startswith("blast_seq_cluster_40_") for name in cluster_names):
        method = "blastp_connected_components"
    else:
        method = "fallback_pairwise_identity_proxy"
    return {
        "status": "pass" if leakage_errors == 0 else "fail",
        "sample_counts": {name: len(group) for name, group in by_split.items()},
        "class_balance": {
            name: dict(Counter(str(r.get("host_genus", "")) for r in group)) for name, group in by_split.items()
        },
        "overlap": overlap,
        "leakage_error_count": leakage_errors,
        "per_genus": per_genus,
        "sequence_cluster_method": method,
        "formal_note": "Formal split requires BLAST-backed <=40% connected components plus phage-group disjointness.",
    }


def load_split_df(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.out_root) / "split_manifest.csv"
    return pd.read_csv(path).fillna("")


def labels_for(df: pd.DataFrame) -> tuple[LabelEncoder, np.ndarray]:
    enc = LabelEncoder()
    y = enc.fit_transform(df["host_genus"].astype(str))
    return enc, y


def metric_row(model: str, representation: str, params: Mapping[str, Any], y_true: np.ndarray, pred: np.ndarray, enc: LabelEncoder, split: str, seed: int, runtime: float) -> dict[str, Any]:
    return {
        "model": model,
        "representation": representation,
        "hyperparameters": json.dumps(params, sort_keys=True),
        f"{split}_macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        f"{split}_weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        f"{split}_balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "seed": seed,
        "runtime": runtime,
    }


def composition_features(seqs: Sequence[str]) -> np.ndarray:
    rows = []
    for seq in seqs:
        seq = str(seq).upper()
        n = max(1, len(seq))
        rows.append([len(seq), *[seq.count(aa) / n for aa in AA]])
    return np.array(rows, dtype=float)


def run_sequence_baselines(args: argparse.Namespace) -> dict[str, Any]:
    df = load_split_df(args)
    enc, y = labels_for(df)
    train = df["split"] == "train"
    val = df["split"] == "validation"
    test = df["split"] == "test"
    rows = []
    preds_dir = Path(args.out_root) / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    majority = Counter(y[train]).most_common(1)[0][0]
    for split_name, mask in [("validation", val), ("test", test)]:
        pred = np.full(mask.sum(), majority)
        rows.append(metric_row("majority_class", "host_prior", {}, y[mask], pred, enc, split_name, 0, 0.0))
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    for seed in seeds:
        started = time.time()
        x = composition_features(df["sequence"].astype(str))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed))
        clf.fit(x[train], y[train])
        for split_name, mask in [("validation", val), ("test", test)]:
            pred = clf.predict(x[mask])
            rows.append(metric_row("logistic_regression", "length_aa_composition", {"C": 1.0}, y[mask], pred, enc, split_name, seed, time.time() - started))
        started = time.time()
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=max(1, args.n_jobs))
        rf.fit(x[train], y[train])
        for split_name, mask in [("validation", val), ("test", test)]:
            pred = rf.predict(x[mask])
            rows.append(metric_row("random_forest", "length_aa_composition", {"n_estimators": 300}, y[mask], pred, enc, split_name, seed, time.time() - started))
    seqs = df["sequence"].astype(str).str.replace(" ", "", regex=False)
    for k in [1, 2, 3, 4]:
        for c in [0.1, 1.0, 10.0]:
            started = time.time()
            clf = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False, min_df=1),
                LogisticRegression(max_iter=2000, C=c, class_weight="balanced", random_state=seeds[0] if seeds else 0),
            )
            clf.fit(seqs[train], y[train])
            val_pred = clf.predict(seqs[val])
            test_pred = clf.predict(seqs[test])
            rows.append(metric_row("logistic_regression", f"aa_{k}mer_tfidf", {"C": c, "k": k}, y[val], val_pred, enc, "validation", seeds[0] if seeds else 0, time.time() - started))
            rows.append(metric_row("logistic_regression", f"aa_{k}mer_tfidf", {"C": c, "k": k}, y[test], test_pred, enc, "test", seeds[0] if seeds else 0, time.time() - started))
    fields = ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "seed", "runtime"]
    normalized = []
    paired: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["representation"], row["hyperparameters"], int(row["seed"]))
        paired.setdefault(key, {"model": row["model"], "representation": row["representation"], "hyperparameters": row["hyperparameters"], "seed": row["seed"], "runtime": row["runtime"]})
        paired[key].update({k: v for k, v in row.items() if k not in {"model", "representation", "hyperparameters", "seed", "runtime"}})
    normalized = list(paired.values())
    write_csv(Path(args.out_root) / "baseline_results.csv", normalized, fields)
    return {"status": "complete", "rows": len(normalized)}


def load_embedding_frame(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, header=None, nrows=3, low_memory=False)
    headerless = bool(pd.notna(pd.to_numeric(sample.iloc[0, 2], errors="coerce")))
    if headerless:
        df = pd.read_csv(path, header=None, low_memory=False)
        cols = ["rbp_id", "host_genus", *[f"emb_{i}" for i in range(df.shape[1] - 2)]]
        df.columns = cols
        return df
    df = pd.read_csv(path, low_memory=False)
    id_col = detect_column(df.columns, ["rbp_id", "protein_id", "accession"])
    host_col = detect_column(df.columns, ["host_genus", "host", "genus"])
    if not id_col:
        raise ValueError(f"could not detect embedding ID column in {path}")
    emb_cols = [c for c in df.columns if c not in {id_col, host_col} and pd.api.types.is_numeric_dtype(df[c])]
    out = df.rename(columns={id_col: "rbp_id"}).copy()
    if host_col:
        out = out.rename(columns={host_col: "host_genus"})
    else:
        out["host_genus"] = ""
    return out[["rbp_id", "host_genus", *emb_cols]]


def run_embedding_heads(args: argparse.Namespace) -> dict[str, Any]:
    split_df = load_split_df(args)
    split_df["rbp_key"] = split_df["rbp_id"].map(normalize_key)
    embedding_specs = [
        ("SaProt", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_relaxed_r3.csv"),
        ("SaProt_sequence_masked", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_seq_mask_relaxed_r3.csv"),
        ("SaProt_structure_masked", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_struct_mask_relaxed_r3.csv"),
        ("ProtT5", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_prostt5_relaxed_r3.csv"),
        ("PST", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_pst_relaxed_r3.csv"),
        ("SaProt_training_csv", PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_eskapee_training.csv"),
    ]
    rows = []
    per_genus = []
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    for model_name, path in embedding_specs:
        if not path.exists():
            rows.append({"model": model_name, "representation": "published_embedding", "status": "NOT_AVAILABLE", "not_available_reason": f"missing {path}"})
            continue
        try:
            emb = load_embedding_frame(path)
        except Exception as exc:
            rows.append({"model": model_name, "representation": "published_embedding", "status": "NOT_AVAILABLE", "not_available_reason": f"load_failed: {exc}"})
            continue
        emb["rbp_key"] = emb["rbp_id"].map(normalize_key)
        merged = split_df.merge(emb.drop(columns=["host_genus"], errors="ignore"), on="rbp_key", how="inner", suffixes=("", "_emb"))
        emb_cols = [c for c in merged.columns if str(c).startswith("emb_")]
        if not emb_cols:
            emb_cols = [c for c in merged.columns if c not in split_df.columns and pd.api.types.is_numeric_dtype(merged[c])]
        if merged.empty or not emb_cols:
            rows.append({"model": model_name, "representation": "published_embedding", "status": "NOT_AVAILABLE", "not_available_reason": "no overlap with split manifest or no numeric embedding columns"})
            continue
        enc, y = labels_for(merged)
        train = merged["split"] == "train"
        val = merged["split"] == "validation"
        test = merged["split"] == "test"
        x = merged[emb_cols].astype(float).to_numpy()
        for seed in seeds:
            best: tuple[float, float, LogisticRegression] | None = None
            for c in [0.01, 0.1, 1.0, 10.0]:
                started = time.time()
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=c, class_weight="balanced", random_state=seed))
                clf.fit(x[train], y[train])
                val_pred = clf.predict(x[val])
                val_f1 = f1_score(y[val], val_pred, average="macro", zero_division=0)
                if best is None or val_f1 > best[0]:
                    best = (val_f1, time.time() - started, clf)
            assert best is not None
            clf = best[2]
            test_pred = clf.predict(x[test])
            val_pred = clf.predict(x[val])
            params = {"selected_C": json.loads(clf.steps[-1][1].get_params()["C"].__repr__())}
            rows.append(
                {
                    "model": model_name,
                    "representation": "published_embedding",
                    "status": "complete",
                    "hyperparameters": json.dumps(params, sort_keys=True),
                    "validation_macro_f1": f1_score(y[val], val_pred, average="macro", zero_division=0),
                    "test_macro_f1": f1_score(y[test], test_pred, average="macro", zero_division=0),
                    "weighted_f1": f1_score(y[test], test_pred, average="weighted", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y[test], test_pred),
                    "seed": seed,
                    "runtime": best[1],
                    "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    "n_test": int(test.sum()),
                    "prediction_path": "",
                    "not_available_reason": "",
                }
            )
            recalls = recall_score(y[test], test_pred, average=None, zero_division=0)
            f1s = f1_score(y[test], test_pred, average=None, zero_division=0)
            for idx, genus in enumerate(enc.classes_):
                per_genus.append({"model": model_name, "seed": seed, "host_genus": genus, "test_f1": f1s[idx], "test_recall": recalls[idx]})
    fields = ["model", "representation", "status", "hyperparameters", "validation_macro_f1", "test_macro_f1", "weighted_f1", "balanced_accuracy", "seed", "runtime", "n_train", "n_validation", "n_test", "prediction_path", "not_available_reason"]
    write_csv(Path(args.out_root) / "plm_results.csv", rows, fields)
    write_csv(Path(args.out_root) / "per_genus_metrics.csv", per_genus, ["model", "seed", "host_genus", "test_f1", "test_recall"])
    return {"status": "complete", "rows": len(rows)}


def external_metric_row(model: str, representation: str, params: Mapping[str, Any], true_labels: Sequence[str], pred_labels: Sequence[str], split: str, runtime: float, status: str = "complete", reason: str = "") -> dict[str, Any]:
    enc = LabelEncoder()
    y_true = enc.fit_transform(list(true_labels))
    label_to_idx = {label: idx for idx, label in enumerate(enc.classes_)}
    majority = Counter(true_labels).most_common(1)[0][0] if true_labels else ""
    y_pred = np.array([label_to_idx.get(label, label_to_idx.get(majority, 0)) for label in pred_labels])
    return {
        "model": model,
        "representation": representation,
        "hyperparameters": json.dumps(params, sort_keys=True),
        f"{split}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "runtime": runtime,
        "seed": "",
        "status": status,
        "not_available_reason": reason,
    }


def parse_retrieval_predictions(path: Path, train_hosts: Mapping[str, str], query_ids: Sequence[str], majority: str) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    if path.exists():
        with path.open(errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                qid, sid = parts[0], parts[1]
                host = train_hosts.get(normalize_key(sid), train_hosts.get(sid, ""))
                if not host:
                    continue
                try:
                    score = float(parts[-1])
                except ValueError:
                    score = 0.0
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, host)
    return {qid: best.get(qid, (0.0, majority))[1] for qid in query_ids}


def run_blast_family_baseline(args: argparse.Namespace, df: pd.DataFrame, program: str, model_name: str) -> list[dict[str, Any]]:
    env = command_env(args)
    exe = shutil.which(program, path=env["PATH"])
    makeblastdb = shutil.which("makeblastdb", path=env["PATH"])
    if not exe or not makeblastdb:
        return [{"model": model_name, "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": f"missing {program}/makeblastdb"}]
    work = Path(args.out_root) / "external_baselines" / program
    train_rows = df[df["split"] == "train"].to_dict("records")
    train_fasta = work / "train.faa"
    db_prefix = work / "train_db"
    write_fasta(train_fasta, train_rows)
    run_logged([makeblastdb, "-in", str(train_fasta), "-dbtype", "prot", "-out", str(db_prefix)], args, timeout=args.blast_timeout_sec)
    train_hosts = {normalize_key(str(row["rbp_id"])): str(row["host_genus"]) for row in train_rows}
    majority = Counter(str(row["host_genus"]) for row in train_rows).most_common(1)[0][0]
    rows: list[dict[str, Any]] = []
    for split_name in ["validation", "test"]:
        query_rows = df[df["split"] == split_name].to_dict("records")
        query_fasta = work / f"{split_name}.faa"
        out_tsv = work / f"{split_name}.tsv"
        query_ids = [str(row["rbp_id"]) for row in query_rows]
        write_fasta(query_fasta, query_rows)
        command = [
            exe,
            "-query",
            str(query_fasta),
            "-db",
            str(db_prefix),
            "-out",
            str(out_tsv),
            "-outfmt",
            "6 qseqid sseqid pident qcovs evalue bitscore",
            "-max_target_seqs",
            str(args.retrieval_max_targets),
            "-num_threads",
            str(max(1, args.n_jobs)),
        ]
        if program == "psiblast":
            command.extend(["-num_iterations", str(args.psiblast_iterations)])
        started = time.time()
        result = run_logged(command, args, timeout=args.blast_timeout_sec)
        if result["returncode"] != 0:
            rows.append({"model": model_name, "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": f"{program} failed: {result['log_tail']}"})
            continue
        preds = parse_retrieval_predictions(out_tsv, train_hosts, query_ids, majority)
        rows.append(external_metric_row(model_name, "external_retrieval", {"program": program}, [str(row["host_genus"]) for row in query_rows], [preds[qid] for qid in query_ids], split_name, time.time() - started))
    return rows


def parse_hmmer_tblout(path: Path, majority: str) -> dict[str, tuple[float, str]]:
    best: dict[str, tuple[float, str]] = {}
    if not path.exists():
        return best
    host = path.stem.replace("validation_", "").replace("test_", "")
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            try:
                score = float(parts[5])
            except ValueError:
                score = 0.0
            if qid not in best or score > best[qid][0]:
                best[qid] = (score, host or majority)
    return best


def run_hmmer_baseline(args: argparse.Namespace, df: pd.DataFrame) -> list[dict[str, Any]]:
    env = command_env(args)
    hmmbuild = shutil.which("hmmbuild", path=env["PATH"])
    hmmsearch = shutil.which("hmmsearch", path=env["PATH"])
    if not hmmbuild or not hmmsearch:
        return [{"model": "HMMER_remote_homology", "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": "missing hmmbuild/hmmsearch"}]
    work = Path(args.out_root) / "external_baselines" / "hmmer"
    train = df[df["split"] == "train"].copy()
    majority = train["host_genus"].astype(str).value_counts().idxmax()
    profiles: dict[str, Path] = {}
    for host, group in train.groupby("host_genus"):
        seq_row = group.sort_values("sequence_length", ascending=False).iloc[0].to_dict()
        aln = work / f"{host}.faa"
        hmm = work / f"{host}.hmm"
        write_fasta(aln, [seq_row])
        result = run_logged([hmmbuild, "--amino", str(hmm), str(aln)], args, timeout=args.hmmer_timeout_sec)
        if result["returncode"] == 0 and hmm.exists():
            profiles[str(host)] = hmm
    if not profiles:
        return [{"model": "HMMER_remote_homology", "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": "no HMM profiles built"}]
    rows: list[dict[str, Any]] = []
    for split_name in ["validation", "test"]:
        query = df[df["split"] == split_name]
        query_rows = query.to_dict("records")
        query_fasta = work / f"{split_name}.faa"
        write_fasta(query_fasta, query_rows)
        best: dict[str, tuple[float, str]] = {}
        started = time.time()
        for host, hmm in profiles.items():
            tbl = work / f"{split_name}_{host}.tblout"
            result = run_logged([hmmsearch, "--tblout", str(tbl), str(hmm), str(query_fasta)], args, timeout=args.hmmer_timeout_sec)
            if result["returncode"] not in {0, 1}:
                continue
            for qid, scored in parse_hmmer_tblout(tbl, majority).items():
                score, _ = scored
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, host)
        query_ids = [str(row["rbp_id"]) for row in query_rows]
        preds = [best.get(qid, (0.0, majority))[1] for qid in query_ids]
        rows.append(external_metric_row("HMMER_remote_homology", "external_retrieval", {"profiles": "one_longest_train_sequence_per_host"}, query["host_genus"].astype(str).tolist(), preds, split_name, time.time() - started))
    return rows


def run_foldseek_baseline(args: argparse.Namespace, df: pd.DataFrame) -> list[dict[str, Any]]:
    env = command_env(args)
    foldseek = shutil.which("foldseek", path=env["PATH"])
    if not foldseek:
        return [{"model": "Foldseek_structure_topk_vote", "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": "missing foldseek"}]
    work = Path(args.out_root) / "external_baselines" / "foldseek"
    train = df[(df["split"] == "train") & df["structure_path"].astype(str).ne("")]
    train_hosts = {normalize_key(Path(str(row["structure_path"])).name): str(row["host_genus"]) for row in train.to_dict("records")}
    majority = train["host_genus"].astype(str).value_counts().idxmax() if not train.empty else df[df["split"] == "train"]["host_genus"].astype(str).value_counts().idxmax()
    train_dir = work / "train_pdb"
    shutil.rmtree(train_dir, ignore_errors=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    for _, row in train.iterrows():
        src = Path(str(row["structure_path"]))
        if src.exists():
            dst = train_dir / src.name
            if not dst.exists():
                os.symlink(src, dst)
    rows: list[dict[str, Any]] = []
    for split_name in ["validation", "test"]:
        query = df[(df["split"] == split_name) & df["structure_path"].astype(str).ne("")]
        query_dir = work / f"{split_name}_pdb"
        shutil.rmtree(query_dir, ignore_errors=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        query_ids = []
        for _, row in query.iterrows():
            src = Path(str(row["structure_path"]))
            if src.exists():
                dst = query_dir / src.name
                if not dst.exists():
                    os.symlink(src, dst)
                query_ids.append(normalize_key(src.name))
        if not query_ids:
            rows.append({"model": "Foldseek_structure_topk_vote", "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": f"no {split_name} structures"})
            continue
        out_tsv = work / f"{split_name}.tsv"
        tmp = work / f"{split_name}_tmp"
        started = time.time()
        result = run_logged(
            [
                foldseek,
                "easy-search",
                str(query_dir),
                str(train_dir),
                str(out_tsv),
                str(tmp),
                "--threads",
                str(max(1, args.n_jobs)),
                "--format-output",
                "query,target,evalue,bits",
            ],
            args,
            timeout=args.foldseek_timeout_sec,
        )
        if result["returncode"] != 0:
            rows.append({"model": "Foldseek_structure_topk_vote", "representation": "external_retrieval", "status": "NOT_AVAILABLE", "not_available_reason": f"foldseek failed: {result['log_tail']}"})
            continue
        best: dict[str, tuple[float, str]] = {}
        with out_tsv.open(errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                qid = normalize_key(parts[0])
                sid = normalize_key(parts[1])
                host = train_hosts.get(sid, "")
                if not host:
                    continue
                try:
                    score = float(parts[3])
                except ValueError:
                    score = 0.0
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, host)
        pred = [best.get(qid, (0.0, majority))[1] for qid in query_ids]
        true = []
        for _, row in query.iterrows():
            src = Path(str(row["structure_path"]))
            if src.exists():
                true.append(str(row["host_genus"]))
        rows.append(external_metric_row("Foldseek_structure_topk_vote", "external_retrieval", {"program": "foldseek easy-search"}, true, pred, split_name, time.time() - started))
    return rows


def run_external_baselines(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    missing = set(audit.get("mandatory_missing_tools", []))
    split_path = Path(args.out_root) / "split_manifest.csv"
    if missing:
        for model, tool in [("BLASTp_nearest_host", "blastp"), ("PSI-BLAST_topk_host_vote", "psiblast"), ("HMMER_remote_homology", "hmmsearch"), ("Foldseek_structure_topk_vote", "foldseek")]:
            rows.append({"model": model, "representation": "external_retrieval", "hyperparameters": "{}", "validation_macro_f1": "", "test_macro_f1": "", "weighted_f1": "", "balanced_accuracy": "", "runtime": "", "seed": "", "status": "NOT_AVAILABLE", "not_available_reason": f"missing required tool: {tool}" if tool in missing else "blocked_until_all_mandatory_tools_pass"})
    elif not split_path.exists() or not csv_has_rows(split_path):
        for model in ["BLASTp_nearest_host", "PSI-BLAST_topk_host_vote", "HMMER_remote_homology", "Foldseek_structure_topk_vote"]:
            rows.append({"model": model, "representation": "external_retrieval", "hyperparameters": "{}", "validation_macro_f1": "", "test_macro_f1": "", "weighted_f1": "", "balanced_accuracy": "", "runtime": "", "seed": "", "status": "NOT_AVAILABLE", "not_available_reason": "blocked_until_dataset_and_strict_split_are_valid"})
    else:
        df = load_split_df(args)
        rows.extend(run_blast_family_baseline(args, df, "blastp", "BLASTp_nearest_host"))
        rows.extend(run_blast_family_baseline(args, df, "psiblast", "PSI-BLAST_topk_host_vote"))
        rows.extend(run_hmmer_baseline(args, df))
        rows.extend(run_foldseek_baseline(args, df))
    path = Path(args.out_root) / "external_baseline_status.csv"
    write_csv(path, rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "test_macro_f1", "weighted_f1", "balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    return {"status": "complete", "path": str(path)}


def smoke_test(args: argparse.Namespace, static: Mapping[str, Any], dataset: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    df = load_split_df(args)
    checks.append({"check": "dataset_non_empty", "status": "pass" if len(df) > 0 else "fail", "rows": int(len(df))})
    checks.append({"check": "split_audit", "status": "pass" if split.get("status") == "pass" else "fail", "leakage_error_count": split.get("leakage_error_count", "")})
    missing = list(static.get("mandatory_missing_tools", []))
    checks.append({"check": "mandatory_tools", "status": "pass" if not missing else "fail", "missing": missing})

    smoke_root = Path(args.out_root) / "smoke"
    train = df[df["split"] == "train"].head(max(10, min(50, len(df)))).to_dict("records")
    test = df[df["split"] != "train"].head(max(5, min(20, len(df)))).to_dict("records")
    if train and test and not missing:
        train_faa = smoke_root / "train.faa"
        test_faa = smoke_root / "test.faa"
        db = smoke_root / "blast_db"
        write_fasta(train_faa, train)
        write_fasta(test_faa, test)
        makeblastdb = shutil.which("makeblastdb", path=command_env(args)["PATH"])
        blastp = shutil.which("blastp", path=command_env(args)["PATH"])
        hmmbuild = shutil.which("hmmbuild", path=command_env(args)["PATH"])
        hmmsearch = shutil.which("hmmsearch", path=command_env(args)["PATH"])
        foldseek = shutil.which("foldseek", path=command_env(args)["PATH"])
        blast_ok = False
        if makeblastdb and blastp:
            run_logged([makeblastdb, "-in", str(train_faa), "-dbtype", "prot", "-out", str(db)], args, timeout=120)
            result = run_logged([blastp, "-query", str(test_faa), "-db", str(db), "-out", str(smoke_root / "blast.tsv"), "-outfmt", "6 qseqid sseqid pident bitscore", "-max_target_seqs", "1"], args, timeout=120)
            blast_ok = result["returncode"] == 0 and (smoke_root / "blast.tsv").exists()
        checks.append({"check": "blastp", "status": "pass" if blast_ok else "fail"})
        hmmer_ok = False
        if hmmbuild and hmmsearch and train:
            one = [train[0]]
            aln = smoke_root / "one.faa"
            hmm = smoke_root / "one.hmm"
            tbl = smoke_root / "hmmer.tblout"
            write_fasta(aln, one)
            build = run_logged([hmmbuild, "--amino", str(hmm), str(aln)], args, timeout=120)
            search = run_logged([hmmsearch, "--tblout", str(tbl), str(hmm), str(test_faa)], args, timeout=120) if build["returncode"] == 0 else {"returncode": 1}
            hmmer_ok = build["returncode"] == 0 and search["returncode"] in {0, 1} and tbl.exists()
        checks.append({"check": "hmmer", "status": "pass" if hmmer_ok else "fail"})
        fold_rows = [row for row in df.to_dict("records") if str(row.get("structure_path", "")) and Path(str(row.get("structure_path", ""))).exists()]
        foldseek_ok = False
        if foldseek and len(fold_rows) >= 2:
            qdir = smoke_root / "foldseek_query"
            tdir = smoke_root / "foldseek_target"
            shutil.rmtree(qdir, ignore_errors=True)
            shutil.rmtree(tdir, ignore_errors=True)
            qdir.mkdir(parents=True, exist_ok=True)
            tdir.mkdir(parents=True, exist_ok=True)
            os.symlink(Path(str(fold_rows[0]["structure_path"])), qdir / Path(str(fold_rows[0]["structure_path"])).name)
            os.symlink(Path(str(fold_rows[1]["structure_path"])), tdir / Path(str(fold_rows[1]["structure_path"])).name)
            result = run_logged([foldseek, "easy-search", str(qdir), str(tdir), str(smoke_root / "foldseek.tsv"), str(smoke_root / "foldseek_tmp"), "--format-output", "query,target,evalue,bits"], args, timeout=180)
            foldseek_ok = result["returncode"] == 0 and (smoke_root / "foldseek.tsv").exists()
        checks.append({"check": "foldseek", "status": "pass" if foldseek_ok else "fail", "available_structures": len(fold_rows)})

    saprot_path = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_relaxed_r3.csv"
    emb_ok = False
    if saprot_path.exists():
        try:
            emb = load_embedding_frame(saprot_path)
            emb_ok = not emb.empty
        except Exception:
            emb_ok = False
    checks.append({"check": "saprot_embedding_load", "status": "pass" if emb_ok else "fail", "path": str(saprot_path)})

    classifier_ok = False
    if len(df) >= 10 and df["host_genus"].nunique() >= 2:
        try:
            sample = df.groupby("host_genus", group_keys=False).head(5)
            x = composition_features(sample["sequence"].astype(str))
            enc, y = labels_for(sample)
            clf = LogisticRegression(max_iter=500)
            clf.fit(x, y)
            classifier_ok = len(clf.predict(x)) == len(sample)
        except Exception:
            classifier_ok = False
    checks.append({"check": "classifier_execution", "status": "pass" if classifier_ok else "fail"})
    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    payload = {
        "created_at": now_utc(),
        "status": status,
        "dataset_rbp_count": dataset.get("rbp_count", 0),
        "checks": checks,
    }
    write_json(Path(args.out_root) / "smoke_test_report.json", payload)
    if status != "pass" and args.formal:
        raise RuntimeError(f"smoke test failed: {checks}")
    return payload


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_root)
    dataset = read_json(out / "dataset_audit.json")
    split = read_json(out / "split_audit.json")
    static = read_json(out / "static_audit.json")
    install_report = read_json(out / "tool_install_report.json")
    baseline_rows = list(csv.DictReader((out / "baseline_results.csv").open())) if (out / "baseline_results.csv").exists() else []
    plm_rows = list(csv.DictReader((out / "plm_results.csv").open())) if (out / "plm_results.csv").exists() else []
    external_rows = list(csv.DictReader((out / "external_baseline_status.csv").open())) if (out / "external_baseline_status.csv").exists() else []

    def best(rows: Sequence[Mapping[str, str]], models: Iterable[str] | None = None) -> dict[str, Any]:
        chosen = []
        model_set = set(models or [])
        for row in rows:
            if model_set and row.get("model") not in model_set:
                continue
            try:
                val = float(row.get("test_macro_f1", "nan"))
            except ValueError:
                continue
            if math.isfinite(val):
                chosen.append((val, row))
        return dict(max(chosen, key=lambda item: item[0])[1]) if chosen else {}

    best_simple = best(baseline_rows)
    saprot = best(plm_rows, ["SaProt", "SaProt_training_csv"])
    best_seq_plm = best(plm_rows, ["ProtT5"])
    foldseek_rows = [r for r in external_rows if "Foldseek" in r.get("model", "")]
    foldseek_available = foldseek_rows and foldseek_rows[0].get("status") not in {"NOT_AVAILABLE", ""}
    mandatory_missing = static.get("mandatory_missing_tools", [])
    missing_evidence = []
    if dataset.get("status") != "valid":
        missing_evidence.append("PHIStruct dataset universe is not available/frozen.")
    if split.get("status") != "pass":
        missing_evidence.append("Strict split audit did not pass.")
    if mandatory_missing:
        missing_evidence.append(f"Missing mandatory tools: {', '.join(mandatory_missing)}.")
    if not foldseek_available:
        missing_evidence.append("Foldseek structure-only baseline has not produced validation/test metrics.")
    if not saprot:
        missing_evidence.append("SaProt published embedding/head metrics are not available.")
    final_status = "INSUFFICIENT_EVIDENCE"
    if not missing_evidence and saprot and best_simple:
        saprot_f1 = float(saprot["test_macro_f1"])
        simple_f1 = float(best_simple["test_macro_f1"])
        if saprot_f1 <= simple_f1:
            final_status = "SIMPLE_BASELINE_SATURATED"
        else:
            final_status = "INSUFFICIENT_EVIDENCE"
    payload = {
        "created_at": now_utc(),
        "status": final_status,
        "formal": bool(args.formal),
        "dataset": dataset,
        "split": split,
        "static_audit": static,
        "tool_install_report": install_report,
        "best_simple_baseline": best_simple,
        "best_homology_baseline": {},
        "foldseek": foldseek_rows[0] if foldseek_rows else {},
        "best_sequence_plm": best_seq_plm,
        "saprot": saprot,
        "saprot_excess_over_strongest_non_foundation": "",
        "saprot_excess_over_foldseek": "",
        "bootstrap_summary": {"status": "NOT_AVAILABLE", "reason": "requires frozen comparable prediction files"},
        "multi_seed_stability": {"status": "NOT_AVAILABLE" if not saprot else "partial_from_embedding_heads"},
        "missing_evidence": missing_evidence,
        "next_action": "resolve missing evidence" if final_status == "INSUFFICIENT_EVIDENCE" else "stop PHIStruct and move to EvoMIL",
    }
    write_json(summary_json_path(args), payload)
    lines = [
        "# PHIStruct / SaProt Qualification Summary",
        "",
        f"- Status: `{final_status}`",
        f"- Dataset: `{dataset.get('rbp_count', 0)}` RBPs, `{dataset.get('phage_count', 0)}` phages, `{dataset.get('host_genus_count', 0)}` host genera",
        f"- Split: `{split.get('sample_counts', {})}`",
        f"- Split audit: `{split.get('status', 'missing')}`",
        f"- Best simple baseline: `{best_simple.get('model', 'NA')}` / `{best_simple.get('test_macro_f1', 'NA')}`",
        f"- Foldseek: `{(foldseek_rows[0] if foldseek_rows else {}).get('test_macro_f1', 'NA')}`",
        f"- Best sequence PLM: `{best_seq_plm.get('model', 'NA')}` / `{best_seq_plm.get('test_macro_f1', 'NA')}`",
        f"- SaProt: `{saprot.get('model', 'NA')}` / `{saprot.get('test_macro_f1', 'NA')}`",
        "",
        "## Missing Evidence",
        "",
        *(f"- {item}" for item in missing_evidence),
    ]
    summary_md_path(args).write_text("\n".join(lines) + "\n")
    return payload


def write_run_metadata(args: argparse.Namespace) -> None:
    write_metadata(
        Path(args.out_root) / "experiment_registry_metadata.json",
        build_run_metadata(
            args=args,
            source_checkpoint="PHIStruct/SaProt published assets",
            data_paths=[
                Path(args.out_root) / "phistruct_external_assets.json",
                Path(args.out_root) / "dataset_manifest.csv",
                Path(args.out_root) / "split_manifest.csv",
            ],
            extra={
                "phase": "phistruct_saprot_qualification",
                "experiment_id": args.experiment_id,
                "official_repo": OFFICIAL_REPO,
                "official_repo_commit": OFFICIAL_REPO_COMMIT,
                "status_path": str(status_path(args)),
                "summary_report_json": str(summary_json_path(args)),
            },
        ),
    )


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    registry = load_registry(args)
    update_status(args, "running", "workflow_start")
    write_run_metadata(args)
    static = run_stage(args, registry, "static_audit", "static_audit", Path(args.out_root) / "static_audit.json", lambda: static_audit(args), validator=json_ok)
    if static is None:
        static = read_json(Path(args.out_root) / "static_audit.json")
    install_report = run_stage(args, registry, "tool_install", "tool_install", Path(args.out_root) / "tool_install_report.json", lambda: install_missing_tools(args, static), validator=json_ok, skippable=False)
    static = static_audit(args)
    run_stage(args, registry, "asset_manifest", "asset_manifest", Path(args.out_root) / "phistruct_external_assets.json", lambda: asset_manifest(args), validator=json_ok)
    run_stage(args, registry, "asset_acquisition", "asset_acquisition", Path(args.out_root) / "asset_acquisition_report.json", lambda: acquire_assets(args), validator=json_ok, skippable=False)
    dataset = run_stage(args, registry, "dataset_manifest", "dataset_manifest", Path(args.out_root) / "dataset_manifest.csv", lambda: build_dataset_manifest(args), validator=None, skippable=False)
    if dataset is None:
        dataset = read_json(Path(args.out_root) / "dataset_audit.json")
    if dataset.get("status") == "valid" and not static.get("mandatory_missing_tools"):
        split = run_stage(args, registry, "strict_split", "strict_split", Path(args.out_root) / "split_manifest.csv", lambda: build_split(args), validator=csv_has_rows, skippable=False)
        if split is None:
            split = read_json(Path(args.out_root) / "split_audit.json")
        if split.get("status") == "pass":
            run_stage(args, registry, "smoke_test", "smoke_test", Path(args.out_root) / "smoke_test_report.json", lambda: smoke_test(args, static, dataset, split), validator=json_ok, skippable=False)
            run_stage(args, registry, "sequence_baselines", "sequence_baselines", Path(args.out_root) / "baseline_results.csv", lambda: run_sequence_baselines(args), validator=csv_has_rows, skippable=False)
            run_stage(args, registry, "embedding_heads", "embedding_heads", Path(args.out_root) / "plm_results.csv", lambda: run_embedding_heads(args), validator=csv_has_rows, skippable=False)
            run_stage(args, registry, "external_baseline_status", "external_baseline_status", Path(args.out_root) / "external_baseline_status.csv", lambda: run_external_baselines(args, static), validator=csv_has_rows, skippable=False)
        else:
            update_task(args, registry, "formal_evaluation", "formal_evaluation", "blocked", reason="leakage_audit_failed")
    else:
        reason = "dataset_not_available" if dataset.get("status") != "valid" else f"mandatory_tools_missing:{static.get('mandatory_missing_tools', [])}"
        update_task(args, registry, "strict_split", "strict_split", "blocked", reason=reason)
        run_stage(args, registry, "external_baseline_status", "external_baseline_status", Path(args.out_root) / "external_baseline_status.csv", lambda: run_external_baselines(args, static), validator=csv_has_rows, skippable=False)
    summary = run_stage(args, registry, "summary", "summary", summary_json_path(args), lambda: summarize(args), validator=json_ok, skippable=False)
    update_status(args, "complete" if summary and summary.get("status") in FINAL_STATUSES else "failed", "workflow_end", final_status=(summary or {}).get("status", ""))
    return summary or read_json(summary_json_path(args))


def launch(args: argparse.Namespace) -> int:
    session = args.screen_name or f"phistruct_qualification_{compact_time()}"
    log_path = Path(args.screen_log or PROJECT_ROOT / "logs" / f"{session}.screen.log")
    if not shutil.which("screen"):
        raise RuntimeError("screen is not installed")
    existing = subprocess.check_output(["screen", "-ls"], text=True, stderr=subprocess.STDOUT)
    if f".{session}" in existing:
        raise RuntimeError(f"screen session already exists: {session}")
    cmd = [
        args.python,
        "-u",
        "phase2/phistruct_qualification.py",
        "--execute",
        "--formal" if args.formal else "--smoke",
        "--out-root",
        str(args.out_root),
        "--log-file",
        str(args.log_file),
        "--experiment-id",
        args.experiment_id,
        "--seeds",
        args.seeds,
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--n-jobs",
        str(args.n_jobs),
        "--stop-on-low-disk-gb",
        str(args.stop_on_low_disk_gb),
        "--download-retries",
        str(args.download_retries),
        "--git-clone-timeout-sec",
        str(args.git_clone_timeout_sec),
        "--download-timeout-sec",
        str(args.download_timeout_sec),
        "--tool-root",
        str(args.tool_root),
        "--tool-download-timeout-sec",
        str(args.tool_download_timeout_sec),
        "--conda-install-timeout-sec",
        str(args.conda_install_timeout_sec),
        "--dependency-retries",
        str(args.dependency_retries),
        "--dependency-retry-delay-sec",
        str(args.dependency_retry_delay_sec),
        "--blast-timeout-sec",
        str(args.blast_timeout_sec),
        "--hmmer-timeout-sec",
        str(args.hmmer_timeout_sec),
        "--foldseek-timeout-sec",
        str(args.foldseek_timeout_sec),
        "--archive-extract-timeout-sec",
        str(args.archive_extract_timeout_sec),
    ]
    if args.auto_download_assets:
        cmd.append("--auto-download-assets")
    if args.install_missing_tools:
        cmd.append("--install-missing-tools")
    if args.resume:
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    shell_cmd = " ".join(subprocess.list2cmdline([item]) for item in cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["screen", "-L", "-Logfile", str(log_path), "-dmS", session, "bash", "-lc", f"cd {subprocess.list2cmdline([str(PROJECT_ROOT)])} && exec {shell_cmd}"])
    payload = {
        "created_at": now_utc(),
        "status": "launched",
        "screen_session": session,
        "screen_log": str(log_path),
        "command": cmd,
        "out_root": str(Path(args.out_root)),
        "status_path": str(status_path(args)),
        "registry_path": str(registry_path(args)),
    }
    write_json(Path(args.out_root) / "launch_report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch-screen", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--formal", action="store_true", default=True)
    parser.add_argument("--smoke", dest="formal", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--auto-download-assets", action="store_true")
    parser.add_argument("--install-missing-tools", action="store_true")
    parser.add_argument("--conda-env", default="UT-p1")
    parser.add_argument("--python", default=DEFAULT_PYTHON if Path(DEFAULT_PYTHON).exists() else sys.executable)
    parser.add_argument("--tool-root", default=str(DEFAULT_TOOL_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--screen-name", default="")
    parser.add_argument("--screen-log", default="")
    parser.add_argument("--experiment-id", default=f"phistruct_saprot_qualification_{compact_time()}")
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--download-timeout-sec", type=int, default=7200)
    parser.add_argument("--tool-download-timeout-sec", type=int, default=7200)
    parser.add_argument("--conda-install-timeout-sec", type=int, default=7200)
    parser.add_argument("--dependency-retries", type=int, default=3)
    parser.add_argument("--dependency-retry-delay-sec", type=int, default=120)
    parser.add_argument("--git-clone-timeout-sec", type=int, default=180)
    parser.add_argument("--blast-timeout-sec", type=int, default=7200)
    parser.add_argument("--hmmer-timeout-sec", type=int, default=3600)
    parser.add_argument("--foldseek-timeout-sec", type=int, default=7200)
    parser.add_argument("--archive-extract-timeout-sec", type=int, default=7200)
    parser.add_argument("--cluster-max-targets", type=int, default=1000)
    parser.add_argument("--cluster-min-qcov", type=float, default=70.0)
    parser.add_argument("--retrieval-max-targets", type=int, default=10)
    parser.add_argument("--psiblast-iterations", type=int, default=2)
    parser.add_argument("--stop-on-low-disk-gb", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.launch_screen:
        raise SystemExit(launch(args))
    if not args.execute:
        print("Nothing to do. Use --launch-screen or --execute.")
        return
    summary = run_workflow(args)
    print(json.dumps({"summary_status": summary.get("status"), "summary_report": str(summary_json_path(args))}, indent=2))


if __name__ == "__main__":
    main()
