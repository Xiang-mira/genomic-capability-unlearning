"""ProteinGym ESM2-150M minimum viable benchmark qualification workflow.

This controller is intentionally isolated from the historical HVUE/Evo runs. It
reuses the repository's Phase 2 metadata/checkpoint conventions while adding a
ProteinGym-specific data path, deterministic split builder, strong mutation
feature baselines, cached ESM2 pilot scoring, and a strictly sequential LoRA
qualification stage.
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
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, file_sha256, stable_hash, write_metadata
from phase1.build_refseq_family_target_dataset import download_file as download_url_file

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - only used on non-torch import hosts.
    torch = None
    nn = None

if torch is not None:
    from phase2.checkpoint_io import atomic_save_safetensors
    from phase2.lora_utils import LoRALinear, count_total, count_trainable, freeze_all


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/protein_48h_esm2_qualification"
DEFAULT_SMOKE_ROOT = PROJECT_ROOT / "data/phase2/protein_48h_esm2_qualification_smoke"
DEFAULT_DMS_DIR = PROJECT_ROOT / "data/proteingym/DMS_substitutions"
DEFAULT_METADATA = PROJECT_ROOT / "data/proteingym/DMS_substitutions.csv"
DEFAULT_PUBLIC_PREDICTIONS = PROJECT_ROOT / "data/proteingym/public_predictions"
DEFAULT_MSA_DIR = PROJECT_ROOT / "data/proteingym/MSA_files"
DEFAULT_ESM2_MODEL = "facebook/esm2_t30_150M_UR50D"
DEFAULT_LOG = PROJECT_ROOT / "logs/protein_48h_esm2_qualification.log"
PROTEINGYM_ZENODO_RECORD_ID = "15293562"
PROTEINGYM_ZENODO_API = f"https://zenodo.org/api/records/{PROTEINGYM_ZENODO_RECORD_ID}"
REQUIRED_PROTEINGYM_FILES = {
    "DMS_ProteinGym_substitutions.zip",
    "DMS_substitutions.csv",
}
PROTEINGYM_PUBLIC_SCORE_ARCHIVE = "zero_shot_substitutions_scores.zip"
EVOLUTIONARY_BASELINE_COLUMNS = (
    "EVmutation",
    "DeepSequence_single",
    "DeepSequence_ensemble",
    "EVE_single",
    "EVE_ensemble",
    "GEMME",
    "VESPA",
    "VESPAl",
    "VespaG",
    "RSALOR",
    "S2F_MSA",
)
ESM2_REPRESENTATION_TYPES = (
    "mutation_position",
    "wt_mutant_position_difference",
    "local_window",
    "whole_sequence_mean",
)
ESM2_READOUTS = ("linear_regression", "ridge_regression")
RESULT_LABELS = {
    "SATURATED_BY_SIMPLE_BASELINE",
    "SATURATED_BY_EVOLUTIONARY_BASELINE",
    "RANDOM_SPLIT_ONLY",
    "INVALID_POSITION_HELD_OUT_SPLIT",
    "DATA_LEAKAGE_DETECTED",
    "FROZEN_PLM_SIGNAL",
    "LORA_RECOVERABLE_SIGNAL",
    "PRELIMINARILY_QUALIFIED",
    "INSUFFICIENT_EVIDENCE",
    "NUMERICAL_FAILURE",
    "NOT_AVAILABLE",
}
STATES = {"pending", "running", "complete", "valid", "invalid", "failed", "skipped"}

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
VOLUME = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}
CHARGE = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}
BLOSUM62_ROWS = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, 0, -3, -2, 0, -2, -1, 0, -4],
    "R": [0, 5, -2, -3, 1, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -1, -3, -2, -3, -1, -2, 0, -4],
    "N": [-2, 0, 6, 1, -3, -1, -1, 0, -3, 0, -2, -3, -2, 1, 0, -1, -4, -2, -3, 3, 4, -1, -4],
    "D": [-1, -2, 1, 6, -3, -2, -1, -1, -3, -1, -4, -3, -3, 0, -1, -1, -4, -3, -3, 4, 1, -1, -4],
    "C": [-2, -3, -3, -3, 9, -3, -3, -1, -1, -3, -1, -1, -1, -1, -1, -1, -2, -2, -1, -3, -3, -3, -4],
    "Q": [0, 1, 0, 0, -3, 5, 2, -2, -3, 1, 0, -3, -1, 0, -1, -2, -2, -1, -2, 0, 3, 1, -4],
    "E": [-2, 0, 0, 2, -4, 2, 5, -2, -3, 1, -2, -3, -2, 0, -1, -2, -3, -2, -2, 1, 4, 1, -4],
    "G": [0, -2, 0, -1, -3, -2, -2, 6, -4, -2, -4, -2, -3, 0, -2, -2, -2, -3, -3, -1, -2, -1, -4],
    "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -1, -2, -3, -2, -1, -2, -2, -2, 2, -3, 0, 0, -1, -4],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, 1, 3, -2, -1, 0, -3, -1, 3, -3, -3, -1, -4],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, 2, 1, -2, -1, 0, -2, -1, 1, -4, -3, -1, -4],
    "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, 0, -1, -1, -3, -2, -2, 0, 1, 1, -4],
    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, -1, -1, 0, -1, -1, 1, -3, -1, -1, -4],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -2, -2, 1, 3, -1, -3, -3, -1, -4],
    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -4, -3, -2, -2, -1, -2, -4],
    "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, -3, -2, -2, 0, 0, 0, -4],
    "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, 0, -1, -1, 0, -4],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -4, -3, -2, -4],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -3, -2, -1, -4],
    "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, -3, -2, -1, -4],
}
BLOSUM_ORDER = list("ARNDCQEGHILKMFPSTWYVBZX")

INVENTORY_FIELDS = [
    "assay_id",
    "assay_name",
    "protein_name",
    "phenotype_type",
    "dms_path",
    "wild_type_sequence",
    "sequence_length",
    "total_sample_count",
    "valid_single_substitution_count",
    "valid_single_substitution_proportion",
    "score_mean",
    "score_std",
    "score_q05",
    "score_q25",
    "score_q50",
    "score_q75",
    "score_q95",
    "score_dynamic_range",
    "missing_value_proportion",
    "mutation_position_coverage",
    "mutation_position_coverage_proportion",
    "duplicate_mutations_exist",
    "msa_available",
    "conservation_or_evolutionary_prediction_available",
    "public_plm_prediction_available",
    "compatible_with_esm2_length",
    "compatible_with_available_gpu_memory",
    "data_quality_score",
    "entered_static_candidate_set",
    "selected_pilot",
    "selection_reason",
    "exclusion_reason",
]
BASELINE_FIELDS = [
    "assay_id",
    "split_type",
    "baseline",
    "baseline_family",
    "status",
    "selection_split",
    "selected_alpha",
    "direction_sign",
    "n_train",
    "n_val",
    "n_test",
    "val_spearman",
    "test_spearman",
    "val_mse",
    "test_mse",
    "mse_improvement_over_global_mean",
    "prediction_path",
    "is_strongest_available_non_plm",
    "not_available_reason",
]
ESM2_FIELDS = [
    "assay_id",
    "split_type",
    "method",
    "readout",
    "status",
    "model_name",
    "model_hash",
    "layer",
    "representation_type",
    "selection_split",
    "selected_alpha",
    "n_train",
    "n_val",
    "n_test",
    "val_spearman",
    "test_spearman",
    "val_mse",
    "test_mse",
    "strongest_non_plm_baseline",
    "baseline_val_spearman",
    "baseline_test_spearman",
    "val_excess",
    "test_excess",
    "position_bootstrap_ci_low",
    "position_bootstrap_ci_high",
    "prediction_path",
    "cache_key",
    "not_available_reason",
]
LORA_FIELDS = [
    "assay_id",
    "stage",
    "run_id",
    "seed",
    "rank",
    "learning_rate",
    "status",
    "selection_split",
    "n_train",
    "n_val",
    "n_test",
    "val_spearman",
    "test_spearman",
    "val_mse",
    "test_mse",
    "strongest_non_plm_baseline",
    "baseline_val_spearman",
    "baseline_test_spearman",
    "val_excess",
    "test_excess",
    "position_bootstrap_ci_low",
    "position_bootstrap_ci_high",
    "adapter_hash",
    "head_hash",
    "checkpoint_selection_evidence",
    "prediction_path",
    "not_available_reason",
]


@dataclass
class VariantRecord:
    assay_id: str
    sample_id: str
    mutation: str
    wt: str
    position: int
    mut: str
    score: float
    mutated_sequence: str
    wt_sequence: str


@dataclass
class AssayInput:
    assay_id: str
    assay_name: str
    protein_name: str
    phenotype_type: str
    dms_path: Path
    metadata: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(fval):
        return ""
    return repr(fval)


def write_json(path: Path, payload: object, *, overwrite: bool = False) -> None:
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


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Sequence[str], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_float(row.get(field)) for field in fields})
    os.replace(tmp, path)


def append_log(args: argparse.Namespace, message: str) -> None:
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_utc()}] {message}\n"
    with log_path.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def status_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "protein_48h_status.json"


def registry_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "protein_48h_registry.json"


def load_registry(args: argparse.Namespace) -> dict[str, Any]:
    return read_json(registry_path(args)) or {
        "workflow": {
            "name": "protein_48h_esm2_qualification",
            "created_at": now_utc(),
            "out_root": str(Path(args.out_root)),
            "max_concurrent_gpu_jobs": 1,
            "formal": bool(args.formal),
            "mock_esm2": bool(args.mock_esm2),
        },
        "tasks": [],
    }


def save_registry(args: argparse.Namespace, registry: dict[str, Any]) -> None:
    registry["workflow"]["updated_at"] = now_utc()
    write_json(registry_path(args), registry, overwrite=True)


def update_task(
    args: argparse.Namespace,
    registry: dict[str, Any],
    task_id: str,
    *,
    stage: str,
    status: str,
    **extra: Any,
) -> None:
    if status not in STATES:
        raise ValueError(f"Invalid registry status: {status}")
    tasks = registry.setdefault("tasks", [])
    row = None
    for candidate in tasks:
        if candidate.get("task_id") == task_id:
            row = candidate
            break
    if row is None:
        row = {
            "task_id": task_id,
            "stage": stage,
            "assay_id": extra.pop("assay_id", ""),
            "model": extra.pop("model", ""),
            "split": extra.pop("split", ""),
            "run": extra.pop("run", ""),
            "seed": extra.pop("seed", ""),
            "pid": os.getpid(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", str(args.cuda_visible_devices)),
            "attempts": 0,
            "created_at": now_utc(),
        }
        tasks.append(row)
    row["stage"] = stage
    row["status"] = status
    row["updated_at"] = now_utc()
    if status == "running":
        row["attempts"] = int(row.get("attempts", 0)) + 1
        row["started_at"] = now_utc()
    if status in {"complete", "valid", "invalid", "failed", "skipped"}:
        row["completed_at"] = now_utc()
    row.update(extra)
    save_registry(args, registry)
    write_json(
        status_path(args),
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": stage,
            "task_id": task_id,
            "registry_path": str(registry_path(args)),
            **extra,
        },
        overwrite=True,
    )


def stage_done(path: Path, validator: Any | None = None) -> bool:
    if not path.exists():
        return False
    if validator is None:
        return True
    try:
        return bool(validator(path))
    except Exception:
        return False


def run_stage(
    args: argparse.Namespace,
    registry: dict[str, Any],
    task_id: str,
    stage: str,
    output_path: Path,
    func: Any,
    *,
    validator: Any | None = None,
    skippable: bool = True,
) -> Any:
    if args.resume and skippable and stage_done(output_path, validator):
        update_task(args, registry, task_id, stage=stage, status="skipped", reason="resume_output_valid", output_path=str(output_path))
        append_log(args, f"skip {stage}: output already valid at {output_path}")
        return None
    last_exc = None
    for attempt in range(int(args.max_retries) + 1):
        update_task(args, registry, task_id, stage=stage, status="running", output_path=str(output_path), retry_attempt=attempt)
        try:
            result = func()
            if validator is not None and not validator(output_path):
                raise RuntimeError(f"Validation failed for stage {stage}: {output_path}")
            update_task(args, registry, task_id, stage=stage, status="complete", output_path=str(output_path))
            append_log(args, f"complete {stage}: {output_path}")
            return result
        except Exception as exc:  # noqa: PERF203 - explicit retry metadata is useful here.
            last_exc = exc
            update_task(
                args,
                registry,
                task_id,
                stage=stage,
                status="failed",
                output_path=str(output_path),
                exception_type=type(exc).__name__,
                exception=str(exc),
                retry_attempt=attempt,
            )
            append_log(args, f"failed {stage} attempt={attempt}: {type(exc).__name__}: {exc}")
            if attempt >= int(args.max_retries):
                break
    raise RuntimeError(f"Stage {stage} failed after retries") from last_exc


def csv_nonempty_or_header(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return False
    return True


def json_object(path: Path) -> bool:
    payload = read_json(path)
    return isinstance(payload, dict)


def normalize_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def blosum_score(wt: str, mut: str) -> float:
    if wt not in BLOSUM62_ROWS or mut not in BLOSUM_ORDER:
        return float("nan")
    return float(BLOSUM62_ROWS[wt][BLOSUM_ORDER.index(mut)])


def charge(aa: str) -> float:
    return float(CHARGE.get(aa, 0.0))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spearman_metric(y_true: np.ndarray, pred: np.ndarray) -> float:
    if y_true.size < 2 or pred.size < 2:
        return float("nan")
    if np.nanstd(y_true) == 0 or np.nanstd(pred) == 0:
        return float("nan")
    value = spearmanr(y_true, pred, nan_policy="omit").correlation
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def mse_metric(y_true: np.ndarray, pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(mean_squared_error(y_true, pred))


def direction_sign_from_validation(y_val: np.ndarray, pred_val: np.ndarray) -> int:
    corr = spearman_metric(y_val, pred_val)
    if math.isfinite(corr) and corr < 0:
        return -1
    return 1


def apply_direction(pred: np.ndarray, sign: int) -> np.ndarray:
    return pred * float(sign)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = safe_read_csv(path)
    result: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        payload = {str(key): (None if pd.isna(value) else value) for key, value in row.items()}
        keys = []
        for column in ("DMS_id", "DMS_filename", "assay_id", "target_name", "protein_name", "UniProt_ID"):
            value = payload.get(column)
            if value is not None and str(value).strip():
                keys.append(normalize_id(str(value)))
                keys.append(normalize_id(Path(str(value)).stem))
        for key in set(keys):
            result[key] = payload
    return result


def find_dms_files(dms_dir: Path) -> list[Path]:
    if dms_dir.is_file():
        return [dms_dir]
    if not dms_dir.exists():
        return []
    return sorted(path for path in dms_dir.rglob("*.csv") if path.is_file())


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only to verify upstream Zenodo md5 checksums.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum(value: str | None) -> tuple[str, str] | None:
    if not value or ":" not in value:
        return None
    algo, digest = value.split(":", 1)
    return algo.lower(), digest.strip().lower()


def checksum_ok(path: Path, expected: str | None) -> bool:
    parsed = parse_checksum(expected)
    if parsed is None:
        return True
    algo, digest = parsed
    if algo == "md5":
        return md5_file(path) == digest
    if algo == "sha256":
        return file_sha256(path) == digest
    return True


def looks_like_html_error(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        head = handle.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title>404" in head[:256]


def fetch_proteingym_record(api_url: str, retries: int = 3) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            request = urllib.request.Request(api_url, headers={"User-Agent": "UT-project1-ProteinGym-ESM2-qualification"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: PERF203 - authoritative metadata fetch should tolerate transient TLS failures.
            last_exc = exc
            if attempt + 1 < max(1, retries):
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch ProteinGym record metadata from {api_url}") from last_exc


def zenodo_file_map(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("key")): dict(item) for item in record.get("files", [])}


def validate_metadata_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "invalid", "reason": "metadata_csv_missing"}
    if looks_like_html_error(path):
        return {"status": "invalid", "reason": "metadata_csv_empty_or_html_error"}
    try:
        df = pd.read_csv(path, nrows=10)
    except Exception as exc:
        return {"status": "invalid", "reason": f"metadata_csv_malformed:{type(exc).__name__}:{exc}"}
    required_any = any(column in df.columns for column in ("DMS_id", "DMS_filename", "assay_id", "target_name"))
    has_sequence = any(column in df.columns for column in ("target_seq", "wildtype_sequence", "wild_type_sequence", "sequence", "WT_sequence"))
    if not required_any:
        return {"status": "invalid", "reason": "metadata_csv_missing_assay_identifier_columns", "columns": list(df.columns)}
    return {
        "status": "valid",
        "rows_sampled": int(len(df)),
        "columns": list(df.columns),
        "has_wild_type_sequence_column": bool(has_sequence),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_one_assay_for_download(assay: AssayInput) -> dict[str, Any]:
    if looks_like_html_error(assay.dms_path):
        return {
            "assay_id": assay.assay_id,
            "path": str(assay.dms_path),
            "status": "invalid",
            "reason": "empty_or_html_error_file",
            "valid_single_substitution_count": 0,
        }
    try:
        rows, meta = read_assay_records(assay)
    except Exception as exc:
        return {
            "assay_id": assay.assay_id,
            "path": str(assay.dms_path),
            "status": "invalid",
            "reason": f"parser_error:{type(exc).__name__}:{exc}",
            "valid_single_substitution_count": 0,
        }
    reason = ""
    if not rows:
        reason = meta.get("parse_error") or "no_valid_single_substitution_records"
    wt_sequence = meta.get("wt_sequence") or (rows[0].wt_sequence if rows else metadata_sequence(assay.metadata))
    return {
        "assay_id": assay.assay_id,
        "path": str(assay.dms_path),
        "status": "valid" if rows and wt_sequence else "invalid",
        "reason": "" if rows and wt_sequence else reason or "wild_type_sequence_unresolved",
        "total_sample_count": int(meta.get("total_sample_count", 0)),
        "valid_single_substitution_count": len(rows),
        "wt_sequence_length": len(wt_sequence),
        "mutation_column": meta.get("mutation_column", ""),
        "score_column": meta.get("score_column", ""),
        "sequence_column": meta.get("sequence_column", ""),
        "sha256": file_sha256(assay.dms_path) if assay.dms_path.exists() else "",
        "size_bytes": assay.dms_path.stat().st_size if assay.dms_path.exists() else 0,
    }


def validate_proteingym_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dms_dir = Path(args.dms_dir)
    metadata_path = Path(args.metadata_csv)
    metadata_status = validate_metadata_csv(metadata_path)
    assays = build_assay_inputs(args)
    assay_rows = [validate_one_assay_for_download(assay) for assay in assays]
    valid_rows = [row for row in assay_rows if row["status"] == "valid"]
    total_valid = sum(int(row.get("valid_single_substitution_count", 0)) for row in valid_rows)
    invalid_rows = [row for row in assay_rows if row["status"] != "valid"]
    status = (
        "valid"
        if metadata_status.get("status") == "valid" and valid_rows and total_valid > 0
        else "invalid"
    )
    reasons = []
    if metadata_status.get("status") != "valid":
        reasons.append(str(metadata_status.get("reason", "metadata_invalid")))
    if not find_dms_files(dms_dir):
        reasons.append("no_dms_csv_files_found")
    if not valid_rows:
        reasons.append("no_parser_valid_single_substitution_assays")
    return {
        "status": status,
        "reason": ";".join(reasons),
        "dms_dir": str(dms_dir),
        "metadata_csv": str(metadata_path),
        "metadata": metadata_status,
        "discovered_assays": len(assay_rows),
        "valid_assays": len(valid_rows),
        "invalid_assays": len(invalid_rows),
        "valid_single_substitution_records": int(total_valid),
        "assays_sample": assay_rows[:25],
        "invalid_assays_sample": invalid_rows[:25],
        "dataset_fingerprint": stable_hash(
            {
                "metadata_sha256": metadata_status.get("sha256", ""),
                "assays": [
                    (row.get("assay_id"), row.get("sha256"), row.get("valid_single_substitution_count"))
                    for row in valid_rows
                ],
            }
        )
        if valid_rows
        else "invalid",
    }


def write_download_report_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# ProteinGym Download And Validation Report",
        "",
        f"- Validation status: `{report.get('validation_status')}`.",
        f"- Official source: `{report.get('official_source')}`.",
        f"- Dataset version/revision: `{report.get('dataset_version')}` / `{report.get('dataset_revision')}`.",
        f"- Discovered assays: `{report.get('discovered_assays')}`.",
        f"- Valid single-substitution records: `{report.get('valid_single_substitution_records')}`.",
        f"- Reused files: `{len(report.get('reused_files', []))}`.",
        f"- Downloaded files: `{len(report.get('downloaded_files', []))}`.",
        f"- Retained files: `{len(report.get('retained_files', []))}`.",
        f"- Removed temporary files: `{len(report.get('removed_temporary_files', []))}`.",
        "",
        "## Optional Resources",
        "",
    ]
    for row in report.get("optional_resources", []):
        lines.append(f"- `{row.get('key')}`: `{row.get('status')}` - {row.get('reason')}")
    path.write_text("\n".join(lines) + "\n")


def download_report_valid(path: Path) -> bool:
    payload = read_json(path)
    return payload.get("validation_status") == "valid"


def download_report_files_intact(path: Path, args: argparse.Namespace) -> bool:
    payload = read_json(path)
    if payload.get("validation_status") != "valid":
        return False
    if not payload.get("dataset_fingerprint") or payload.get("dataset_fingerprint") == "invalid":
        return False
    dms_files = find_dms_files(Path(args.dms_dir))
    if not dms_files:
        return False
    discovered = payload.get("discovered_assays")
    if discovered not in ("", None) and int(discovered) != len(dms_files):
        return False
    retained_rows = payload.get("retained_files", [])
    retained_by_path = {}
    for row in retained_rows:
        row_path = row.get("path")
        if not row_path:
            continue
        retained_by_path[str(Path(row_path).resolve())] = row
    required_paths = [Path(args.metadata_csv), *dms_files]
    for required_path in required_paths:
        if not required_path.exists() or looks_like_html_error(required_path):
            return False
        row = retained_by_path.get(str(required_path.resolve()))
        if not row:
            return False
        expected_size = row.get("size_bytes")
        if expected_size not in ("", None) and int(expected_size) != required_path.stat().st_size:
            return False
        expected_sha = row.get("sha256")
        if expected_sha and file_sha256(required_path) != expected_sha:
            return False
    return True


def invalid_data_quarantine_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "proteingym_invalid_inputs"


def quarantine_invalid_inputs(args: argparse.Namespace, reason: str) -> list[dict[str, Any]]:
    moved = []
    timestamp = now_utc().replace(":", "").replace("+", "_")
    target_root = invalid_data_quarantine_root(args) / timestamp
    for source in [Path(args.dms_dir), Path(args.metadata_csv)]:
        if not source.exists():
            continue
        target = target_root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({"source": str(source), "quarantine_path": str(target), "reason": reason})
    return moved


def safe_extract_proteingym_zip(archive_path: Path, staging_root: Path) -> list[Path]:
    if looks_like_html_error(archive_path):
        raise RuntimeError(f"Downloaded archive is empty or HTML error page: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Downloaded substitutions archive is not a valid zip: {archive_path}")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive_path) as handle:
        members = [member for member in handle.infolist() if not member.is_dir()]
        if not members:
            raise RuntimeError(f"ProteinGym substitutions archive is empty: {archive_path}")
        for member in members:
            name = Path(member.filename)
            if member.file_size <= 0:
                continue
            if name.suffix.lower() != ".csv":
                continue
            target = staging_root / name.name
            with handle.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if looks_like_html_error(target):
                raise RuntimeError(f"Extracted ProteinGym CSV looks like an HTML error page: {target}")
            extracted.append(target)
    if not extracted:
        raise RuntimeError(f"No CSV files were extracted from ProteinGym substitutions archive: {archive_path}")
    return extracted


def replace_canonical_dms_dir(args: argparse.Namespace, staging_csvs: Sequence[Path]) -> None:
    dms_dir = Path(args.dms_dir)
    parent = dms_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent / f"{dms_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for source in staging_csvs:
        shutil.copy2(source, tmp_dir / source.name)
    backup = None
    if dms_dir.exists():
        backup = parent / f"{dms_dir.name}.replace_backup.{os.getpid()}"
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(dms_dir, backup)
    os.replace(tmp_dir, dms_dir)
    if backup and backup.exists():
        shutil.rmtree(backup)


def download_required_proteingym_files(args: argparse.Namespace, record: Mapping[str, Any]) -> dict[str, Any]:
    files = zenodo_file_map(record)
    missing = REQUIRED_PROTEINGYM_FILES - set(files)
    if missing:
        raise RuntimeError(f"Zenodo record is missing required ProteinGym files: {sorted(missing)}")
    cache_dir = Path(args.dms_dir).parent / "download_cache"
    staging_dir = Path(args.dms_dir).parent / "extract_staging"
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    removed = []
    retained = []

    archive_info = files["DMS_ProteinGym_substitutions.zip"]
    metadata_info = files["DMS_substitutions.csv"]
    archive_path = cache_dir / "DMS_ProteinGym_substitutions.zip"
    metadata_tmp = cache_dir / "DMS_substitutions.csv"

    for info, path in [(archive_info, archive_path), (metadata_info, metadata_tmp)]:
        if path.exists() and checksum_ok(path, info.get("checksum")) and not looks_like_html_error(path):
            downloaded.append({"path": str(path), "source": info.get("links", {}).get("self"), "status": "reused_cache", "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
            continue
        if path.exists():
            path.unlink()
        download_url_file(info["links"]["self"], str(path), desc=f"ProteinGym {info['key']}", retries=args.download_retries)
        if looks_like_html_error(path):
            raise RuntimeError(f"Downloaded ProteinGym file is empty or HTML error page: {path}")
        if not checksum_ok(path, info.get("checksum")):
            raise RuntimeError(f"Checksum mismatch for downloaded ProteinGym file: {path}")
        downloaded.append({"path": str(path), "source": info.get("links", {}).get("self"), "status": "downloaded", "sha256": file_sha256(path), "size_bytes": path.stat().st_size})

    staging_csvs = safe_extract_proteingym_zip(archive_path, staging_dir)
    replace_canonical_dms_dir(args, staging_csvs)
    metadata_path = Path(args.metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_metadata = metadata_path.with_suffix(metadata_path.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(metadata_tmp, tmp_metadata)
    os.replace(tmp_metadata, metadata_path)
    retained.append({"path": str(metadata_path), "source_key": "DMS_substitutions.csv", "sha256": file_sha256(metadata_path), "size_bytes": metadata_path.stat().st_size})
    retained.extend(
        {
            "path": str(path),
            "source_key": "DMS_ProteinGym_substitutions.zip",
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in find_dms_files(Path(args.dms_dir))
    )
    if archive_path.exists() and args.remove_validated_archives:
        archive_hash = file_sha256(archive_path)
        archive_size = archive_path.stat().st_size
        archive_path.unlink()
        removed.append({"path": str(archive_path), "sha256": archive_hash, "size_bytes": archive_size, "reason": "validated_extracted_archive_removed"})
    if staging_dir.exists():
        staging_hash = stable_hash([(str(path.relative_to(staging_dir)), file_sha256(path)) for path in sorted(staging_dir.rglob("*")) if path.is_file()])
        shutil.rmtree(staging_dir)
        removed.append({"path": str(staging_dir), "sha256": staging_hash, "reason": "validated_extraction_staging_removed"})
    return {"downloaded_files": downloaded, "retained_files": retained, "removed_temporary_files": removed}


def optional_proteingym_resources(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = zenodo_file_map(record)
    result = []
    for key in ("zero_shot_substitutions_scores.zip", "DMS_msa_files.zip"):
        info = files.get(key, {})
        result.append(
            {
                "key": key,
                "size_bytes": info.get("size", ""),
                "checksum": info.get("checksum", ""),
                "status": "deferred",
                "reason": (
                    "not downloaded during initial ingestion; current workflow only requires per-assay substitution CSVs and metadata before prescreening. "
                    "Public prediction or MSA resources remain optional and should be downloaded only after selected pilot assays require them."
                ),
            }
        )
    return result


def local_path_from_url(url: str) -> Path | None:
    parsed = urlparse(str(url))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def fetch_url_range(url: str, start: int, end: int, retries: int = 3) -> bytes:
    local_path = local_path_from_url(url)
    if local_path is not None:
        with local_path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "UT-project1-ProteinGym-ESM2-qualification",
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status_code = getattr(response, "status", None)
                if start > 0 and status_code == 200:
                    raise RuntimeError("server ignored byte-range request")
                payload = response.read()
            if len(payload) == end - start + 1:
                return payload
            if start == 0 and len(payload) >= end - start + 1:
                return payload[start : end + 1]
            raise RuntimeError(f"range request returned {len(payload)} bytes, expected {end - start + 1}")
        except Exception as exc:  # noqa: PERF203 - bounded retry preserves source metadata.
            last_exc = exc
            if attempt + 1 < max(1, retries):
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed range request for {url} bytes={start}-{end}") from last_exc


def remote_zip_central_directory(url: str, archive_size: int, retries: int = 3) -> dict[str, dict[str, Any]]:
    local_path = local_path_from_url(url)
    if local_path is not None:
        with zipfile.ZipFile(local_path) as handle:
            return {
                info.filename: {
                    "filename": info.filename,
                    "compression_method": info.compress_type,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "crc": info.CRC,
                    "local_header_offset": info.header_offset,
                }
                for info in handle.infolist()
                if not info.is_dir()
            }
    tail_size = min(int(archive_size), 66000)
    tail = fetch_url_range(url, int(archive_size) - tail_size, int(archive_size) - 1, retries=retries)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise RuntimeError("ProteinGym public score archive has no ZIP end-of-central-directory record")
    eocd = tail[eocd_offset : eocd_offset + 22]
    if len(eocd) < 22:
        raise RuntimeError("ProteinGym public score archive has a truncated ZIP end-of-central-directory record")
    sig, _disk_no, _cd_disk, _disk_entries, total_entries, cd_size, cd_offset, _comment_len = struct.unpack("<4sHHHHIIH", eocd)
    if sig != b"PK\x05\x06" or total_entries <= 0 or cd_size <= 0:
        raise RuntimeError("ProteinGym public score archive central directory is malformed or empty")
    cd = fetch_url_range(url, int(cd_offset), int(cd_offset + cd_size - 1), retries=retries)
    entries: dict[str, dict[str, Any]] = {}
    cursor = 0
    while cursor + 46 <= len(cd):
        header = cd[cursor : cursor + 46]
        fields = struct.unpack("<4sHHHHHHIIIHHHHHII", header)
        if fields[0] != b"PK\x01\x02":
            break
        name_len, extra_len, comment_len = int(fields[10]), int(fields[11]), int(fields[12])
        name_start = cursor + 46
        name_end = name_start + name_len
        name = cd[name_start:name_end].decode("utf-8")
        if not name.endswith("/"):
            entries[name] = {
                "filename": name,
                "compression_method": int(fields[4]),
                "crc": int(fields[7]),
                "compressed_size": int(fields[8]),
                "uncompressed_size": int(fields[9]),
                "local_header_offset": int(fields[16]),
            }
        cursor = name_end + extra_len + comment_len
    if not entries:
        raise RuntimeError("ProteinGym public score archive central directory contained no files")
    return entries


def extract_zip_member_by_range(url: str, entry: Mapping[str, Any], target_path: Path, retries: int = 3) -> dict[str, Any]:
    local_path = local_path_from_url(url)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    if local_path is not None:
        with zipfile.ZipFile(local_path) as handle:
            with handle.open(str(entry["filename"])) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        offset = int(entry["local_header_offset"])
        local_header = fetch_url_range(url, offset, offset + 30 - 1, retries=retries)
        fields = struct.unpack("<4sHHHHHIIIHH", local_header)
        if fields[0] != b"PK\x03\x04":
            raise RuntimeError(f"ZIP local header not found for {entry['filename']}")
        name_len, extra_len = int(fields[9]), int(fields[10])
        data_start = offset + 30 + name_len + extra_len
        compressed = fetch_url_range(url, data_start, data_start + int(entry["compressed_size"]) - 1, retries=retries)
        method = int(entry["compression_method"])
        if method == 0:
            payload = compressed
        elif method == 8:
            payload = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"Unsupported ZIP compression method {method} for {entry['filename']}")
        if len(payload) != int(entry["uncompressed_size"]):
            raise RuntimeError(f"Extracted size mismatch for {entry['filename']}")
        if zlib.crc32(payload) & 0xFFFFFFFF != int(entry["crc"]):
            raise RuntimeError(f"CRC mismatch for {entry['filename']}")
        tmp.write_bytes(payload)
    if looks_like_html_error(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Extracted public score file is empty or an HTML error page: {target_path}")
    os.replace(tmp, target_path)
    return {
        "path": str(target_path),
        "member": str(entry["filename"]),
        "sha256": file_sha256(target_path),
        "size_bytes": target_path.stat().st_size,
        "compressed_size": int(entry.get("compressed_size", 0)),
        "crc32": f"{int(entry.get('crc', 0)):08x}",
    }


def validate_evolutionary_prediction_file(path: Path, assay_id: str, records: Sequence[VariantRecord]) -> dict[str, Any]:
    if not path.exists():
        return {"assay_id": assay_id, "path": str(path), "status": "missing", "reason": "file_missing"}
    if looks_like_html_error(path):
        return {"assay_id": assay_id, "path": str(path), "status": "invalid", "reason": "empty_or_html_error"}
    try:
        df = safe_read_csv(path)
    except Exception as exc:
        return {"assay_id": assay_id, "path": str(path), "status": "invalid", "reason": f"malformed_csv:{type(exc).__name__}:{exc}"}
    mutation_col = pick_column(df.columns, ["mutant", "mutation", "variant", "mutations"])
    if mutation_col is None:
        return {"assay_id": assay_id, "path": str(path), "status": "invalid", "reason": "mutation_column_missing", "columns": list(df.columns)}
    record_mutations = {record.mutation for record in records}
    matched = df[df[mutation_col].astype(str).isin(record_mutations)] if record_mutations else df.iloc[0:0]
    method_counts: dict[str, int] = {}
    for column in EVOLUTIONARY_BASELINE_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(matched[column], errors="coerce")
        count = int(np.isfinite(values.to_numpy(dtype=np.float64)).sum())
        if count:
            method_counts[column] = count
    if not method_counts or max(method_counts.values()) < 3:
        return {
            "assay_id": assay_id,
            "path": str(path),
            "status": "invalid",
            "reason": "no_matching_finite_allowed_evolutionary_score_column",
            "columns": list(df.columns),
            "matched_mutations": int(len(matched)),
            "method_counts": method_counts,
        }
    return {
        "assay_id": assay_id,
        "path": str(path),
        "status": "valid",
        "mutation_column": mutation_col,
        "rows": int(len(df)),
        "matched_mutations": int(len(matched)),
        "method_counts": method_counts,
        "allowed_columns_present": [column for column in EVOLUTIONARY_BASELINE_COLUMNS if column in df.columns],
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def public_score_target_path(args: argparse.Namespace, assay_id: str) -> Path:
    return Path(args.public_predictions_dir) / "ProteinGym_zero_shot_substitution_scores" / f"{assay_id}.csv"


def find_existing_valid_public_score(args: argparse.Namespace, assay_id: str, records: Sequence[VariantRecord]) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [public_score_target_path(args, assay_id)]
    root = Path(args.public_predictions_dir)
    if root.exists():
        assay_key = normalize_id(assay_id).lower()
        candidates.extend(path for path in root.rglob("*.csv") if assay_key in normalize_id(path.name).lower())
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        validation = validate_evolutionary_prediction_file(path, assay_id, records)
        if validation.get("status") == "valid":
            return path, validation
    return None, None


def ensure_evolutionary_baseline_resources(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_path = out_root / "protein_48h_evolutionary_baseline_report.json"
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    pilot_ids = list(split_payload.get("pilot_assay_ids", []))
    allow_remote_fetch = bool(pilot_ids and args.auto_download_proteingym and not args.skip_proteingym_download)
    record = fetch_proteingym_record(args.proteingym_source_api, retries=args.download_retries) if allow_remote_fetch else {"id": "", "doi": "", "metadata": {"version": "local"}, "files": []}
    files = zenodo_file_map(record)
    archive_info = files.get(PROTEINGYM_PUBLIC_SCORE_ARCHIVE, {})
    archive_url = archive_info.get("links", {}).get("self", "")
    archive_size = int(archive_info.get("size") or 0)
    archive_checksum = archive_info.get("checksum", "")
    reused: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    per_assay: dict[str, Any] = {}

    entries: dict[str, dict[str, Any]] = {}
    if pilot_ids and archive_url and archive_size:
        entries = remote_zip_central_directory(archive_url, archive_size, retries=args.download_retries)
    elif pilot_ids and allow_remote_fetch:
        unavailable.append(
            {
                "resource": PROTEINGYM_PUBLIC_SCORE_ARCHIVE,
                "status": "NOT_AVAILABLE",
                "reason": "official ProteinGym record did not expose the public zero-shot substitution score archive",
            }
        )

    for assay_id in pilot_ids:
        records = records_by_assay.get(assay_id, [])
        existing_path, existing_validation = find_existing_valid_public_score(args, assay_id, records)
        if existing_path and existing_validation:
            reused.append({"assay_id": assay_id, **existing_validation})
            per_assay[assay_id] = {"status": "valid", "source": "reused_existing_file", "validation": existing_validation}
            continue
        target_path = public_score_target_path(args, assay_id)
        member_name = f"{assay_id}.csv"
        entry = entries.get(member_name)
        if entry is None:
            matches = [row for name, row in entries.items() if Path(name).name == member_name]
            entry = matches[0] if matches else None
        if entry is None or not archive_url:
            reason = f"member {member_name} was not found in {PROTEINGYM_PUBLIC_SCORE_ARCHIVE}"
            unavailable.append({"assay_id": assay_id, "status": "NOT_AVAILABLE", "reason": reason})
            per_assay[assay_id] = {"status": "NOT_AVAILABLE", "reason": reason}
            continue
        extraction = extract_zip_member_by_range(archive_url, entry, target_path, retries=args.download_retries)
        validation = validate_evolutionary_prediction_file(target_path, assay_id, records)
        if validation.get("status") != "valid":
            reason = str(validation.get("reason", "extracted_file_invalid"))
            unavailable.append({"assay_id": assay_id, "status": "NOT_AVAILABLE", "reason": reason, "validation": validation})
            per_assay[assay_id] = {"status": "NOT_AVAILABLE", "reason": reason, "validation": validation}
            continue
        extracted.append({"assay_id": assay_id, **extraction, "validation": validation})
        per_assay[assay_id] = {"status": "valid", "source": "official_public_score_archive", "validation": validation}

    retained = [
        {
            "assay_id": assay_id,
            "path": row["validation"]["path"],
            "sha256": row["validation"].get("sha256", ""),
            "size_bytes": row["validation"].get("size_bytes", ""),
            "method_counts": row["validation"].get("method_counts", {}),
        }
        for assay_id, row in per_assay.items()
        if row.get("status") == "valid"
    ]
    payload = {
        "created_at": now_utc(),
        "official_source": f"https://zenodo.org/records/{record.get('id', PROTEINGYM_ZENODO_RECORD_ID)}" if record.get("id") else "",
        "source_api": args.proteingym_source_api,
        "source_doi": record.get("doi", ""),
        "dataset_version": record.get("metadata", {}).get("version", ""),
        "dataset_revision": str(record.get("id", "")),
        "archive": {
            "key": PROTEINGYM_PUBLIC_SCORE_ARCHIVE,
            "source_url": archive_url,
            "size_bytes": archive_size,
            "checksum": archive_checksum,
            "retained_complete_archive": False,
            "extraction_strategy": "HTTP byte-range extraction of selected assay CSV members only",
        },
        "pilot_assays": pilot_ids,
        "allowed_evolutionary_columns": list(EVOLUTIONARY_BASELINE_COLUMNS),
        "status": "valid" if all(per_assay.get(assay_id, {}).get("status") == "valid" for assay_id in pilot_ids) else "partial",
        "per_assay": per_assay,
        "downloaded_and_retained_files": retained,
        "reused_files": reused,
        "extracted_files": extracted,
        "removed_temporary_files": [],
        "unavailable_optional_resources": unavailable,
        "storage_policy": {
            "canonical_public_prediction_dir": str(Path(args.public_predictions_dir)),
            "complete_public_score_archive_retained": False,
            "complete_assay_files_copied_to_run_dirs": False,
        },
    }
    write_json(report_path, payload, overwrite=True)
    write_evolutionary_resource_report_md(out_root / "protein_48h_evolutionary_baseline_report.md", payload)
    return payload


def write_evolutionary_resource_report_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# ProteinGym Evolutionary Baseline Resource Report",
        "",
        f"- Status: `{payload.get('status')}`.",
        f"- Official source: `{payload.get('official_source')}`.",
        f"- Dataset version/revision: `{payload.get('dataset_version')}` / `{payload.get('dataset_revision')}`.",
        f"- Archive: `{payload.get('archive', {}).get('key')}`.",
        f"- Selected pilot assays: `{', '.join(payload.get('pilot_assays', []))}`.",
        f"- Retained files: `{len(payload.get('downloaded_and_retained_files', []))}`.",
        f"- Reused files: `{len(payload.get('reused_files', []))}`.",
        "",
        "## Per-Assay Status",
        "",
    ]
    for assay_id, row in sorted(payload.get("per_assay", {}).items()):
        validation = row.get("validation", {})
        methods = validation.get("method_counts", {})
        lines.append(f"- `{assay_id}`: `{row.get('status')}` methods `{json.dumps(methods, sort_keys=True)}`.")
    unavailable = payload.get("unavailable_optional_resources", [])
    if unavailable:
        lines.extend(["", "## Unavailable", ""])
        for row in unavailable:
            lines.append(f"- `{row.get('assay_id', row.get('resource', 'resource'))}`: {row.get('reason')}")
    path.write_text("\n".join(lines) + "\n")


def ensure_proteingym_data(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_path = out_root / "protein_48h_proteingym_download_report.json"
    existing_report = read_json(report_path)
    validation = validate_proteingym_dataset(args)
    reused_files = []
    downloaded_payload = {"downloaded_files": [], "retained_files": [], "removed_temporary_files": []}
    quarantined = []
    if validation["status"] != "valid" and not args.auto_download_proteingym:
        record = {"id": "", "doi": "", "metadata": {"version": ""}, "files": []}
        source = args.proteingym_source_api
    elif validation["status"] == "valid" and args.skip_proteingym_download:
        record = {"id": "local-fixture", "doi": "", "metadata": {"version": "local-fixture"}, "files": []}
        source = "local prevalidated ProteinGym-style fixture"
    elif validation["status"] == "valid" and existing_report.get("validation_status") == "valid" and download_report_files_intact(report_path, args):
        record = {
            "id": existing_report.get("dataset_revision", ""),
            "doi": existing_report.get("source_doi", ""),
            "metadata": {"version": existing_report.get("dataset_version", "")},
            "files": [],
        }
        source = str(existing_report.get("official_source") or args.proteingym_source_api)
    else:
        record = fetch_proteingym_record(args.proteingym_source_api, retries=args.download_retries)
        source = f"https://zenodo.org/records/{record.get('id', PROTEINGYM_ZENODO_RECORD_ID)}"
    if validation["status"] == "valid":
        reused_files = [
            {"path": str(Path(args.metadata_csv)), "sha256": file_sha256(Path(args.metadata_csv)), "size_bytes": Path(args.metadata_csv).stat().st_size},
            *[
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in find_dms_files(Path(args.dms_dir))[:500]
            ],
        ]
    else:
        if not args.auto_download_proteingym:
            report = build_download_report(args, record, validation, source, reused_files, downloaded_payload, quarantined, existing_report)
            write_json(report_path, report, overwrite=True)
            write_download_report_md(out_root / "protein_48h_proteingym_download_report.md", report)
            return report
        if Path(args.dms_dir).exists() or Path(args.metadata_csv).exists():
            quarantined = quarantine_invalid_inputs(args, validation.get("reason", "invalid_existing_proteingym_inputs"))
        downloaded_payload = download_required_proteingym_files(args, record)
        validation = validate_proteingym_dataset(args)
    report = build_download_report(args, record, validation, source, reused_files, downloaded_payload, quarantined, existing_report)
    write_json(report_path, report, overwrite=True)
    write_download_report_md(out_root / "protein_48h_proteingym_download_report.md", report)
    if validation["status"] != "valid":
        raise RuntimeError(f"ProteinGym data validation failed: {validation.get('reason')}")
    return report


def build_download_report(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    validation: Mapping[str, Any],
    source: str,
    reused_files: Sequence[Mapping[str, Any]],
    downloaded_payload: Mapping[str, Any],
    quarantined: Sequence[Mapping[str, Any]],
    existing_report: Mapping[str, Any],
) -> dict[str, Any]:
    files = zenodo_file_map(record)
    retained = list(downloaded_payload.get("retained_files", []))
    if not retained and validation.get("status") == "valid":
        retained = [
            {"path": str(Path(args.metadata_csv)), "sha256": file_sha256(Path(args.metadata_csv)), "size_bytes": Path(args.metadata_csv).stat().st_size},
            *[
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in find_dms_files(Path(args.dms_dir))
            ],
        ]
    return {
        "created_at": now_utc(),
        "official_source": source,
        "source_api": args.proteingym_source_api,
        "source_doi": record.get("doi", ""),
        "dataset_version": record.get("metadata", {}).get("version", ""),
        "dataset_revision": str(record.get("id", "")),
        "required_source_files": {
            key: {
                "size_bytes": files.get(key, {}).get("size", ""),
                "checksum": files.get(key, {}).get("checksum", ""),
                "download_url": files.get(key, {}).get("links", {}).get("self", ""),
            }
            for key in sorted(REQUIRED_PROTEINGYM_FILES)
        }
        if files
        else existing_report.get("required_source_files", {}),
        "validation_status": validation.get("status"),
        "validation_reason": validation.get("reason", ""),
        "discovered_assays": validation.get("discovered_assays", 0),
        "valid_assays": validation.get("valid_assays", 0),
        "invalid_assays": validation.get("invalid_assays", 0),
        "valid_single_substitution_records": validation.get("valid_single_substitution_records", 0),
        "dataset_fingerprint": validation.get("dataset_fingerprint", "invalid"),
        "validation": validation,
        "reused_files": list(reused_files),
        "downloaded_files": list(downloaded_payload.get("downloaded_files", [])),
        "retained_files": retained,
        "removed_temporary_files": list(downloaded_payload.get("removed_temporary_files", [])),
        "quarantined_invalid_inputs": list(quarantined),
        "optional_resources": optional_proteingym_resources(record),
        "previous_report_status": existing_report.get("validation_status", ""),
        "storage_policy": {
            "canonical_dms_dir": str(args.dms_dir),
            "canonical_metadata_csv": str(args.metadata_csv),
            "archives_removed_after_validation": bool(args.remove_validated_archives),
            "complete_assay_files_copied_to_run_dirs": False,
        },
    }


def snapshot_previous_blocked_execution(args: argparse.Namespace, registry: dict[str, Any], download_report: Mapping[str, Any]) -> None:
    out_root = Path(args.out_root)
    summary_path = out_root / "protein_48h_summary_report.json"
    summary = read_json(summary_path)
    if summary.get("input_status", {}).get("status") != "NOT_AVAILABLE":
        return
    fingerprint = str(download_report.get("dataset_fingerprint", ""))
    snapshots = registry.setdefault("blocked_executions", [])
    if any(row.get("dataset_fingerprint") == fingerprint and row.get("source_summary") == str(summary_path) for row in snapshots):
        return
    timestamp = now_utc().replace(":", "").replace("+", "_")
    snapshot_root = out_root / "blocked_executions" / timestamp
    snapshot_root.mkdir(parents=True, exist_ok=True)
    active_paths = [
        "protein_48h_candidate_inventory.csv",
        "protein_48h_candidate_ranking.csv",
        "protein_48h_valid_records.json",
        "protein_48h_split_manifest.json",
        "protein_48h_split_audit.json",
        "protein_48h_baseline_results.csv",
        "protein_48h_baseline_report.md",
        "protein_48h_baseline_predictions",
        "protein_48h_esm2_pilot_metrics.csv",
        "protein_48h_esm2_pilot_predictions",
        "protein_48h_advancement_gate.json",
        "protein_48h_lora_metrics.csv",
        "protein_48h_lora_predictions",
        "protein_48h_summary_report.json",
        "protein_48h_summary_report.md",
        "protein_48h_artifact_audit.json",
    ]
    moved = []
    for rel in active_paths:
        source = out_root / rel
        if not source.exists():
            continue
        target = snapshot_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({"source": str(source), "snapshot_path": str(target)})
    snapshot_payload = {
        "created_at": now_utc(),
        "reason": "previous_missing_proteingym_data_execution_preserved_before_real_dataset_restart",
        "dataset_fingerprint": fingerprint,
        "source_summary": str(summary_path),
        "moved_artifacts": moved,
        "previous_permitted_conclusion": summary.get("permitted_conclusion", ""),
        "previous_input_status": summary.get("input_status", {}),
    }
    write_json(snapshot_root / "blocked_execution_snapshot.json", snapshot_payload, overwrite=True)
    snapshots.append(snapshot_payload)
    save_registry(args, registry)


def task_id_for_data(args: argparse.Namespace, base: str) -> str:
    suffix = getattr(args, "data_run_suffix", "")
    return f"{base}_{suffix}" if suffix else base


def pick_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for col in columns:
        key = str(col).lower()
        if any(candidate.lower() in key for candidate in candidates):
            return str(col)
    return None


def assay_id_for_file(path: Path, metadata: Mapping[str, Mapping[str, Any]]) -> str:
    stem = normalize_id(path.stem)
    row = metadata.get(stem) or metadata.get(normalize_id(path.name)) or {}
    for key in ("DMS_id", "assay_id", "target_name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return normalize_id(str(value))
    return stem


def metadata_for_file(path: Path, metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for key in (normalize_id(path.stem), normalize_id(path.name)):
        if key in metadata:
            return dict(metadata[key])
    return {}


def metadata_sequence(row: Mapping[str, Any]) -> str:
    for key in ("target_seq", "wildtype_sequence", "wild_type_sequence", "sequence", "WT_sequence"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return clean_protein_sequence(str(value))
    return ""


def clean_protein_sequence(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch in AA_SET)


def parse_single_mutation(value: Any) -> tuple[str, int, str, str] | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    parts = re.split(r"[:;,]", text)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) != 1:
        return None
    match = MUTATION_RE.match(parts[0])
    if not match:
        return None
    wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
    if wt not in AA_SET or mut not in AA_SET or wt == mut or pos <= 0:
        return None
    return wt, pos, mut, f"{wt}{pos}{mut}"


def infer_wt_sequence(rows: pd.DataFrame, mutation_col: str, sequence_col: str | None, metadata_row: Mapping[str, Any]) -> str:
    wt = metadata_sequence(metadata_row)
    if wt:
        return wt
    if sequence_col is None:
        return ""
    for _, row in rows.head(1000).iterrows():
        parsed = parse_single_mutation(row.get(mutation_col))
        if parsed is None:
            continue
        wt_res, pos, _mut_res, _mutation = parsed
        seq = clean_protein_sequence(str(row.get(sequence_col, "")))
        if pos <= len(seq):
            chars = list(seq)
            chars[pos - 1] = wt_res
            return "".join(chars)
    return ""


def read_assay_records(assay: AssayInput) -> tuple[list[VariantRecord], dict[str, Any]]:
    df = safe_read_csv(assay.dms_path)
    mutation_col = pick_column(df.columns, ["mutant", "mutation", "variant", "mutations"])
    score_col = pick_column(df.columns, ["DMS_score", "score", "fitness", "measurement", "target", "label"])
    sequence_col = pick_column(df.columns, ["mutated_sequence", "mutant_sequence", "sequence"])
    if mutation_col is None or score_col is None:
        return [], {
            "total_sample_count": int(len(df)),
            "missing_value_proportion": 1.0,
            "parse_error": f"missing required mutation/score columns in {assay.dms_path}",
        }
    wt_sequence = infer_wt_sequence(df, mutation_col, sequence_col, assay.metadata)
    records: list[VariantRecord] = []
    missing_scores = 0
    seen_ids: set[str] = set()
    for idx, row in df.iterrows():
        parsed = parse_single_mutation(row.get(mutation_col))
        raw_score = pd.to_numeric(pd.Series([row.get(score_col)]), errors="coerce").iloc[0]
        if pd.isna(raw_score):
            missing_scores += 1
            continue
        if parsed is None:
            continue
        wt, pos, mut, mutation = parsed
        if not wt_sequence or pos > len(wt_sequence) or wt_sequence[pos - 1] != wt:
            continue
        if sequence_col is not None and not pd.isna(row.get(sequence_col)):
            mutated_sequence = clean_protein_sequence(str(row.get(sequence_col)))
        else:
            chars = list(wt_sequence)
            chars[pos - 1] = mut
            mutated_sequence = "".join(chars)
        if len(mutated_sequence) != len(wt_sequence):
            continue
        if mutated_sequence[pos - 1] != mut:
            chars = list(wt_sequence)
            chars[pos - 1] = mut
            mutated_sequence = "".join(chars)
        base_id = str(row.get("id") or row.get("sample_id") or mutation)
        sample_id = normalize_id(base_id)
        if sample_id in seen_ids:
            sample_id = normalize_id(f"{base_id}_{idx}")
        seen_ids.add(sample_id)
        records.append(
            VariantRecord(
                assay_id=assay.assay_id,
                sample_id=sample_id,
                mutation=mutation,
                wt=wt,
                position=pos,
                mut=mut,
                score=float(raw_score),
                mutated_sequence=mutated_sequence,
                wt_sequence=wt_sequence,
            )
        )
    return records, {
        "total_sample_count": int(len(df)),
        "missing_value_proportion": float(missing_scores / max(len(df), 1)),
        "wt_sequence": wt_sequence,
        "score_column": score_col,
        "mutation_column": mutation_col,
        "sequence_column": sequence_col or "",
    }


def matching_aux_files(root: Path, assay: AssayInput) -> list[Path]:
    if not root.exists():
        return []
    needles = {
        normalize_id(assay.assay_id).lower(),
        normalize_id(assay.dms_path.stem).lower(),
        normalize_id(str(assay.metadata.get("DMS_filename", ""))).lower(),
        normalize_id(str(assay.metadata.get("MSA_filename", ""))).lower(),
    }
    needles = {needle for needle in needles if needle}
    matches = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        normalized = normalize_id(path.name).lower()
        if any(needle and needle in normalized for needle in needles):
            matches.append(path)
    return sorted(matches)


def inspect_public_prediction_columns(files: Sequence[Path]) -> tuple[bool, bool]:
    has_evolutionary = False
    has_plm = False
    for path in files[:5]:
        if path.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        for col in df.columns:
            key = str(col).lower()
            if any(token in key for token in ("esm", "plm", "proteinmpnn", "prott5", "protbert")):
                has_plm = True
            if str(col) in EVOLUTIONARY_BASELINE_COLUMNS:
                has_evolutionary = True
    return has_evolutionary, has_plm


def build_assay_inputs(args: argparse.Namespace) -> list[AssayInput]:
    metadata = load_metadata(Path(args.metadata_csv))
    assays = []
    for path in find_dms_files(Path(args.dms_dir)):
        row = metadata_for_file(path, metadata)
        assay_id = assay_id_for_file(path, metadata)
        assays.append(
            AssayInput(
                assay_id=assay_id,
                assay_name=str(row.get("DMS_id") or row.get("assay_name") or path.stem),
                protein_name=str(row.get("target_name") or row.get("protein_name") or row.get("UniProt_ID") or path.stem),
                phenotype_type=str(row.get("selection_type") or row.get("phenotype_type") or row.get("DMS_binarization_cutoff") or "unknown"),
                dms_path=path,
                metadata=row,
            )
        )
    return assays


def assay_inventory_row(args: argparse.Namespace, assay: AssayInput) -> tuple[dict[str, Any], list[VariantRecord]]:
    records, parse_meta = read_assay_records(assay)
    scores = np.array([record.score for record in records], dtype=np.float64)
    positions = {record.position for record in records}
    duplicates = len({record.mutation for record in records}) < len(records)
    wt_sequence = parse_meta.get("wt_sequence") or (records[0].wt_sequence if records else metadata_sequence(assay.metadata))
    seq_len = len(wt_sequence)
    pred_files = matching_aux_files(Path(args.public_predictions_dir), assay)
    msa_files = matching_aux_files(Path(args.msa_dir), assay)
    public_evo, public_plm = inspect_public_prediction_columns(pred_files)
    valid_prop = len(records) / max(int(parse_meta.get("total_sample_count", 0)), 1)
    score_std = float(np.std(scores)) if scores.size else float("nan")
    q = np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95]) if scores.size else [float("nan")] * 5
    dynamic_range = float(np.nanmax(scores) - np.nanmin(scores)) if scores.size else float("nan")
    compatible_length = bool(seq_len and seq_len <= args.esm2_max_length)
    gpu_ok = bool(args.mock_esm2 or args.device == "cpu" or detect_single_gpu(args))
    quality = 0.0
    if scores.size:
        quality += min(math.log10(len(records) + 1), 4.0)
        quality += 2.0 * valid_prop
        quality += min(max(score_std, 0.0), 5.0)
        quality += min(len(positions) / max(seq_len, 1), 1.0)
        quality -= float(parse_meta.get("missing_value_proportion", 1.0))
        quality += 0.5 if public_evo or msa_files else 0.0
        quality += 0.25 if compatible_length else -2.0
        quality -= 0.25 if duplicates else 0.0
    exclusion_reasons = []
    if len(records) < args.min_valid_samples:
        exclusion_reasons.append("too_few_valid_single_substitutions")
    if not scores.size or not math.isfinite(score_std) or score_std < args.min_score_std:
        exclusion_reasons.append("insufficient_score_dynamic_range")
    if parse_meta.get("missing_value_proportion", 1.0) > args.max_missing_proportion:
        exclusion_reasons.append("high_missing_value_rate")
    if not compatible_length:
        exclusion_reasons.append("sequence_exceeds_esm2_length_policy")
    if valid_prop < args.min_valid_proportion:
        exclusion_reasons.append("low_single_substitution_fraction")
    row = {
        "assay_id": assay.assay_id,
        "assay_name": assay.assay_name,
        "protein_name": assay.protein_name,
        "phenotype_type": assay.phenotype_type,
        "dms_path": str(assay.dms_path),
        "wild_type_sequence": wt_sequence,
        "sequence_length": seq_len,
        "total_sample_count": int(parse_meta.get("total_sample_count", 0)),
        "valid_single_substitution_count": len(records),
        "valid_single_substitution_proportion": valid_prop,
        "score_mean": float(np.mean(scores)) if scores.size else "",
        "score_std": score_std,
        "score_q05": q[0],
        "score_q25": q[1],
        "score_q50": q[2],
        "score_q75": q[3],
        "score_q95": q[4],
        "score_dynamic_range": dynamic_range,
        "missing_value_proportion": parse_meta.get("missing_value_proportion", 1.0),
        "mutation_position_coverage": len(positions),
        "mutation_position_coverage_proportion": len(positions) / max(seq_len, 1),
        "duplicate_mutations_exist": duplicates,
        "msa_available": bool(msa_files),
        "conservation_or_evolutionary_prediction_available": bool(public_evo or msa_files),
        "public_plm_prediction_available": bool(public_plm),
        "compatible_with_esm2_length": compatible_length,
        "compatible_with_available_gpu_memory": gpu_ok,
        "data_quality_score": quality,
        "entered_static_candidate_set": False,
        "selected_pilot": False,
        "selection_reason": "",
        "exclusion_reason": ";".join(exclusion_reasons),
    }
    return row, records


def detect_single_gpu(args: argparse.Namespace) -> bool:
    if str(args.device).startswith("cpu"):
        return True
    if torch is None:
        return False
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() == 1
    except Exception:
        return False


def write_hvue_summary(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    fairness_path = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_input_fairness.json"
    comparison_path = PROJECT_ROOT / "data/phase2/stage1_baseline_alignment_20260729/kmer_input_fairness_comparison.csv"
    calibration_path = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727/base_calibration_final_report.json"
    multiseed_path = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727/base_calibration_excess_recalculated.csv"
    formal_report_path = PROJECT_ROOT / "data/phase2/stage1_formal_target_manifests/stage1_formal_target_manifest_report.json"
    fairness = read_json(fairness_path)
    calibration = read_json(calibration_path)
    formal_report = read_json(formal_report_path)
    comparison_rows = []
    if comparison_path.exists():
        with comparison_path.open(newline="") as handle:
            comparison_rows = list(csv.DictReader(handle))
    multi_rows = []
    if multiseed_path.exists():
        with multiseed_path.open(newline="") as handle:
            multi_rows = list(csv.DictReader(handle))
    matched_row = next((row for row in comparison_rows if row.get("config_id") == "earlier_matched_input_count_first512_cgrid10"), {})
    full_row = next((row for row in comparison_rows if row.get("config_id") == "full_sequence_strong_hashing_cgrid100"), {})
    best = calibration.get("best_observed_run_by_auroc_excess", {})
    matched_auroc = float(matched_row.get("auroc") or fairness.get("matched_input_baseline") or "nan")
    full_auroc = float(full_row.get("auroc") or calibration.get("canonical_baseline", {}).get("test_auroc") or "nan")
    raw_best_auroc = float(best.get("raw_test_auroc", "nan"))
    payload = {
        "created_at": now_utc(),
        "status": "read_only_hvue_benchmark_limitation_frozen",
        "matched_input_kmer": {
            "auroc": matched_auroc,
            "mcc": float(matched_row.get("mcc") or "nan"),
            "feature_policy": matched_row.get("feature_implementation", "count_first512"),
            "sequence_length_policy": matched_row.get("sequence_length_policy") or fairness.get("kmer_0849_sequence_length_policy", ""),
        },
        "full_sequence_strong_kmer": {
            "auroc": full_auroc,
            "mcc": float(full_row.get("mcc") or calibration.get("canonical_baseline", {}).get("test_mcc") or "nan"),
            "feature_policy": full_row.get("feature_implementation", "hashing_l2_scaled"),
            "sequence_length_policy": full_row.get("sequence_length_policy") or fairness.get("kmer_0893_sequence_length_policy", ""),
        },
        "best_fresh_lora_result": best,
        "multi_seed_fresh_lora_results": multi_rows,
        "excess_values": {
            "best_lora_minus_matched_input_kmer_auroc": raw_best_auroc - matched_auroc if math.isfinite(raw_best_auroc) and math.isfinite(matched_auroc) else None,
            "best_lora_minus_full_sequence_strong_kmer_auroc": raw_best_auroc - full_auroc if math.isfinite(raw_best_auroc) and math.isfinite(full_auroc) else None,
            "reported_matched_input_context": "approximately +0.0436 AUROC",
            "reported_full_sequence_context": "approximately +0.00025 AUROC",
        },
        "input_length_truncation_and_feature_policies": fairness,
        "threshold_selection_policy": calibration.get("canonical_baseline", {}).get("threshold_policy") or full_row.get("threshold_policy", "validation_selected_mcc"),
        "split_and_manifest": {
            "split_type": "cluster_disjoint",
            "manifest": formal_report.get("merged_manifest") or fairness.get("manifest_path", ""),
            "manifest_sha256": fairness.get("manifest_sha256", ""),
        },
        "interpretation": {
            "why_original_margin_changes": (
                "The earlier positive margin used a matched-input k-mer baseline constrained to the same prefix-truncated input budget as the LoRA "
                "pipeline. The later strong baseline uses full-sequence 3-6-mer hashing with a wider C grid and reaches essentially the same AUROC as "
                "the best fresh-LoRA run, so the apparent model-specific margin largely vanishes under the stronger sequence-composition comparator."
            ),
            "hvue_formal_tar_status": "do_not_launch_new_formal_hvue_tar_lat_or_lora_subspace_training",
            "artifact_roles": {
                "TAR": "method_prototype; diagnostic_experiment; benchmark_limitation_evidence",
                "LAT": "method_prototype; diagnostic_experiment; benchmark_limitation_evidence",
                "Experiment3": "method_prototype; diagnostic_experiment; benchmark_limitation_evidence",
            },
        },
        "source_artifacts": {
            "input_fairness": str(fairness_path),
            "baseline_alignment": str(comparison_path),
            "calibration_final_report": str(calibration_path),
            "multi_seed_lora": str(multiseed_path),
            "formal_manifest_report": str(formal_report_path),
        },
    }
    write_json(out_root / "protein_48h_hvue_benchmark_limitation_summary.json", payload)
    lines = [
        "# HVUE Benchmark-Limitation Summary",
        "",
        f"- Matched-input k-mer AUROC/MCC: `{payload['matched_input_kmer']['auroc']}` / `{payload['matched_input_kmer']['mcc']}`.",
        f"- Full-sequence strong k-mer AUROC/MCC: `{payload['full_sequence_strong_kmer']['auroc']}` / `{payload['full_sequence_strong_kmer']['mcc']}`.",
        f"- Best fresh-LoRA run: `{best.get('run_id', 'unknown')}` with AUROC `{best.get('raw_test_auroc', '')}`.",
        f"- Best LoRA excess over matched-input k-mer: `{payload['excess_values']['best_lora_minus_matched_input_kmer_auroc']}`.",
        f"- Best LoRA excess over full-sequence strong k-mer: `{payload['excess_values']['best_lora_minus_full_sequence_strong_kmer_auroc']}`.",
        "",
        "HVUE is frozen as benchmark-limitation evidence. New formal HVUE TAR, LAT, LoRA-subspace, and high-cost training are not part of this workflow.",
    ]
    write_json(out_root / "protein_48h_hvue_benchmark_limitation_summary.metadata.json", build_run_metadata(args=args, data_paths=payload["source_artifacts"].values(), extra={"phase": "protein_48h_hvue_summary"}))
    md_path = out_root / "protein_48h_hvue_benchmark_limitation_summary.md"
    if not md_path.exists():
        md_path.write_text("\n".join(lines) + "\n")
    return payload


def frozen_protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "created_at": now_utc(),
        "status": "canonical_frozen_before_test_readout",
        "workflow": "ProteinGym-ESM2-150M minimum viable benchmark qualification",
        "max_concurrent_gpu_jobs": 1,
        "gpu_policy": {
            "cuda_visible_devices": str(args.cuda_visible_devices),
            "strict_sequential_execution": True,
            "ddp_allowed": False,
            "concurrent_gpu_jobs_allowed": 1,
        },
        "input_roots": {
            "dms_dir": str(args.dms_dir),
            "metadata_csv": str(args.metadata_csv),
            "public_predictions_dir": str(args.public_predictions_dir),
            "msa_dir": str(args.msa_dir),
            "proteingym_source_api": str(args.proteingym_source_api),
        },
        "stage_order": [
            "proteingym_data_download_and_validation",
            "hvue_summary",
            "static_prescreen",
            "split_creation",
            "baseline_evaluation",
            "esm2_pilot",
            "advancement_gate",
            "lora_calibration",
            "lora_multiseed_confirmation",
            "final_aggregation",
        ],
        "prescreen": {
            "max_static_candidate_set": args.max_static_candidates,
            "max_pilot_assays": args.max_pilot_assays,
            "min_valid_samples": args.min_valid_samples,
            "min_score_std": args.min_score_std,
            "max_missing_proportion": args.max_missing_proportion,
            "min_valid_proportion": args.min_valid_proportion,
            "esm2_max_length": args.esm2_max_length,
        },
        "splits": {
            "seed": args.split_seed,
            "random_split": {"train": args.train_fraction, "val": args.val_fraction, "test": args.test_fraction},
            "position_heldout_split": {"train": args.train_fraction, "val": args.val_fraction, "test": args.test_fraction},
            "primary_split": "position_heldout",
            "no_duplicate_mutation_across_splits": True,
            "position_sets_mutually_disjoint": True,
        },
        "baselines": [
            "wt_residue",
            "mutant_residue",
            "wt_plus_mutant_residue",
            "mutation_position",
            "normalized_position",
            "residue_pair_plus_position",
            "blosum_substitution",
            "charge_change",
            "hydrophobicity_change",
            "residue_volume_change",
            "site_specific_mean",
            "site_independent_pair_mean",
            "strongest_available_msa_or_evolutionary_prediction",
        ],
        "metrics": {
            "primary": "spearman",
            "secondary": "mse",
            "correlation_direction": "higher_is_better_after_validation_orienting",
            "model_excess": "model_spearman - strongest_available_non_plm_baseline_spearman",
            "mse_improvement": "baseline_mse - model_mse",
        },
        "baseline_selection": {
            "selection_split": "val",
            "test_set_used_for_selection": False,
            "strongest_non_plm_rule": "highest validation Spearman among complete non-PLM baselines; tie-break lower validation MSE then baseline name",
        },
        "esm2_pilot": {
            "model": args.esm2_model,
            "new_esm2_650m_allowed": False,
            "zero_shot_definition": "masked wild-type sequence log p(mutant residue) minus log p(wild-type residue) at the mutated position",
            "tokenizer_policy": "HuggingFace ESM tokenizer; one amino acid per residue; special CLS token means 1-indexed mutation position maps to token index position",
            "sequence_policy": f"full sequence when length <= {args.esm2_max_length}; longer assays excluded in this round",
            "frozen_representations": ["mutation_position", "wt_mutant_position_difference", "local_window", "whole_sequence_mean"],
            "readouts": ["linear_regression", "ridge_regression"],
            "cache_policy": "cache key includes model hash, assay ID, sequence hash, split manifest hash, layer, representation type, and preprocessing policy",
        },
        "advancement_gate": {
            "uses_test_for_advancement": False,
            "primary_threshold": args.frozen_excess_threshold,
            "positive_margin_rule": "at least one ESM2 zero-shot or frozen readout has positive validation excess over the strongest available non-PLM baseline on position-held-out split",
            "preferred_margin_rule": f"validation Spearman excess >= {args.frozen_excess_threshold}",
            "bootstrap_rule": "position-level validation bootstrap CI lower bound above zero, or explicitly provisional positive trend",
            "ranking_rule": "sort by pass status, validation excess, validation Spearman, data quality score, assay_id; do not use test metrics for advancement",
            "tie_breaking_rule": "higher validation excess, then higher validation Spearman, then higher valid sample count, then lexical assay_id",
        },
        "lora": {
            "max_lora_assays": args.max_lora_assays,
            "calibration_seed": args.calibration_seed,
            "formal_seeds": args.formal_seeds,
            "rank_grid": [8, 16],
            "learning_rate_grid": [1e-5, 5e-5],
            "bf16": True,
            "target_modules": args.lora_target_modules,
            "checkpoint_selection": "validation Spearman only; no test-based checkpoint selection",
            "fresh_head_each_seed": True,
            "full_finetuning_allowed": False,
        },
        "excluded_work": [
            "ESM2-650M",
            "supervised_ESM1v_finetuning",
            "ESM2_full_finetuning",
            "family_disjoint_multi_protein_joint_training",
            "TAR",
            "vanilla_LAT",
            "LoRA_shaped_LAT",
            "LoRA_subspace_targeting",
            "large_lora_grid",
            "multi_seed_lora_all_pilots",
            "complete_retain_benchmark",
            "new_formal_HVUE_training",
        ],
        "storage": {
            "shared_model_dir": str(Path(args.out_root) / "shared_models"),
            "cache_dir": str(Path(args.out_root) / "caches"),
            "do_not_store_full_lora_merged_models": True,
            "retain_only_best_and_latest_training_checkpoints": True,
        },
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def write_frozen_protocol(args: argparse.Namespace) -> dict[str, Any]:
    payload = frozen_protocol_payload(args)
    write_json(Path(args.out_root) / "protein_48h_frozen_protocol.json", payload, overwrite=True)
    return payload


def prescreen_assays(args: argparse.Namespace) -> dict[str, Any]:
    assays = build_assay_inputs(args)
    inventory_rows: list[dict[str, Any]] = []
    records_by_assay: dict[str, list[VariantRecord]] = {}
    for assay in assays:
        row, records = assay_inventory_row(args, assay)
        inventory_rows.append(row)
        records_by_assay[assay.assay_id] = records
    sorted_rows = sorted(inventory_rows, key=lambda row: (-float(row.get("data_quality_score") or 0), str(row.get("assay_id"))))
    candidate_rows = [row for row in sorted_rows if not row.get("exclusion_reason")]
    static_ids = {row["assay_id"] for row in candidate_rows[: args.max_static_candidates]}
    pilot_ids = {row["assay_id"] for row in candidate_rows[: args.max_pilot_assays]}
    ranking_rows = []
    for rank, row in enumerate(sorted_rows, start=1):
        row = dict(row)
        row["entered_static_candidate_set"] = row["assay_id"] in static_ids
        row["selected_pilot"] = row["assay_id"] in pilot_ids
        if row["selected_pilot"]:
            row["selection_reason"] = "top_quality_candidate_passing_static_filters_within_three_pilot_limit"
        elif row["entered_static_candidate_set"]:
            row["selection_reason"] = "static_candidate_not_in_top_three_pilots"
        elif not row["exclusion_reason"]:
            row["exclusion_reason"] = "outside_top_static_candidate_limit"
        row["candidate_rank"] = rank
        ranking_rows.append(row)
    inventory_path = Path(args.out_root) / "protein_48h_candidate_inventory.csv"
    ranking_path = Path(args.out_root) / "protein_48h_candidate_ranking.csv"
    fields = [*INVENTORY_FIELDS, "candidate_rank"]
    write_csv(inventory_path, ranking_rows, fields)
    write_csv(ranking_path, ranking_rows, fields)
    records_path = Path(args.out_root) / "protein_48h_valid_records.json"
    record_payload = {
        assay_id: [
            {
                "assay_id": record.assay_id,
                "sample_id": record.sample_id,
                "mutation": record.mutation,
                "wt": record.wt,
                "position": record.position,
                "mut": record.mut,
                "score": record.score,
                "mutated_sequence": record.mutated_sequence,
                "wt_sequence": record.wt_sequence,
            }
            for record in records
        ]
        for assay_id, records in records_by_assay.items()
    }
    write_json(records_path, record_payload)
    return {
        "assays_seen": len(assays),
        "static_candidate_count": len(static_ids),
        "pilot_ids": sorted(pilot_ids),
        "records_path": str(records_path),
        "inventory_rows": ranking_rows,
    }


def load_valid_records(out_root: Path) -> dict[str, list[VariantRecord]]:
    payload = read_json(out_root / "protein_48h_valid_records.json")
    result: dict[str, list[VariantRecord]] = {}
    for assay_id, rows in payload.items():
        result[assay_id] = [
            VariantRecord(
                assay_id=row["assay_id"],
                sample_id=row["sample_id"],
                mutation=row["mutation"],
                wt=row["wt"],
                position=int(row["position"]),
                mut=row["mut"],
                score=float(row["score"]),
                mutated_sequence=row["mutated_sequence"],
                wt_sequence=row["wt_sequence"],
            )
            for row in rows
        ]
    return result


def assign_grouped_split(groups: list[Any], group_to_count: Mapping[Any, int], fractions: tuple[float, float, float], seed: int) -> dict[Any, str]:
    rng = random.Random(seed)
    shuffled = sorted(groups, key=lambda value: stable_hash([str(value), seed]))
    rng.shuffle(shuffled)
    total = sum(group_to_count[group] for group in shuffled)
    targets = {
        "train": total * fractions[0],
        "val": total * fractions[1],
        "test": total * fractions[2],
    }
    assignments: dict[Any, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for group in shuffled:
        deficits = {split: targets[split] - counts[split] for split in counts}
        split = max(deficits, key=lambda name: (deficits[name], -counts[name]))
        assignments[group] = split
        counts[split] += group_to_count[group]
    for required in ("train", "val", "test"):
        if counts[required] == 0 and shuffled:
            largest = max((split for split in counts if counts[split] > 1), key=lambda split: counts[split], default="train")
            move_group = next(group for group, split in assignments.items() if split == largest)
            assignments[move_group] = required
            counts[required] += group_to_count[move_group]
            counts[largest] -= group_to_count[move_group]
    return assignments


def split_records(records: list[VariantRecord], split_type: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    fractions = (args.train_fraction, args.val_fraction, args.test_fraction)
    if split_type == "position_heldout":
        groups = sorted({record.position for record in records})
        group_to_count = {pos: sum(1 for record in records if record.position == pos) for pos in groups}
        assignments = assign_grouped_split(groups, group_to_count, fractions, args.split_seed)
        return [
            {
                "assay_id": record.assay_id,
                "sample_id": record.sample_id,
                "mutation": record.mutation,
                "position": record.position,
                "wt": record.wt,
                "mut": record.mut,
                "score": record.score,
                "split_type": split_type,
                "split": assignments[record.position],
            }
            for record in records
        ]
    groups = sorted({record.mutation for record in records})
    group_to_count = {mutation: sum(1 for record in records if record.mutation == mutation) for mutation in groups}
    assignments = assign_grouped_split(groups, group_to_count, fractions, args.split_seed + 1009)
    return [
        {
            "assay_id": record.assay_id,
            "sample_id": record.sample_id,
            "mutation": record.mutation,
            "position": record.position,
            "wt": record.wt,
            "mut": record.mut,
            "score": record.score,
            "split_type": split_type,
            "split": assignments[record.mutation],
        }
        for record in records
    ]


def split_audit_for_entries(entries: list[dict[str, Any]], wt_sequence: str) -> dict[str, Any]:
    by_split = {split: [row for row in entries if row["split"] == split] for split in ("train", "val", "test")}
    pos_sets = {split: {row["position"] for row in rows} for split, rows in by_split.items()}
    mut_sets = {split: {row["mutation"] for row in rows} for split, rows in by_split.items()}
    score_stats = {}
    unreliable = []
    for split, rows in by_split.items():
        scores = np.array([float(row["score"]) for row in rows], dtype=np.float64)
        score_stats[split] = {
            "n": len(rows),
            "mean": float(np.mean(scores)) if scores.size else None,
            "std": float(np.std(scores)) if scores.size else None,
            "min": float(np.min(scores)) if scores.size else None,
            "max": float(np.max(scores)) if scores.size else None,
        }
        if len(rows) < 5 or (scores.size and np.std(scores) == 0):
            unreliable.append(split)
    def overlap(a: str, b: str, sets: Mapping[str, set[Any]]) -> list[Any]:
        return sorted(sets[a].intersection(sets[b]))
    pair_dist = {
        split: {
            f"{row['wt']}->{row['mut']}": sum(1 for item in rows if item["wt"] == row["wt"] and item["mut"] == row["mut"])
            for row in rows
        }
        for split, rows in by_split.items()
    }
    audit = {
        "counts": {split: len(rows) for split, rows in by_split.items()},
        "position_overlap": {
            "train_val": overlap("train", "val", pos_sets),
            "train_test": overlap("train", "test", pos_sets),
            "val_test": overlap("val", "test", pos_sets),
        },
        "duplicate_mutation_overlap": {
            "train_val": overlap("train", "val", mut_sets),
            "train_test": overlap("train", "test", mut_sets),
            "val_test": overlap("val", "test", mut_sets),
        },
        "protein_overlap": "same_single_protein_assay_expected",
        "mutation_pair_distribution": pair_dist,
        "wild_type_residue_distribution": {
            split: {aa: sum(1 for row in rows if row["wt"] == aa) for aa in AA}
            for split, rows in by_split.items()
        },
        "mutant_residue_distribution": {
            split: {aa: sum(1 for row in rows if row["mut"] == aa) for aa in AA}
            for split, rows in by_split.items()
        },
        "score_distribution": score_stats,
        "unreliable_label_distribution_splits": unreliable,
        "sequence_length": len(wt_sequence),
    }
    leakage = any(audit["position_overlap"].values()) if entries and entries[0]["split_type"] == "position_heldout" else False
    leakage = leakage or any(audit["duplicate_mutation_overlap"].values())
    audit["status"] = "valid" if not leakage and not unreliable else "invalid"
    audit["leakage_detected"] = bool(leakage)
    return audit


def create_splits(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    ranking_rows = list(csv.DictReader((out_root / "protein_48h_candidate_ranking.csv").open()))
    pilot_ids = [row["assay_id"] for row in ranking_rows if str(row.get("selected_pilot", "")).lower() == "true"]
    all_entries: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for assay_id in pilot_ids:
        records = records_by_assay.get(assay_id, [])
        if not records:
            continue
        wt_sequence = records[0].wt_sequence
        audits[assay_id] = {}
        for split_type in ("random", "position_heldout"):
            entries = split_records(records, split_type, args)
            all_entries.extend(entries)
            audits[assay_id][split_type] = split_audit_for_entries(entries, wt_sequence)
    payload = {
        "created_at": now_utc(),
        "split_seed": args.split_seed,
        "primary_split": "position_heldout",
        "pilot_assay_ids": pilot_ids,
        "entries": all_entries,
    }
    payload["manifest_hash"] = stable_hash(payload)
    write_json(out_root / "protein_48h_split_manifest.json", payload)
    write_json(out_root / "protein_48h_split_audit.json", {"created_at": now_utc(), "assays": audits, "manifest_hash": payload["manifest_hash"]})
    return payload


def rows_for_split(records: list[VariantRecord], split_entries: list[dict[str, Any]], split_type: str) -> list[tuple[VariantRecord, str]]:
    by_key = {
        (row.get("assay_id", ""), row["sample_id"], row["mutation"]): row["split"]
        for row in split_entries
        if row["split_type"] == split_type
    }
    legacy_by_key = {
        (row["sample_id"], row["mutation"]): row["split"]
        for row in split_entries
        if row["split_type"] == split_type and not row.get("assay_id")
    }
    result = []
    for record in records:
        split = by_key.get((record.assay_id, record.sample_id, record.mutation))
        if split is None:
            split = legacy_by_key.get((record.sample_id, record.mutation))
        if split:
            result.append((record, split))
    return result


def feature_dicts(records: Sequence[VariantRecord], baseline: str) -> list[dict[str, float]]:
    result = []
    for record in records:
        norm_pos = record.position / max(len(record.wt_sequence), 1)
        if baseline == "wt_residue":
            result.append({f"wt={record.wt}": 1.0})
        elif baseline == "mutant_residue":
            result.append({f"mut={record.mut}": 1.0})
        elif baseline == "wt_plus_mutant_residue":
            result.append({f"wt={record.wt}": 1.0, f"mut={record.mut}": 1.0})
        elif baseline == "mutation_position":
            result.append({"position": float(record.position)})
        elif baseline == "normalized_position":
            result.append({"normalized_position": norm_pos})
        elif baseline == "residue_pair_plus_position":
            result.append({f"pair={record.wt}->{record.mut}": 1.0, "normalized_position": norm_pos})
        elif baseline == "blosum_substitution":
            result.append({"blosum62": blosum_score(record.wt, record.mut)})
        elif baseline == "charge_change":
            result.append({"charge_change": charge(record.mut) - charge(record.wt)})
        elif baseline == "hydrophobicity_change":
            result.append({"hydrophobicity_change": HYDROPATHY[record.mut] - HYDROPATHY[record.wt]})
        elif baseline == "residue_volume_change":
            result.append({"volume_change": VOLUME[record.mut] - VOLUME[record.wt]})
        else:
            raise ValueError(f"Unknown feature baseline: {baseline}")
    return result


def split_arrays(rows: list[tuple[VariantRecord, str]]) -> tuple[list[VariantRecord], np.ndarray, np.ndarray]:
    records = [record for record, _split in rows]
    y = np.array([record.score for record in records], dtype=np.float64)
    splits = np.array([split for _record, split in rows])
    return records, y, splits


def fit_ridge_baseline(records: list[VariantRecord], y: np.ndarray, splits: np.ndarray, baseline: str) -> tuple[dict[str, Any], np.ndarray]:
    dict_rows = feature_dicts(records, baseline)
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(dict_rows)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if train.sum() < 2 or val.sum() < 1 or test.sum() < 1:
        raise ValueError("insufficient split rows for baseline")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_val = scaler.transform(x[val])
    x_test = scaler.transform(x[test])
    best_model = None
    best_alpha = None
    best_val = -float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y[train])
        pred_val = model.predict(x_val)
        val_spear = spearman_metric(y[val], pred_val)
        score = val_spear if math.isfinite(val_spear) else -float("inf")
        if score > best_val:
            best_model = model
            best_alpha = alpha
            best_val = score
    if best_model is None:
        best_model = Ridge(alpha=1.0).fit(x_train, y[train])
        best_alpha = 1.0
    pred = np.full_like(y, fill_value=float(np.mean(y[train])), dtype=np.float64)
    pred[train] = best_model.predict(x_train)
    pred[val] = best_model.predict(x_val)
    pred[test] = best_model.predict(x_test)
    sign = direction_sign_from_validation(y[val], pred[val])
    pred = apply_direction(pred, sign)
    return {"selected_alpha": best_alpha, "direction_sign": sign}, pred


def mean_lookup_baseline(records: list[VariantRecord], y: np.ndarray, splits: np.ndarray, key_func: Any) -> tuple[dict[str, Any], np.ndarray]:
    train = splits == "train"
    val = splits == "val"
    global_mean = float(np.mean(y[train])) if train.sum() else 0.0
    buckets: dict[Any, list[float]] = {}
    for record, score, is_train in zip(records, y, train):
        if is_train:
            buckets.setdefault(key_func(record), []).append(float(score))
    lookup = {key: float(np.mean(values)) for key, values in buckets.items()}
    pred = np.array([lookup.get(key_func(record), global_mean) for record in records], dtype=np.float64)
    sign = direction_sign_from_validation(y[val], pred[val])
    pred = apply_direction(pred, sign)
    return {"selected_alpha": "", "direction_sign": sign}, pred


def load_public_prediction_table(args: argparse.Namespace, assay_id: str, records: Sequence[VariantRecord]) -> list[tuple[str, str, np.ndarray]]:
    root = Path(args.public_predictions_dir)
    if not root.exists():
        return []
    assay_key = normalize_id(assay_id).lower()
    files = [path for path in root.rglob("*.csv") if assay_key in normalize_id(path.name).lower()]
    if not files:
        return []
    mutation_to_index = {record.mutation: idx for idx, record in enumerate(records)}
    rows: list[tuple[str, str, np.ndarray]] = []
    for path in files:
        try:
            df = safe_read_csv(path)
        except Exception:
            continue
        mutation_col = pick_column(df.columns, ["mutant", "mutation", "variant", "mutations"])
        if mutation_col is None:
            continue
        for col in EVOLUTIONARY_BASELINE_COLUMNS:
            if col not in df.columns or col == mutation_col:
                continue
            values = np.full(len(records), np.nan, dtype=np.float64)
            for _, row in df[[mutation_col, col]].dropna().iterrows():
                mutation = str(row[mutation_col]).strip()
                if mutation in mutation_to_index:
                    try:
                        values[mutation_to_index[mutation]] = float(row[col])
                    except (TypeError, ValueError):
                        pass
            if np.isfinite(values).sum() >= 3:
                rows.append((f"public_evolutionary:{col}", str(path), values))
    return rows


def write_prediction_csv(
    path: Path,
    records: Sequence[VariantRecord],
    splits: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    baseline: str,
    *,
    overwrite: bool = False,
) -> None:
    rows = [
        {
            "assay_id": record.assay_id,
            "sample_id": record.sample_id,
            "mutation": record.mutation,
            "position": record.position,
            "split": split,
            "label": score,
            "prediction": prediction,
            "method": baseline,
        }
        for record, split, score, prediction in zip(records, splits, y, pred)
    ]
    write_csv(path, rows, ["assay_id", "sample_id", "mutation", "position", "split", "label", "prediction", "method"], overwrite=overwrite)


def expected_prediction_keys(records: Sequence[VariantRecord]) -> list[tuple[str, str]]:
    return [(record.sample_id, record.mutation) for record in records]


def validate_prediction_csv(
    path: Path,
    records: Sequence[VariantRecord],
    splits: np.ndarray,
    *,
    expected_method: str = "",
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "reason": "file_missing"}
    if looks_like_html_error(path):
        return {"path": str(path), "status": "invalid", "reason": "empty_or_html_error"}
    required = {"assay_id", "sample_id", "mutation", "position", "split", "label", "prediction", "method"}
    try:
        df = safe_read_csv(path)
    except Exception as exc:
        return {"path": str(path), "status": "invalid", "reason": f"malformed_csv:{type(exc).__name__}:{exc}"}
    missing_columns = sorted(required - set(map(str, df.columns)))
    if missing_columns:
        return {"path": str(path), "status": "invalid", "reason": "missing_columns", "missing_columns": missing_columns}
    if len(df) != len(records):
        return {"path": str(path), "status": "invalid", "reason": "row_count_mismatch", "rows": int(len(df)), "expected_rows": len(records)}
    keys = expected_prediction_keys(records)
    expected_key_set = set(keys)
    df_keys = list(zip(df["sample_id"].astype(str), df["mutation"].astype(str)))
    if len(set(df_keys)) != len(df_keys):
        return {"path": str(path), "status": "invalid", "reason": "duplicate_sample_mutation_keys"}
    if set(df_keys) != expected_key_set:
        return {"path": str(path), "status": "invalid", "reason": "sample_or_mutation_key_mismatch"}
    by_key = {key: idx for idx, key in enumerate(df_keys)}
    pred = np.full(len(records), np.nan, dtype=np.float64)
    labels = np.full(len(records), np.nan, dtype=np.float64)
    observed_splits = []
    for idx, key in enumerate(keys):
        row = df.iloc[by_key[key]]
        pred[idx] = numeric(row["prediction"], float("nan"))
        labels[idx] = numeric(row["label"], float("nan"))
        observed_splits.append(str(row["split"]))
    if not np.isfinite(pred).all() or not np.isfinite(labels).all():
        return {"path": str(path), "status": "invalid", "reason": "nonfinite_label_or_prediction"}
    expected_labels = np.array([record.score for record in records], dtype=np.float64)
    if not np.allclose(labels, expected_labels, rtol=1e-7, atol=1e-7):
        return {"path": str(path), "status": "invalid", "reason": "label_values_do_not_match_records"}
    if list(map(str, splits.tolist())) != observed_splits:
        return {"path": str(path), "status": "invalid", "reason": "split_assignment_mismatch"}
    if expected_method:
        methods = {str(value) for value in df["method"].dropna().unique()}
        if methods and expected_method not in methods:
            return {"path": str(path), "status": "invalid", "reason": "method_name_mismatch", "methods": sorted(methods), "expected_method": expected_method}
    return {
        "path": str(path),
        "status": "valid",
        "rows": int(len(df)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_valid_prediction_values(path: Path, records: Sequence[VariantRecord], splits: np.ndarray, *, expected_method: str = "") -> tuple[np.ndarray | None, dict[str, Any]]:
    validation = validate_prediction_csv(path, records, splits, expected_method=expected_method)
    if validation.get("status") != "valid":
        return None, validation
    df = safe_read_csv(path)
    by_key = {(str(row["sample_id"]), str(row["mutation"])): numeric(row["prediction"], float("nan")) for _, row in df.iterrows()}
    pred = np.array([by_key[(record.sample_id, record.mutation)] for record in records], dtype=np.float64)
    return pred, validation


def reuse_or_write_prediction(
    args: argparse.Namespace,
    path: Path,
    records: Sequence[VariantRecord],
    splits: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    method: str,
) -> np.ndarray:
    existing, validation = load_valid_prediction_values(path, records, splits, expected_method=method)
    if existing is not None:
        return existing
    if validation.get("status") == "invalid":
        quarantine_invalid_artifact(args, path, str(validation.get("reason", "invalid_prediction")))
    write_prediction_csv(path, records, splits, y, pred, method, overwrite=True)
    return pred


def validate_feature_cache(path: Path, records: Sequence[VariantRecord]) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "reason": "cache_missing"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data:
                return {"path": str(path), "status": "invalid", "reason": "features_array_missing"}
            features = data["features"]
            if features.ndim != 2 or features.shape[0] != len(records):
                return {"path": str(path), "status": "invalid", "reason": "feature_shape_mismatch", "shape": list(features.shape), "expected_rows": len(records)}
            if not np.isfinite(features).all():
                return {"path": str(path), "status": "invalid", "reason": "nonfinite_features"}
            if "sample_ids" in data:
                sample_ids = [str(value) for value in data["sample_ids"].tolist()]
                expected = [record.sample_id for record in records]
                if sample_ids != expected:
                    return {"path": str(path), "status": "invalid", "reason": "sample_id_order_mismatch"}
    except Exception as exc:
        return {"path": str(path), "status": "invalid", "reason": f"malformed_npz:{type(exc).__name__}:{exc}"}
    return {"path": str(path), "status": "valid", "rows": len(records), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def quarantine_invalid_artifact(args: argparse.Namespace, path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    out_root = Path(args.out_root)
    try:
        rel = path.relative_to(out_root)
    except ValueError:
        rel = Path(path.name)
    timestamp = now_utc().replace(":", "").replace("+", "_")
    target = out_root / "resume_invalid_artifacts" / timestamp / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    write_json(target.with_suffix(target.suffix + ".quarantine.json"), {"created_at": now_utc(), "source_path": str(path), "quarantine_path": str(target), "reason": reason}, overwrite=True)
    return target


def metric_row(
    assay_id: str,
    split_type: str,
    baseline: str,
    family: str,
    status: str,
    records: Sequence[VariantRecord],
    y: np.ndarray,
    splits: np.ndarray,
    pred: np.ndarray,
    meta: Mapping[str, Any],
    prediction_path: Path,
    reason: str = "",
) -> dict[str, Any]:
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    global_pred = np.full(test.sum(), float(np.mean(y[train])) if train.sum() else 0.0)
    test_mse = mse_metric(y[test], pred[test]) if test.sum() else float("nan")
    return {
        "assay_id": assay_id,
        "split_type": split_type,
        "baseline": baseline,
        "baseline_family": family,
        "status": status,
        "selection_split": "val",
        "selected_alpha": meta.get("selected_alpha", ""),
        "direction_sign": meta.get("direction_sign", ""),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "val_spearman": spearman_metric(y[val], pred[val]) if val.sum() else "",
        "test_spearman": spearman_metric(y[test], pred[test]) if test.sum() else "",
        "val_mse": mse_metric(y[val], pred[val]) if val.sum() else "",
        "test_mse": test_mse,
        "mse_improvement_over_global_mean": mse_metric(y[test], global_pred) - test_mse if test.sum() else "",
        "prediction_path": str(prediction_path),
        "is_strongest_available_non_plm": False,
        "not_available_reason": reason,
    }


def strongest_baseline_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    complete = [row for row in rows if row.get("status") == "complete"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in complete:
        by_key.setdefault((row["assay_id"], row["split_type"]), []).append(row)
    result = {}
    for key, candidates in by_key.items():
        def sort_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
            val_s = float(row.get("val_spearman") or -999.0)
            val_m = float(row.get("val_mse") or 999999.0)
            return (val_s, -val_m, str(row.get("baseline")))
        result[key] = max(candidates, key=sort_key)
    return result


def strongest_rows_by_family(rows: list[dict[str, Any]], family: str) -> dict[tuple[str, str], dict[str, Any]]:
    return strongest_baseline_rows([row for row in rows if row.get("baseline_family") == family])


def snapshot_existing_artifact(out_root: Path, path: Path, category: str, reason: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    timestamp = now_utc().replace(":", "").replace("+", "_")
    target = out_root / "artifact_revisions" / category / timestamp / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    payload = {
        "created_at": now_utc(),
        "source_path": str(path),
        "snapshot_path": str(target),
        "reason": reason,
        "sha256": file_sha256(target),
        "size_bytes": target.stat().st_size,
    }
    write_json(target.with_suffix(target.suffix + ".revision.json"), payload, overwrite=True)
    return payload


def baseline_strength_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    simple = strongest_rows_by_family(rows, "simple")
    evolutionary = strongest_rows_by_family(rows, "evolutionary")
    overall = strongest_baseline_rows(rows)
    assays = sorted({row.get("assay_id", "") for row in rows if row.get("assay_id")})
    result: dict[str, Any] = {}
    for assay_id in assays:
        split_payload: dict[str, Any] = {}
        for split_type in ("random", "position_heldout"):
            key = (assay_id, split_type)
            evo_row = evolutionary.get(key)
            unavailable = [
                row
                for row in rows
                if row.get("assay_id") == assay_id
                and row.get("split_type") == split_type
                and row.get("baseline_family") == "evolutionary"
                and row.get("status") == "NOT_AVAILABLE"
            ]
            split_payload[split_type] = {
                "strongest_simple_baseline": simple.get(key, {}),
                "strongest_evolutionary_baseline": evo_row or {},
                "strongest_overall_non_PLM_baseline": overall.get(key, {}),
                "evolutionary_status": "complete" if evo_row else ("NOT_AVAILABLE" if unavailable else "missing"),
                "evolutionary_not_available_reason": unavailable[0].get("not_available_reason", "") if unavailable else "",
                "strong_baseline_evidence": "complete" if evo_row else "provisional_strongest_available_without_evolutionary",
            }
        result[assay_id] = split_payload
    return result


def write_evolutionary_baseline_summary(args: argparse.Namespace, rows: list[dict[str, Any]], resource_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_path = out_root / "protein_48h_evolutionary_baseline_report.json"
    payload = dict(resource_report or read_json(report_path) or {})
    payload.update(
        {
            "updated_at": now_utc(),
            "baseline_revision_status": "validated",
            "baseline_strength_by_assay": baseline_strength_payload(rows),
            "baseline_result_path": str(out_root / "protein_48h_baseline_results.csv"),
            "baseline_result_sha256": file_sha256(out_root / "protein_48h_baseline_results.csv") if (out_root / "protein_48h_baseline_results.csv").exists() else "",
        }
    )
    write_json(report_path, payload, overwrite=True)
    write_evolutionary_baseline_summary_md(out_root / "protein_48h_evolutionary_baseline_report.md", payload)
    return payload


def write_evolutionary_baseline_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# ProteinGym Evolutionary Baseline Report",
        "",
        f"- Status: `{payload.get('status')}`.",
        f"- Official source: `{payload.get('official_source')}`.",
        f"- Dataset version/revision: `{payload.get('dataset_version')}` / `{payload.get('dataset_revision')}`.",
        f"- Baseline revision status: `{payload.get('baseline_revision_status', 'pending')}`.",
        f"- Retained public-score files: `{len(payload.get('downloaded_and_retained_files', []))}`.",
        "",
        "## Strongest Baselines",
        "",
    ]
    for assay_id, split_rows in sorted(payload.get("baseline_strength_by_assay", {}).items()):
        pos = split_rows.get("position_heldout", {})
        simple = pos.get("strongest_simple_baseline", {})
        evo = pos.get("strongest_evolutionary_baseline", {})
        overall = pos.get("strongest_overall_non_PLM_baseline", {})
        lines.append(
            f"- `{assay_id}` position-held-out: simple `{simple.get('baseline', '')}` "
            f"evolutionary `{evo.get('baseline', 'NOT_AVAILABLE')}` overall `{overall.get('baseline', '')}` "
            f"evidence `{pos.get('strong_baseline_evidence')}`."
        )
    unavailable = payload.get("unavailable_optional_resources", [])
    if unavailable:
        lines.extend(["", "## Unavailable Resources", ""])
        for row in unavailable:
            lines.append(f"- `{row.get('assay_id', row.get('resource', 'resource'))}`: {row.get('reason')}")
    path.write_text("\n".join(lines) + "\n")


def evaluate_baselines(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    pilot_ids = list(split_payload.get("pilot_assay_ids", []))
    resource_report = read_json(out_root / "protein_48h_evolutionary_baseline_report.json")
    if not resource_report or any(resource_report.get("per_assay", {}).get(assay_id, {}).get("status") != "valid" for assay_id in pilot_ids):
        resource_report = ensure_evolutionary_baseline_resources(args)
    records_by_assay = load_valid_records(out_root)
    split_entries = split_payload.get("entries", [])
    rows: list[dict[str, Any]] = []
    pred_root = out_root / "protein_48h_baseline_predictions"
    for assay_id in split_payload.get("pilot_assay_ids", []):
        records_all = records_by_assay.get(assay_id, [])
        for split_type in ("random", "position_heldout"):
            paired = rows_for_split(records_all, split_entries, split_type)
            if not paired:
                continue
            records, y, splits = split_arrays(paired)
            for baseline in [
                "wt_residue",
                "mutant_residue",
                "wt_plus_mutant_residue",
                "mutation_position",
                "normalized_position",
                "residue_pair_plus_position",
                "blosum_substitution",
                "charge_change",
                "hydrophobicity_change",
                "residue_volume_change",
            ]:
                meta, pred = fit_ridge_baseline(records, y, splits, baseline)
                pred_path = pred_root / assay_id / split_type / f"{baseline}.csv"
                write_prediction_csv(pred_path, records, splits, y, pred, baseline)
                rows.append(metric_row(assay_id, split_type, baseline, "simple", "complete", records, y, splits, pred, meta, pred_path))
            for baseline, key_func in [
                ("site_specific_mean", lambda record: record.position),
                ("site_independent_pair_mean", lambda record: (record.wt, record.mut)),
            ]:
                meta, pred = mean_lookup_baseline(records, y, splits, key_func)
                pred_path = pred_root / assay_id / split_type / f"{baseline}.csv"
                write_prediction_csv(pred_path, records, splits, y, pred, baseline)
                rows.append(metric_row(assay_id, split_type, baseline, "simple", "complete", records, y, splits, pred, meta, pred_path))
            public_rows = load_public_prediction_table(args, assay_id, records)
            if public_rows:
                for baseline, source, raw_pred in public_rows:
                    finite = np.isfinite(raw_pred)
                    pred = np.where(finite, raw_pred, np.nanmean(raw_pred[finite]))
                    sign = direction_sign_from_validation(y[splits == "val"], pred[splits == "val"])
                    pred = apply_direction(pred, sign)
                    pred_path = pred_root / assay_id / split_type / f"{normalize_id(baseline)}.csv"
                    write_prediction_csv(pred_path, records, splits, y, pred, baseline)
                    rows.append(metric_row(assay_id, split_type, baseline, "evolutionary", "complete", records, y, splits, pred, {"direction_sign": sign, "selected_alpha": source}, pred_path))
            else:
                rows.append(
                    {
                        "assay_id": assay_id,
                        "split_type": split_type,
                        "baseline": "strongest_available_msa_or_evolutionary_prediction",
                        "baseline_family": "evolutionary",
                        "status": "NOT_AVAILABLE",
                        "selection_split": "val",
                        "selected_alpha": "",
                        "direction_sign": "",
                        "n_train": int((splits == "train").sum()),
                        "n_val": int((splits == "val").sum()),
                        "n_test": int((splits == "test").sum()),
                        "val_spearman": "",
                        "test_spearman": "",
                        "val_mse": "",
                        "test_mse": "",
                        "mse_improvement_over_global_mean": "",
                        "prediction_path": "",
                        "is_strongest_available_non_plm": False,
                        "not_available_reason": "no matching MSA/evolutionary public prediction table was found",
                    }
                )
    strongest = strongest_baseline_rows(rows)
    for row in rows:
        key = (row.get("assay_id"), row.get("split_type"))
        if key in strongest and row.get("baseline") == strongest[key].get("baseline"):
            row["is_strongest_available_non_plm"] = True
    baseline_path = out_root / "protein_48h_baseline_results.csv"
    if baseline_path.exists():
        snapshot_existing_artifact(out_root, baseline_path, "baseline_results", "baseline_recomputed_after_evolutionary_resource_validation")
    write_csv(baseline_path, rows, BASELINE_FIELDS, overwrite=True)
    write_baseline_report(out_root, rows)
    write_evolutionary_baseline_summary(args, rows, resource_report)
    return {"rows": rows, "strongest": strongest}


def write_baseline_report(out_root: Path, rows: list[dict[str, Any]]) -> None:
    strongest = strongest_baseline_rows(rows)
    lines = ["# ProteinGym Strong Baseline Report", ""]
    if not rows:
        lines.append("No pilot assays were available, so no baselines were run.")
    for (assay_id, split_type), row in sorted(strongest.items()):
        lines.append(
            f"- `{assay_id}` `{split_type}` strongest available non-PLM baseline: "
            f"`{row['baseline']}` val Spearman `{row.get('val_spearman')}` test Spearman `{row.get('test_spearman')}`."
        )
    unavailable = [row for row in rows if row.get("status") == "NOT_AVAILABLE"]
    if unavailable:
        lines.extend(["", "## Not Available", ""])
        for row in unavailable:
            lines.append(f"- `{row['assay_id']}` `{row['split_type']}` `{row['baseline']}`: {row.get('not_available_reason')}")
    path = out_root / "protein_48h_baseline_report.md"
    path.write_text("\n".join(lines) + "\n")


def load_predictions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def bootstrap_delta_by_position(
    model_pred_path: str | Path,
    baseline_pred_path: str | Path,
    split: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not model_pred_path or not baseline_pred_path or not Path(model_pred_path).exists() or not Path(baseline_pred_path).exists():
        return None, None
    model = load_predictions(model_pred_path)
    baseline = load_predictions(baseline_pred_path)
    merged = model.merge(
        baseline[["sample_id", "mutation", "split", "prediction"]].rename(columns={"prediction": "baseline_prediction"}),
        on=["sample_id", "mutation", "split"],
        how="inner",
    )
    merged = merged[merged["split"] == split]
    if merged.empty:
        return None, None
    positions = sorted(merged["position"].unique().tolist())
    if len(positions) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(max(1, n_bootstrap)):
        sampled = rng.choice(positions, size=len(positions), replace=True)
        boot = pd.concat([merged[merged["position"] == pos] for pos in sampled], ignore_index=True)
        y = boot["label"].to_numpy(dtype=float)
        model_s = spearman_metric(y, boot["prediction"].to_numpy(dtype=float))
        base_s = spearman_metric(y, boot["baseline_prediction"].to_numpy(dtype=float))
        if math.isfinite(model_s) and math.isfinite(base_s):
            deltas.append(model_s - base_s)
    if not deltas:
        return None, None
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def model_hash_for_args(args: argparse.Namespace) -> str:
    model = str(args.esm2_model)
    path = Path(model)
    if path.exists():
        if path.is_file():
            return file_sha256(path)
        files = sorted(item for item in path.rglob("*") if item.is_file() and item.name in {"config.json", "pytorch_model.bin", "model.safetensors"})
        return stable_hash([(str(item.relative_to(path)), file_sha256(item)) for item in files]) if files else stable_hash({"path": str(path)})
    return stable_hash({"hf_model_id": model})


def mock_signal(record: VariantRecord) -> float:
    left = record.wt_sequence[max(0, record.position - 4) : record.position - 1]
    right = record.wt_sequence[record.position : min(len(record.wt_sequence), record.position + 3)]
    context = sum((AA.index(ch) + 1) * (idx + 1) for idx, ch in enumerate(left + right) if ch in AA_SET)
    mut_term = AA.index(record.mut) - AA.index(record.wt)
    return float(0.07 * context + 0.5 * mut_term)


def fit_readout_features(features: np.ndarray, y: np.ndarray, splits: np.ndarray, readout: str) -> tuple[dict[str, Any], np.ndarray]:
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train])
    x_val = scaler.transform(features[val])
    x_test = scaler.transform(features[test])
    if readout == "linear_regression":
        model = LinearRegression()
        model.fit(x_train, y[train])
        selected_alpha = ""
    else:
        best_model = None
        selected_alpha = None
        best_score = -float("inf")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha)
            model.fit(x_train, y[train])
            score = spearman_metric(y[val], model.predict(x_val))
            score = score if math.isfinite(score) else -float("inf")
            if score > best_score:
                best_model = model
                selected_alpha = alpha
                best_score = score
        model = best_model or Ridge(alpha=1.0).fit(x_train, y[train])
        selected_alpha = selected_alpha or 1.0
    pred = np.full_like(y, fill_value=float(np.mean(y[train])), dtype=np.float64)
    pred[train] = model.predict(x_train)
    pred[val] = model.predict(x_val)
    pred[test] = model.predict(x_test)
    sign = direction_sign_from_validation(y[val], pred[val])
    return {"selected_alpha": selected_alpha, "direction_sign": sign}, apply_direction(pred, sign)


def mock_features(records: Sequence[VariantRecord], rep_type: str) -> np.ndarray:
    signals = np.array([mock_signal(record) for record in records], dtype=np.float64)
    positions = np.array([record.position / max(len(record.wt_sequence), 1) for record in records], dtype=np.float64)
    mut = np.array([AA.index(record.mut) for record in records], dtype=np.float64)
    wt = np.array([AA.index(record.wt) for record in records], dtype=np.float64)
    if rep_type == "whole_sequence_mean":
        return np.stack([0.15 * signals + positions, positions, mut * 0.02], axis=1)
    if rep_type == "wt_mutant_position_difference":
        return np.stack([signals, mut - wt, positions], axis=1)
    if rep_type == "local_window":
        return np.stack([signals, signals**2 * 0.01, mut - wt, positions], axis=1)
    return np.stack([signals, mut, wt, positions], axis=1)


def baseline_lookup(out_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(csv.DictReader((out_root / "protein_48h_baseline_results.csv").open())) if (out_root / "protein_48h_baseline_results.csv").exists() else []
    return strongest_baseline_rows(rows)


def process_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def esm2_expected_prediction_specs(args: argparse.Namespace, assay_id: str, split_type: str) -> list[dict[str, Any]]:
    pred_root = Path(args.out_root) / "protein_48h_esm2_pilot_predictions"
    specs = [
        {
            "method": "zero_shot",
            "readout": "wild_type_relative_masked_log_odds",
            "representation_type": "masked_log_odds",
            "path": pred_root / assay_id / split_type / "zero_shot.csv",
            "cache_key": stable_hash(["mock", assay_id, split_type, "zero"]) if args.mock_esm2 else "",
        }
    ]
    for rep_type in ESM2_REPRESENTATION_TYPES:
        for readout in ESM2_READOUTS:
            specs.append(
                {
                    "method": "frozen_representation",
                    "readout": readout,
                    "representation_type": rep_type,
                    "path": pred_root / assay_id / split_type / f"{rep_type}_{readout}.csv",
                    "cache_key": "",
                }
            )
    return specs


def esm2_cache_key(args: argparse.Namespace, assay_id: str, split_type: str, rep_type: str, records: Sequence[VariantRecord], split_hash: str) -> str:
    if args.mock_esm2:
        return stable_hash(["mock", assay_id, split_type, rep_type, args.esm2_layer])
    model_hash = model_hash_for_args(args)
    return stable_hash(
        {
            "model_hash": model_hash,
            "assay_id": assay_id,
            "sequence_hash": stable_hash([record.wt_sequence for record in records]),
            "split_manifest_hash": split_hash,
            "layer": args.esm2_layer,
            "representation_type": rep_type,
            "preprocessing_policy": f"length<={args.esm2_max_length};local_window={args.local_window}",
        }
    )


def lora_run_id(stage: str, assay_id: str, rank: int, lr: float, seed: int) -> str:
    return f"{stage}_{assay_id}_r{rank}_lr{lr:g}_seed{seed}".replace(".", "p")


def validate_lora_artifacts(args: argparse.Namespace, assay_id: str, rank: int, lr: float, seed: int, stage: str) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    paired = rows_for_split(records_by_assay.get(assay_id, []), split_payload.get("entries", []), "position_heldout")
    if not paired:
        return {"status": "missing", "reason": "split_records_missing"}
    records, _y, splits = split_arrays(paired)
    run_id = lora_run_id(stage, assay_id, rank, lr, seed)
    pred_path = out_root / "protein_48h_lora_predictions" / assay_id / f"{run_id}.csv"
    run_dir = out_root / "lora_runs" / assay_id / run_id
    pred, pred_validation = load_valid_prediction_values(pred_path, records, splits)
    adapter_path = run_dir / "adapter.safetensors"
    head_path = run_dir / "head.safetensors"
    evidence_path = run_dir / "checkpoint_selection_evidence.json"
    if pred is None:
        return {"status": "missing" if pred_validation.get("status") == "missing" else "invalid", "run_id": run_id, "prediction_validation": pred_validation}
    missing = [str(path) for path in (adapter_path, head_path, evidence_path) if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {"status": "invalid", "run_id": run_id, "reason": "checkpoint_artifact_missing_or_empty", "missing": missing}
    evidence = read_json(evidence_path)
    if not evidence.get("selection_metric") or not evidence.get("adapter_path") or not evidence.get("head_path"):
        return {"status": "invalid", "run_id": run_id, "reason": "checkpoint_selection_evidence_incomplete", "evidence_path": str(evidence_path)}
    return {
        "status": "valid",
        "run_id": run_id,
        "prediction_path": str(pred_path),
        "adapter_path": str(adapter_path),
        "head_path": str(head_path),
        "evidence_path": str(evidence_path),
        "prediction_sha256": file_sha256(pred_path),
        "adapter_hash": file_sha256(adapter_path),
        "head_hash": file_sha256(head_path),
        "evidence": evidence,
    }


def validate_resume_state(args: argparse.Namespace, registry: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    split_hash = split_payload.get("manifest_hash", stable_hash(split_payload))
    valid_predictions: list[dict[str, Any]] = []
    invalid_predictions: list[dict[str, Any]] = []
    missing_predictions: list[dict[str, Any]] = []
    valid_caches: list[dict[str, Any]] = []
    invalid_caches: list[dict[str, Any]] = []
    missing_caches: list[dict[str, Any]] = []
    for assay_id in split_payload.get("pilot_assay_ids", []):
        records_all = records_by_assay.get(assay_id, [])
        for split_type in ("random", "position_heldout"):
            paired = rows_for_split(records_all, split_payload.get("entries", []), split_type)
            if not paired:
                continue
            records, _y, splits = split_arrays(paired)
            for spec in esm2_expected_prediction_specs(args, assay_id, split_type):
                path = Path(spec["path"])
                validation = validate_prediction_csv(path, records, splits)
                row = {
                    "assay_id": assay_id,
                    "split_type": split_type,
                    "method": spec["method"],
                    "readout": spec["readout"],
                    "representation_type": spec["representation_type"],
                    **validation,
                }
                if validation.get("status") == "valid":
                    valid_predictions.append(row)
                elif validation.get("status") == "missing":
                    missing_predictions.append(row)
                else:
                    invalid_predictions.append(row)
            for rep_type in ESM2_REPRESENTATION_TYPES:
                cache_key = esm2_cache_key(args, assay_id, split_type, rep_type, records, split_hash)
                cache_path = out_root / "caches/esm2_frozen" / f"{cache_key}.npz"
                validation = validate_feature_cache(cache_path, records)
                row = {"assay_id": assay_id, "split_type": split_type, "representation_type": rep_type, "cache_key": cache_key, **validation}
                if validation.get("status") == "valid":
                    valid_caches.append(row)
                elif validation.get("status") == "missing":
                    missing_caches.append(row)
                else:
                    invalid_caches.append(row)

    stale = []
    for task in registry.get("tasks", []):
        if task.get("status") == "running" and not process_alive(task.get("pid")):
            stale.append(dict(task))
            task["status"] = "failed"
            task["completed_at"] = now_utc()
            task["stale_running_reconciled_at"] = now_utc()
            task["stale_running_reconciliation_reason"] = "recorded PID is no longer active after pause/resume validation"
    if stale:
        registry.setdefault("stale_running_reconciliations", []).extend(stale)
        save_registry(args, registry)

    gate = read_json(out_root / "protein_48h_advancement_gate.json")
    lora_candidates = gate.get("lora_candidate_order", [])[: args.max_lora_assays]
    lora_valid = []
    lora_missing_or_invalid = []
    for assay_id in lora_candidates:
        for rank in (8, 16):
            for lr in (1e-5, 5e-5):
                row = validate_lora_artifacts(args, assay_id, rank, lr, args.calibration_seed, "calibration")
                (lora_valid if row.get("status") == "valid" else lora_missing_or_invalid).append({"assay_id": assay_id, "stage": "calibration", "rank": rank, "learning_rate": lr, "seed": args.calibration_seed, **row})
        for seed in [int(part) for part in str(args.formal_seeds).split(",") if part.strip()]:
            for rank in (8, 16):
                for lr in (1e-5, 5e-5):
                    row = validate_lora_artifacts(args, assay_id, rank, lr, seed, "formal")
                    if row.get("status") == "valid":
                        lora_valid.append({"assay_id": assay_id, "stage": "formal", "rank": rank, "learning_rate": lr, "seed": seed, **row})

    payload = {
        "created_at": now_utc(),
        "out_root": str(out_root),
        "status": "valid" if not invalid_predictions and not invalid_caches else "requires_rerun",
        "expected_esm2_prediction_outputs": len(valid_predictions) + len(invalid_predictions) + len(missing_predictions),
        "valid_reused_esm2_prediction_outputs": len(valid_predictions),
        "invalid_esm2_prediction_outputs": len(invalid_predictions),
        "missing_esm2_prediction_outputs": len(missing_predictions),
        "valid_reused_frozen_caches": len(valid_caches),
        "invalid_frozen_caches": len(invalid_caches),
        "missing_frozen_caches": len(missing_caches),
        "valid_reused_artifacts": valid_predictions[:500],
        "invalid_or_incomplete_artifacts": [*invalid_predictions, *invalid_caches][:500],
        "outputs_scheduled_for_rerun": [*invalid_predictions, *missing_predictions, *invalid_caches, *missing_caches][:500],
        "cache_hits": valid_caches[:500],
        "registry_corrections": stale,
        "lora_valid_artifacts": lora_valid[:200],
        "lora_missing_or_invalid_artifacts": lora_missing_or_invalid[:200],
        "evidence_completed_suites_not_recomputed": "valid prediction files and cache hashes are recorded before baseline-dependent aggregate metrics are refreshed",
    }
    write_json(out_root / "protein_48h_resume_validation.json", payload, overwrite=True)
    write_resume_validation_md(out_root / "protein_48h_resume_validation.md", payload)
    return payload


def write_resume_validation_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# ProteinGym Resume Validation Report",
        "",
        f"- Status: `{payload.get('status')}`.",
        f"- Expected ESM2 prediction outputs: `{payload.get('expected_esm2_prediction_outputs')}`.",
        f"- Valid reused ESM2 predictions: `{payload.get('valid_reused_esm2_prediction_outputs')}`.",
        f"- Invalid ESM2 predictions: `{payload.get('invalid_esm2_prediction_outputs')}`.",
        f"- Missing ESM2 predictions: `{payload.get('missing_esm2_prediction_outputs')}`.",
        f"- Valid frozen cache hits: `{payload.get('valid_reused_frozen_caches')}`.",
        f"- Invalid frozen caches: `{payload.get('invalid_frozen_caches')}`.",
        f"- Missing frozen caches: `{payload.get('missing_frozen_caches')}`.",
        f"- Registry stale-running corrections: `{len(payload.get('registry_corrections', []))}`.",
        f"- Valid LoRA artifacts discovered: `{len(payload.get('lora_valid_artifacts', []))}`.",
        "",
        payload.get("evidence_completed_suites_not_recomputed", ""),
    ]
    path.write_text("\n".join(lines) + "\n")


def esm2_metric_row(
    args: argparse.Namespace,
    assay_id: str,
    split_type: str,
    method: str,
    readout: str,
    records: Sequence[VariantRecord],
    y: np.ndarray,
    splits: np.ndarray,
    pred: np.ndarray,
    meta: Mapping[str, Any],
    prediction_path: Path,
    cache_key: str,
    strongest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    val = splits == "val"
    test = splits == "test"
    baseline_val = float(strongest.get("val_spearman") or "nan") if strongest else float("nan")
    baseline_test = float(strongest.get("test_spearman") or "nan") if strongest else float("nan")
    ci_low, ci_high = (None, None)
    if strongest and strongest.get("prediction_path"):
        ci_low, ci_high = bootstrap_delta_by_position(prediction_path, strongest["prediction_path"], "val", args.n_bootstrap, args.bootstrap_seed)
    val_s = spearman_metric(y[val], pred[val]) if val.sum() else float("nan")
    test_s = spearman_metric(y[test], pred[test]) if test.sum() else float("nan")
    return {
        "assay_id": assay_id,
        "split_type": split_type,
        "method": method,
        "readout": readout,
        "status": "complete",
        "model_name": args.esm2_model if not args.mock_esm2 else "mock_esm2_150m",
        "model_hash": model_hash_for_args(args) if not args.mock_esm2 else stable_hash("mock_esm2_150m"),
        "layer": args.esm2_layer,
        "representation_type": meta.get("representation_type", ""),
        "selection_split": "val",
        "selected_alpha": meta.get("selected_alpha", ""),
        "n_train": int((splits == "train").sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "val_spearman": val_s,
        "test_spearman": test_s,
        "val_mse": mse_metric(y[val], pred[val]) if val.sum() else "",
        "test_mse": mse_metric(y[test], pred[test]) if test.sum() else "",
        "strongest_non_plm_baseline": strongest.get("baseline", "") if strongest else "",
        "baseline_val_spearman": baseline_val if math.isfinite(baseline_val) else "",
        "baseline_test_spearman": baseline_test if math.isfinite(baseline_test) else "",
        "val_excess": val_s - baseline_val if math.isfinite(val_s) and math.isfinite(baseline_val) else "",
        "test_excess": test_s - baseline_test if math.isfinite(test_s) and math.isfinite(baseline_test) else "",
        "position_bootstrap_ci_low": ci_low,
        "position_bootstrap_ci_high": ci_high,
        "prediction_path": str(prediction_path),
        "cache_key": cache_key,
        "not_available_reason": "",
    }


def evaluate_mock_esm2(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    strongest = baseline_lookup(out_root)
    rows: list[dict[str, Any]] = []
    pred_root = out_root / "protein_48h_esm2_pilot_predictions"
    for assay_id in split_payload.get("pilot_assay_ids", []):
        records_all = records_by_assay.get(assay_id, [])
        for split_type in ("random", "position_heldout"):
            paired = rows_for_split(records_all, split_payload.get("entries", []), split_type)
            if not paired:
                continue
            records, y, splits = split_arrays(paired)
            key = (assay_id, split_type)
            base = strongest.get(key)
            zero_pred = np.array([mock_signal(record) for record in records], dtype=np.float64)
            sign = direction_sign_from_validation(y[splits == "val"], zero_pred[splits == "val"])
            zero_pred = apply_direction(zero_pred, sign)
            pred_path = pred_root / assay_id / split_type / "zero_shot.csv"
            zero_pred = reuse_or_write_prediction(args, pred_path, records, splits, y, zero_pred, "zero_shot")
            rows.append(esm2_metric_row(args, assay_id, split_type, "zero_shot", "wild_type_relative_masked_log_odds", records, y, splits, zero_pred, {"selected_alpha": "", "representation_type": "masked_log_odds"}, pred_path, stable_hash(["mock", assay_id, split_type, "zero"]), base))
            for rep_type in ESM2_REPRESENTATION_TYPES:
                features = mock_features(records, rep_type)
                cache_key = stable_hash(["mock", assay_id, split_type, rep_type, args.esm2_layer])
                cache_path = out_root / "caches/esm2_frozen" / f"{cache_key}.npz"
                cache_validation = validate_feature_cache(cache_path, records)
                if cache_validation.get("status") == "invalid":
                    quarantine_invalid_artifact(args, cache_path, str(cache_validation.get("reason", "invalid_mock_feature_cache")))
                if cache_validation.get("status") != "valid":
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_cache = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                    with tmp_cache.open("wb") as handle:
                        np.savez_compressed(handle, features=features.astype(np.float32), sample_ids=np.array([r.sample_id for r in records]))
                    os.replace(tmp_cache, cache_path)
                for readout in ESM2_READOUTS:
                    meta, pred = fit_readout_features(features, y, splits, readout)
                    meta["representation_type"] = rep_type
                    pred_path = pred_root / assay_id / split_type / f"{rep_type}_{readout}.csv"
                    pred = reuse_or_write_prediction(args, pred_path, records, splits, y, pred, f"{rep_type}_{readout}")
                    rows.append(esm2_metric_row(args, assay_id, split_type, "frozen_representation", readout, records, y, splits, pred, meta, pred_path, cache_key, base))
    return rows


def load_transformers_model(args: argparse.Namespace, masked: bool = False) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    cache_dir = Path(args.out_root) / "shared_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model, cache_dir=str(cache_dir), local_files_only=args.local_files_only)
    model_cls = AutoModelForMaskedLM if masked else AutoModel
    model = model_cls.from_pretrained(args.esm2_model, cache_dir=str(cache_dir), local_files_only=args.local_files_only)
    return tokenizer, model


def esm_token_index(position: int) -> int:
    return int(position)


def run_zero_shot_esm2(args: argparse.Namespace, records: Sequence[VariantRecord], splits: np.ndarray) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is unavailable")
    tokenizer, model = load_transformers_model(args, masked=True)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    preds = []
    batch_size = max(1, int(args.esm2_batch_size))
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            masked_sequences = []
            token_positions = []
            wt_ids = []
            mut_ids = []
            for record in batch:
                chars = list(record.wt_sequence)
                chars[record.position - 1] = tokenizer.mask_token
                masked_sequences.append("".join(chars))
                token_positions.append(esm_token_index(record.position))
                wt_ids.append(tokenizer.convert_tokens_to_ids(record.wt))
                mut_ids.append(tokenizer.convert_tokens_to_ids(record.mut))
            encoded = tokenizer(masked_sequences, return_tensors="pt", padding=True, truncation=True, max_length=args.esm2_max_length + 2)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float().log_softmax(dim=-1)
            for row_idx, tok_idx in enumerate(token_positions):
                preds.append(float(logits[row_idx, tok_idx, mut_ids[row_idx]] - logits[row_idx, tok_idx, wt_ids[row_idx]]))
    return np.array(preds, dtype=np.float64)


def extract_real_esm2_features(args: argparse.Namespace, records: Sequence[VariantRecord], rep_type: str, cache_key: str) -> np.ndarray:
    cache_path = Path(args.out_root) / "caches/esm2_frozen" / f"{cache_key}.npz"
    if cache_path.exists():
        validation = validate_feature_cache(cache_path, records)
        if validation.get("status") == "valid":
            with np.load(cache_path, allow_pickle=False) as data:
                return data["features"].astype(np.float32)
        quarantine_invalid_artifact(args, cache_path, str(validation.get("reason", "invalid_feature_cache")))
    if torch is None:
        raise RuntimeError("torch is unavailable")
    tokenizer, model = load_transformers_model(args, masked=False)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    features = []
    batch_size = max(1, int(args.esm2_batch_size))
    window = int(args.local_window)
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            sequences = []
            for record in batch:
                if rep_type in {"wt_mutant_position_difference"}:
                    sequences.extend([record.wt_sequence, record.mutated_sequence])
                elif rep_type in {"mutation_position"}:
                    sequences.append(record.mutated_sequence)
                else:
                    sequences.append(record.mutated_sequence)
            encoded = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=args.esm2_max_length + 2)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[int(args.esm2_layer)].float()
            cursor = 0
            for record in batch:
                tok = esm_token_index(record.position)
                if rep_type == "wt_mutant_position_difference":
                    wt_vec = hidden[cursor, tok].detach().cpu().numpy()
                    mut_vec = hidden[cursor + 1, tok].detach().cpu().numpy()
                    features.append(mut_vec - wt_vec)
                    cursor += 2
                elif rep_type == "local_window":
                    lo = max(1, tok - window)
                    hi = min(hidden.shape[1] - 1, tok + window + 1)
                    features.append(hidden[cursor, lo:hi].mean(dim=0).detach().cpu().numpy())
                    cursor += 1
                elif rep_type == "whole_sequence_mean":
                    attention = encoded["attention_mask"][cursor].bool()
                    valid_hidden = hidden[cursor, attention]
                    if valid_hidden.shape[0] > 2:
                        valid_hidden = valid_hidden[1:-1]
                    features.append(valid_hidden.mean(dim=0).detach().cpu().numpy())
                    cursor += 1
                else:
                    features.append(hidden[cursor, tok].detach().cpu().numpy())
                    cursor += 1
    arr = np.asarray(features, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_cache.open("wb") as handle:
        np.savez_compressed(handle, features=arr, sample_ids=np.array([record.sample_id for record in records]))
    os.replace(tmp_cache, cache_path)
    return arr


def evaluate_real_esm2(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    split_hash = split_payload.get("manifest_hash", stable_hash(split_payload))
    strongest = baseline_lookup(out_root)
    previous_metric_path = out_root / "protein_48h_esm2_pilot_metrics.csv"
    previous_rows = {str(row.get("prediction_path")): row for row in csv.DictReader(previous_metric_path.open())} if previous_metric_path.exists() else {}
    rows: list[dict[str, Any]] = []
    pred_root = out_root / "protein_48h_esm2_pilot_predictions"
    model_hash = model_hash_for_args(args)
    for assay_id in split_payload.get("pilot_assay_ids", []):
        records_all = records_by_assay.get(assay_id, [])
        for split_type in ("random", "position_heldout"):
            paired = rows_for_split(records_all, split_payload.get("entries", []), split_type)
            if not paired:
                continue
            records, y, splits = split_arrays(paired)
            key = (assay_id, split_type)
            base = strongest.get(key)
            try:
                pred_path = pred_root / assay_id / split_type / "zero_shot.csv"
                zero_pred, validation = load_valid_prediction_values(pred_path, records, splits, expected_method="zero_shot")
                if zero_pred is None:
                    if validation.get("status") == "invalid":
                        quarantine_invalid_artifact(args, pred_path, str(validation.get("reason", "invalid_zero_shot_prediction")))
                    zero_pred = run_zero_shot_esm2(args, records, splits)
                    sign = direction_sign_from_validation(y[splits == "val"], zero_pred[splits == "val"])
                    zero_pred = apply_direction(zero_pred, sign)
                    write_prediction_csv(pred_path, records, splits, y, zero_pred, "zero_shot", overwrite=True)
                rows.append(esm2_metric_row(args, assay_id, split_type, "zero_shot", "wild_type_relative_masked_log_odds", records, y, splits, zero_pred, {"selected_alpha": "", "representation_type": "masked_log_odds"}, pred_path, stable_hash([model_hash, assay_id, split_hash, split_type, "zero"]), base))
            except Exception as exc:
                rows.append(unavailable_esm2_row(args, assay_id, split_type, "zero_shot", str(exc), base))
            for rep_type in ESM2_REPRESENTATION_TYPES:
                cache_key = esm2_cache_key(args, assay_id, split_type, rep_type, records, split_hash)
                cached_readouts: dict[str, np.ndarray] = {}
                missing_readouts: list[str] = []
                for readout in ESM2_READOUTS:
                    pred_path = pred_root / assay_id / split_type / f"{rep_type}_{readout}.csv"
                    pred, validation = load_valid_prediction_values(pred_path, records, splits, expected_method=f"{rep_type}_{readout}")
                    if pred is None:
                        missing_readouts.append(readout)
                        if validation.get("status") == "invalid":
                            quarantine_invalid_artifact(args, pred_path, str(validation.get("reason", "invalid_esm2_prediction")))
                    else:
                        cached_readouts[readout] = pred
                        old = previous_rows.get(str(pred_path), {})
                        meta = {
                            "selected_alpha": old.get("selected_alpha", ""),
                            "representation_type": rep_type,
                        }
                        rows.append(esm2_metric_row(args, assay_id, split_type, "frozen_representation", readout, records, y, splits, pred, meta, pred_path, cache_key, base))
                if not missing_readouts:
                    continue
                try:
                    features = extract_real_esm2_features(args, records, rep_type, cache_key)
                    for readout in missing_readouts:
                        meta, pred = fit_readout_features(features, y, splits, readout)
                        meta["representation_type"] = rep_type
                        pred_path = pred_root / assay_id / split_type / f"{rep_type}_{readout}.csv"
                        write_prediction_csv(pred_path, records, splits, y, pred, f"{rep_type}_{readout}", overwrite=True)
                        rows.append(esm2_metric_row(args, assay_id, split_type, "frozen_representation", readout, records, y, splits, pred, meta, pred_path, cache_key, base))
                except Exception as exc:
                    rows.append(unavailable_esm2_row(args, assay_id, split_type, f"frozen_{rep_type}", str(exc), base))
    return rows


def unavailable_esm2_row(args: argparse.Namespace, assay_id: str, split_type: str, method: str, reason: str, strongest: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "assay_id": assay_id,
        "split_type": split_type,
        "method": method,
        "readout": "",
        "status": "NOT_AVAILABLE",
        "model_name": args.esm2_model,
        "model_hash": model_hash_for_args(args),
        "layer": args.esm2_layer,
        "representation_type": "",
        "selection_split": "val",
        "selected_alpha": "",
        "n_train": "",
        "n_val": "",
        "n_test": "",
        "val_spearman": "",
        "test_spearman": "",
        "val_mse": "",
        "test_mse": "",
        "strongest_non_plm_baseline": strongest.get("baseline", "") if strongest else "",
        "baseline_val_spearman": strongest.get("val_spearman", "") if strongest else "",
        "baseline_test_spearman": strongest.get("test_spearman", "") if strongest else "",
        "val_excess": "",
        "test_excess": "",
        "position_bootstrap_ci_low": "",
        "position_bootstrap_ci_high": "",
        "prediction_path": "",
        "cache_key": "",
        "not_available_reason": reason,
    }


def run_esm2_pilot(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    rows: list[dict[str, Any]]
    if not split_payload.get("pilot_assay_ids"):
        rows = []
    elif args.mock_esm2:
        rows = evaluate_mock_esm2(args)
    else:
        rows = evaluate_real_esm2(args)
    (out_root / "protein_48h_esm2_pilot_predictions").mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "protein_48h_esm2_pilot_metrics.csv"
    if metrics_path.exists():
        snapshot_existing_artifact(out_root, metrics_path, "esm2_pilot_metrics", "esm2_metrics_recomputed_after_baseline_revision_without_recomputing_valid_predictions")
    write_csv(metrics_path, rows, ESM2_FIELDS, overwrite=True)
    return {"rows": rows}


def numeric(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def gate_and_rank(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    inventory = list(csv.DictReader((out_root / "protein_48h_candidate_ranking.csv").open())) if (out_root / "protein_48h_candidate_ranking.csv").exists() else []
    audits = read_json(out_root / "protein_48h_split_audit.json").get("assays", {})
    baseline_rows = list(csv.DictReader((out_root / "protein_48h_baseline_results.csv").open())) if (out_root / "protein_48h_baseline_results.csv").exists() else []
    esm_rows = list(csv.DictReader((out_root / "protein_48h_esm2_pilot_metrics.csv").open())) if (out_root / "protein_48h_esm2_pilot_metrics.csv").exists() else []
    strongest = strongest_baseline_rows(baseline_rows)
    quality = {row["assay_id"]: numeric(row.get("data_quality_score"), 0.0) for row in inventory}
    valid_count = {row["assay_id"]: int(numeric(row.get("valid_single_substitution_count"), 0.0)) for row in inventory}
    decisions = []
    for assay_id in [row["assay_id"] for row in inventory if str(row.get("selected_pilot", "")).lower() == "true"]:
        labels = []
        reasons = []
        position_audit = audits.get(assay_id, {}).get("position_heldout", {})
        if position_audit.get("status") != "valid":
            labels.append("INVALID_POSITION_HELD_OUT_SPLIT")
            reasons.append("position-held-out split failed integrity or label-distribution audit")
        if position_audit.get("leakage_detected"):
            labels.append("DATA_LEAKAGE_DETECTED")
            reasons.append("position or duplicate-mutation leakage detected")
        strong = strongest.get((assay_id, "position_heldout"))
        if not strong:
            labels.append("NOT_AVAILABLE")
            reasons.append("no complete strongest non-PLM baseline was available")
        assay_esm = [row for row in esm_rows if row.get("assay_id") == assay_id and row.get("split_type") == "position_heldout" and row.get("status") == "complete"]
        random_esm = [row for row in esm_rows if row.get("assay_id") == assay_id and row.get("split_type") == "random" and row.get("status") == "complete"]
        if not assay_esm:
            labels.append("NOT_AVAILABLE")
            reasons.append("no complete ESM2 zero-shot or frozen representation result was available")
        best = max(assay_esm, key=lambda row: (numeric(row.get("val_excess"), -999.0), numeric(row.get("val_spearman"), -999.0)), default=None)
        best_random = max(random_esm, key=lambda row: numeric(row.get("val_excess"), -999.0), default=None)
        best_val_excess = numeric(best.get("val_excess") if best else "", -999.0)
        best_test_excess = numeric(best.get("test_excess") if best else "", -999.0)
        ci_low = numeric(best.get("position_bootstrap_ci_low") if best else "", float("nan"))
        provisional = best_val_excess > 0 and (not math.isfinite(ci_low) or ci_low <= 0)
        passes = bool(
            position_audit.get("status") == "valid"
            and strong
            and best is not None
            and best_val_excess > 0
            and (best_val_excess >= args.frozen_excess_threshold or provisional)
        )
        if best is not None and best_val_excess > 0:
            labels.append("FROZEN_PLM_SIGNAL")
            if provisional:
                reasons.append("frozen ESM2 has a positive validation trend but bootstrap support is provisional")
        else:
            family = strong.get("baseline_family", "simple") if strong else "simple"
            labels.append("SATURATED_BY_EVOLUTIONARY_BASELINE" if family == "evolutionary" else "SATURATED_BY_SIMPLE_BASELINE")
            reasons.append("no ESM2 pilot method exceeded the strongest available non-PLM baseline on validation")
        if best_random is not None and numeric(best_random.get("val_excess"), -999.0) > 0 and best_val_excess <= 0:
            labels.append("RANDOM_SPLIT_ONLY")
            reasons.append("positive excess on random split did not transfer to position-held-out validation")
        if not passes:
            labels.append("INSUFFICIENT_EVIDENCE")
        decisions.append(
            {
                "assay_id": assay_id,
                "labels": sorted(set(labels), key=lambda label: sorted(RESULT_LABELS).index(label)),
                "advance_to_lora": passes,
                "provisional": provisional,
                "best_model_method": best.get("method", "") if best else "",
                "best_model_readout": best.get("readout", "") if best else "",
                "best_model_val_spearman": best.get("val_spearman", "") if best else "",
                "best_model_test_spearman": best.get("test_spearman", "") if best else "",
                "best_model_val_excess": best.get("val_excess", "") if best else "",
                "best_model_test_excess": best.get("test_excess", "") if best else "",
                "strongest_baseline": strong.get("baseline", "") if strong else "",
                "strongest_baseline_family": strong.get("baseline_family", "") if strong else "",
                "baseline_val_spearman": strong.get("val_spearman", "") if strong else "",
                "baseline_test_spearman": strong.get("test_spearman", "") if strong else "",
                "position_bootstrap_ci_low": best.get("position_bootstrap_ci_low", "") if best else "",
                "position_bootstrap_ci_high": best.get("position_bootstrap_ci_high", "") if best else "",
                "data_quality_score": quality.get(assay_id, 0.0),
                "valid_single_substitution_count": valid_count.get(assay_id, 0),
                "decision_reason": "; ".join(dict.fromkeys(reasons)),
                "supporting_artifact_path": best.get("prediction_path", "") if best else "",
            }
        )
    decisions = sorted(
        decisions,
        key=lambda row: (
            not row["advance_to_lora"],
            -numeric(row.get("best_model_val_excess"), -999.0),
            -numeric(row.get("best_model_val_spearman"), -999.0),
            -row.get("valid_single_substitution_count", 0),
            row["assay_id"],
        ),
    )
    lora_candidates = [row["assay_id"] for row in decisions if row["advance_to_lora"]][: args.max_lora_assays]
    payload = {
        "created_at": now_utc(),
        "selection_split": "val",
        "test_used_for_advancement": False,
        "ranking_rule": "validation excess, validation Spearman, valid sample count, assay_id",
        "lora_candidate_order": lora_candidates,
        "decisions": decisions,
    }
    write_json(out_root / "protein_48h_advancement_gate.json", payload, overwrite=True)
    return payload


class Esm2PositionRegressor(nn.Module if nn is not None else object):
    def __init__(self, base_model: Any, hidden_dim: int):
        super().__init__()
        self.base_model = base_model
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: Any, attention_mask: Any, positions: Any) -> Any:
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch, positions]
        return self.head(pooled.float()).squeeze(-1)


def replace_child(parent: Any, child_name: str, module: Any) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora_named_modules(model: Any, target_terms: Sequence[str], rank: int, alpha: int, dropout: float) -> tuple[list[Any], list[str]]:
    freeze_all(model)
    terms = [term.strip() for term in target_terms if term.strip()]
    params = []
    names = []
    for module_name, module in list(model.named_modules()):
        if not module_name or not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if terms and not any(term in module_name for term in terms):
            continue
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
        lora = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        replace_child(parent, parts[-1], lora)
        params.extend([lora.lora_A, lora.lora_B])
        names.append(module_name)
    if not params:
        raise ValueError(f"No ESM2 Linear modules matched LoRA targets: {terms}")
    for param in params:
        param.requires_grad_(True)
    return params, names


def batch_records_for_lora(records: Sequence[VariantRecord], indices: Sequence[int], tokenizer: Any, args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    batch = [records[int(idx)] for idx in indices]
    encoded = tokenizer([record.mutated_sequence for record in batch], return_tensors="pt", padding=True, truncation=True, max_length=args.esm2_max_length + 2)
    device = torch.device(args.device)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    positions = torch.tensor([esm_token_index(record.position) for record in batch], dtype=torch.long, device=device)
    labels = torch.tensor([record.score for record in batch], dtype=torch.float32, device=device)
    return input_ids, attention_mask, positions, labels


def evaluate_lora_model(model: Any, records: Sequence[VariantRecord], indices: np.ndarray, tokenizer: Any, args: argparse.Namespace) -> np.ndarray:
    preds = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), args.lora_eval_batch_size):
            batch_idx = indices[start : start + args.lora_eval_batch_size]
            input_ids, attention_mask, positions, _labels = batch_records_for_lora(records, batch_idx, tokenizer, args)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = model(input_ids, attention_mask, positions)
            preds.extend(pred.detach().float().cpu().numpy().tolist())
    return np.array(preds, dtype=np.float64)


def train_real_lora_run(args: argparse.Namespace, assay_id: str, rank: int, lr: float, seed: int, stage: str) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    paired = rows_for_split(records_by_assay[assay_id], split_payload.get("entries", []), "position_heldout")
    records, y, splits = split_arrays(paired)
    run_id = lora_run_id(stage, assay_id, rank, lr, seed)
    pred_path = out_root / "protein_48h_lora_predictions" / assay_id / f"{run_id}.csv"
    run_dir = out_root / "lora_runs" / assay_id / run_id
    existing = validate_lora_artifacts(args, assay_id, rank, lr, seed, stage)
    if existing.get("status") == "valid":
        pred, validation = load_valid_prediction_values(pred_path, records, splits)
        if pred is None:
            raise RuntimeError(f"validated LoRA artifact could not be reloaded: {validation}")
        return lora_metric_row(
            args,
            assay_id,
            stage,
            run_id,
            seed,
            rank,
            lr,
            records,
            y,
            splits,
            pred,
            pred_path,
            str(existing.get("adapter_hash", "")),
            str(existing.get("head_hash", "")),
            existing.get("evidence", {}),
        )
    pred_validation = existing.get("prediction_validation", {})
    if pred_validation.get("status") == "invalid":
        quarantine_invalid_artifact(args, pred_path, str(pred_validation.get("reason", "invalid_lora_prediction")))
    if torch is None:
        raise RuntimeError("torch is unavailable")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    tokenizer, base_model = load_transformers_model(args, masked=False)
    target_terms = [term.strip() for term in str(args.lora_target_modules).split(",") if term.strip()]
    adapter_params, module_names = inject_lora_named_modules(base_model, target_terms, rank=rank, alpha=rank * 2, dropout=args.lora_dropout)
    hidden_dim = int(getattr(base_model.config, "hidden_size"))
    model = Esm2PositionRegressor(base_model, hidden_dim).to(args.device)
    for param in model.head.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW([*adapter_params, *model.head.parameters()], lr=lr, weight_decay=args.lora_weight_decay)
    best_state = None
    best_val = -float("inf")
    best_step = 0
    stale = 0
    step = 0
    rng = np.random.default_rng(seed)
    batch_size = max(1, int(args.lora_batch_size))
    while step < args.lora_max_steps:
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            step += 1
            model.train()
            batch_idx = order[start : start + batch_size]
            input_ids, attention_mask, positions, labels = batch_records_for_lora(records, batch_idx, tokenizer, args)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = model(input_ids, attention_mask, positions)
                loss = torch.mean((pred.float() - labels.float()) ** 2) / max(1, args.grad_accum_steps)
            loss.backward()
            if step % max(1, args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.lora_eval_every == 0 or step == args.lora_max_steps:
                val_pred = evaluate_lora_model(model, records, val_idx, tokenizer, args)
                val_score = spearman_metric(y[val_idx], val_pred)
                if math.isfinite(val_score) and val_score > best_val:
                    best_val = val_score
                    best_step = step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if "lora_" in key or key.startswith("head.")}
                    stale = 0
                else:
                    stale += 1
                    if stale >= args.lora_patience:
                        step = args.lora_max_steps
                        break
            if step >= args.lora_max_steps:
                break
    if best_state:
        model.load_state_dict(best_state, strict=False)
    all_pred = np.full_like(y, fill_value=float(np.mean(y[train_idx])), dtype=np.float64)
    all_pred[train_idx] = evaluate_lora_model(model, records, train_idx, tokenizer, args)
    all_pred[val_idx] = evaluate_lora_model(model, records, val_idx, tokenizer, args)
    all_pred[test_idx] = evaluate_lora_model(model, records, test_idx, tokenizer, args)
    write_prediction_csv(pred_path, records, splits, y, all_pred, run_id, overwrite=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_tensors = {key: value.detach().cpu() for key, value in model.state_dict().items() if "lora_" in key}
    head_tensors = {key: value.detach().cpu() for key, value in model.state_dict().items() if key.startswith("head.")}
    adapter_path = run_dir / "adapter.safetensors"
    head_path = run_dir / "head.safetensors"
    atomic_save_safetensors(adapter_tensors, str(adapter_path), metadata={"checkpoint_policy": "adapter", "assay_id": assay_id, "run_id": run_id, "module_names": ",".join(module_names)})
    atomic_save_safetensors(head_tensors, str(head_path), metadata={"checkpoint_policy": "head", "assay_id": assay_id, "run_id": run_id})
    evidence = {
        "best_step": best_step,
        "selection_metric": "val_spearman",
        "selection_score": best_val,
        "lora_modules": module_names,
        "trainable_params": count_trainable(model),
        "total_params": count_total(model),
        "adapter_path": str(adapter_path),
        "head_path": str(head_path),
    }
    write_json(run_dir / "checkpoint_selection_evidence.json", evidence, overwrite=True)
    return lora_metric_row(args, assay_id, stage, run_id, seed, rank, lr, records, y, splits, all_pred, pred_path, file_sha256(adapter_path), file_sha256(head_path), evidence)


def lora_metric_row(
    args: argparse.Namespace,
    assay_id: str,
    stage: str,
    run_id: str,
    seed: int,
    rank: int,
    lr: float,
    records: Sequence[VariantRecord],
    y: np.ndarray,
    splits: np.ndarray,
    pred: np.ndarray,
    pred_path: Path,
    adapter_hash: str,
    head_hash: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    strongest = baseline_lookup(Path(args.out_root)).get((assay_id, "position_heldout"))
    val = splits == "val"
    test = splits == "test"
    val_s = spearman_metric(y[val], pred[val])
    test_s = spearman_metric(y[test], pred[test])
    base_val = numeric(strongest.get("val_spearman") if strongest else "", float("nan"))
    base_test = numeric(strongest.get("test_spearman") if strongest else "", float("nan"))
    ci_low, ci_high = (None, None)
    if strongest and strongest.get("prediction_path"):
        ci_low, ci_high = bootstrap_delta_by_position(pred_path, strongest["prediction_path"], "test", args.n_bootstrap, args.bootstrap_seed + seed)
    return {
        "assay_id": assay_id,
        "stage": stage,
        "run_id": run_id,
        "seed": seed,
        "rank": rank,
        "learning_rate": lr,
        "status": "complete",
        "selection_split": "val",
        "n_train": int((splits == "train").sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "val_spearman": val_s,
        "test_spearman": test_s,
        "val_mse": mse_metric(y[val], pred[val]),
        "test_mse": mse_metric(y[test], pred[test]),
        "strongest_non_plm_baseline": strongest.get("baseline", "") if strongest else "",
        "baseline_val_spearman": base_val if math.isfinite(base_val) else "",
        "baseline_test_spearman": base_test if math.isfinite(base_test) else "",
        "val_excess": val_s - base_val if math.isfinite(val_s) and math.isfinite(base_val) else "",
        "test_excess": test_s - base_test if math.isfinite(test_s) and math.isfinite(base_test) else "",
        "position_bootstrap_ci_low": ci_low,
        "position_bootstrap_ci_high": ci_high,
        "adapter_hash": adapter_hash,
        "head_hash": head_hash,
        "checkpoint_selection_evidence": json.dumps(evidence, sort_keys=True),
        "prediction_path": str(pred_path),
        "not_available_reason": "",
    }


def mock_lora_rows(args: argparse.Namespace, assay_id: str) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    records_by_assay = load_valid_records(out_root)
    split_payload = read_json(out_root / "protein_48h_split_manifest.json")
    paired = rows_for_split(records_by_assay[assay_id], split_payload.get("entries", []), "position_heldout")
    records, y, splits = split_arrays(paired)
    rows = []
    calibration = []
    for rank in (8, 16):
        for lr in (1e-5, 5e-5):
            pred = np.array([mock_signal(record) + 0.02 * rank - 100.0 * abs(lr - 5e-5) for record in records])
            pred += np.random.default_rng(args.calibration_seed + rank).normal(0, 0.01, len(records))
            pred_path = out_root / "protein_48h_lora_predictions" / assay_id / f"calibration_r{rank}_lr{lr:g}_seed{args.calibration_seed}.csv"
            write_prediction_csv(pred_path, records, splits, y, pred, "mock_lora_calibration")
            row = lora_metric_row(args, assay_id, "calibration", f"mock_calibration_{assay_id}_r{rank}_lr{lr:g}_seed{args.calibration_seed}", args.calibration_seed, rank, lr, records, y, splits, pred, pred_path, stable_hash(["adapter", assay_id, rank, lr]), stable_hash(["head", assay_id, rank, lr]), {"mock": True, "selection_metric": "val_spearman"})
            calibration.append(row)
            rows.append(row)
    best = max(calibration, key=lambda row: numeric(row.get("val_spearman"), -999.0))
    selected_rank = int(best["rank"])
    selected_lr = float(best["learning_rate"])
    for seed in [int(part) for part in str(args.formal_seeds).split(",") if part.strip()]:
        pred = np.array([mock_signal(record) for record in records])
        pred += np.random.default_rng(seed).normal(0, 0.015, len(records))
        pred_path = out_root / "protein_48h_lora_predictions" / assay_id / f"formal_r{selected_rank}_lr{selected_lr:g}_seed{seed}.csv"
        write_prediction_csv(pred_path, records, splits, y, pred, "mock_lora_formal")
        rows.append(lora_metric_row(args, assay_id, "formal", f"mock_formal_{assay_id}_r{selected_rank}_lr{selected_lr:g}_seed{seed}", seed, selected_rank, selected_lr, records, y, splits, pred, pred_path, stable_hash(["adapter", assay_id, selected_rank, selected_lr, seed]), stable_hash(["head", assay_id, selected_rank, selected_lr, seed]), {"mock": True, "selected_from_calibration": best["run_id"]}))
    return rows


def parse_evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def lora_complete_qualification(
    args: argparse.Namespace,
    assay_id: str,
    assay_rows: Sequence[Mapping[str, Any]],
    gate_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_root = Path(args.out_root)
    labels = set(gate_decision.get("labels", []) if gate_decision else [])
    reasons: list[str] = []
    audits = read_json(out_root / "protein_48h_split_audit.json").get("assays", {})
    position_audit = audits.get(assay_id, {}).get("position_heldout", {})
    if position_audit.get("status") != "valid":
        labels.add("INVALID_POSITION_HELD_OUT_SPLIT")
        reasons.append("position-held-out split is not valid")
    if position_audit.get("leakage_detected"):
        labels.add("DATA_LEAKAGE_DETECTED")
        reasons.append("leakage audit detected position or duplicate-mutation overlap")

    baseline_rows = list(csv.DictReader((out_root / "protein_48h_baseline_results.csv").open())) if (out_root / "protein_48h_baseline_results.csv").exists() else []
    strongest = strongest_baseline_rows(baseline_rows).get((assay_id, "position_heldout"))
    strength = baseline_strength_payload(baseline_rows).get(assay_id, {}).get("position_heldout", {})
    baseline_status = strength.get("strong_baseline_evidence", "missing")
    if not strongest:
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("no strongest non-PLM baseline was available")
    if baseline_status == "missing":
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("baseline status is missing rather than complete or explicitly classified")

    required_seeds = [int(part) for part in str(args.formal_seeds).split(",") if part.strip()]
    formal = [row for row in assay_rows if row.get("assay_id") == assay_id and row.get("stage") == "formal" and row.get("status") == "complete"]
    formal_by_seed = {int(numeric(row.get("seed"), -1)): row for row in formal if numeric(row.get("seed"), -1) >= 0}
    missing_seeds = [seed for seed in required_seeds if seed not in formal_by_seed]
    if missing_seeds:
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append(f"missing formal LoRA seed(s): {missing_seeds}")

    formal_ordered = [formal_by_seed[seed] for seed in required_seeds if seed in formal_by_seed]
    excesses = [numeric(row.get("test_excess"), float("nan")) for row in formal_ordered]
    finite_excess = [value for value in excesses if math.isfinite(value)]
    if len(finite_excess) != len(required_seeds):
        labels.add("NUMERICAL_FAILURE")
        reasons.append("one or more formal LoRA excess values is non-finite")
    same_positive_direction = bool(finite_excess and len(finite_excess) == len(required_seeds) and all(value > 0 for value in finite_excess))
    mean_excess = float(np.nanmean(excesses)) if excesses else float("nan")
    std_excess = float(np.nanstd(excesses)) if excesses else float("nan")
    worst_excess = float(np.nanmin(excesses)) if excesses else float("nan")
    if not same_positive_direction:
        labels.add("INSUFFICIENT_EVIDENCE")
        family = strongest.get("baseline_family", "simple") if strongest else "simple"
        labels.add("SATURATED_BY_EVOLUTIONARY_BASELINE" if family == "evolutionary" else "SATURATED_BY_SIMPLE_BASELINE")
        reasons.append("formal LoRA excess values are not all positive in the same direction")
    if not math.isfinite(mean_excess) or mean_excess < args.frozen_excess_threshold:
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("mean formal LoRA Spearman excess does not meet the frozen provisional threshold")
    if math.isfinite(worst_excess) and worst_excess <= -args.contradictory_excess_tolerance:
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("worst seed produces a contradictory negative excess")

    ci_lows = [numeric(row.get("position_bootstrap_ci_low"), float("nan")) for row in formal_ordered]
    ci_highs = [numeric(row.get("position_bootstrap_ci_high"), float("nan")) for row in formal_ordered]
    bootstrap_pass = bool(ci_lows and len(ci_lows) == len(required_seeds) and all(math.isfinite(value) and value > 0 for value in ci_lows))
    if not bootstrap_pass and not (args.mock_esm2 or args.mock_lora):
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("position-level bootstrap confidence interval lower bound is not above zero for every formal seed")

    checkpoint_ok = True
    for row in formal_ordered:
        evidence = parse_evidence(row.get("checkpoint_selection_evidence"))
        selection_metric = str(evidence.get("selection_metric", ""))
        if "test" in selection_metric.lower():
            checkpoint_ok = False
        if not evidence.get("mock") and (selection_metric != "val_spearman" or not evidence.get("adapter_path") or not evidence.get("head_path")):
            checkpoint_ok = False
    if not checkpoint_ok:
        labels.add("INSUFFICIENT_EVIDENCE")
        reasons.append("checkpoint-selection evidence is missing, incomplete, or not validation-based")

    qualified = bool(
        position_audit.get("status") == "valid"
        and not position_audit.get("leakage_detected")
        and strongest
        and baseline_status != "missing"
        and not missing_seeds
        and same_positive_direction
        and math.isfinite(mean_excess)
        and mean_excess >= args.frozen_excess_threshold
        and (not math.isfinite(worst_excess) or worst_excess > -args.contradictory_excess_tolerance)
        and (bootstrap_pass or args.mock_esm2 or args.mock_lora)
        and checkpoint_ok
    )
    if qualified:
        labels.discard("INSUFFICIENT_EVIDENCE")
        labels.discard("SATURATED_BY_SIMPLE_BASELINE")
        labels.discard("SATURATED_BY_EVOLUTIONARY_BASELINE")
        labels.update({"LORA_RECOVERABLE_SIGNAL", "PRELIMINARILY_QUALIFIED"})
    else:
        labels.discard("PRELIMINARILY_QUALIFIED")
        if "NUMERICAL_FAILURE" not in labels:
            labels.add("INSUFFICIENT_EVIDENCE")
    return {
        "assay_id": assay_id,
        "qualified": qualified,
        "formal_seed_count": len(formal_ordered),
        "required_formal_seeds": required_seeds,
        "missing_formal_seeds": missing_seeds,
        "same_positive_direction": same_positive_direction,
        "mean_excess": mean_excess if math.isfinite(mean_excess) else "",
        "std_excess": std_excess if math.isfinite(std_excess) else "",
        "worst_seed_excess": worst_excess if math.isfinite(worst_excess) else "",
        "bootstrap_ci_low_min": min(ci_lows) if ci_lows and all(math.isfinite(value) for value in ci_lows) else "",
        "bootstrap_ci_high_max": max(ci_highs) if ci_highs and all(math.isfinite(value) for value in ci_highs) else "",
        "bootstrap_pass": bootstrap_pass,
        "checkpoint_selection_ok": checkpoint_ok,
        "baseline_status": baseline_status,
        "strongest_baseline": strongest.get("baseline", "") if strongest else "",
        "strongest_baseline_family": strongest.get("baseline_family", "") if strongest else "",
        "labels": sorted(labels, key=lambda label: sorted(RESULT_LABELS).index(label)),
        "reasons": "; ".join(dict.fromkeys(reasons)),
    }


def run_lora_stage(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    gate = read_json(out_root / "protein_48h_advancement_gate.json")
    candidates = gate.get("lora_candidate_order", [])[: args.max_lora_assays]
    gate_decisions = {row.get("assay_id"): row for row in gate.get("decisions", [])}
    rows: list[dict[str, Any]] = []
    (out_root / "protein_48h_lora_predictions").mkdir(parents=True, exist_ok=True)
    if not candidates:
        write_csv(out_root / "protein_48h_lora_metrics.csv", rows, LORA_FIELDS, overwrite=True)
        return {"rows": rows, "attempted_assays": []}
    attempted = []
    qualifications = []
    for assay_id in candidates[:2]:
        attempted.append(assay_id)
        try:
            if args.mock_esm2 or args.mock_lora:
                assay_rows = mock_lora_rows(args, assay_id)
            else:
                assay_rows = []
                calibration = []
                for rank in (8, 16):
                    for lr in (1e-5, 5e-5):
                        row = train_real_lora_run(args, assay_id, rank, lr, args.calibration_seed, "calibration")
                        assay_rows.append(row)
                        calibration.append(row)
                best = max(calibration, key=lambda row: numeric(row.get("val_spearman"), -999.0))
                for seed in [int(part) for part in str(args.formal_seeds).split(",") if part.strip()]:
                    assay_rows.append(train_real_lora_run(args, assay_id, int(best["rank"]), float(best["learning_rate"]), seed, "formal"))
            rows.extend(assay_rows)
            qualification = lora_complete_qualification(args, assay_id, assay_rows, gate_decisions.get(assay_id, {}))
            qualifications.append(qualification)
            if qualification.get("qualified") and "PRELIMINARILY_QUALIFIED" in qualification.get("labels", []):
                break
        except Exception as exc:
            rows.append(
                {
                    "assay_id": assay_id,
                    "stage": "lora",
                    "run_id": "",
                    "seed": "",
                    "rank": "",
                    "learning_rate": "",
                    "status": "NOT_AVAILABLE",
                    "selection_split": "val",
                    "n_train": "",
                    "n_val": "",
                    "n_test": "",
                    "val_spearman": "",
                    "test_spearman": "",
                    "val_mse": "",
                    "test_mse": "",
                    "strongest_non_plm_baseline": "",
                    "baseline_val_spearman": "",
                    "baseline_test_spearman": "",
                    "val_excess": "",
                    "test_excess": "",
                    "position_bootstrap_ci_low": "",
                    "position_bootstrap_ci_high": "",
                    "adapter_hash": "",
                    "head_hash": "",
                    "checkpoint_selection_evidence": "",
                    "prediction_path": "",
                    "not_available_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    metrics_path = out_root / "protein_48h_lora_metrics.csv"
    if metrics_path.exists():
        snapshot_existing_artifact(out_root, metrics_path, "lora_metrics", "lora_metrics_refreshed_after_resume_validation")
    write_csv(metrics_path, rows, LORA_FIELDS, overwrite=True)
    write_json(out_root / "protein_48h_lora_qualification_evidence.json", {"created_at": now_utc(), "attempted_assays": attempted, "qualifications": qualifications}, overwrite=True)
    return {"rows": rows, "attempted_assays": attempted, "qualifications": qualifications}


def classify_final(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    gate = read_json(out_root / "protein_48h_advancement_gate.json")
    lora_rows = list(csv.DictReader((out_root / "protein_48h_lora_metrics.csv").open())) if (out_root / "protein_48h_lora_metrics.csv").exists() else []
    decisions = gate.get("decisions", [])
    final = []
    for decision in decisions:
        assay_id = decision["assay_id"]
        evidence = lora_complete_qualification(args, assay_id, lora_rows, decision)
        row = dict(decision)
        row["labels"] = evidence["labels"]
        row["lora_seed_count"] = evidence["formal_seed_count"]
        row["lora_test_excess_mean"] = evidence["mean_excess"]
        row["lora_test_excess_std"] = evidence["std_excess"]
        row["lora_worst_seed_excess"] = evidence["worst_seed_excess"]
        row["lora_bootstrap_ci_low_min"] = evidence["bootstrap_ci_low_min"]
        row["lora_bootstrap_ci_high_max"] = evidence["bootstrap_ci_high_max"]
        row["complete_lora_gate_passed"] = evidence["qualified"]
        row["complete_lora_gate_reason"] = evidence["reasons"]
        final.append(row)
    return {"decisions": final, "preliminarily_qualified": [row["assay_id"] for row in final if "PRELIMINARILY_QUALIFIED" in row.get("labels", [])]}


def artifact_audit(args: argparse.Namespace, final_payload: Mapping[str, Any]) -> dict[str, Any]:
    out_root = Path(args.out_root)
    files = []
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".tmp"):
            continue
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(out_root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload = {
        "created_at": now_utc(),
        "out_root": str(out_root),
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
        "deletions": [],
        "storage_policy": frozen_protocol_payload(args)["storage"],
        "final_decision": final_payload,
    }
    write_json(out_root / "protein_48h_artifact_audit.json", payload, overwrite=True)
    return payload


def write_summary(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    inventory = list(csv.DictReader((out_root / "protein_48h_candidate_ranking.csv").open())) if (out_root / "protein_48h_candidate_ranking.csv").exists() else []
    baseline_rows = list(csv.DictReader((out_root / "protein_48h_baseline_results.csv").open())) if (out_root / "protein_48h_baseline_results.csv").exists() else []
    esm_rows = list(csv.DictReader((out_root / "protein_48h_esm2_pilot_metrics.csv").open())) if (out_root / "protein_48h_esm2_pilot_metrics.csv").exists() else []
    lora_rows = list(csv.DictReader((out_root / "protein_48h_lora_metrics.csv").open())) if (out_root / "protein_48h_lora_metrics.csv").exists() else []
    gate = read_json(out_root / "protein_48h_advancement_gate.json")
    final = classify_final(args)
    pilots = [row["assay_id"] for row in inventory if str(row.get("selected_pilot", "")).lower() == "true"]
    candidate_set = [row["assay_id"] for row in inventory if str(row.get("entered_static_candidate_set", "")).lower() == "true"]
    strongest = strongest_baseline_rows(baseline_rows)
    saturated = [
        {
            "assay_id": row["assay_id"],
            "labels": row.get("labels", []),
            "baseline": row.get("strongest_baseline", ""),
            "reason": row.get("decision_reason", ""),
        }
        for row in final["decisions"]
        if "SATURATED_BY_SIMPLE_BASELINE" in row.get("labels", []) or "SATURATED_BY_EVOLUTIONARY_BASELINE" in row.get("labels", [])
    ]
    best_frozen = max(
        [row for row in esm_rows if row.get("status") == "complete" and row.get("split_type") == "position_heldout"],
        key=lambda row: numeric(row.get("val_excess"), -999.0),
        default={},
    )
    whole = [row for row in esm_rows if row.get("representation_type") == "whole_sequence_mean" and row.get("split_type") == "position_heldout"]
    non_whole = [row for row in esm_rows if row.get("representation_type") not in {"whole_sequence_mean", ""} and row.get("split_type") == "position_heldout"]
    whole_best = max([numeric(row.get("val_excess"), float("nan")) for row in whole], default=float("nan"))
    non_whole_best = max([numeric(row.get("val_excess"), float("nan")) for row in non_whole], default=float("nan"))
    lora_formal = [row for row in lora_rows if row.get("stage") == "formal" and row.get("status") == "complete"]
    best_calibration = max([row for row in lora_rows if row.get("stage") == "calibration" and row.get("status") == "complete"], key=lambda row: numeric(row.get("val_spearman"), -999.0), default={})
    dms_files = find_dms_files(Path(args.dms_dir))
    input_status = {
        "dms_dir": str(args.dms_dir),
        "metadata_csv": str(args.metadata_csv),
        "protein_gym_dms_files_found": len(dms_files),
        "status": "available" if dms_files else "NOT_AVAILABLE",
        "reason": "" if dms_files else f"no ProteinGym DMS CSV files found under {args.dms_dir}",
    }
    failure_reasons = {
        row["assay_id"]: row.get("decision_reason", "")
        for row in final["decisions"]
        if "PRELIMINARILY_QUALIFIED" not in row.get("labels", [])
    }
    if not failure_reasons and not candidate_set:
        failure_reasons = {"WORKFLOW_INPUTS": f"NOT_AVAILABLE: {input_status['reason']}"}
    answers = {
        "1_static_candidate_count": len(candidate_set),
        "2_pilot_selection_limit": f"{len(pilots)} pilot assay(s) selected; controller enforces max_pilot_assays={args.max_pilot_assays} and prioritizes data quality over phenotype diversity.",
        "3_strongest_baseline_by_pilot": {f"{assay_id}:position_heldout": strongest.get((assay_id, "position_heldout"), {}).get("baseline", "") for assay_id in pilots},
        "4_saturated_assays": saturated,
        "5_random_vs_position_heldout": summarize_random_vs_position(esm_rows),
        "6_zero_shot_positive_margin": any(row.get("method") == "zero_shot" and numeric(row.get("val_excess"), -999.0) > 0 for row in esm_rows if row.get("split_type") == "position_heldout"),
        "7_best_frozen_representation": best_frozen.get("representation_type", "") or best_frozen.get("method", ""),
        "8_whole_sequence_pooling_dilution": "yes" if math.isfinite(whole_best) and math.isfinite(non_whole_best) and whole_best < non_whole_best else "not_established",
        "9_any_assay_advanced_to_lora": bool(gate.get("lora_candidate_order")),
        "10_best_lora_development_config": {
            "rank": best_calibration.get("rank", ""),
            "learning_rate": best_calibration.get("learning_rate", ""),
            "validation_spearman": best_calibration.get("val_spearman", ""),
        },
        "11_three_seed_stability": summarize_lora_stability(lora_formal),
        "12_preliminarily_qualified_pair_found": bool(final["preliminarily_qualified"]),
        "13_failure_reasons": failure_reasons,
        "14_escalation_justified": (
            "conditionally justified for a next-stage confirmation plan, not as a completed benchmark claim"
            if final["preliminarily_qualified"]
            else "not justified unless at least one preliminary qualified ESM2-150M pair exists"
        ),
        "15_prerequisites_for_tar_lat_subspace_transfer": "established" if final["preliminarily_qualified"] else "not established",
    }
    payload = {
        "created_at": now_utc(),
        "workflow_status": "complete",
        "formal": bool(args.formal),
        "mock_esm2": bool(args.mock_esm2),
        "out_root": str(out_root),
        "candidate_set_assays": candidate_set,
        "input_status": input_status,
        "pilot_assays": pilots,
        "lora_candidate_order": gate.get("lora_candidate_order", []),
        "result_classification": final["decisions"],
        "preliminarily_qualified": final["preliminarily_qualified"],
        "answers": answers,
        "permitted_conclusion": (
            "A candidate task-model pair has been identified that passes the preliminary checks."
            if final["preliminarily_qualified"]
            else "No candidate task-model pair has been identified that passes the preliminary strong-baseline, position-held-out, and fresh-LoRA checks."
        ),
    }
    write_json(out_root / "protein_48h_summary_report.json", payload, overwrite=True)
    write_summary_md(out_root / "protein_48h_summary_report.md", payload)
    artifact_audit(args, payload)
    return payload


def summarize_random_vs_position(esm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    assays = sorted({row.get("assay_id", "") for row in esm_rows if row.get("assay_id")})
    for assay_id in assays:
        random_best = max([numeric(row.get("val_excess"), float("nan")) for row in esm_rows if row.get("assay_id") == assay_id and row.get("split_type") == "random"], default=float("nan"))
        pos_best = max([numeric(row.get("val_excess"), float("nan")) for row in esm_rows if row.get("assay_id") == assay_id and row.get("split_type") == "position_heldout"], default=float("nan"))
        result[assay_id] = {"best_random_val_excess": random_best if math.isfinite(random_best) else "", "best_position_heldout_val_excess": pos_best if math.isfinite(pos_best) else ""}
    return result


def summarize_lora_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run", "seed_count": 0}
    excesses = [numeric(row.get("test_excess"), float("nan")) for row in rows]
    return {
        "status": "stable_positive" if all(math.isfinite(value) and value > 0 for value in excesses) and len(rows) >= 3 else "insufficient_or_unstable",
        "seed_count": len(rows),
        "mean_test_excess": float(np.nanmean(excesses)),
        "std_test_excess": float(np.nanstd(excesses)),
        "worst_seed_excess": float(np.nanmin(excesses)),
    }


def write_summary_md(path: Path, payload: Mapping[str, Any]) -> None:
    answers = payload["answers"]
    lines = [
        "# ProteinGym ESM2-150M 48h Qualification Summary",
        "",
        f"- Workflow status: `{payload['workflow_status']}`.",
        f"- Static candidate assays: `{answers['1_static_candidate_count']}`.",
        f"- Pilot assays: `{', '.join(payload['pilot_assays']) if payload['pilot_assays'] else 'none'}`.",
        f"- Advanced to LoRA: `{', '.join(payload['lora_candidate_order']) if payload['lora_candidate_order'] else 'none'}`.",
        f"- Preliminary qualified pairs: `{', '.join(payload['preliminarily_qualified']) if payload['preliminarily_qualified'] else 'none'}`.",
        "",
        "## Required Answers",
        "",
    ]
    for key, value in answers.items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Permitted Conclusion", "", str(payload["permitted_conclusion"])])
    path.write_text("\n".join(lines) + "\n")


def write_empty_downstream_outputs(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    write_json(out_root / "protein_48h_split_manifest.json", {"created_at": now_utc(), "primary_split": "position_heldout", "pilot_assay_ids": [], "entries": [], "manifest_hash": stable_hash([])})
    write_json(out_root / "protein_48h_split_audit.json", {"created_at": now_utc(), "assays": {}, "manifest_hash": stable_hash([])})
    write_csv(out_root / "protein_48h_baseline_results.csv", [], BASELINE_FIELDS)
    write_baseline_report(out_root, [])
    (out_root / "protein_48h_esm2_pilot_predictions").mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "protein_48h_esm2_pilot_metrics.csv", [], ESM2_FIELDS)
    write_json(out_root / "protein_48h_advancement_gate.json", {"created_at": now_utc(), "lora_candidate_order": [], "decisions": [], "reason": "no pilot assays available"})
    (out_root / "protein_48h_lora_predictions").mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "protein_48h_lora_metrics.csv", [], LORA_FIELDS)


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def require_storage(args: argparse.Namespace) -> None:
    free = free_disk_gb(Path(args.out_root))
    if free < args.stop_on_low_disk_gb:
        raise RuntimeError(f"low disk before workflow: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G")


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    require_storage(args)
    registry = load_registry(args)
    write_metadata(
        Path(args.out_root) / "protein_48h_workflow_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[args.dms_dir, args.metadata_csv, args.public_predictions_dir, args.msa_dir],
            extra={
                "phase": "protein_48h_esm2_qualification",
                "max_concurrent_gpu_jobs": 1,
                "formal": bool(args.formal),
            },
        ),
    )
    download_report = run_stage(
        args,
        registry,
        "stage0_proteingym_data",
        "proteingym_data_download_and_validation",
        Path(args.out_root) / "protein_48h_proteingym_download_report.json",
        lambda: ensure_proteingym_data(args),
        validator=lambda path: (
            download_report_valid(path) and download_report_files_intact(path, args)
        )
        or (not args.auto_download_proteingym and json_object(path)),
    )
    if download_report is None:
        download_report = read_json(Path(args.out_root) / "protein_48h_proteingym_download_report.json")
    if download_report.get("validation_status") == "valid":
        args.data_run_suffix = str(download_report.get("dataset_fingerprint", ""))[:12]
        snapshot_previous_blocked_execution(args, registry, download_report)
    run_stage(args, registry, "stage0_hvue_summary", "hvue_summary", Path(args.out_root) / "protein_48h_hvue_benchmark_limitation_summary.json", lambda: write_hvue_summary(args), validator=json_object)
    run_stage(args, registry, "stage0_frozen_protocol", "frozen_protocol", Path(args.out_root) / "protein_48h_frozen_protocol.json", lambda: write_frozen_protocol(args), validator=json_object, skippable=False)
    prescreen = run_stage(args, registry, task_id_for_data(args, "stage1_static_prescreen"), "static_prescreen", Path(args.out_root) / "protein_48h_candidate_inventory.csv", lambda: prescreen_assays(args), validator=csv_nonempty_or_header)
    if prescreen is None:
        prescreen_payload = {"pilot_ids": []}
        ranking = Path(args.out_root) / "protein_48h_candidate_ranking.csv"
        if ranking.exists():
            prescreen_payload["pilot_ids"] = [row["assay_id"] for row in csv.DictReader(ranking.open()) if str(row.get("selected_pilot", "")).lower() == "true"]
    else:
        prescreen_payload = prescreen
    if not prescreen_payload.get("pilot_ids"):
        update_task(args, registry, task_id_for_data(args, "stage2_to_stage4_skipped"), stage="post_prescreen_gate", status="skipped", reason="no pilot assays passed static filters")
        write_empty_downstream_outputs(args)
        summary = write_summary(args)
        update_task(args, registry, task_id_for_data(args, "stage_final_summary"), stage="final_aggregation", status="complete", output_path=str(Path(args.out_root) / "protein_48h_summary_report.json"))
        return summary
    run_stage(args, registry, task_id_for_data(args, "stage2_splits"), "split_creation", Path(args.out_root) / "protein_48h_split_manifest.json", lambda: create_splits(args), validator=json_object)
    run_stage(args, registry, task_id_for_data(args, "stage2_resume_validation"), "resume_validation", Path(args.out_root) / "protein_48h_resume_validation.json", lambda: validate_resume_state(args, registry), validator=json_object, skippable=False)
    run_stage(args, registry, task_id_for_data(args, "stage2_evolutionary_resources"), "evolutionary_baseline_resources", Path(args.out_root) / "protein_48h_evolutionary_baseline_report.json", lambda: ensure_evolutionary_baseline_resources(args), validator=json_object, skippable=False)
    run_stage(args, registry, task_id_for_data(args, "stage2_baselines"), "baseline_evaluation", Path(args.out_root) / "protein_48h_baseline_results.csv", lambda: evaluate_baselines(args), validator=csv_nonempty_or_header, skippable=False)
    run_stage(args, registry, task_id_for_data(args, "stage3_esm2_pilot"), "esm2_pilot", Path(args.out_root) / "protein_48h_esm2_pilot_metrics.csv", lambda: run_esm2_pilot(args), validator=csv_nonempty_or_header, skippable=False)
    run_stage(args, registry, task_id_for_data(args, "stage3_advancement_gate"), "advancement_gate", Path(args.out_root) / "protein_48h_advancement_gate.json", lambda: gate_and_rank(args), validator=json_object, skippable=False)
    gate = read_json(Path(args.out_root) / "protein_48h_advancement_gate.json")
    if not gate.get("lora_candidate_order"):
        update_task(args, registry, task_id_for_data(args, "stage4_lora_skipped"), stage="lora_qualification", status="skipped", reason="no assay passed frozen ESM2 advancement gate")
        (Path(args.out_root) / "protein_48h_lora_predictions").mkdir(parents=True, exist_ok=True)
        write_csv(Path(args.out_root) / "protein_48h_lora_metrics.csv", [], LORA_FIELDS)
    else:
        run_stage(args, registry, task_id_for_data(args, "stage4_lora"), "lora_qualification", Path(args.out_root) / "protein_48h_lora_metrics.csv", lambda: run_lora_stage(args), validator=csv_nonempty_or_header, skippable=False)
    summary = write_summary(args)
    update_task(args, registry, task_id_for_data(args, "stage_final_summary"), stage="final_aggregation", status="complete", output_path=str(Path(args.out_root) / "protein_48h_summary_report.json"))
    return summary


def make_smoke_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    dms_dir = root / "fixture/DMS_substitutions"
    dms_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "fixture/DMS_substitutions.csv"
    msa_dir = root / "fixture/MSA_files"
    pred_dir = root / "fixture/public_predictions"
    msa_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260804)
    wt = "".join(rng.choice(AA) for _ in range(90))
    rows = []
    for pos in range(3, 78):
        wt_res = wt[pos - 1]
        for offset in (3, 7):
            mut = AA[(AA.index(wt_res) + offset) % len(AA)]
            chars = list(wt)
            chars[pos - 1] = mut
            mutation = f"{wt_res}{pos}{mut}"
            left = wt[max(0, pos - 4) : pos - 1]
            right = wt[pos : min(len(wt), pos + 3)]
            context = sum((AA.index(ch) + 1) * (idx + 1) for idx, ch in enumerate(left + right))
            score = 0.07 * context + 0.5 * (AA.index(mut) - AA.index(wt_res))
            rows.append({"mutant": mutation, "mutated_sequence": "".join(chars), "DMS_score": score})
    write_csv(dms_dir / "SMOKE_SIGNAL.csv", rows, ["mutant", "mutated_sequence", "DMS_score"])
    metadata_rows = [{"DMS_id": "SMOKE_SIGNAL", "DMS_filename": "SMOKE_SIGNAL.csv", "target_name": "synthetic_context_signal", "selection_type": "activity", "target_seq": wt}]
    write_csv(metadata_path, metadata_rows, ["DMS_id", "DMS_filename", "target_name", "selection_type", "target_seq"])
    return dms_dir, metadata_path, pred_dir, msa_dir


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.out_root)
    dms_dir, metadata, pred_dir, msa_dir = make_smoke_fixture(root)
    args.dms_dir = str(dms_dir)
    args.metadata_csv = str(metadata)
    args.public_predictions_dir = str(pred_dir)
    args.msa_dir = str(msa_dir)
    args.mock_esm2 = True
    args.mock_lora = True
    args.formal = False
    args.skip_proteingym_download = True
    args.min_valid_samples = min(args.min_valid_samples, 20)
    args.n_bootstrap = min(args.n_bootstrap, 100)
    args.max_retries = 0
    return run_workflow(args)


def launch(args: argparse.Namespace) -> int:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        "-u",
        "phase2/proteingym_esm2_qualification.py",
        "run",
        "--out-root",
        str(args.out_root),
        "--dms-dir",
        str(args.dms_dir),
        "--metadata-csv",
        str(args.metadata_csv),
        "--public-predictions-dir",
        str(args.public_predictions_dir),
        "--msa-dir",
        str(args.msa_dir),
        "--esm2-model",
        str(args.esm2_model),
        "--device",
        str(args.device),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--resume",
        "--formal",
    ]
    cmd.extend(["--proteingym-source-api", str(args.proteingym_source_api)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if not args.auto_download_proteingym:
        cmd.append("--no-auto-download-proteingym")
    if args.skip_proteingym_download:
        cmd.append("--skip-proteingym-download")
    with Path(args.log_file).open("a") as log:
        log.write(f"[{now_utc()}] launcher cmd={' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(Path(args.out_root) / "protein_48h_launcher_status.json", {"created_at": now_utc(), "pid": proc.pid, "cmd": cmd, "log_file": str(args.log_file)}, overwrite=True)
    print(proc.pid)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--dms-dir", default=str(DEFAULT_DMS_DIR))
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA))
    parser.add_argument("--public-predictions-dir", default=str(DEFAULT_PUBLIC_PREDICTIONS))
    parser.add_argument("--msa-dir", default=str(DEFAULT_MSA_DIR))
    parser.add_argument("--proteingym-source-api", default=PROTEINGYM_ZENODO_API)
    parser.add_argument("--auto-download-proteingym", dest="auto_download_proteingym", action="store_true", default=True)
    parser.add_argument("--no-auto-download-proteingym", dest="auto_download_proteingym", action="store_false")
    parser.add_argument("--skip-proteingym-download", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--remove-validated-archives", action="store_true", default=True)
    parser.add_argument("--keep-validated-archives", dest="remove_validated_archives", action="store_false")
    parser.add_argument("--esm2-model", default=DEFAULT_ESM2_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--mock-esm2", action="store_true")
    parser.add_argument("--mock-lora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--stop-on-low-disk-gb", type=float, default=10.0)
    parser.add_argument("--max-static-candidates", type=int, default=20)
    parser.add_argument("--max-pilot-assays", type=int, default=3)
    parser.add_argument("--max-lora-assays", type=int, default=2)
    parser.add_argument("--min-valid-samples", type=int, default=96)
    parser.add_argument("--min-valid-proportion", type=float, default=0.75)
    parser.add_argument("--min-score-std", type=float, default=1e-8)
    parser.add_argument("--max-missing-proportion", type=float, default=0.2)
    parser.add_argument("--esm2-max-length", type=int, default=1022)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--esm2-layer", type=int, default=-1)
    parser.add_argument("--esm2-batch-size", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--frozen-excess-threshold", type=float, default=0.03)
    parser.add_argument("--contradictory-excess-tolerance", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--formal-seeds", default="42,43,44")
    parser.add_argument("--lora-target-modules", default="query,value")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-batch-size", type=int, default=2)
    parser.add_argument("--lora-eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-max-steps", type=int, default=500)
    parser.add_argument("--lora-eval-every", type=int, default=50)
    parser.add_argument("--lora-patience", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_common_args(run)
    download = sub.add_parser("download-data")
    add_common_args(download)
    smoke = sub.add_parser("smoke")
    add_common_args(smoke)
    smoke.set_defaults(out_root=str(DEFAULT_SMOKE_ROOT), mock_esm2=True, mock_lora=True)
    launch_parser = sub.add_parser("launch")
    add_common_args(launch_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "download-data":
        Path(args.out_root).mkdir(parents=True, exist_ok=True)
        ensure_proteingym_data(args)
    elif args.command == "launch":
        raise SystemExit(launch(args))
    else:
        run_workflow(args)


if __name__ == "__main__":
    main()
