"""VPF-PLM strict qualification controller.

This controller reuses the existing phase-2 qualification style to run the
final viral benchmark candidate:

PHROG protein family / remote-homology disjoint viral protein function
prediction with the official VPF-PLM ProtBERT-BFD representation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from phase2.signed_bootstrap import paired_grouped_prediction_bootstrap
from phase2.vpf_plm_compat import predict_probabilities
DEFAULT_PYTHON = "/home/teacher1/miniconda3/envs/UT-p1/bin/python"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/vpf_plm_qualification"
DEFAULT_LOG = PROJECT_ROOT / "logs/vpf_plm_qualification.log"
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "data/external/vpf_plm"
DEFAULT_BUCKET_ROOT = DEFAULT_EXTERNAL_ROOT / "official_bucket"
DEFAULT_INFERENCE_REPO = DEFAULT_EXTERNAL_ROOT / "viral-protein-function-plm-main"
DEFAULT_ANALYSIS_REPO = DEFAULT_EXTERNAL_ROOT / "viral-protein-function-annotation-with-protein-language-model-main"
DEFAULT_LOCAL_TOOL_BIN = PROJECT_ROOT / "tools/vpf_local/bin"
DEFAULT_LOCAL_TOOL_USR_BIN = PROJECT_ROOT / "tools/vpf_local/usr/bin"
DEFAULT_TOOL_ROOT = PROJECT_ROOT / "tools/vpf_local/official"
OFFICIAL_MODEL_ZENODO = "https://doi.org/10.5281/zenodo.10182746"
OFFICIAL_ANALYSIS_ZENODO = "https://doi.org/10.5281/zenodo.10182750"
OFFICIAL_PAPER = "https://doi.org/10.1038/s41564-023-01584-8"
PHROG_INDEX_URL = "https://storage.googleapis.com/viral_protein_family_plm_embeddings/phrogs/PHROG_index_revised_v4_10292022.csv"
PHROG_INDEX_NAME = "PHROG_index_revised_v4_10292022.csv"
PHROG_FAA_PREFIX = "https://storage.googleapis.com/viral_protein_family_plm_embeddings/phrogs/faa_downloaded_04052022"
PHROG_EMBED_PREFIX = "https://storage.googleapis.com/viral_protein_family_plm_embeddings/phrogs/protbert_bfd_embeddings_phrog"
PHROG_CENTROID_PREFIX = "https://storage.googleapis.com/viral_protein_family_plm_embeddings/phrogs/phrog_family_centroid"
BLAST_URL = "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz"
FOLDSEEK_URLS = [
    "https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz",
    "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz",
]
AA = "ACDEFGHIKLMNPQRSTVWY"
PC_GROUPS = {
    "A": "H",
    "C": "S",
    "D": "N",
    "E": "N",
    "F": "A",
    "G": "H",
    "H": "B",
    "I": "A",
    "K": "B",
    "L": "A",
    "M": "A",
    "N": "P",
    "P": "T",
    "Q": "P",
    "R": "B",
    "S": "P",
    "T": "P",
    "V": "A",
    "W": "A",
    "Y": "A",
    "X": "X",
}
KNOWN_NONUNKNOWN_CATEGORIES = [
    "DNA, RNA and nucleotide metabolism",
    "connector",
    "head and packaging",
    "integration and excision",
    "lysis",
    "moron, auxiliary metabolic gene and host takeover",
    "other",
    "tail",
    "transcription regulation",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def valid_json(path: Path) -> bool:
    return path.exists() and bool(read_json(path))


def valid_csv(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        return False
    with path.open() as handle:
        return max(0, sum(1 for _ in handle) - 1) >= min_rows


def run_cmd(
    command: Sequence[str],
    *,
    cwd: Path = PROJECT_ROOT,
    timeout: int = 3600,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=dict(env) if env else None,
    )
    return {
        "returncode": proc.returncode,
        "runtime_sec": round(time.time() - started, 2),
        "output_tail": proc.stdout[-8000:],
    }


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{now_utc()}] {message}"
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log_file).open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def status_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "vpf_controller_status.json"


def registry_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "vpf_experiment_registry.json"


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


def path_revision(path: Path) -> str:
    if (path / ".git").exists():
        result = run_cmd(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=60)
        if result["returncode"] == 0:
            return result["output_tail"].strip().splitlines()[-1]
    if path.is_file():
        return f"sha256:{sha256_file(path)}"
    return ""


def resume_ready_json(path: Path) -> bool:
    return valid_json(path)


def resume_ready_csv(path: Path, min_rows: int = 1) -> bool:
    return valid_csv(path, min_rows=min_rows)


def run_stage_with_resume(
    args: argparse.Namespace,
    stage: str,
    fn,
    ready,
    **status_extra: Any,
) -> Any:
    if args.resume and ready():
        update_status(args, "running", stage, resumed=True, skipped=True, **status_extra)
        log(args, f"resume skip {stage}")
        return {"status": "skipped", "stage": stage}
    update_status(args, "running", stage, **status_extra)
    log(args, f"running {stage}")
    return fn(args)


def ensure_download(url: str, dest: Path, *, timeout: int, quiet: bool = False) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "returncode": 0,
            "runtime_sec": 0.0,
            "output_tail": f"reused existing file {dest}",
            "attempts": [{"method": "reuse", "returncode": 0, "runtime_sec": 0.0, "output_tail": f"reused {dest}"}],
        }
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempts: list[dict[str, Any]] = []
    wget_base = [
        "wget",
        "--tries=5",
        "--retry-connrefused",
        "--waitretry=2",
        "--timeout=30",
        "--read-timeout=120",
        "--no-http-keep-alive",
        "-O",
        str(tmp),
        url,
    ]
    if quiet:
        wget_base.insert(1, "-q")
    for _ in range(3):
        result = run_cmd(wget_base, timeout=timeout)
        attempts.append({"method": "wget", **result})
        if result["returncode"] == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, dest)
            return {"returncode": 0, "runtime_sec": sum(a["runtime_sec"] for a in attempts), "output_tail": attempts[-1]["output_tail"], "attempts": attempts}
        tmp.unlink(missing_ok=True)
        time.sleep(2)
    curl_cmd = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout),
        "--http1.1",
        "-o",
        str(tmp),
        url,
    ]
    if quiet:
        curl_cmd.insert(1, "--silent")
        curl_cmd.insert(2, "--show-error")
    result = run_cmd(curl_cmd, timeout=timeout)
    attempts.append({"method": "curl", **result})
    if result["returncode"] == 0 and tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, dest)
        return {"returncode": 0, "runtime_sec": sum(a["runtime_sec"] for a in attempts), "output_tail": attempts[-1]["output_tail"], "attempts": attempts}
    tmp.unlink(missing_ok=True)
    try:
        import requests

        started = time.time()
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, dest)
            attempts.append(
                {
                    "method": "requests",
                    "returncode": 0,
                    "runtime_sec": round(time.time() - started, 2),
                    "output_tail": f"downloaded {url} via requests fallback",
                }
            )
            return {"returncode": 0, "runtime_sec": sum(a["runtime_sec"] for a in attempts), "output_tail": attempts[-1]["output_tail"], "attempts": attempts}
    except Exception as exc:
        attempts.append(
            {
                "method": "requests",
                "returncode": 1,
                "runtime_sec": 0.0,
                "output_tail": repr(exc),
            }
        )
    tmp.unlink(missing_ok=True)
    tail = "\n".join(f"[{a['method']}] rc={a['returncode']} {a['output_tail']}" for a in attempts[-4:])
    return {"returncode": 1, "runtime_sec": sum(a["runtime_sec"] for a in attempts), "output_tail": tail, "attempts": attempts}


def tool_roots(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.tool_root)
    return {
        "root": root,
        "blast": root / "ncbi-blast",
        "foldseek": root / "foldseek",
        "downloads": root / "downloads",
        "tmp": root / "tmp",
    }


def command_env(args: argparse.Namespace) -> dict[str, str]:
    roots = tool_roots(args)
    env = os.environ.copy()
    path_parts = [
        str(roots["blast"] / "bin"),
        str(roots["foldseek"] / "bin"),
        str(Path(args.local_tool_bin)),
        str(Path(args.local_tool_usr_bin)),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    ld_parts = [
        str(Path(args.local_tool_bin).parent / "usr/lib/ncbi-blast+"),
        str(Path(args.local_tool_bin).parent / "usr/lib/x86_64-linux-gnu"),
        str(Path(args.local_tool_bin).parent / "usr/lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(part for part in ld_parts if part)
    env.setdefault("TF_USE_LEGACY_KERAS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def tool_version(name: str, args: argparse.Namespace) -> dict[str, Any]:
    exe = shutil.which(name, path=command_env(args)["PATH"])
    if not exe:
        return {"path": "", "ok": False, "version": ""}
    for version_args in (["-version"], ["--version"], ["version"], ["-h"]):
        result = run_cmd([exe, *version_args], timeout=120, env=command_env(args))
        if result["returncode"] == 0 or result["output_tail"]:
            return {"path": exe, "ok": result["returncode"] == 0, "version": result["output_tail"].splitlines()[:2]}
    return {"path": exe, "ok": True, "version": []}


def install_official_blast(args: argparse.Namespace) -> dict[str, Any]:
    roots = tool_roots(args)
    archive = roots["downloads"] / "ncbi-blast-2.17.0+-x64-linux.tar.gz"
    if not archive.exists() or archive.stat().st_size == 0:
        download = ensure_download(BLAST_URL, archive, timeout=args.tool_download_timeout_sec)
        if download["returncode"] != 0:
            return {"tool": "blast", "status": "failed", "download": download}
    extract_dir = roots["tmp"] / "blast_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract = run_cmd(["tar", "-xzf", str(archive), "-C", str(extract_dir)], timeout=600)
    candidates = sorted(extract_dir.glob("ncbi-blast-*"))
    if not candidates:
        return {"tool": "blast", "status": "failed", "extract": extract}
    shutil.rmtree(roots["blast"], ignore_errors=True)
    shutil.move(str(candidates[0]), str(roots["blast"]))
    ok = (roots["blast"] / "bin/blastp").exists()
    return {"tool": "blast", "status": "complete" if ok else "failed", "path": str(roots["blast"] / "bin")}


def install_official_foldseek(args: argparse.Namespace) -> dict[str, Any]:
    roots = tool_roots(args)
    archive = roots["downloads"] / "foldseek-linux-avx2.tar.gz"
    if not archive.exists() or archive.stat().st_size == 0:
        last = None
        for url in FOLDSEEK_URLS:
            last = ensure_download(url, archive, timeout=args.tool_download_timeout_sec)
            if last["returncode"] == 0 and archive.exists() and archive.stat().st_size > 0:
                break
            archive.unlink(missing_ok=True)
        if last and (not archive.exists() or archive.stat().st_size == 0):
            return {"tool": "foldseek", "status": "failed", "download": last}
    extract_dir = roots["tmp"] / "foldseek_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract = run_cmd(["tar", "-xzf", str(archive), "-C", str(extract_dir)], timeout=600)
    candidates = sorted(extract_dir.glob("foldseek"))
    if not candidates:
        return {"tool": "foldseek", "status": "failed", "extract": extract}
    shutil.rmtree(roots["foldseek"], ignore_errors=True)
    shutil.move(str(candidates[0]), str(roots["foldseek"]))
    ok = (roots["foldseek"] / "bin/foldseek").exists()
    return {"tool": "foldseek", "status": "complete" if ok else "failed", "path": str(roots["foldseek"] / "bin")}


def ensure_local_tools(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blast = tool_version("blastp", args)
    if not blast["path"] or "libmbedcrypto.so.7" in " ".join(blast.get("version", [])):
        rows.append(install_official_blast(args))
    else:
        rows.append({"tool": "blast", "status": "already_available", "path": blast["path"]})
    foldseek = tool_version("foldseek", args)
    if foldseek["path"]:
        rows.append({"tool": "foldseek", "status": "already_available", "path": foldseek["path"]})
    else:
        rows.append({"tool": "foldseek", "status": "skipped_optional", "reason": "structure control is allowed to remain pending while the primary strict sequence/homology benchmark runs"})
    payload = {
        "created_at": now_utc(),
        "attempts": rows,
        "resolved_tools": {
            name: tool_version(name, args)
            for name in ["screen", "blastp", "psiblast", "makeblastdb", "hmmsearch", "hhsearch", "foldseek", "mmseqs"]
        },
    }
    payload["warnings"] = []
    if not payload["resolved_tools"]["foldseek"]["path"]:
        payload["warnings"].append("foldseek_not_installed_during_tool_bootstrap")
    write_json(Path(args.out_root) / "vpf_tool_bootstrap.json", payload)
    return payload


def ensure_bucket_index(args: argparse.Namespace) -> Path:
    root = Path(args.bucket_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / PHROG_INDEX_NAME
    if path.exists():
        return path
    result = ensure_download(PHROG_INDEX_URL, path, timeout=1800)
    if result["returncode"] != 0 or not path.exists():
        raise RuntimeError(f"failed to download PHROG index: {result['output_tail']}")
    return path


def load_phrog_index(args: argparse.Namespace) -> pd.DataFrame:
    path = ensure_bucket_index(args)
    df = pd.read_csv(path, low_memory=False)
    df["functional_category"] = df["Category"].fillna("").astype(str).str.strip()
    df["functional_category_revised_v4"] = df.get("revised_category_v4", "").fillna("").astype(str).str.strip()
    df["phrog_family_id"] = df["#phrog"].astype(str).str.strip()
    df["family_size"] = pd.to_numeric(df["#prot"], errors="coerce").fillna(0).astype(int)
    df["avg_sequence_length"] = pd.to_numeric(df["Avg_#AA"], errors="coerce")
    return df


def load_label_order(inference_repo: Path) -> list[str]:
    classes_path = inference_repo / "model/model_unknown_80_07092023_lb.pkl"
    with classes_path.open("rb") as handle:
        obj = pickle.load(handle)
    if hasattr(obj, "classes_"):
        return [str(x) for x in obj.classes_]
    raise RuntimeError(f"unsupported label encoder type in {classes_path}")


def append_asset(
    rows: list[dict[str, Any]],
    *,
    name: str,
    asset_type: str,
    source: str,
    local_path: Path | None,
    version: str = "",
    revision: str = "",
    license_name: str = "",
    notes: str = "",
) -> None:
    checksum = ""
    if local_path and local_path.exists() and local_path.is_file():
        checksum = sha256_file(local_path)
    rows.append(
        {
            "asset_name": name,
            "asset_type": asset_type,
            "source": source,
            "version": version,
            "revision": revision,
            "checksum": checksum,
            "local_path": str(local_path) if local_path else "",
            "download_date": now_utc(),
            "license": license_name,
            "notes": notes,
        }
    )


def asset_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    inference_repo = Path(args.inference_repo)
    analysis_repo = Path(args.analysis_repo)
    bucket_index = ensure_bucket_index(args)
    rows: list[dict[str, Any]] = []
    append_asset(
        rows,
        name="VPF-PLM inference repository",
        asset_type="source_code",
        source=OFFICIAL_MODEL_ZENODO,
        local_path=inference_repo,
        version="zenodo-10182746",
        revision=path_revision(inference_repo),
        license_name="repository LICENSE",
        notes="Local extracted official inference repository.",
    )
    append_asset(
        rows,
        name="VPF-PLM analysis repository",
        asset_type="source_code",
        source=OFFICIAL_ANALYSIS_ZENODO,
        local_path=analysis_repo,
        version="zenodo-10182750",
        revision=path_revision(analysis_repo),
        license_name="repository LICENSE",
        notes="Local extracted official analysis/training repository.",
    )
    append_asset(
        rows,
        name="Official released classifier",
        asset_type="model",
        source=OFFICIAL_MODEL_ZENODO,
        local_path=inference_repo / "model/model_unknown_80_07092023/saved_model.pb",
        version="model_unknown_80_07092023",
        license_name="repository LICENSE",
        notes="Official TensorFlow SavedModel classifier for released inference pipeline.",
    )
    append_asset(
        rows,
        name="Official released label encoder",
        asset_type="label_encoder",
        source=OFFICIAL_MODEL_ZENODO,
        local_path=inference_repo / "model/model_unknown_80_07092023_lb.pkl",
        version="model_unknown_80_07092023_lb",
        license_name="repository LICENSE",
        notes="Recovered class ordering for official predictions.",
    )
    append_asset(
        rows,
        name="PHROG index revised v4 2022-10-29",
        asset_type="dataset_index",
        source=PHROG_INDEX_URL,
        local_path=bucket_index,
        version="revised_v4_10292022",
        license_name="official bucket terms",
        notes="Primary metadata table referenced by official training notebook.",
    )
    append_asset(
        rows,
        name="Precomputed HHsearch score files for known annotated PHROGs",
        asset_type="remote_homology_scores",
        source=OFFICIAL_ANALYSIS_ZENODO,
        local_path=analysis_repo / "figure2/figure_2b",
        version="bundled_with_analysis_repo",
        license_name="repository LICENSE",
        notes="Used to construct remote-homology graph and fair train-only HHsearch baseline.",
    )
    payload = {
        "created_at": now_utc(),
        "paper": OFFICIAL_PAPER,
        "official_model_zenodo": OFFICIAL_MODEL_ZENODO,
        "official_analysis_zenodo": OFFICIAL_ANALYSIS_ZENODO,
        "assets": rows,
        "label_order": load_label_order(inference_repo),
        "bucket_resources": {
            "phrog_index_revised_v4": PHROG_INDEX_URL,
            "phrog_faa_prefix": PHROG_FAA_PREFIX,
            "phrog_protbert_embeddings_prefix": PHROG_EMBED_PREFIX,
            "phrog_family_centroid_prefix": PHROG_CENTROID_PREFIX,
        },
    }
    write_json(out_root / "vpf_plm_external_assets.json", payload)
    return payload


def protocol_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    inference_repo = Path(args.inference_repo)
    label_order = load_label_order(inference_repo)
    payload = {
        "created_at": now_utc(),
        "official_prediction_unit": {
            "public_inference_repo": "individual proteins from FASTA",
            "strict_experiment_interpretation": "individual proteins grouped by PHROG family and remote-homology cluster",
            "evidence": [
                "scripts/predict_function.py emits per-protein predictions keyed by FASTA identifiers",
                "official training notebook loads protbert_bfd_embeddings_phrog/{phrog}.pkl and samples multiple sequence embeddings per family",
                "family centroids are available separately and are therefore not the primary official training unit",
            ],
        },
        "source_sequences": "Official PHROG family protein FASTA files from the official GCP bucket.",
        "family_definition": "PHROG family identifier (#phrog / phrog_#) from PHROG_index_revised_v4_10292022.csv.",
        "functional_label_definition": "High-level PHROG categories; formal strict benchmark excludes unknown function from primary evaluation.",
        "official_plm_representation": {
            "model_name": "Rostlab/prot_bert_bfd",
            "tokenizer": "BertTokenizer.from_pretrained(..., do_lower_case=False)",
            "encoder": "BertModel.from_pretrained('Rostlab/prot_bert_bfd')",
            "sequence_cleaning": "upper-case, remove spaces/dashes, replace U/Z/O/B with X",
            "tokenization": "space-separated amino acids",
            "truncation": "max_length=1024",
            "pooling": "mean over token embeddings excluding CLS/SEP with attention-mask weighting",
            "embedding_dimension": 1024,
        },
        "official_downstream_classifier": {
            "model_path": "model/model_unknown_80_07092023/",
            "label_encoder_path": "model/model_unknown_80_07092023_lb.pkl",
            "class_order": label_order,
            "architecture": [
                "Dense(512, relu)",
                "Dropout(0.2)",
                "Dense(256, relu)",
                "Dropout(0.2)",
                "Dense(128, relu)",
                "Dense(10, softmax)",
            ],
            "framework": "TensorFlow / Keras",
            "training_note": "Released pretrained head is reserved for sanity/reproduction; formal strict result trains a fresh head on frozen strict train only.",
        },
        "decision_thresholds": {
            "default": "argmax / highest probability class",
            "optional_efam_calibration": True,
            "calibration_usage": "Allowed for official-output reproduction only; not primary strict qualification.",
        },
        "strict_experiment_freeze": {
            "delta_sign_convention": "model_metric - baseline_metric",
            "grouping_minimum": "PHROG family",
            "preferred_bootstrap_unit": "remote-homology cluster",
        },
    }
    write_json(out_root / "vpf_plm_official_protocol_audit.json", payload)
    return payload


@dataclass
class HHScores:
    query: str
    target: str
    probability: float
    score: float
    columns: int
    target_length: int


def hhsearch_score_root(args: argparse.Namespace) -> Path:
    return Path(args.analysis_repo) / "figure2/figure_2b"


def parse_score_file(path: Path) -> list[HHScores]:
    query = path.stem
    rows: list[HHScores] = []
    start = False
    with path.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not start:
                if line.startswith("TARGET"):
                    start = True
                continue
            if not line.strip():
                continue
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 9:
                continue
            target = parts[0]
            if not target.startswith("phrog_"):
                continue
            try:
                rows.append(
                    HHScores(
                        query=query,
                        target=target,
                        target_length=int(parts[-7]),
                        columns=int(parts[-6]),
                        probability=float(parts[-3]),
                        score=float(parts[-2]),
                    )
                )
            except Exception:
                continue
    return rows


def build_remote_homology_clusters(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    df = load_phrog_index(args)
    known = df[df["functional_category"].isin(KNOWN_NONUNKNOWN_CATEGORIES)].copy()
    family_info = {
        row.phrog_family_id: {
            "functional_category": row.functional_category,
            "family_size": int(row.family_size),
        }
        for row in known.itertuples()
    }
    parent = {fam: fam for fam in family_info}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    edge_rows: list[dict[str, Any]] = []
    root = hhsearch_score_root(args)
    for score_file in sorted(root.glob("*/*/*.scores")):
        for score in parse_score_file(score_file):
            if score.query == score.target:
                continue
            if score.query not in family_info or score.target not in family_info:
                continue
            if score.probability < args.hhsearch_probability_threshold:
                continue
            if score.columns < args.hhsearch_min_columns:
                continue
            union(score.query, score.target)
            edge_rows.append(
                {
                    "query_family": score.query,
                    "target_family": score.target,
                    "probability": score.probability,
                    "score": score.score,
                    "aligned_columns": score.columns,
                    "target_length": score.target_length,
                }
            )
    cluster_map: dict[str, str] = {}
    root_to_cluster: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for fam in sorted(family_info):
        root_id = find(fam)
        if root_id not in root_to_cluster:
            root_to_cluster[root_id] = f"rhc_{len(root_to_cluster) + 1:05d}"
        cluster_id = root_to_cluster[root_id]
        cluster_map[fam] = cluster_id
        rows.append(
            {
                "phrog_family_id": fam,
                "remote_homology_cluster_id": cluster_id,
                "functional_category": family_info[fam]["functional_category"],
                "family_size": family_info[fam]["family_size"],
            }
        )
    write_csv(
        out_root / "vpf_remote_homology_clusters.csv",
        rows,
        ["phrog_family_id", "remote_homology_cluster_id", "functional_category", "family_size"],
    )
    write_csv(
        out_root / "vpf_remote_homology_edges.csv",
        edge_rows,
        ["query_family", "target_family", "probability", "score", "aligned_columns", "target_length"],
    )
    component_sizes = Counter(row["remote_homology_cluster_id"] for row in rows)
    summary = {
        "created_at": now_utc(),
        "family_count": len(rows),
        "cluster_count": len(component_sizes),
        "hhsearch_probability_threshold": args.hhsearch_probability_threshold,
        "hhsearch_min_columns": args.hhsearch_min_columns,
        "largest_cluster_size": max(component_sizes.values()) if component_sizes else 0,
        "singleton_clusters": int(sum(1 for size in component_sizes.values() if size == 1)),
        "edge_count": len(edge_rows),
    }
    write_json(out_root / "vpf_remote_homology_cluster_summary.json", summary)
    return summary


def dataset_freeze(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    df = load_phrog_index(args)
    known = df[df["functional_category"].isin(KNOWN_NONUNKNOWN_CATEGORIES)].copy()
    rows = []
    for row in known.itertuples():
        rows.append(
            {
                "protein_id": "",
                "sequence": "",
                "sequence_length": "",
                "phrog_family_id": row.phrog_family_id,
                "functional_category": row.functional_category,
                "family_size": int(row.family_size),
                "representative_sequence": "",
                "embedding_path": "",
                "structure_path": "",
                "source": "PHROG_index_revised_v4_10292022.csv",
                "dataset_version": "revised_v4_10292022",
                "annotation_source": "official PHROG revised category v4",
                "row_unit": "family_metadata_only",
                "prediction_unit": "individual protein",
                "avg_sequence_length": row.avg_sequence_length,
                "annotation": str(getattr(row, "revised_annotation_v4", "") or getattr(row, "Annotation", "") or ""),
                "pfam_hit": str(getattr(row, "Pfam_hit", "") or ""),
                "go_hit": str(getattr(row, "GO_hit", "") or ""),
                "ko_hit": str(getattr(row, "KO_hit", "") or ""),
            }
        )
    write_csv(
        out_root / "vpf_dataset_manifest.csv",
        rows,
        [
            "protein_id",
            "sequence",
            "sequence_length",
            "phrog_family_id",
            "functional_category",
            "family_size",
            "representative_sequence",
            "embedding_path",
            "structure_path",
            "source",
            "dataset_version",
            "annotation_source",
            "row_unit",
            "prediction_unit",
            "avg_sequence_length",
            "annotation",
            "pfam_hit",
            "go_hit",
            "ko_hit",
        ],
    )
    category_counts = known["functional_category"].value_counts().to_dict()
    protein_counts = known.groupby("functional_category")["family_size"].sum().astype(int).to_dict()
    payload = {
        "created_at": now_utc(),
        "family_count": int(len(known)),
        "category_count": int(known["functional_category"].nunique()),
        "per_category_family_counts": category_counts,
        "per_category_protein_counts": protein_counts,
        "note": "Family metadata frozen before sequence/embedding acquisition.",
    }
    write_json(out_root / "vpf_dataset_freeze_summary.json", payload)
    return payload


def split_clusters(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    dataset = pd.read_csv(out_root / "vpf_dataset_manifest.csv")
    clusters = pd.read_csv(out_root / "vpf_remote_homology_clusters.csv")
    frame = dataset.merge(clusters, on=["phrog_family_id", "functional_category", "family_size"], how="inner")
    cluster_rows = []
    for cluster_id, group in frame.groupby("remote_homology_cluster_id"):
        fam_counts = group["functional_category"].value_counts().to_dict()
        prot_counts = group.groupby("functional_category")["family_size"].sum().astype(int).to_dict()
        cluster_rows.append(
            {
                "remote_homology_cluster_id": cluster_id,
                "family_count": int(len(group)),
                "protein_count": int(group["family_size"].sum()),
                "family_counts_by_category": fam_counts,
                "protein_counts_by_category": prot_counts,
            }
        )
    totals_by_cat = frame["functional_category"].value_counts().to_dict()
    split_targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    current = {split: Counter() for split in split_targets}
    assignment: dict[str, str] = {}
    total_families = len(frame)
    target_total_by_split = {split_name: split_targets[split_name] * total_families for split_name in split_targets}
    target_cat_by_split = {
        split_name: {cat: split_targets[split_name] * count for cat, count in totals_by_cat.items()}
        for split_name in split_targets
    }

    def score_candidate(split_name: str, row: dict[str, Any]) -> tuple[float, float, float, int, str]:
        current_total = sum(current[split_name].values())
        after_total = current_total + row["family_count"]
        total_overflow = max(0.0, after_total - target_total_by_split[split_name])
        category_need = 0.0
        category_overflow = 0.0
        for cat, add_count in row["family_counts_by_category"].items():
            before = current[split_name][cat]
            target = target_cat_by_split[split_name][cat]
            category_need += max(0.0, target - before)
            category_overflow += max(0.0, before + add_count - target)
        priority = {"train": 0, "validation": 1, "test": 2}[split_name]
        return (-max(0.0, target_total_by_split[split_name] - current_total), total_overflow + category_overflow, -category_need, priority, split_name)

    for row in sorted(cluster_rows, key=lambda r: (-r["family_count"], -r["protein_count"], r["remote_homology_cluster_id"])):
        chosen = min(score_candidate(split_name, row) for split_name in split_targets)[4]
        assignment[row["remote_homology_cluster_id"]] = chosen
        current[chosen].update(row["family_counts_by_category"])

    frame["split"] = frame["remote_homology_cluster_id"].map(assignment)
    rows = frame[["phrog_family_id", "functional_category", "family_size", "remote_homology_cluster_id", "split"]].sort_values(["split", "functional_category", "phrog_family_id"])
    write_csv(
        out_root / "vpf_split_manifest.csv",
        rows.to_dict(orient="records"),
        ["phrog_family_id", "functional_category", "family_size", "remote_homology_cluster_id", "split"],
    )
    per_cat = rows.groupby(["split", "functional_category"])["phrog_family_id"].count().astype(int).reset_index()
    summary = {
        "created_at": now_utc(),
        "split_counts_families": rows.groupby("split")["phrog_family_id"].count().astype(int).to_dict(),
        "cluster_count": int(rows["remote_homology_cluster_id"].nunique()),
        "per_split_category_family_counts": [
            {
                "split": str(r["split"]),
                "functional_category": str(r["functional_category"]),
                "family_count": int(r["phrog_family_id"]),
            }
            for r in per_cat.to_dict(orient="records")
        ],
    }
    write_json(out_root / "vpf_split_plan_summary.json", summary)
    return summary


def family_asset_dir(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.out_root) / "cache"
    return {
        "root": root,
        "fasta": root / "family_fastas",
        "embed": root / "family_embeddings",
        "centroid": root / "family_centroids",
    }


def parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open() as handle:
        for record in SeqIO.parse(handle, "fasta"):
            rows.append((str(record.id), str(record.seq).upper()))
    return rows


def acquire_sequences(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    cache = family_asset_dir(args)
    for p in cache.values():
        p.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(out_root / "vpf_split_manifest.csv")
    metadata = pd.read_csv(out_root / "vpf_dataset_manifest.csv")
    meta_by_family = metadata.set_index("phrog_family_id").to_dict(orient="index")
    manifest_path = out_root / "vpf_sequence_manifest.csv"
    progress_path = out_root / "vpf_sequence_acquisition_progress.json"
    previous_progress = read_json(progress_path)
    completed = set(previous_progress.get("completed_families", []))
    failure_counts: dict[str, int] = {
        str(k): int(v) for k, v in previous_progress.get("failure_counts", {}).items()
    }
    last_errors: dict[str, str] = {
        str(k): str(v) for k, v in previous_progress.get("last_errors", {}).items()
    }
    existing_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open() as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_families = {row["phrog_family_id"] for row in existing_rows}
    if existing_families:
        completed.update(existing_families)
    rows = existing_rows[:]
    family_rows = split.sort_values(["split", "phrog_family_id"]).to_dict(orient="records")
    warnings: list[str] = []
    started = time.time()
    processed = 0
    no_progress_passes = 0
    total_families = len(family_rows)
    field_order = [
        "protein_id",
        "sequence",
        "sequence_length",
        "phrog_family_id",
        "functional_category",
        "family_size",
        "remote_homology_cluster_id",
        "split",
        "embedding_family_path",
        "embedding_row_index",
        "family_centroid_path",
        "source_fasta_path",
        "source",
        "dataset_version",
        "annotation_source",
        "annotation",
        "pfam_hit",
        "go_hit",
        "ko_hit",
    ]
    while len(completed) < total_families:
        pass_progress = 0
        pass_failures: list[str] = []
        pending_rows = [row for row in family_rows if str(row["phrog_family_id"]) not in completed]
        pending_rows.sort(key=lambda row: (failure_counts.get(str(row["phrog_family_id"]), 0), row["split"], str(row["phrog_family_id"])))
        for fam_row in pending_rows:
            family = str(fam_row["phrog_family_id"])
            fasta_path = cache["fasta"] / f"{family}.faa"
            embed_path = cache["embed"] / f"{family}.pkl"
            centroid_path = cache["centroid"] / f"{family}.pkl"
            try:
                if not fasta_path.exists():
                    result = ensure_download(f"{PHROG_FAA_PREFIX}/{family}.faa", fasta_path, timeout=args.family_download_timeout_sec, quiet=True)
                    if result["returncode"] != 0:
                        raise RuntimeError(f"failed to download FASTA: {result['output_tail']}")
                if not embed_path.exists():
                    result = ensure_download(f"{PHROG_EMBED_PREFIX}/{family}.pkl", embed_path, timeout=args.family_download_timeout_sec, quiet=True)
                    if result["returncode"] != 0:
                        raise RuntimeError(f"failed to download embeddings: {result['output_tail']}")
                if args.fetch_family_centroids and not centroid_path.exists():
                    ensure_download(f"{PHROG_CENTROID_PREFIX}/{family}.pkl", centroid_path, timeout=args.family_download_timeout_sec, quiet=True)
                seq_records = parse_fasta_records(fasta_path)
                with embed_path.open("rb") as handle:
                    emb = pickle.load(handle)
                if not isinstance(emb, np.ndarray):
                    raise TypeError(f"unexpected embedding object: {type(emb)}")
                usable = min(len(seq_records), int(emb.shape[0]))
                if usable == 0:
                    raise RuntimeError("no usable proteins")
                if len(seq_records) != int(emb.shape[0]):
                    warnings.append(f"{family}: fasta_count={len(seq_records)} embedding_count={int(emb.shape[0])}; truncated_to={usable}")
                meta = meta_by_family[family]
                for idx in range(usable):
                    protein_id, sequence = seq_records[idx]
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "sequence": sequence,
                            "sequence_length": len(sequence),
                            "phrog_family_id": family,
                            "functional_category": fam_row["functional_category"],
                            "family_size": int(fam_row["family_size"]),
                            "remote_homology_cluster_id": fam_row["remote_homology_cluster_id"],
                            "split": fam_row["split"],
                            "embedding_family_path": str(embed_path),
                            "embedding_row_index": idx,
                            "family_centroid_path": str(centroid_path) if centroid_path.exists() else "",
                            "source_fasta_path": str(fasta_path),
                            "source": "official GCS PHROG family FASTA + ProtBERT embeddings",
                            "dataset_version": "revised_v4_10292022",
                            "annotation_source": meta["annotation_source"],
                            "annotation": meta["annotation"],
                            "pfam_hit": meta["pfam_hit"],
                            "go_hit": meta["go_hit"],
                            "ko_hit": meta["ko_hit"],
                        }
                    )
                completed.add(family)
                failure_counts.pop(family, None)
                last_errors.pop(family, None)
                processed += 1
                pass_progress += 1
                if processed % 25 == 0:
                    write_csv(manifest_path, rows, field_order)
                    write_json(
                        progress_path,
                        {
                            "updated_at": now_utc(),
                            "completed_families": sorted(completed),
                            "pending_families": [str(row["phrog_family_id"]) for row in family_rows if str(row["phrog_family_id"]) not in completed],
                            "failure_counts": failure_counts,
                            "last_errors": last_errors,
                            "warning_count": len(warnings),
                        },
                    )
                    log(args, f"sequence acquisition progress: {len(completed)}/{total_families} families")
            except Exception as exc:
                failure_counts[family] = failure_counts.get(family, 0) + 1
                last_errors[family] = str(exc)
                pass_failures.append(family)
                continue
        write_csv(manifest_path, rows, field_order)
        write_json(
            progress_path,
            {
                "updated_at": now_utc(),
                "completed_families": sorted(completed),
                "pending_families": [str(row["phrog_family_id"]) for row in family_rows if str(row["phrog_family_id"]) not in completed],
                "failure_counts": failure_counts,
                "last_errors": last_errors,
                "warning_count": len(warnings),
                "no_progress_passes": no_progress_passes,
            },
        )
        if len(completed) >= total_families:
            break
        if pass_progress > 0:
            no_progress_passes = 0
            log(args, f"sequence acquisition progress: {len(completed)}/{total_families} families")
            continue
        no_progress_passes += 1
        retry_count = min(len(pass_failures), 20)
        log(
            args,
            "sequence acquisition retry pass without progress; "
            f"pending={total_families - len(completed)} no_progress_passes={no_progress_passes} "
            f"sample_failed_families={pass_failures[:retry_count]}",
        )
        if no_progress_passes >= args.family_max_no_progress_passes:
            sample = {family: last_errors.get(family, "") for family in sorted(pass_failures)[:10]}
            raise RuntimeError(
                "sequence acquisition exhausted retry budget without progress; "
                f"pending_families={total_families - len(completed)} sample_errors={sample}"
            )
        time.sleep(args.family_retry_sleep_sec)
    write_csv(manifest_path, rows, field_order)
    report = {
        "created_at": now_utc(),
        "family_count": int(len(completed)),
        "protein_count": int(len(rows)),
        "warning_count": len(warnings),
        "warnings_sample": warnings[:100],
        "runtime_sec": round(time.time() - started, 2),
        "cache_root": str(cache["root"]),
    }
    write_json(
        progress_path,
        {
            "updated_at": now_utc(),
            "completed_families": sorted(completed),
            "pending_families": [],
            "failure_counts": {},
            "last_errors": {},
            "warning_count": len(warnings),
            "no_progress_passes": 0,
        },
    )
    write_json(out_root / "vpf_sequence_acquisition_report.json", report)
    return report


def sequence_leakage_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    df = pd.read_csv(out_root / "vpf_sequence_manifest.csv")
    leakage = {
        "protein_id_overlap": False,
        "exact_sequence_overlap": False,
        "phrog_family_overlap": False,
        "remote_homology_cluster_overlap": False,
        "duplicate_embedding_row_overlap": False,
        "duplicate_structure_overlap": False,
        "label_leakage_from_path": False,
    }
    by_split = {name: frame.copy() for name, frame in df.groupby("split")}
    for a, a_df in by_split.items():
        for b, b_df in by_split.items():
            if a >= b:
                continue
            leakage["protein_id_overlap"] |= bool(set(a_df["protein_id"]) & set(b_df["protein_id"]))
            leakage["exact_sequence_overlap"] |= bool(set(a_df["sequence"]) & set(b_df["sequence"]))
            leakage["phrog_family_overlap"] |= bool(set(a_df["phrog_family_id"]) & set(b_df["phrog_family_id"]))
            leakage["remote_homology_cluster_overlap"] |= bool(set(a_df["remote_homology_cluster_id"]) & set(b_df["remote_homology_cluster_id"]))
            leakage["duplicate_embedding_row_overlap"] |= bool(set(zip(a_df["embedding_family_path"], a_df["embedding_row_index"])) & set(zip(b_df["embedding_family_path"], b_df["embedding_row_index"])))
    payload = {
        "created_at": now_utc(),
        "status": "PASS" if not any(bool(v) for v in leakage.values()) else "FAIL",
        "protein_count": int(len(df)),
        "family_count": int(df["phrog_family_id"].nunique()),
        "remote_homology_cluster_count": int(df["remote_homology_cluster_id"].nunique()),
        "sample_counts": df["split"].value_counts().to_dict(),
        "leakage_checks": leakage,
        "category_counts_by_split": {
            split_name: frame["functional_category"].value_counts().to_dict()
            for split_name, frame in by_split.items()
        },
        "max_train_test_sequence_similarity": "not estimated beyond exact-match duplicate audit; remote-homology disjoint split enforced by HHsearch connected components",
    }
    write_json(out_root / "vpf_split_audit.json", payload)
    return payload


def dependency_preflight(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    commands = ["screen", "blastp", "psiblast", "makeblastdb", "hmmsearch", "hhsearch", "foldseek", "mmseqs"]
    tools = {name: tool_version(name, args) for name in commands}
    py = Path(args.python)
    module_status = {}
    for mod in ["tensorflow", "torch", "transformers", "sklearn", "pandas", "numpy", "Bio", "requests"]:
        result = run_cmd([str(py), "-c", f"import importlib; importlib.import_module('{mod}'); print('OK')"], timeout=120, env=command_env(args))
        module_status[mod] = "OK" if result["returncode"] == 0 else f"MISSING: {result['output_tail'].strip()}"
    hh_root = hhsearch_score_root(args)
    payload = {
        "created_at": now_utc(),
        "tools": tools,
        "python_modules": module_status,
        "hhsearch_score_files_present": len(list(hh_root.glob("*/*/*.scores"))),
        "critical_blockers": [],
        "warnings": [],
    }
    if not tools["screen"]["path"]:
        payload["critical_blockers"].append("screen_missing")
    if not Path(args.inference_repo).exists():
        payload["critical_blockers"].append("official_inference_repo_missing")
    if not Path(args.analysis_repo).exists():
        payload["critical_blockers"].append("official_analysis_repo_missing")
    if not tools["blastp"]["path"] or not tools["makeblastdb"]["path"]:
        payload["critical_blockers"].append("blast_missing")
    if not tools["psiblast"]["path"]:
        payload["warnings"].append("psiblast_missing")
    if not tools["hhsearch"]["path"]:
        payload["critical_blockers"].append("hhsearch_missing")
    if not tools["foldseek"]["path"]:
        payload["warnings"].append("foldseek_missing")
    if "MISSING" in module_status["tensorflow"]:
        payload["critical_blockers"].append("tensorflow_missing")
    write_json(out_root / "vpf_preflight_dependency_audit.json", payload)
    return payload


def official_sanity_report(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    repo = Path(args.inference_repo)
    sanity_root = out_root / "official_sanity"
    sanity_root.mkdir(parents=True, exist_ok=True)
    test_faa = repo / "test/test.faa"
    compare_pred = pd.read_csv(repo / "test/test_out_compare/test_function_predictions.csv")
    compare_prob = pd.read_csv(repo / "test/test_out_compare/test_functional_probabilities.csv", index_col=0)
    embed_result = run_cmd(
        [
            args.python,
            str(repo / "scripts/embed_faa.py"),
            "--faa_path",
            str(test_faa),
            "--save_dir",
            str(sanity_root),
        ],
        cwd=repo,
        timeout=args.official_sanity_timeout_sec,
        env=command_env(args),
    )
    pred_result = run_cmd(
        [
            args.python,
            str(repo / "scripts/predict_function.py"),
            "--save_dir",
            str(sanity_root),
            "--embed_path",
            str(sanity_root / "test_embeddings_dict.pkl"),
        ],
        cwd=repo,
        timeout=args.official_sanity_timeout_sec,
        env=command_env(args),
    )
    pred_path = sanity_root / "test_function_predictions.csv"
    prob_path = sanity_root / "test_functional_probabilities.csv"
    status = "SANITY_FAIL"
    reason = "official sanity outputs missing"
    max_prob_abs_diff = None
    prediction_match_rate = 0.0
    if pred_path.exists() and prob_path.exists():
        pred_df = pd.read_csv(pred_path)
        prob_df = pd.read_csv(prob_path, index_col=0)
        joined = pred_df.merge(compare_pred, on="protein_id", suffixes=("_new", "_ref"))
        prediction_match_rate = float(np.mean(joined["class_phrog_new"].astype(str) == joined["class_phrog_ref"].astype(str))) if len(joined) else 0.0
        common_cols = [col for col in compare_prob.columns if col in prob_df.columns]
        compare_aligned = compare_prob.loc[prob_df.index, common_cols].astype(float)
        prob_aligned = prob_df.loc[compare_aligned.index, common_cols].astype(float)
        max_prob_abs_diff = float(np.max(np.abs(compare_aligned.to_numpy() - prob_aligned.to_numpy())))
        if prediction_match_rate == 1.0 and max_prob_abs_diff <= 1e-4:
            status = "SANITY_PASS"
            reason = "official embedding + classifier pipeline reproduced reference outputs within tolerance"
        else:
            status = "SANITY_WARNING"
            reason = "official pipeline ran, but output drift exceeded strict tolerance"
    payload = {
        "created_at": now_utc(),
        "status": status,
        "reason": reason,
        "official_test_fasta": str(test_faa),
        "output_dir": str(sanity_root),
        "embed_returncode": embed_result["returncode"],
        "predict_returncode": pred_result["returncode"],
        "embed_output_tail": embed_result["output_tail"],
        "predict_output_tail": pred_result["output_tail"],
        "prediction_match_rate": prediction_match_rate,
        "max_probability_absolute_difference": max_prob_abs_diff,
        "strict_formal_may_proceed": status in {"SANITY_PASS", "SANITY_WARNING"},
    }
    write_json(out_root / "vpf_official_sanity_report.json", payload)
    return payload


def load_sequence_df(args: argparse.Namespace) -> pd.DataFrame:
    return pd.read_csv(Path(args.out_root) / "vpf_sequence_manifest.csv")


def labels_in_order() -> list[str]:
    return list(KNOWN_NONUNKNOWN_CATEGORIES)


def metric_row(
    model: str,
    representation: str,
    params: Mapping[str, Any],
    y_true_val: Sequence[str],
    y_pred_val: Sequence[str],
    y_true_test: Sequence[str],
    y_pred_test: Sequence[str],
    labels: Sequence[str],
    runtime: float,
    *,
    seed: str = "",
    status: str = "complete",
    reason: str = "",
) -> dict[str, Any]:
    return {
        "model": model,
        "representation": representation,
        "hyperparameters": json.dumps(dict(params), sort_keys=True),
        "validation_macro_f1": f1_score(y_true_val, y_pred_val, labels=list(labels), average="macro", zero_division=0),
        "validation_weighted_f1": f1_score(y_true_val, y_pred_val, labels=list(labels), average="weighted", zero_division=0),
        "validation_balanced_accuracy": balanced_accuracy_score(y_true_val, y_pred_val),
        "test_macro_f1": f1_score(y_true_test, y_pred_test, labels=list(labels), average="macro", zero_division=0),
        "test_weighted_f1": f1_score(y_true_test, y_pred_test, labels=list(labels), average="weighted", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_true_test, y_pred_test),
        "runtime": runtime,
        "seed": seed,
        "status": status,
        "not_available_reason": reason,
    }


def prediction_rows(
    model: str,
    representation: str,
    split_name: str,
    frame: pd.DataFrame,
    predictions: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for (_, row), pred in zip(frame.iterrows(), predictions):
        rows.append(
            {
                "model": model,
                "representation": representation,
                "split": split_name,
                "protein_id": str(row["protein_id"]),
                "phrog_family_id": str(row["phrog_family_id"]),
                "remote_homology_cluster_id": str(row["remote_homology_cluster_id"]),
                "true_label": str(row["functional_category"]),
                "predicted_label": str(pred),
            }
        )
    return rows


def aa_composition_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    hydrophobic = set("AILMFWVY")
    basic = set("KRH")
    acidic = set("DE")
    polar = set("STNQCGP")
    for row in df.itertuples():
        seq = str(row.sequence)
        n = max(1, len(seq))
        feats.append(
            [
                len(seq),
                sum(ch in hydrophobic for ch in seq) / n,
                sum(ch in basic for ch in seq) / n,
                sum(ch in acidic for ch in seq) / n,
                sum(ch in polar for ch in seq) / n,
                *[seq.count(aa) / n for aa in AA],
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def run_simple_baselines(args: argparse.Namespace) -> dict[str, Any]:
    df = load_sequence_df(args)
    labels = labels_in_order()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y_train = train["functional_category"].astype(str).to_numpy()
    y_val = val["functional_category"].astype(str).to_numpy()
    y_test = test["functional_category"].astype(str).to_numpy()
    rows = []
    preds = []
    majority = Counter(y_train).most_common(1)[0][0]
    val_major = [majority] * len(val)
    test_major = [majority] * len(test)
    rows.append(metric_row("majority_class", "class_prior", {}, y_val, val_major, y_test, test_major, labels, 0.0))
    preds.extend(prediction_rows("majority_class", "class_prior", "validation", val, val_major))
    preds.extend(prediction_rows("majority_class", "class_prior", "test", test, test_major))
    x_train = aa_composition_features(train)
    x_val = aa_composition_features(val)
    x_test = aa_composition_features(test)
    best_lr = None
    for c in [0.01, 0.1, 1.0, 10.0]:
        started = time.time()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=c, class_weight="balanced", multi_class="auto"))
        clf.fit(x_train, y_train)
        val_pred = clf.predict(x_val)
        score = f1_score(y_val, val_pred, labels=labels, average="macro", zero_division=0)
        if best_lr is None or score > best_lr[0]:
            best_lr = (score, c, clf, time.time() - started, val_pred)
    assert best_lr is not None
    test_pred = best_lr[2].predict(x_test)
    rows.append(metric_row("logistic_regression", "length_aa_composition_physchem", {"C": best_lr[1]}, y_val, best_lr[4], y_test, test_pred, labels, best_lr[3]))
    preds.extend(prediction_rows("logistic_regression", "length_aa_composition_physchem", "validation", val, best_lr[4]))
    preds.extend(prediction_rows("logistic_regression", "length_aa_composition_physchem", "test", test, test_pred))
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=args.split_seed, n_jobs=max(1, args.n_jobs))
    started = time.time()
    rf.fit(x_train, y_train)
    val_rf = rf.predict(x_val)
    test_rf = rf.predict(x_test)
    rows.append(metric_row("random_forest", "length_aa_composition_physchem", {"n_estimators": 300}, y_val, val_rf, y_test, test_rf, labels, time.time() - started))
    preds.extend(prediction_rows("random_forest", "length_aa_composition_physchem", "validation", val, val_rf))
    preds.extend(prediction_rows("random_forest", "length_aa_composition_physchem", "test", test, test_rf))
    write_csv(Path(args.out_root) / "vpf_simple_baselines.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_simple_baseline_predictions.csv", preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    return {"status": "complete", "rows": len(rows)}


def run_kmer_baselines(args: argparse.Namespace) -> dict[str, Any]:
    df = load_sequence_df(args)
    labels = labels_in_order()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    rows = []
    preds = []
    for k in [1, 2, 3, 4]:
        best = None
        for c in [0.1, 1.0, 10.0]:
            started = time.time()
            clf = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False, min_df=1),
                LogisticRegression(max_iter=2000, C=c, class_weight="balanced"),
            )
            clf.fit(train["sequence"].astype(str), train["functional_category"].astype(str))
            val_pred = clf.predict(val["sequence"].astype(str))
            score = f1_score(val["functional_category"], val_pred, labels=labels, average="macro", zero_division=0)
            if best is None or score > best[0]:
                best = (score, c, clf, time.time() - started, val_pred)
        assert best is not None
        test_pred = best[2].predict(test["sequence"].astype(str))
        representation = f"aa_{k}mer_tfidf"
        rows.append(metric_row("logistic_regression", representation, {"C": best[1], "k": k}, val["functional_category"], best[4], test["functional_category"], test_pred, labels, best[3]))
        preds.extend(prediction_rows("logistic_regression", representation, "validation", val, best[4]))
        preds.extend(prediction_rows("logistic_regression", representation, "test", test, test_pred))
    write_csv(Path(args.out_root) / "vpf_kmer_baselines.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_kmer_baseline_predictions.csv", preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    return {"status": "complete", "rows": len(rows)}


def write_fasta(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row['protein_id']}\n{row['sequence']}\n")


def parse_retrieval_predictions(path: Path, train_labels: Mapping[str, str], query_ids: Sequence[str], majority: str) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    if path.exists():
        with path.open(errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                qid, sid = parts[0], parts[1]
                label = train_labels.get(sid, "")
                if not label:
                    continue
                try:
                    score = float(parts[-1])
                except ValueError:
                    score = 0.0
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, label)
    return {qid: best.get(qid, (0.0, majority))[1] for qid in query_ids}


def run_blast_like(args: argparse.Namespace, program: str, model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = load_sequence_df(args)
    labels = labels_in_order()
    env = command_env(args)
    exe = shutil.which(program, path=env["PATH"])
    makeblastdb = shutil.which("makeblastdb", path=env["PATH"])
    if not exe or not makeblastdb:
        return (
            [{"model": model_name, "representation": "external_retrieval", "hyperparameters": "{}", "validation_macro_f1": "", "validation_weighted_f1": "", "validation_balanced_accuracy": "", "test_macro_f1": "", "test_weighted_f1": "", "test_balanced_accuracy": "", "runtime": "", "seed": "", "status": "NOT_AVAILABLE", "not_available_reason": f"missing {program}/makeblastdb"}],
            [],
        )
    work = Path(args.out_root) / "external_baselines" / program
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    train_fasta = work / "train.faa"
    db_prefix = work / "train_db"
    write_fasta(train_fasta, train.to_dict(orient="records"))
    run_cmd([makeblastdb, "-in", str(train_fasta), "-dbtype", "prot", "-out", str(db_prefix)], timeout=args.blast_timeout_sec, env=env)
    train_labels = dict(zip(train["protein_id"].astype(str), train["functional_category"].astype(str)))
    majority = train["functional_category"].astype(str).value_counts().idxmax()
    rows = []
    preds = []
    val_pred_labels = None
    for split_name, frame in [("validation", val), ("test", test)]:
        query_fasta = work / f"{split_name}.faa"
        out_tsv = work / f"{split_name}.tsv"
        write_fasta(query_fasta, frame.to_dict(orient="records"))
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
        result = run_cmd(command, timeout=args.blast_timeout_sec, env=env)
        if result["returncode"] != 0:
            return (
                [{"model": model_name, "representation": "external_retrieval", "hyperparameters": "{}", "validation_macro_f1": "", "validation_weighted_f1": "", "validation_balanced_accuracy": "", "test_macro_f1": "", "test_weighted_f1": "", "test_balanced_accuracy": "", "runtime": "", "seed": "", "status": "NOT_AVAILABLE", "not_available_reason": f"{program} failed: {result['output_tail']}"}],
                [],
            )
        pred_map = parse_retrieval_predictions(out_tsv, train_labels, frame["protein_id"].astype(str).tolist(), majority)
        pred_labels = [pred_map[qid] for qid in frame["protein_id"].astype(str)]
        preds.extend(prediction_rows(model_name, "external_retrieval", split_name, frame, pred_labels))
        if split_name == "validation":
            val_pred_labels = pred_labels
            runtime = time.time() - started
        else:
            assert val_pred_labels is not None
            rows.append(metric_row(model_name, "external_retrieval", {"program": program}, val["functional_category"], val_pred_labels, test["functional_category"], pred_labels, labels, runtime))
    return rows, preds


def run_hhsearch_family_baseline(args: argparse.Namespace) -> dict[str, Any]:
    df = load_sequence_df(args)
    labels = labels_in_order()
    edges = pd.read_csv(Path(args.out_root) / "vpf_remote_homology_edges.csv")
    family_split = pd.read_csv(Path(args.out_root) / "vpf_split_manifest.csv")
    fam_to_split = dict(zip(family_split["phrog_family_id"].astype(str), family_split["split"].astype(str)))
    fam_to_label = dict(zip(family_split["phrog_family_id"].astype(str), family_split["functional_category"].astype(str)))
    train_fams = {fam for fam, split_name in fam_to_split.items() if split_name == "train"}
    majority = family_split[family_split["split"] == "train"]["functional_category"].astype(str).value_counts().idxmax()
    best_target: dict[str, tuple[float, str]] = {}
    for row in edges.itertuples():
        query = str(row.query_family)
        target = str(row.target_family)
        if target not in train_fams:
            continue
        score = float(row.probability) * 1000.0 + float(row.score)
        if query not in best_target or score > best_target[query][0]:
            best_target[query] = (score, target)
    pred_family = {fam: fam_to_label.get(best_target[fam][1], majority) if fam in best_target else majority for fam in fam_to_split}
    rows = []
    preds = []
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    val_pred = [pred_family.get(fam, majority) for fam in val["phrog_family_id"].astype(str)]
    test_pred = [pred_family.get(fam, majority) for fam in test["phrog_family_id"].astype(str)]
    rows.append(metric_row("HHsearch_train_family_transfer", "remote_homology_graph", {"threshold_probability": args.hhsearch_probability_threshold, "min_columns": args.hhsearch_min_columns}, val["functional_category"], val_pred, test["functional_category"], test_pred, labels, 0.0))
    preds.extend(prediction_rows("HHsearch_train_family_transfer", "remote_homology_graph", "validation", val, val_pred))
    preds.extend(prediction_rows("HHsearch_train_family_transfer", "remote_homology_graph", "test", test, test_pred))
    write_csv(Path(args.out_root) / "vpf_hhsearch_baseline.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_hhsearch_baseline_predictions.csv", preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    return {"status": "complete", "rows": len(rows)}


def run_domain_baseline(args: argparse.Namespace) -> dict[str, Any]:
    df = load_sequence_df(args).copy()
    labels = labels_in_order()
    df["domain_text"] = (
        df["pfam_hit"].fillna("").astype(str)
        + " "
        + df["go_hit"].fillna("").astype(str)
        + " "
        + df["ko_hit"].fillna("").astype(str)
    ).str.strip()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    majority = train["functional_category"].astype(str).value_counts().idxmax()
    if not train["domain_text"].str.len().gt(0).any():
        rows = [{"model": "domain_tfidf_lr", "representation": "pfam_go_ko_text", "hyperparameters": "{}", "validation_macro_f1": "", "validation_weighted_f1": "", "validation_balanced_accuracy": "", "test_macro_f1": "", "test_weighted_f1": "", "test_balanced_accuracy": "", "runtime": "", "seed": "", "status": "NOT_AVAILABLE", "not_available_reason": "empty domain metadata"}]
        write_csv(Path(args.out_root) / "vpf_domain_baseline.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
        return {"status": "complete", "rows": len(rows)}
    best = None
    for c in [0.1, 1.0, 10.0]:
        started = time.time()
        clf = make_pipeline(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False, min_df=1),
            LogisticRegression(max_iter=3000, C=c, class_weight="balanced"),
        )
        clf.fit(train["domain_text"], train["functional_category"])
        val_pred = clf.predict(val["domain_text"])
        score = f1_score(val["functional_category"], val_pred, labels=labels, average="macro", zero_division=0)
        if best is None or score > best[0]:
            best = (score, c, clf, time.time() - started, val_pred)
    assert best is not None
    test_pred = best[2].predict(test["domain_text"])
    rows = [metric_row("domain_tfidf_lr", "pfam_go_ko_text", {"C": best[1]}, val["functional_category"], best[4], test["functional_category"], test_pred, labels, best[3])]
    preds = prediction_rows("domain_tfidf_lr", "pfam_go_ko_text", "validation", val, best[4]) + prediction_rows("domain_tfidf_lr", "pfam_go_ko_text", "test", test, test_pred)
    write_csv(Path(args.out_root) / "vpf_domain_baseline.csv", rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_domain_baseline_predictions.csv", preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    return {"status": "complete", "rows": len(rows)}


def load_family_embedding(path: str) -> np.ndarray:
    with Path(path).open("rb") as handle:
        arr = pickle.load(handle)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"expected numpy array embedding family, got {type(arr)} from {path}")
    return np.asarray(arr, dtype=np.float32)


def build_split_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    df = load_sequence_df(args)
    order_frames = {name: frame.reset_index(drop=True) for name, frame in df.groupby("split")}
    xs: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    ys: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for split_name, frame in order_frames.items():
        by_family = frame.groupby("embedding_family_path", sort=False)
        for embed_path, group in by_family:
            arr = load_family_embedding(str(embed_path))
            idx = group["embedding_row_index"].astype(int).to_numpy()
            xs[split_name].append(arr[idx])
            ys[split_name].extend(group["functional_category"].astype(str).tolist())
        if not xs[split_name]:
            raise RuntimeError(f"no embeddings collected for split {split_name}")
    stacked_x = {split_name: np.vstack(xs[split_name]).astype(np.float32) for split_name in xs}
    stacked_y = {split_name: np.asarray(ys[split_name], dtype=object) for split_name in ys}
    return stacked_x, stacked_y, order_frames


def keras_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_vpf_head(num_classes: int, lr: float) -> Any:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1024,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    return model


def run_plm_models(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    xs, ys, frames = build_split_arrays(args)
    labels = labels_in_order()
    enc = LabelEncoder()
    enc.fit(labels)
    y_train_int = enc.transform(ys["train"])
    y_val_int = enc.transform(ys["validation"])
    y_test_int = enc.transform(ys["test"])
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=len(labels))
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=len(labels))
    class_counts = Counter(ys["train"])
    total = float(sum(class_counts.values()))
    class_weight = {enc.transform([label])[0]: total / (len(class_counts) * count) for label, count in class_counts.items()}
    metric_rows = []
    pred_rows = []
    best_seed = None
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        keras_seed(seed)
        model = build_vpf_head(len(labels), args.plm_lr)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
        started = time.time()
        model.fit(
            xs["train"],
            y_train,
            validation_data=(xs["validation"], y_val),
            epochs=args.plm_epochs,
            batch_size=args.plm_batch_size,
            verbose=0,
            class_weight=class_weight,
            callbacks=callbacks,
        )
        val_prob = model.predict(xs["validation"], verbose=0)
        test_prob = model.predict(xs["test"], verbose=0)
        val_pred = enc.inverse_transform(np.argmax(val_prob, axis=1))
        test_pred = enc.inverse_transform(np.argmax(test_prob, axis=1))
        row = metric_row(
            "VPF_PLM_fresh_head",
            "official_protbert_bfd_embeddings",
            {"epochs": args.plm_epochs, "batch_size": args.plm_batch_size, "lr": args.plm_lr},
            ys["validation"],
            val_pred,
            ys["test"],
            test_pred,
            labels,
            time.time() - started,
            seed=str(seed),
        )
        metric_rows.append(row)
        pred_rows.extend(prediction_rows("VPF_PLM_fresh_head", f"official_protbert_bfd_embeddings_seed_{seed}", "validation", frames["validation"], val_pred))
        pred_rows.extend(prediction_rows("VPF_PLM_fresh_head", f"official_protbert_bfd_embeddings_seed_{seed}", "test", frames["test"], test_pred))
        if best_seed is None or float(row["validation_macro_f1"]) > float(best_seed["validation_macro_f1"]):
            best_seed = {
                "seed": seed,
                "validation_macro_f1": float(row["validation_macro_f1"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "test_pred": test_pred.tolist(),
                "val_pred": val_pred.tolist(),
            }
    assert best_seed is not None
    summary = {
        "created_at": now_utc(),
        "primary_seed": int(best_seed["seed"]),
        "primary_validation_macro_f1": best_seed["validation_macro_f1"],
        "primary_test_macro_f1": best_seed["test_macro_f1"],
        "seed_count": len(metric_rows),
        "mean_test_macro_f1": float(np.mean([float(row["test_macro_f1"]) for row in metric_rows])),
        "std_test_macro_f1": float(np.std([float(row["test_macro_f1"]) for row in metric_rows])),
        "min_test_macro_f1": float(np.min([float(row["test_macro_f1"]) for row in metric_rows])),
        "max_test_macro_f1": float(np.max([float(row["test_macro_f1"]) for row in metric_rows])),
    }
    write_csv(Path(args.out_root) / "vpf_plm_seed_results.csv", metric_rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_plm_seed_predictions.csv", pred_rows, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    write_json(Path(args.out_root) / "vpf_plm_primary_seed.json", summary)
    return summary


def collect_baseline_rows(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in [
        Path(args.out_root) / "vpf_simple_baselines.csv",
        Path(args.out_root) / "vpf_kmer_baselines.csv",
        Path(args.out_root) / "vpf_blast_baseline.csv",
        Path(args.out_root) / "vpf_psiblast_baseline.csv",
        Path(args.out_root) / "vpf_hhsearch_baseline.csv",
        Path(args.out_root) / "vpf_domain_baseline.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("no baseline metrics found")
    return pd.concat(frames, ignore_index=True)


def select_strongest_baseline(args: argparse.Namespace) -> dict[str, Any]:
    df = collect_baseline_rows(args)
    df = df[df["status"].fillna("complete").isin(["complete", "COMPLETE"])]
    if df.empty:
        raise RuntimeError("no complete baseline available")
    best = df.sort_values(["validation_macro_f1", "test_macro_f1"], ascending=False).iloc[0].to_dict()
    payload = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best.items()}
    write_json(Path(args.out_root) / "vpf_strongest_baseline.json", payload)
    return payload


def load_prediction_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def bootstrap_and_summarize(args: argparse.Namespace) -> dict[str, Any]:
    strongest = select_strongest_baseline(args)
    plm_summary = read_json(Path(args.out_root) / "vpf_plm_primary_seed.json")
    primary_seed = int(plm_summary["primary_seed"])
    seed_preds = load_prediction_table(Path(args.out_root) / "vpf_plm_seed_predictions.csv")
    seed_preds = seed_preds[(seed_preds["split"] == "test") & seed_preds["representation"].astype(str).str.endswith(f"seed_{primary_seed}")]
    baseline_map = {
        "vpf_simple_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_simple_baseline_predictions.csv") if (Path(args.out_root) / "vpf_simple_baseline_predictions.csv").exists() else pd.DataFrame(),
        "vpf_kmer_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_kmer_baseline_predictions.csv") if (Path(args.out_root) / "vpf_kmer_baseline_predictions.csv").exists() else pd.DataFrame(),
        "vpf_blast_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_blast_baseline_predictions.csv") if (Path(args.out_root) / "vpf_blast_baseline_predictions.csv").exists() else pd.DataFrame(),
        "vpf_psiblast_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_psiblast_baseline_predictions.csv") if (Path(args.out_root) / "vpf_psiblast_baseline_predictions.csv").exists() else pd.DataFrame(),
        "vpf_hhsearch_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_hhsearch_baseline_predictions.csv") if (Path(args.out_root) / "vpf_hhsearch_baseline_predictions.csv").exists() else pd.DataFrame(),
        "vpf_domain_baseline_predictions.csv": load_prediction_table(Path(args.out_root) / "vpf_domain_baseline_predictions.csv") if (Path(args.out_root) / "vpf_domain_baseline_predictions.csv").exists() else pd.DataFrame(),
    }
    baseline_preds = pd.concat([frame for frame in baseline_map.values() if not frame.empty], ignore_index=True)
    strongest_pred = baseline_preds[(baseline_preds["split"] == "test") & (baseline_preds["model"].astype(str) == str(strongest["model"])) & (baseline_preds["representation"].astype(str) == str(strongest["representation"]))]
    merged = seed_preds.merge(
        strongest_pred[["protein_id", "predicted_label"]].rename(columns={"predicted_label": "baseline_pred"}),
        on="protein_id",
        how="inner",
    )
    merged = merged.rename(columns={"predicted_label": "model_pred", "true_label": "true_label"})
    labels = labels_in_order()
    samples, summary = paired_grouped_prediction_bootstrap(
        merged,
        group_col="remote_homology_cluster_id",
        true_col="true_label",
        model_pred_col="model_pred",
        baseline_pred_col="baseline_pred",
        labels=labels,
        scorer=lambda y_true, y_pred, ls: f1_score(list(y_true), list(y_pred), labels=list(ls), average="macro", zero_division=0),
        n_valid=args.n_bootstrap,
        max_attempts=args.bootstrap_max_attempts,
        seed=args.bootstrap_seed,
        model_score_key="plm_macro_f1",
        baseline_score_key="baseline_macro_f1",
        delta_key="delta_plm_minus_baseline",
        bootstrap_unit="remote_homology_cluster_id",
        invalid_reason="bootstrap sample missing at least one functional category",
    )
    summary["strongest_baseline"] = {"model": strongest["model"], "representation": strongest["representation"]}
    write_csv(Path(args.out_root) / "vpf_bootstrap_samples.csv", samples, ["replicate", "plm_macro_f1", "baseline_macro_f1", "delta_plm_minus_baseline"])
    write_json(Path(args.out_root) / "vpf_bootstrap_summary.json", summary)
    return summary


def strongest_baseline_family(preds: pd.DataFrame) -> pd.DataFrame:
    return preds.groupby(["model", "representation"], as_index=False)["test_macro_f1"].max()


def per_category_analysis(args: argparse.Namespace) -> dict[str, Any]:
    strongest = read_json(Path(args.out_root) / "vpf_strongest_baseline.json")
    plm_summary = read_json(Path(args.out_root) / "vpf_plm_primary_seed.json")
    primary_seed = int(plm_summary["primary_seed"])
    plm_preds = pd.read_csv(Path(args.out_root) / "vpf_plm_seed_predictions.csv")
    plm_test = plm_preds[(plm_preds["split"] == "test") & plm_preds["representation"].astype(str).str.endswith(f"seed_{primary_seed}")].copy()
    baseline_preds = []
    for path in [
        Path(args.out_root) / "vpf_simple_baseline_predictions.csv",
        Path(args.out_root) / "vpf_kmer_baseline_predictions.csv",
        Path(args.out_root) / "vpf_blast_baseline_predictions.csv",
        Path(args.out_root) / "vpf_psiblast_baseline_predictions.csv",
        Path(args.out_root) / "vpf_hhsearch_baseline_predictions.csv",
        Path(args.out_root) / "vpf_domain_baseline_predictions.csv",
    ]:
        if path.exists():
            baseline_preds.append(pd.read_csv(path))
    base = pd.concat(baseline_preds, ignore_index=True)
    base = base[(base["split"] == "test") & (base["model"].astype(str) == str(strongest["model"])) & (base["representation"].astype(str) == str(strongest["representation"]))].copy()
    rows = []
    labels = labels_in_order()
    for label in labels:
        plm_true = (plm_test["true_label"].astype(str) == label).astype(int)
        plm_pred = (plm_test["predicted_label"].astype(str) == label).astype(int)
        base_true = (base["true_label"].astype(str) == label).astype(int)
        base_pred = (base["predicted_label"].astype(str) == label).astype(int)
        p_p, p_r, p_f1, support = precision_recall_fscore_support(plm_true, plm_pred, average="binary", zero_division=0)
        b_p, b_r, b_f1, _ = precision_recall_fscore_support(base_true, base_pred, average="binary", zero_division=0)
        rows.append(
            {
                "category": label,
                "support": int(support),
                "precision": p_p,
                "recall": p_r,
                "f1": p_f1,
                "best_plm_f1": p_f1,
                "best_baseline_f1": b_f1,
                "delta": p_f1 - b_f1,
            }
        )
    write_csv(Path(args.out_root) / "vpf_per_category_metrics.csv", rows, ["category", "support", "precision", "recall", "f1", "best_plm_f1", "best_baseline_f1", "delta"])
    cm_plm = confusion_matrix(plm_test["true_label"], plm_test["predicted_label"], labels=labels)
    cm_base = confusion_matrix(base["true_label"], base["predicted_label"], labels=labels)
    cm_rows = []
    for kind, matrix in [("plm_primary", cm_plm), ("strongest_baseline", cm_base)]:
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm_rows.append({"matrix": kind, "true_label": true_label, "predicted_label": pred_label, "count": int(matrix[i, j])})
    write_csv(Path(args.out_root) / "vpf_confusion_matrices.csv", cm_rows, ["matrix", "true_label", "predicted_label", "count"])
    return {"status": "complete", "rows": len(rows)}


def smoke_report(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    split = read_json(out_root / "vpf_split_audit.json")
    preflight = read_json(out_root / "vpf_preflight_dependency_audit.json")
    sanity = read_json(out_root / "vpf_official_sanity_report.json")
    passed = bool(split) and split.get("status") == "PASS" and bool(preflight) and not preflight.get("critical_blockers")
    payload = {
        "created_at": now_utc(),
        "status": "pass" if passed and sanity.get("status") in {"SANITY_PASS", "SANITY_WARNING"} else "fail",
        "checks": {
            "official_metadata_loaded": resume_ready_csv(out_root / "vpf_dataset_manifest.csv"),
            "remote_homology_grouping_built": resume_ready_csv(out_root / "vpf_remote_homology_clusters.csv"),
            "strict_split_built": bool(split),
            "split_audit_passed": split.get("status") == "PASS",
            "official_sanity_passed": sanity.get("status") in {"SANITY_PASS", "SANITY_WARNING"},
            "sequence_manifest_ready": resume_ready_csv(out_root / "vpf_sequence_manifest.csv"),
        },
    }
    write_json(out_root / "vpf_smoke_report.json", payload)
    return payload


def final_summary(args: argparse.Namespace) -> dict[str, Any]:
    baseline = read_json(Path(args.out_root) / "vpf_strongest_baseline.json")
    plm_summary = read_json(Path(args.out_root) / "vpf_plm_primary_seed.json")
    bootstrap = read_json(Path(args.out_root) / "vpf_bootstrap_summary.json")
    split = read_json(Path(args.out_root) / "vpf_split_audit.json")
    tool_audit = read_json(Path(args.out_root) / "vpf_preflight_dependency_audit.json")
    seed_rows = pd.read_csv(Path(args.out_root) / "vpf_plm_seed_results.csv")
    baseline_rows = collect_baseline_rows(args)
    simple_best = baseline_rows[baseline_rows["representation"].astype(str).str.contains("class_prior|composition|kmer|aa_", regex=True, na=False)]["test_macro_f1"].astype(float).max()
    homology_best = baseline_rows[baseline_rows["model"].astype(str).str.contains("BLAST|HHsearch|PSI", regex=True, na=False)]["test_macro_f1"].astype(float).max()
    domain_df = baseline_rows[baseline_rows["model"].astype(str).str.contains("domain", case=False, na=False)]
    domain_best = float(domain_df["test_macro_f1"].astype(float).max()) if not domain_df.empty else None
    structure_available = bool(tool_audit.get("tools", {}).get("foldseek", {}).get("path"))
    strongest_test = float(baseline["test_macro_f1"])
    plm_test = float(plm_summary["primary_test_macro_f1"])
    delta = plm_test - strongest_test
    seed_deltas = [float(row["test_macro_f1"]) - strongest_test for _, row in seed_rows.iterrows()]
    positive_seed_count = int(sum(d > 0 for d in seed_deltas))
    ci_low = bootstrap.get("ci95_low")
    status = "NO_QUALIFYING_HEADROOM"
    if delta <= 0 and strongest_test >= simple_best:
        status = "NO_QUALIFYING_HEADROOM"
    if delta <= 0 and strongest_test >= homology_best:
        status = "HOMOLOGY_BASELINE_SATURATED"
    if delta <= 0 and simple_best >= plm_test:
        status = "SIMPLE_BASELINE_SATURATED"
    if domain_best is not None and domain_best >= plm_test:
        status = "DOMAIN_BASELINE_SATURATED"
    if delta > 0 and (ci_low is None or ci_low <= 0 or positive_seed_count < 4):
        status = "OOD_SIGNAL_UNSTABLE"
    if delta > 0 and delta < 0.03:
        status = "NO_QUALIFYING_HEADROOM"
    if delta >= 0.03 and ci_low is not None and ci_low > 0 and positive_seed_count >= 4:
        status = "PRELIMINARILY_QUALIFIED_PENDING_STRUCTURE_CONTROL" if not structure_available else "PRELIMINARILY_QUALIFIED"
    payload = {
        "created_at": now_utc(),
        "final_status": status,
        "viral_search_status": "CLOSED" if status not in {"PRELIMINARILY_QUALIFIED", "PRELIMINARILY_QUALIFIED_PENDING_STRUCTURE_CONTROL"} else "OPEN_PENDING_NEXT_STAGE_DESIGN",
        "strict_ood_split": split,
        "official_model": {
            "plm": "Rostlab/prot_bert_bfd",
            "representation": "official precomputed ProtBERT-BFD per-protein embeddings + fresh strict-train head",
        },
        "primary_plm": {
            "seed": plm_summary["primary_seed"],
            "test_macro_f1": plm_test,
            "validation_macro_f1": plm_summary["primary_validation_macro_f1"],
        },
        "strongest_non_foundation_baseline": baseline,
        "deltas": {
            "overall": delta,
            "simple": plm_test - float(simple_best),
            "homology": plm_test - float(homology_best),
            "domain": None if domain_best is None else plm_test - domain_best,
            "structure": None,
        },
        "bootstrap_summary": bootstrap,
        "seed_stability": {
            "seed_count": int(len(seed_deltas)),
            "positive_seed_count": positive_seed_count,
            "worst_seed_excess": float(min(seed_deltas)),
            "mean_seed_excess": float(np.mean(seed_deltas)),
            "std_seed_excess": float(np.std(seed_deltas)),
        },
        "structure_control": {
            "available": structure_available,
            "status": "PENDING" if not structure_available else "NOT_RUN_IN_THIS_CONTROLLER",
        },
        "gates": {
            "simple_headroom": plm_test > float(simple_best),
            "homology_headroom": plm_test > float(homology_best),
            "domain_headroom": True if domain_best is None else plm_test > float(domain_best),
            "strict_ood": split.get("status") == "PASS",
            "effect_size_ge_0_03": delta >= 0.03,
            "ci95_lower_gt_0": ci_low is not None and ci_low > 0,
            "seed_stability_4_of_5": positive_seed_count >= 4,
        },
    }
    write_json(Path(args.out_root) / "vpf_summary_report.json", payload)
    lines = [
        "# VPF-PLM Qualification Summary",
        "",
        f"- Final status: `{status}`",
        f"- Primary PLM test macro-F1: `{plm_test:.6f}`",
        f"- Strongest baseline: `{baseline['model']}` / `{baseline['representation']}` / `{strongest_test:.6f}`",
        f"- Overall delta (model - baseline): `{delta:.6f}`",
        f"- Bootstrap 95% CI: `[{bootstrap.get('ci95_low')}, {bootstrap.get('ci95_high')}]`",
        f"- Positive seeds: `{positive_seed_count}/{len(seed_deltas)}`",
        f"- Structure control available: `{structure_available}`",
    ]
    (Path(args.out_root) / "vpf_summary_report.md").write_text("\n".join(lines) + "\n")
    return payload


def run_homology_baselines(args: argparse.Namespace) -> dict[str, Any]:
    blast_rows, blast_preds = run_blast_like(args, "blastp", "BLASTp_nearest_train_protein")
    psiblast_rows, psiblast_preds = run_blast_like(args, "psiblast", "PSI-BLAST_nearest_train_protein")
    write_csv(Path(args.out_root) / "vpf_blast_baseline.csv", blast_rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_blast_baseline_predictions.csv", blast_preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    write_csv(Path(args.out_root) / "vpf_psiblast_baseline.csv", psiblast_rows, ["model", "representation", "hyperparameters", "validation_macro_f1", "validation_weighted_f1", "validation_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_balanced_accuracy", "runtime", "seed", "status", "not_available_reason"])
    write_csv(Path(args.out_root) / "vpf_psiblast_baseline_predictions.csv", psiblast_preds, ["model", "representation", "split", "protein_id", "phrog_family_id", "remote_homology_cluster_id", "true_label", "predicted_label"])
    hh = run_hhsearch_family_baseline(args)
    return {"status": "complete", "blast_rows": len(blast_rows), "psiblast_rows": len(psiblast_rows), "hh_rows": hh["rows"]}


def execute(args: argparse.Namespace) -> None:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    registry = read_json(registry_path(args)) or {
        "created_at": now_utc(),
        "status": "running",
        "experiment_id": stable_hash(f"vpf_plm_qualification|{now_utc()}")[:16],
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], timeout=30)["output_tail"].strip().splitlines()[-1],
        "random_seeds": {"split_seed": args.split_seed, "seed_base": args.seed_base, "n_seeds": args.n_seeds},
        "hhsearch_probability_threshold": args.hhsearch_probability_threshold,
        "hhsearch_min_columns": args.hhsearch_min_columns,
    }
    out_root = Path(args.out_root)
    try:
        write_json(registry_path(args), registry | {"status": "running", "current_stage": "start"})
        log(args, "starting VPF-PLM strict qualification controller")
        run_stage_with_resume(args, "asset_audit", asset_audit, lambda: resume_ready_json(out_root / "vpf_plm_external_assets.json"))
        run_stage_with_resume(args, "protocol_audit", protocol_audit, lambda: resume_ready_json(out_root / "vpf_plm_official_protocol_audit.json"))
        run_stage_with_resume(args, "tool_bootstrap", ensure_local_tools, lambda: resume_ready_json(out_root / "vpf_tool_bootstrap.json"))
        run_stage_with_resume(args, "dependency_preflight", dependency_preflight, lambda: resume_ready_json(out_root / "vpf_preflight_dependency_audit.json"))
        run_stage_with_resume(args, "dataset_freeze", dataset_freeze, lambda: resume_ready_csv(out_root / "vpf_dataset_manifest.csv") and resume_ready_json(out_root / "vpf_dataset_freeze_summary.json"))
        run_stage_with_resume(args, "remote_homology_clusters", build_remote_homology_clusters, lambda: resume_ready_csv(out_root / "vpf_remote_homology_clusters.csv") and resume_ready_json(out_root / "vpf_remote_homology_cluster_summary.json"))
        run_stage_with_resume(args, "strict_split", split_clusters, lambda: resume_ready_csv(out_root / "vpf_split_manifest.csv") and resume_ready_json(out_root / "vpf_split_plan_summary.json"))
        run_stage_with_resume(args, "sequence_acquisition", acquire_sequences, lambda: resume_ready_json(out_root / "vpf_sequence_acquisition_report.json") and resume_ready_csv(out_root / "vpf_sequence_manifest.csv", min_rows=100))
        run_stage_with_resume(args, "sequence_leakage_audit", sequence_leakage_audit, lambda: resume_ready_json(out_root / "vpf_split_audit.json"))
        sanity = run_stage_with_resume(args, "official_sanity", official_sanity_report, lambda: resume_ready_json(out_root / "vpf_official_sanity_report.json"))
        if isinstance(sanity, dict) and sanity.get("status") != "skipped":
            if sanity.get("status") == "SANITY_FAIL":
                raise RuntimeError("official sanity failed")
        run_stage_with_resume(args, "smoke_report", smoke_report, lambda: resume_ready_json(out_root / "vpf_smoke_report.json"))
        run_stage_with_resume(args, "simple_baselines", run_simple_baselines, lambda: resume_ready_csv(out_root / "vpf_simple_baselines.csv"), formal_experiment_started=True)
        run_stage_with_resume(args, "kmer_baselines", run_kmer_baselines, lambda: resume_ready_csv(out_root / "vpf_kmer_baselines.csv"), formal_experiment_started=True)
        run_stage_with_resume(args, "homology_baselines", run_homology_baselines, lambda: resume_ready_csv(out_root / "vpf_blast_baseline.csv") and resume_ready_csv(out_root / "vpf_psiblast_baseline.csv") and resume_ready_csv(out_root / "vpf_hhsearch_baseline.csv"), formal_experiment_started=True)
        run_stage_with_resume(args, "domain_baseline", run_domain_baseline, lambda: resume_ready_csv(out_root / "vpf_domain_baseline.csv"), formal_experiment_started=True)
        run_stage_with_resume(args, "plm_training", run_plm_models, lambda: resume_ready_csv(out_root / "vpf_plm_seed_results.csv") and resume_ready_json(out_root / "vpf_plm_primary_seed.json"), formal_experiment_started=True)
        run_stage_with_resume(args, "bootstrap_summary", bootstrap_and_summarize, lambda: resume_ready_json(out_root / "vpf_bootstrap_summary.json") and resume_ready_csv(out_root / "vpf_bootstrap_samples.csv"), formal_experiment_started=True)
        run_stage_with_resume(args, "per_category_analysis", per_category_analysis, lambda: resume_ready_csv(out_root / "vpf_per_category_metrics.csv") and resume_ready_csv(out_root / "vpf_confusion_matrices.csv"), formal_experiment_started=True)
        summary = run_stage_with_resume(args, "final_summary", final_summary, lambda: resume_ready_json(out_root / "vpf_summary_report.json"), formal_experiment_started=True)
        if isinstance(summary, dict) and summary.get("status") == "skipped":
            summary = read_json(out_root / "vpf_summary_report.json")
        write_json(
            registry_path(args),
            registry
            | {
                "status": "complete",
                "current_stage": "complete",
                "final_status": summary["final_status"],
                "official_repo_revision": path_revision(Path(args.inference_repo)),
                "dataset_version": "PHROG revised v4 2022-10-29",
                "dataset_hash": sha256_file(ensure_bucket_index(args)),
                "split_hash": sha256_file(out_root / "vpf_split_manifest.csv"),
                "plm_name": "Rostlab/prot_bert_bfd",
                "plm_revision": "official precomputed PHROG bucket embeddings",
                "end_time": now_utc(),
            }
        )
        update_status(args, "complete", "complete", final_status=summary["final_status"], formal_experiment_started=True)
    except Exception as exc:
        current_stage = read_json(status_path(args)).get("stage", "unknown")
        write_json(registry_path(args), registry | {"status": "blocked", "current_stage": current_stage, "blocked_at": now_utc(), "blocker": str(exc)})
        update_status(args, "blocked", current_stage, blocker=str(exc))
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
        "--inference-repo",
        str(Path(args.inference_repo)),
        "--analysis-repo",
        str(Path(args.analysis_repo)),
        "--bucket-root",
        str(Path(args.bucket_root)),
        "--tool-root",
        str(Path(args.tool_root)),
        "--local-tool-bin",
        str(Path(args.local_tool_bin)),
        "--local-tool-usr-bin",
        str(Path(args.local_tool_usr_bin)),
        "--log-file",
        str(Path(args.log_file)),
        "--hhsearch-probability-threshold",
        str(args.hhsearch_probability_threshold),
        "--hhsearch-min-columns",
        str(args.hhsearch_min_columns),
        "--split-seed",
        str(args.split_seed),
        "--seed-base",
        str(args.seed_base),
        "--n-seeds",
        str(args.n_seeds),
        "--plm-epochs",
        str(args.plm_epochs),
        "--plm-batch-size",
        str(args.plm_batch_size),
        "--plm-lr",
        str(args.plm_lr),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--bootstrap-max-attempts",
        str(args.bootstrap_max_attempts),
    ]
    cmd.append("--resume" if args.resume else "--no-resume")
    result = run_cmd(cmd, timeout=60, env=command_env(args))
    if result["returncode"] != 0:
        raise RuntimeError(result["output_tail"])
    update_status(args, "running", "screen_launched", screen_name=args.screen_name)
    print(f"launched screen {args.screen_name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-screen", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--screen-name", default="vpf_plm_qualification")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--inference-repo", default=str(DEFAULT_INFERENCE_REPO))
    parser.add_argument("--analysis-repo", default=str(DEFAULT_ANALYSIS_REPO))
    parser.add_argument("--bucket-root", default=str(DEFAULT_BUCKET_ROOT))
    parser.add_argument("--local-tool-bin", default=str(DEFAULT_LOCAL_TOOL_BIN))
    parser.add_argument("--local-tool-usr-bin", default=str(DEFAULT_LOCAL_TOOL_USR_BIN))
    parser.add_argument("--tool-root", default=str(DEFAULT_TOOL_ROOT))
    parser.add_argument("--hhsearch-probability-threshold", type=float, default=90.0)
    parser.add_argument("--hhsearch-min-columns", type=int, default=40)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--family-download-timeout-sec", type=int, default=7200)
    parser.add_argument("--family-retry-sleep-sec", type=int, default=30)
    parser.add_argument("--family-max-no-progress-passes", type=int, default=12)
    parser.add_argument("--tool-download-timeout-sec", type=int, default=7200)
    parser.add_argument("--official-sanity-timeout-sec", type=int, default=7200)
    parser.add_argument("--blast-timeout-sec", type=int, default=14400)
    parser.add_argument("--retrieval-max-targets", type=int, default=1)
    parser.add_argument("--psiblast-iterations", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed-base", type=int, default=20260813)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--plm-epochs", type=int, default=5)
    parser.add_argument("--plm-batch-size", type=int, default=256)
    parser.add_argument("--plm-lr", type=float, default=1e-4)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-max-attempts", type=int, default=100000)
    parser.add_argument("--fetch-family-centroids", action="store_true")
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
