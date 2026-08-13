"""Audit frozen PHIStruct failure mode, then gate EvoMIL launch.

This controller is deliberately conservative. It reconstructs comparable
per-sample PHIStruct predictions from the frozen formal artifacts without
changing the split or retuning hyperparameters, writes the requested failure
audit deliverables, then attempts only the official EvoMIL acquisition/audit
steps needed before any formal EvoMIL benchmark may start.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from phase2.signed_bootstrap import paired_grouped_prediction_bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/teacher1/miniconda3/envs/UT-p1/bin/python"
DEFAULT_PHISTRUCT_ROOT = PROJECT_ROOT / "data/phase2/phistruct_qualification"
DEFAULT_AUDIT_ROOT = DEFAULT_PHISTRUCT_ROOT / "phistruct_failure_audit"
DEFAULT_EVOMIL_ROOT = PROJECT_ROOT / "data/phase2/evomil_qualification"
DEFAULT_LOG = PROJECT_ROOT / "logs/phistruct_failure_audit_evomil_controller.log"
OFFICIAL_EVOMIL_REPO = "https://github.com/liudan111/EvoMIL"
OFFICIAL_EVOMIL_PAPER = "https://doi.org/10.1371/journal.pcbi.1012597"


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


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{now_utc()}] {message}"
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log_file).open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def status_path(args: argparse.Namespace) -> Path:
    return Path(args.audit_root) / "controller_status.json"


def update_status(args: argparse.Namespace, status: str, stage: str, **extra: Any) -> None:
    write_json(
        status_path(args),
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": stage,
            "phistruct_root": str(Path(args.phistruct_root)),
            "audit_root": str(Path(args.audit_root)),
            "evomil_root": str(Path(args.evomil_root)),
            "log_file": str(Path(args.log_file)),
            **extra,
        },
    )


def run_cmd(
    command: Sequence[str],
    *,
    cwd: Path = PROJECT_ROOT,
    timeout: int = 3600,
    retries: int = 0,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    attempts = []
    for attempt in range(retries + 1):
        started = time.time()
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        record = {
            "attempt": attempt + 1,
            "returncode": proc.returncode,
            "runtime_sec": time.time() - started,
            "output_tail": proc.stdout[-4000:],
        }
        attempts.append(record)
        if proc.returncode == 0:
            break
        if attempt < retries:
            time.sleep(retry_sleep)
    return attempts[-1] | {"attempts": attempts}


def normalize_key(value: object) -> str:
    return str(value).strip().split()[0]


def load_split(root: Path) -> pd.DataFrame:
    path = root / "split_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    required = {"rbp_id", "host_genus", "split", "phage_id", "sequence_cluster_40"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"split manifest missing columns: {missing}")
    df["rbp_key"] = df["rbp_id"].map(normalize_key)
    return df


def detect_column(cols: Iterable[str], candidates: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in cols}
    normalized = {str(c).lower().replace(" ", "_"): str(c) for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
        if cand.lower().replace(" ", "_") in normalized:
            return normalized[cand.lower().replace(" ", "_")]
    for cand in candidates:
        cand_lower = cand.lower()
        for col in cols:
            if cand_lower in str(col).lower():
                return str(col)
    return None


def load_embedding_frame(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, header=None, nrows=3, low_memory=False)
    headerless = bool(pd.notna(pd.to_numeric(sample.iloc[0, 2], errors="coerce")))
    if headerless:
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = ["rbp_id", "host_genus", *[f"emb_{i}" for i in range(df.shape[1] - 2)]]
        return df
    df = pd.read_csv(path, low_memory=False)
    id_col = detect_column(df.columns, ["rbp_id", "protein_id", "protein id", "accession"])
    host_col = detect_column(df.columns, ["host_genus", "host", "genus"])
    if not id_col:
        raise ValueError(f"could not detect embedding ID column in {path}")
    emb_cols = [c for c in df.columns if c not in {id_col, host_col} and pd.api.types.is_numeric_dtype(df[c])]
    out = df.rename(columns={id_col: "rbp_id"}).copy()
    out["host_genus"] = out[host_col] if host_col else ""
    return out[["rbp_id", "host_genus", *emb_cols]]


def labels_for(labels: Sequence[str]) -> tuple[LabelEncoder, np.ndarray]:
    enc = LabelEncoder()
    return enc, enc.fit_transform(list(labels))


def metric_bundle(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(list(y_true), list(y_pred))),
    }


def per_genus_rows(model: str, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        list(y_true),
        list(y_pred),
        labels=list(labels),
        zero_division=0,
    )
    rows = []
    for idx, genus in enumerate(labels):
        true_arr = np.asarray(y_true)
        pred_arr = np.asarray(y_pred)
        tp = int(((true_arr == genus) & (pred_arr == genus)).sum())
        fp = int(((true_arr != genus) & (pred_arr == genus)).sum())
        fn = int(((true_arr == genus) & (pred_arr != genus)).sum())
        rows.append(
            {
                "model": model,
                "host_genus": genus,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def confusion_rows(model: str, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[dict[str, Any]]:
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(labels))
    rows = []
    for i, true_label in enumerate(labels):
        row = {"model": model, "true_host": true_label}
        for j, pred_label in enumerate(labels):
            row[f"pred_{pred_label}"] = int(cm[i, j])
        rows.append(row)
    return rows


def selected_saprot_c(root: Path) -> float:
    path = root / "plm_results.csv"
    if not path.exists():
        return 0.01
    df = pd.read_csv(path)
    rows = df[(df["model"] == "SaProt") & (df["status"] == "complete")].copy()
    if rows.empty:
        return 0.01
    rows["validation_macro_f1_num"] = pd.to_numeric(rows["validation_macro_f1"], errors="coerce")
    best = rows.sort_values("validation_macro_f1_num", ascending=False).iloc[0]
    try:
        params = json.loads(str(best["hyperparameters"]))
        return float(params.get("selected_C", 0.01))
    except Exception:
        return 0.01


def reconstruct_saprot(args: argparse.Namespace, split_df: pd.DataFrame, labels: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    emb_path = PROJECT_ROOT / "data/external/phistruct/experiments/data/inphared/structure/rbp_saprot_relaxed_r3.csv"
    emb = load_embedding_frame(emb_path)
    emb["rbp_key"] = emb["rbp_id"].map(normalize_key)
    merged = split_df.merge(emb.drop(columns=["host_genus"], errors="ignore"), on="rbp_key", how="inner", suffixes=("", "_emb"))
    emb_cols = [c for c in merged.columns if str(c).startswith("emb_")]
    if not emb_cols:
        split_cols = set(split_df.columns)
        emb_cols = [c for c in merged.columns if c not in split_cols and pd.api.types.is_numeric_dtype(merged[c])]
    if merged.empty or not emb_cols:
        raise RuntimeError("SaProt embeddings do not overlap frozen split")
    c_value = selected_saprot_c(Path(args.phistruct_root))
    train = merged["split"] == "train"
    val = merged["split"] == "validation"
    test = merged["split"] == "test"
    enc, y = labels_for(merged["host_genus"].astype(str).tolist())
    x = merged[emb_cols].astype(float).to_numpy()
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=c_value, class_weight="balanced", random_state=args.seed),
    )
    clf.fit(x[train], y[train])
    pred_idx = clf.predict(x)
    merged = merged.copy()
    merged["saprot_pred"] = enc.inverse_transform(pred_idx)
    pred_df = merged[["rbp_id", "rbp_key", "host_genus", "phage_id", "sequence_cluster_40", "split", "saprot_pred"]].copy()
    metrics = {
        "model": "SaProt",
        "prediction_reconstruction": "frozen_split_retrained_head_fixed_selected_C_no_retuning",
        "selected_C": c_value,
        "embedding_path": str(emb_path),
        "embedding_sha256": sha256_file(emb_path),
        "n_train": int(train.sum()),
        "n_validation": int(val.sum()),
        "n_test": int(test.sum()),
        "validation": metric_bundle(merged.loc[val, "host_genus"], merged.loc[val, "saprot_pred"], labels),
        "test": metric_bundle(merged.loc[test, "host_genus"], merged.loc[test, "saprot_pred"], labels),
    }
    return pred_df, metrics


def parse_blast_predictions(root: Path, split_df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    blast_path = root / "external_baselines/blastp" / f"{split_name}.tsv"
    train = split_df[split_df["split"] == "train"].copy()
    query = split_df[split_df["split"] == split_name].copy()
    train_hosts = {normalize_key(row.rbp_id): str(row.host_genus) for row in train.itertuples()}
    train_phages = {normalize_key(row.rbp_id): str(row.phage_id) for row in train.itertuples()}
    train_clusters = {normalize_key(row.rbp_id): str(row.sequence_cluster_40) for row in train.itertuples()}
    majority = train["host_genus"].astype(str).value_counts().idxmax()
    best: dict[str, dict[str, Any]] = {}
    if blast_path.exists():
        with blast_path.open(errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                qid, sid, pident, qcov, evalue, bitscore = parts[:6]
                try:
                    score = float(bitscore)
                except ValueError:
                    score = -math.inf
                qkey = normalize_key(qid)
                if qkey not in best or score > float(best[qkey]["bitscore"]):
                    best[qkey] = {
                        "query_id": qid,
                        "top_hit_id": sid,
                        "top_hit_host": train_hosts.get(normalize_key(sid), majority),
                        "top_hit_phage_id": train_phages.get(normalize_key(sid), ""),
                        "top_hit_sequence_cluster_40": train_clusters.get(normalize_key(sid), ""),
                        "pident": pident,
                        "query_coverage": qcov,
                        "evalue": evalue,
                        "bitscore": score,
                    }
    rows = []
    for row in query.itertuples():
        qkey = normalize_key(row.rbp_id)
        hit = best.get(qkey)
        pred = hit["top_hit_host"] if hit else majority
        rows.append(
            {
                "split": split_name,
                "query_id": row.rbp_id,
                "true_host": row.host_genus,
                "predicted_host": pred,
                "top_hit_id": "" if not hit else hit["top_hit_id"],
                "top_hit_host": "" if not hit else hit["top_hit_host"],
                "top_hit_phage_id": "" if not hit else hit["top_hit_phage_id"],
                "top_hit_sequence_cluster_40": "" if not hit else hit["top_hit_sequence_cluster_40"],
                "query_phage_id": row.phage_id,
                "query_sequence_cluster_40": row.sequence_cluster_40,
                "pident": "" if not hit else hit["pident"],
                "alignment_length": "",
                "query_coverage": "" if not hit else hit["query_coverage"],
                "subject_coverage": "",
                "evalue": "" if not hit else hit["evalue"],
                "bitscore": "" if not hit else hit["bitscore"],
                "correct": bool(pred == row.host_genus),
                "audit_note": "" if hit else "no BLAST hit; majority-host fallback",
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "blast_tsv": str(blast_path),
        "blast_tsv_sha256": sha256_file(blast_path) if blast_path.exists() else "",
        "queries": int(len(query)),
        "queries_with_hit": int(out["top_hit_id"].astype(str).ne("").sum()),
        "fallback_predictions": int(out["top_hit_id"].astype(str).eq("").sum()),
        "fallback_host": majority,
        "note": "Official frozen BLAST outfmt has qseqid/sseqid/pident/qcovs/evalue/bitscore only; alignment length and subject coverage are unavailable.",
    }
    return out, summary


def parse_hmmer_predictions(root: Path, split_df: pd.DataFrame, split_name: str) -> tuple[list[str], dict[str, Any]]:
    work = root / "external_baselines/hmmer"
    train = split_df[split_df["split"] == "train"].copy()
    query = split_df[split_df["split"] == split_name].copy()
    majority = train["host_genus"].astype(str).value_counts().idxmax()
    profiles = sorted(work.glob("*.hmm"))
    best: dict[str, tuple[float, str]] = {}
    tblouts = sorted(work.glob(f"{split_name}_*.tblout"))
    for tbl in tblouts:
        host = tbl.stem.replace(f"{split_name}_", "")
        with tbl.open(errors="ignore") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                qid = normalize_key(parts[0])
                try:
                    score = float(parts[5])
                except ValueError:
                    score = 0.0
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, host)
    preds = [best.get(normalize_key(row.rbp_id), (0.0, majority))[1] for row in query.itertuples()]
    summary = {
        "split": split_name,
        "profile_count": len(profiles),
        "tblout_count": len(tblouts),
        "queries": int(len(query)),
        "hit_queries": int(len(best)),
        "fallback_predictions": int(len(query) - len(best)),
        "fallback_host": majority,
        "prediction_class_distribution": dict(Counter(preds)),
        "status": "pass" if profiles and tblouts and len(preds) == len(query) else "warn",
    }
    return preds, summary


def bootstrap_delta(
    rows: pd.DataFrame,
    labels: Sequence[str],
    *,
    n_valid: int,
    max_attempts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples, generic_summary = paired_grouped_prediction_bootstrap(
        rows,
        group_col="bootstrap_group",
        true_col="true_host",
        model_pred_col="saprot_pred",
        baseline_pred_col="blast_pred",
        labels=labels,
        scorer=lambda y_true, y_pred, lab: f1_score(y_true, y_pred, labels=list(lab), average="macro", zero_division=0),
        n_valid=n_valid,
        max_attempts=max_attempts,
        seed=seed,
        model_score_key="saprot_macro_f1",
        baseline_score_key="blast_macro_f1",
        delta_key="delta_saprot_minus_blast",
        bootstrap_unit="phage_id",
        invalid_reason="bootstrap sample missing at least one host class",
        extra_sample_fields={
            "sampled_groups": lambda _sample, chosen: int(len(chosen)),
            "sampled_rows": lambda sample, _chosen: int(len(sample)),
        },
    )
    summary = {
        "status": generic_summary["status"],
        "bootstrap_unit": generic_summary["bootstrap_unit"],
        "requested_valid_replicates": generic_summary["requested_valid_replicates"],
        "valid_replicates": generic_summary["valid_bootstrap_replicates"],
        "invalid_replicates": generic_summary["invalid_bootstrap_replicates"],
        "attempted_replicates": generic_summary["attempted_bootstrap_replicates"],
        "invalid_reason": generic_summary["invalid_reason"],
        "observed_delta_saprot_minus_blast": generic_summary["observed_delta"],
        "mean_delta_saprot_minus_blast": generic_summary["mean_delta"],
        "median_delta_saprot_minus_blast": generic_summary["median_delta"],
        "ci95_low": generic_summary["ci95_low"],
        "ci95_high": generic_summary["ci95_high"],
        "p_delta_gt_0": generic_summary["p_delta_gt_0"],
        "p_delta_lt_0": generic_summary["p_delta_lt_0"],
        "p_delta_eq_0": generic_summary["p_delta_eq_0"],
    }
    return samples, summary


def summarize_blast_hits(blast_audit: pd.DataFrame, labels: Sequence[str]) -> dict[str, Any]:
    test = blast_audit[blast_audit["split"] == "test"].copy()
    numeric_cols = ["pident", "query_coverage", "bitscore"]
    stats: dict[str, Any] = {}
    for status, frame in [("correct", test[test["correct"] == True]), ("incorrect", test[test["correct"] == False])]:
        stats[status] = {"n": int(len(frame))}
        for col in numeric_cols:
            vals = pd.to_numeric(frame[col], errors="coerce").dropna()
            stats[status][col] = {
                "median": None if vals.empty else float(vals.median()),
                "mean": None if vals.empty else float(vals.mean()),
            }
    per_host = {}
    for host in labels:
        frame = test[test["true_host"].astype(str) == host]
        per_host[host] = {
            "queries": int(len(frame)),
            "hit_rate": float(frame["top_hit_id"].astype(str).ne("").mean()) if len(frame) else None,
            "accuracy": float(frame["correct"].astype(bool).mean()) if len(frame) else None,
        }
    return {
        "test_queries": int(len(test)),
        "test_hit_rate": float(test["top_hit_id"].astype(str).ne("").mean()) if len(test) else 0.0,
        "test_accuracy": float(test["correct"].astype(bool).mean()) if len(test) else 0.0,
        "correct_vs_incorrect_hit_stats": stats,
        "per_host_hit_summary": per_host,
        "unavailable_fields": ["alignment_length", "subject_coverage"],
    }


def run_phistruct_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.phistruct_root)
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    split_df = load_split(root)
    labels = sorted(split_df["host_genus"].astype(str).unique().tolist())
    summary_report = read_json(root / "summary_report.json")

    saprot_df, saprot_info = reconstruct_saprot(args, split_df, labels)
    blast_val, blast_val_summary = parse_blast_predictions(root, split_df, "validation")
    blast_test, blast_test_summary = parse_blast_predictions(root, split_df, "test")
    blast_audit = pd.concat([blast_val, blast_test], ignore_index=True)

    test_join = split_df[split_df["split"] == "test"][
        ["rbp_id", "rbp_key", "host_genus", "phage_id", "sequence_cluster_40"]
    ].merge(saprot_df[["rbp_key", "saprot_pred"]], on="rbp_key", how="left")
    blast_test_small = blast_test.rename(columns={"query_id": "rbp_id", "true_host": "blast_true_host", "predicted_host": "blast_pred"})
    test_join = test_join.merge(blast_test_small[["rbp_id", "blast_pred", "top_hit_id"]], on="rbp_id", how="left")
    test_join = test_join.rename(columns={"host_genus": "true_host"})
    test_join["bootstrap_group"] = test_join["phage_id"].astype(str)
    if test_join["saprot_pred"].isna().any() or test_join["blast_pred"].isna().any():
        raise RuntimeError("missing reconstructed SaProt or BLAST predictions for test rows")

    saprot_metrics = metric_bundle(test_join["true_host"], test_join["saprot_pred"], labels)
    blast_metrics = metric_bundle(test_join["true_host"], test_join["blast_pred"], labels)
    observed_delta = saprot_metrics["macro_f1"] - blast_metrics["macro_f1"]

    saprot_per = per_genus_rows("SaProt", test_join["true_host"], test_join["saprot_pred"], labels)
    blast_per = per_genus_rows("BLASTp_nearest_host", test_join["true_host"], test_join["blast_pred"], labels)
    saprot_by_host = {row["host_genus"]: row for row in saprot_per}
    blast_by_host = {row["host_genus"]: row for row in blast_per}
    comparison = []
    for host in labels:
        comparison.append(
            {
                "host_genus": host,
                "support": saprot_by_host[host]["support"],
                "saprot_f1": saprot_by_host[host]["f1"],
                "blast_f1": blast_by_host[host]["f1"],
                "delta_saprot_minus_blast": saprot_by_host[host]["f1"] - blast_by_host[host]["f1"],
                "saprot_recall": saprot_by_host[host]["recall"],
                "blast_recall": blast_by_host[host]["recall"],
            }
        )

    loo = []
    for host in labels:
        frame = test_join[test_join["true_host"] != host]
        if frame.empty:
            continue
        sap = f1_score(frame["true_host"], frame["saprot_pred"], labels=[x for x in labels if x != host], average="macro", zero_division=0)
        bla = f1_score(frame["true_host"], frame["blast_pred"], labels=[x for x in labels if x != host], average="macro", zero_division=0)
        loo.append(
            {
                "excluded_host_genus": host,
                "remaining_rows": int(len(frame)),
                "saprot_macro_f1": float(sap),
                "blast_macro_f1": float(bla),
                "delta_saprot_minus_blast": float(sap - bla),
            }
        )
    tiny_hosts = [row["host_genus"] for row in saprot_per if row["support"] <= args.tiny_class_support]
    no_tiny = test_join[~test_join["true_host"].isin(tiny_hosts)]
    tiny_sensitivity = {
        "tiny_class_support_threshold": args.tiny_class_support,
        "tiny_hosts": tiny_hosts,
        "rows_after_excluding_tiny": int(len(no_tiny)),
    }
    if len(no_tiny):
        no_tiny_labels = [x for x in labels if x not in tiny_hosts]
        sap = f1_score(no_tiny["true_host"], no_tiny["saprot_pred"], labels=no_tiny_labels, average="macro", zero_division=0)
        bla = f1_score(no_tiny["true_host"], no_tiny["blast_pred"], labels=no_tiny_labels, average="macro", zero_division=0)
        tiny_sensitivity.update(
            {
                "saprot_macro_f1_without_tiny": float(sap),
                "blast_macro_f1_without_tiny": float(bla),
                "delta_saprot_minus_blast_without_tiny": float(sap - bla),
            }
        )

    samples, bootstrap_summary = bootstrap_delta(
        test_join,
        labels,
        n_valid=args.bootstrap_replicates,
        max_attempts=args.bootstrap_max_attempts,
        seed=args.seed,
    )
    bootstrap_summary["observed_delta_saprot_minus_blast"] = observed_delta

    hmmer_val_pred, hmmer_val = parse_hmmer_predictions(root, split_df, "validation")
    hmmer_test_pred, hmmer_test = parse_hmmer_predictions(root, split_df, "test")
    hmmer_sanity = {
        "status": "pass" if hmmer_val["status"] == "pass" and hmmer_test["status"] == "pass" else "warn",
        "validation": hmmer_val,
        "test": hmmer_test,
    }
    if hmmer_test_pred:
        hmmer_sanity["test_metrics"] = metric_bundle(
            split_df[split_df["split"] == "test"]["host_genus"].astype(str),
            hmmer_test_pred,
            labels,
        )

    p_blast_better = bootstrap_summary.get("p_delta_lt_0")
    not_tiny_only = bool(tiny_sensitivity.get("delta_saprot_minus_blast_without_tiny", 1.0) <= 0.0)
    saturated = (
        observed_delta <= 0.0
        and p_blast_better is not None
        and float(p_blast_better) >= args.saturation_probability
        and not_tiny_only
    )
    final_status = "HOMOLOGY_BASELINE_SATURATED" if saturated else "PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED"
    tiny_driver = "NO" if not_tiny_only else "PARTIALLY_OR_YES"

    write_csv(audit_root / "per_genus_saprot_metrics.csv", saprot_per, ["model", "host_genus", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
    write_csv(audit_root / "per_genus_blast_metrics.csv", blast_per, ["model", "host_genus", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
    write_csv(audit_root / "per_genus_comparison.csv", comparison, ["host_genus", "support", "saprot_f1", "blast_f1", "delta_saprot_minus_blast", "saprot_recall", "blast_recall"])
    cm_fields = ["model", "true_host", *[f"pred_{label}" for label in labels]]
    write_csv(audit_root / "saprot_confusion_matrix.csv", confusion_rows("SaProt", test_join["true_host"], test_join["saprot_pred"], labels), cm_fields)
    write_csv(audit_root / "blast_confusion_matrix.csv", confusion_rows("BLASTp_nearest_host", test_join["true_host"], test_join["blast_pred"], labels), cm_fields)
    write_csv(audit_root / "leave_one_genus_out_sensitivity.csv", loo, ["excluded_host_genus", "remaining_rows", "saprot_macro_f1", "blast_macro_f1", "delta_saprot_minus_blast"])
    write_csv(audit_root / "paired_bootstrap_samples.csv", samples, ["replicate", "saprot_macro_f1", "blast_macro_f1", "delta_saprot_minus_blast", "sampled_groups", "sampled_rows"])
    blast_fields = [
        "split",
        "query_id",
        "true_host",
        "predicted_host",
        "top_hit_id",
        "top_hit_host",
        "top_hit_phage_id",
        "top_hit_sequence_cluster_40",
        "query_phage_id",
        "query_sequence_cluster_40",
        "pident",
        "alignment_length",
        "query_coverage",
        "subject_coverage",
        "evalue",
        "bitscore",
        "correct",
        "audit_note",
    ]
    write_csv(audit_root / "blast_hit_audit.csv", blast_audit.to_dict("records"), blast_fields)

    blast_hit_summary = {
        "validation": blast_val_summary,
        "test": blast_test_summary,
        **summarize_blast_hits(blast_audit, labels),
    }
    write_json(audit_root / "paired_bootstrap_summary.json", bootstrap_summary)
    write_json(audit_root / "blast_hit_summary.json", blast_hit_summary)
    write_json(audit_root / "hmmer_sanity_audit.json", hmmer_sanity)

    audit_summary = {
        "created_at": now_utc(),
        "source_phistruct_summary": str(root / "summary_report.json"),
        "source_phistruct_summary_sha256": sha256_file(root / "summary_report.json") if (root / "summary_report.json").exists() else "",
        "dataset": summary_report.get("dataset", {}),
        "split": summary_report.get("split", {}),
        "host_classes": labels,
        "test_host_distribution": dict(Counter(test_join["true_host"])),
        "saprot": saprot_info | {"test": saprot_metrics},
        "blastp": {"test": blast_metrics, "hit_summary": blast_hit_summary},
        "paired_bootstrap": bootstrap_summary,
        "tiny_class_sensitivity": tiny_sensitivity,
        "is_blast_advantage_driven_only_by_tiny_classes": tiny_driver,
        "hmmer_sanity": hmmer_sanity,
        "decision_rule": {
            "saturation_probability_threshold": args.saturation_probability,
            "requires_observed_saprot_minus_blast_delta_lte_0": True,
            "requires_blast_better_bootstrap_probability_gte_threshold": True,
            "requires_advantage_not_removed_by_excluding_tiny_classes": True,
        },
        "final_phistruct_status": final_status,
        "next_action": "attempt official EvoMIL acquisition and qualification gate" if final_status in {"HOMOLOGY_BASELINE_SATURATED", "PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED"} else "stop",
        "scientific_note": "PHIStruct remains unqualified regardless of whether the failure mechanism is statistically resolved.",
    }
    write_json(audit_root / "audit_summary.json", audit_summary)

    lines = [
        "# PHIStruct Failure Audit",
        "",
        f"- Dataset: {summary_report.get('dataset', {}).get('rbp_count', len(split_df))} RBPs; host classes: {', '.join(labels)}",
        f"- Original/reconstructed SaProt test macro-F1: {saprot_metrics['macro_f1']:.6f}",
        f"- Reconstructed BLASTp test macro-F1: {blast_metrics['macro_f1']:.6f}",
        f"- Delta (SaProt - BLASTp): {observed_delta:.6f}",
        f"- Bootstrap valid/invalid: {bootstrap_summary.get('valid_replicates', 0)} / {bootstrap_summary.get('invalid_replicates', 0)}",
        f"- Bootstrap 95% CI: [{bootstrap_summary.get('ci95_low')}, {bootstrap_summary.get('ci95_high')}]",
        f"- P(delta > 0): {bootstrap_summary.get('p_delta_gt_0')}",
        f"- P(delta < 0): {bootstrap_summary.get('p_delta_lt_0')}",
        f"- BLAST tiny-class-only driver: {tiny_driver}",
        f"- HMMER sanity: {hmmer_sanity['status']}",
        f"- Final PHIStruct status: {final_status}",
        "",
        "## Per-Genus Delta",
        "",
    ]
    for row in comparison:
        lines.append(
            f"- {row['host_genus']}: support={row['support']}, SaProt F1={row['saprot_f1']:.6f}, BLAST F1={row['blast_f1']:.6f}, delta={row['delta_saprot_minus_blast']:.6f}"
        )
    (audit_root / "audit_summary.md").write_text("\n".join(lines) + "\n")
    return audit_summary


def git_revision(path: Path) -> str:
    result = run_cmd(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=60)
    if result["returncode"] == 0:
        return result["output_tail"].strip().splitlines()[-1]
    return ""


def acquire_official_evomil(args: argparse.Namespace, phistruct_decision: str) -> dict[str, Any]:
    root = Path(args.evomil_root)
    external_root = PROJECT_ROOT / "data/external/evomil"
    repo = external_root / "EvoMIL"
    root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    update_status(args, "running", "evomil_official_asset_acquisition", phistruct_decision=phistruct_decision)

    clone_result: dict[str, Any]
    if repo.exists() and (repo / ".git").exists():
        fetch = run_cmd(["git", "-C", str(repo), "fetch", "--tags", "--prune"], timeout=600, retries=2)
        clone_result = {"action": "fetch", **fetch}
    else:
        shutil.rmtree(repo, ignore_errors=True)
        clone = run_cmd(["git", "clone", "--depth", "1", OFFICIAL_EVOMIL_REPO, str(repo)], timeout=1200, retries=2)
        clone_result = {"action": "clone", **clone}
    revision = git_revision(repo) if repo.exists() else ""

    official_files = []
    data_candidates = []
    code_candidates = []
    if repo.exists():
        for path in sorted(repo.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            rel = path.relative_to(repo).as_posix()
            suffix = path.suffix.lower()
            size = path.stat().st_size
            if suffix in {".py", ".ipynb", ".r", ".sh"}:
                code_candidates.append(rel)
            if suffix in {".csv", ".tsv", ".txt", ".fa", ".faa", ".fasta", ".fna", ".pkl", ".npz", ".npy", ".xlsx"}:
                data_candidates.append(rel)
                official_files.append(
                    {
                        "asset_name": f"EvoMIL repository file: {rel}",
                        "source_url": f"{OFFICIAL_EVOMIL_REPO}/blob/{revision}/{rel}" if revision else OFFICIAL_EVOMIL_REPO,
                        "revision": revision,
                        "checksum": sha256_file(path),
                        "local_path": str(path),
                        "license": "",
                        "download_date": now_utc(),
                        "size_bytes": size,
                    }
                )

    manifest = {
        "created_at": now_utc(),
        "assets": [
            {
                "asset_name": "Official EvoMIL repository",
                "source_url": OFFICIAL_EVOMIL_REPO,
                "revision": revision,
                "checksum": "",
                "local_path": str(repo),
                "license": "",
                "download_date": now_utc(),
                "clone_result": clone_result,
            },
            {
                "asset_name": "EvoMIL paper",
                "source_url": OFFICIAL_EVOMIL_PAPER,
                "revision": "PLOS Computational Biology article DOI",
                "checksum": "",
                "local_path": "",
                "license": "",
                "download_date": now_utc(),
            },
            *official_files,
        ],
    }
    write_json(root / "evomil_external_assets.json", manifest)

    warnings = []
    blockers = []
    if clone_result["returncode"] != 0:
        if revision and repo.exists():
            warnings.append(f"official EvoMIL git refresh failed transiently; using existing verified checkout: {clone_result['output_tail'][-500:]}")
        else:
            blockers.append(f"official EvoMIL git acquisition failed: {clone_result['output_tail'][-500:]}")
    if not revision:
        blockers.append("could not verify official EvoMIL repository revision")
    if not data_candidates:
        blockers.append("official EvoMIL repository acquisition succeeded, but no dataset/proteome files were found in the repository checkout")
    if not code_candidates:
        blockers.append("official EvoMIL repository checkout did not expose runnable code files")
    has_vhdb_table = any("virushostdb" in item.lower() for item in data_candidates)
    has_protein_fasta = any(Path(item).suffix.lower() in {".fa", ".faa", ".fasta"} for item in data_candidates)
    has_embedding_asset = any(Path(item).suffix.lower() in {".npz", ".npy", ".pkl"} and "esm" in item.lower() for item in data_candidates)
    formal_blockers = []
    if not has_vhdb_table:
        formal_blockers.append("official VHDB virus-host association table not found")
    if not has_protein_fasta:
        formal_blockers.append("official viral protein/proteome FASTA assets are not present in the EvoMIL checkout")
    if not has_embedding_asset:
        formal_blockers.append("official/precomputed ESM-1b protein embedding assets are not present in the EvoMIL checkout")
    formal_blockers.append("strict EvoMIL dataset adapter, proteome-cluster split, leakage audit, smoke test, and mandatory baselines are not yet implemented in this project")

    gate = {
        "created_at": now_utc(),
        "status": "blocked" if blockers or formal_blockers else "asset_acquired",
        "official_repo": OFFICIAL_EVOMIL_REPO,
        "revision": revision,
        "repository_path": str(repo),
        "code_candidate_count": len(code_candidates),
        "data_candidate_count": len(data_candidates),
        "code_candidates": code_candidates[:100],
        "data_candidates": data_candidates[:100],
        "warnings": warnings,
        "blockers": blockers,
        "formal_blockers": formal_blockers,
        "has_vhdb_table": has_vhdb_table,
        "has_protein_fasta": has_protein_fasta,
        "has_embedding_asset": has_embedding_asset,
        "formal_evomil_started": False,
        "reason_not_started": "formal EvoMIL requires a non-empty official dataset manifest, strict split, leakage audit, smoke test, and mandatory baselines",
    }
    write_json(root / "evomil_experiment_registry.json", gate)
    if blockers or formal_blockers:
        update_status(args, "blocked", "evomil_official_asset_gate", blockers=blockers + formal_blockers, warnings=warnings, formal_evomil_started=False)
    return gate


def controller_execute(args: argparse.Namespace) -> None:
    started = time.time()
    update_status(args, "running", "phistruct_failure_audit")
    log(args, "starting PHIStruct frozen-result failure audit")
    audit = run_phistruct_audit(args)
    log(args, f"PHIStruct failure audit complete: {audit['final_phistruct_status']}")
    gate = acquire_official_evomil(args, audit["final_phistruct_status"])
    if gate["status"] == "blocked":
        all_blockers = gate.get("blockers", []) + gate.get("formal_blockers", [])
        log(args, f"EvoMIL formal launch blocked: {'; '.join(all_blockers)}")
        update_status(
            args,
            "blocked",
            "workflow_blocked_before_formal_evomil",
            phistruct_final_status=audit["final_phistruct_status"],
            blockers=all_blockers,
            warnings=gate.get("warnings", []),
            formal_evomil_started=False,
            elapsed_sec=time.time() - started,
        )
    else:
        log(args, "EvoMIL official assets acquired; formal benchmark implementation gate remains pending")
        update_status(
            args,
            "blocked",
            "workflow_blocked_before_formal_evomil",
            phistruct_final_status=audit["final_phistruct_status"],
            blockers=[gate["reason_not_started"]],
            formal_evomil_started=False,
            elapsed_sec=time.time() - started,
        )


def launch_screen(args: argparse.Namespace) -> None:
    screen_name = args.screen_name
    command = [
        "screen",
        "-L",
        "-Logfile",
        str(Path(args.log_file).with_suffix(".screen.log")),
        "-dmS",
        screen_name,
        args.python,
        str(Path(__file__).resolve()),
        "--execute",
        "--phistruct-root",
        str(Path(args.phistruct_root)),
        "--audit-root",
        str(Path(args.audit_root)),
        "--evomil-root",
        str(Path(args.evomil_root)),
        "--log-file",
        str(Path(args.log_file)),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--bootstrap-max-attempts",
        str(args.bootstrap_max_attempts),
        "--seed",
        str(args.seed),
        "--tiny-class-support",
        str(args.tiny_class_support),
        "--saturation-probability",
        str(args.saturation_probability),
    ]
    result = run_cmd(command, timeout=60)
    if result["returncode"] != 0:
        raise RuntimeError(result["output_tail"])
    update_status(args, "running", "screen_launched", screen_name=screen_name, formal_evomil_started=False)
    print(f"launched screen {screen_name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-screen", action="store_true")
    parser.add_argument("--screen-name", default="phistruct_audit_evomil_controller")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--phistruct-root", default=str(DEFAULT_PHISTRUCT_ROOT))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--evomil-root", default=str(DEFAULT_EVOMIL_ROOT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-max-attempts", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny-class-support", type=int, default=10)
    parser.add_argument("--saturation-probability", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.launch_screen:
        launch_screen(args)
        return
    if not args.execute:
        raise SystemExit("use --launch-screen or --execute")
    controller_execute(args)


if __name__ == "__main__":
    main()
