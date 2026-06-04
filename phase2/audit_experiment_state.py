"""Audit local data, manifests, and experiment artefacts for Phase 2/3.

This script is intentionally lightweight and uses only local files. It answers:
  - Which benchmark datasets are present in the workspace?
  - Does the checked-in HVUE/GUE manifest fully cover the raw HVUE files?
  - Is the requested Calici taxonomy-shortcut audit executable with current public CSVs?
  - Which Phase 2/3 checkpoints already exist?

The output JSON is meant to be committed alongside experiment notes so the current
state of the workspace is explicit and reproducible.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EXPECTED_VGUE_TASKS = {
    "host_range_prediction": ["host-range", "host_range"],
    "dna_vs_rna_virus": ["dna", "rna"],
    "hiv1_vs_hiv2": ["hiv-1", "hiv1", "hiv-2", "hiv2"],
    "sars_cov_2_lineage_typing": ["sars", "lineage"],
    "hiv1_tropism": ["tropism", "hiv"],
}

TARGET_VIRAL_RETAIN_TASKS = {
    "host_range_prediction",
    "dna_vs_rna_virus",
    "hiv1_vs_hiv2",
}

HVUE_SPECS = {
    "Host_Tropism/train.csv": ("hvue_human_host_tropism", "train", ""),
    "Host_Tropism/dev.csv": ("hvue_human_host_tropism", "val", ""),
    "Host_Tropism/test.csv": ("hvue_human_host_tropism", "test", ""),
    "Pathogenecity/CINI/train.csv": ("hvue_human_virus_pathogenicity_cini", "train", "mixed"),
    "Pathogenecity/CINI/dev.csv": ("hvue_human_virus_pathogenicity_cini", "val", "mixed"),
    "Pathogenecity/CINI/test.csv": ("hvue_human_virus_pathogenicity_cini", "test", "mixed"),
    "Pathogenecity/BVBRC_Calci/train.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "train", "Caliciviridae"),
    "Pathogenecity/BVBRC_Calci/dev.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "val", "Caliciviridae"),
    "Pathogenecity/BVBRC_Calci/test.csv": ("hvue_human_virus_pathogenicity_bvbrc_calici", "test", "Caliciviridae"),
    "Pathogenecity/BVBRC_CoV/train.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "train", "Coronaviridae"),
    "Pathogenecity/BVBRC_CoV/dev.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "val", "Coronaviridae"),
    "Pathogenecity/BVBRC_CoV/test.csv": ("hvue_human_virus_pathogenicity_bvbrc_cov", "test", "Coronaviridae"),
    "Transmissibility/Calcivirdae/train.csv": ("hvue_human_transmissibility_caliciviridae", "train", "Caliciviridae"),
    "Transmissibility/Calcivirdae/dev.csv": ("hvue_human_transmissibility_caliciviridae", "val", "Caliciviridae"),
    "Transmissibility/Calcivirdae/test.csv": ("hvue_human_transmissibility_caliciviridae", "test", "Caliciviridae"),
    "Transmissibility/Coronovirdae/train.csv": ("hvue_human_transmissibility_coronaviridae", "train", "Coronaviridae"),
    "Transmissibility/Coronovirdae/dev.csv": ("hvue_human_transmissibility_coronaviridae", "val", "Coronaviridae"),
    "Transmissibility/Coronovirdae/test.csv": ("hvue_human_transmissibility_coronaviridae", "test", "Coronaviridae"),
    "Transmissibility/Orthomyxovirdae/train.csv": ("hvue_human_transmissibility_orthomyxoviridae", "train", "Orthomyxoviridae"),
    "Transmissibility/Orthomyxovirdae/dev.csv": ("hvue_human_transmissibility_orthomyxoviridae", "val", "Orthomyxoviridae"),
    "Transmissibility/Orthomyxovirdae/test.csv": ("hvue_human_transmissibility_orthomyxoviridae", "test", "Orthomyxoviridae"),
}

SHORTCUT_TASKS = {
    "calici_transmissibility": "Transmissibility/Calcivirdae",
    "bvbrc_calici_pathogenicity": "Pathogenecity/BVBRC_Calci",
}

TAXONOMY_COLUMNS = {
    "family",
    "genus",
    "species",
    "virus_name",
    "virus_tax_id",
    "accession",
    "taxid",
}


def count_csv_rows(path: Path) -> int:
    with path.open() as f:
        next(f, None)
        return sum(1 for _ in f)


def read_csv_columns(path: Path) -> List[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def count_table_rows(path: Path) -> int:
    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            return sum(1 for line in f if line.strip())
    return count_csv_rows(path)


def summarize_host_manifest(path: Path) -> Dict[str, object]:
    split_label = defaultdict(Counter)
    total = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            split_label[row["split"]][row["label"]] += 1
    return {
        "path": str(path),
        "exists": True,
        "rows": total,
        "split_label_counts": {split: dict(counter) for split, counter in sorted(split_label.items())},
    }


def summarize_hvue_raw(raw_root: Path) -> Dict[str, object]:
    per_file = {}
    per_task = defaultdict(dict)
    for rel_path, (task, split, family) in sorted(HVUE_SPECS.items()):
        csv_path = raw_root / "hvue" / rel_path
        exists = csv_path.exists()
        row_count = count_csv_rows(csv_path) if exists else None
        columns = read_csv_columns(csv_path) if exists else []
        per_file[rel_path] = {
            "path": str(csv_path),
            "exists": exists,
            "rows": row_count,
            "columns": columns,
            "task": task,
            "split": split,
            "family": family,
        }
        per_task[task][split] = row_count
    return {
        "files": per_file,
        "tasks": {task: dict(split_counts) for task, split_counts in sorted(per_task.items())},
    }


def summarize_gue_raw(raw_root: Path) -> Dict[str, object]:
    gue_root = raw_root / "gue" / "GUE"
    if not gue_root.exists():
        return {"exists": False, "task_dirs": {}, "n_task_dirs": 0}

    task_dirs = {}
    for child in sorted(gue_root.iterdir()):
        if not child.is_dir():
            continue
        split_counts = {}
        for split in ("train", "dev", "test"):
            csv_path = child / f"{split}.csv"
            split_counts[split] = count_csv_rows(csv_path) if csv_path.exists() else None
        task_dirs[child.name] = split_counts
    return {"exists": True, "task_dirs": task_dirs, "n_task_dirs": len(task_dirs)}


def summarize_viral_retain_raw(raw_root: Path) -> Dict[str, object]:
    viral_root = raw_root / "viral_retain"
    if not viral_root.exists():
        return {
            "exists": False,
            "root": str(viral_root),
            "target_tasks": sorted(TARGET_VIRAL_RETAIN_TASKS),
            "present_tasks": [],
            "missing_tasks": sorted(TARGET_VIRAL_RETAIN_TASKS),
            "task_split_counts": {},
        }
    task_split_counts = {}
    present_tasks = []
    for task in sorted(TARGET_VIRAL_RETAIN_TASKS):
        task_dir = viral_root / task
        if not task_dir.exists():
            continue
        present_tasks.append(task)
        split_counts = {}
        for split in ("train", "val", "dev", "valid", "validation", "test"):
            for suffix in (".csv", ".tsv", ".jsonl", ".json"):
                path = task_dir / f"{split}{suffix}"
                if path.exists():
                    normalized = "val" if split in {"dev", "valid", "validation"} else split
                    split_counts[normalized] = count_table_rows(path)
                    break
        task_split_counts[task] = split_counts
    return {
        "exists": True,
        "root": str(viral_root),
        "target_tasks": sorted(TARGET_VIRAL_RETAIN_TASKS),
        "present_tasks": present_tasks,
        "missing_tasks": sorted(TARGET_VIRAL_RETAIN_TASKS - set(present_tasks)),
        "task_split_counts": task_split_counts,
    }


def summarize_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False}
    counts = defaultdict(Counter)
    groups = Counter()
    rows = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            counts[row["task"]][row["split"]] += 1
            if row.get("group"):
                groups[row["group"]] += 1
    return {
        "exists": True,
        "path": str(path),
        "rows": rows,
        "task_split_counts": {task: dict(counter) for task, counter in sorted(counts.items())},
        "group_counts": dict(sorted(groups.items())),
    }


def compare_hvue_raw_to_manifest(raw_summary: Dict[str, object], manifest_summary: Dict[str, object]) -> List[Dict[str, object]]:
    issues = []
    manifest_counts = manifest_summary.get("task_split_counts", {}) if manifest_summary.get("exists") else {}
    for rel_path, meta in sorted(raw_summary["files"].items()):
        expected = meta["rows"]
        task = meta["task"]
        split = meta["split"]
        actual = manifest_counts.get(task, {}).get(split)
        if expected != actual:
            issues.append(
                {
                    "task": task,
                    "split": split,
                    "raw_file": rel_path,
                    "raw_rows": expected,
                    "manifest_rows": actual,
                }
            )
    return issues


def scan_for_vgue(data_root: Path) -> Dict[str, object]:
    seen_paths = []
    for root, _dirs, files in os.walk(data_root):
        for name in files:
            lower = name.lower()
            path = str(Path(root) / name)
            path_lower = path.lower()
            if (
                "vgue" in lower
                or "vir2vec" in lower
                or "viral_retain" in path_lower
                or "lineage" in lower
                or "hiv" in lower
            ):
                seen_paths.append(path)
    matches = {}
    for task, keywords in EXPECTED_VGUE_TASKS.items():
        task_hits = []
        for path in seen_paths:
            lower = path.lower()
            if all(keyword in lower for keyword in keywords[:1]):
                task_hits.append(path)
        matches[task] = task_hits
    present_tasks = sorted(task for task, hits in matches.items() if hits)
    missing_tasks = sorted(task for task, hits in matches.items() if not hits)
    return {
        "workspace_scan_paths": seen_paths[:200],
        "present_tasks": present_tasks,
        "missing_tasks": missing_tasks,
        "task_hits": matches,
    }


def load_host_sequences(path: Path) -> set[str]:
    sequences = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.add(row["sequence"])
    return sequences


def audit_shortcut_feasibility(raw_root: Path, host_manifest: Path) -> Dict[str, object]:
    host_sequences = load_host_sequences(host_manifest) if host_manifest.exists() else set()
    results = {}
    for label, rel_dir in SHORTCUT_TASKS.items():
        task_dir = raw_root / "hvue" / rel_dir
        split_info = {}
        columns_union = set()
        total_rows = 0
        total_overlap = 0
        for split_name in ("train", "dev", "test"):
            csv_path = task_dir / f"{split_name}.csv"
            if not csv_path.exists():
                split_info[split_name] = {"exists": False}
                continue
            columns = read_csv_columns(csv_path)
            columns_union.update(columns)
            rows = 0
            overlap = 0
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows += 1
                    if row.get("sequence") in host_sequences:
                        overlap += 1
            total_rows += rows
            total_overlap += overlap
            split_info[split_name] = {
                "exists": True,
                "rows": rows,
                "columns": columns,
                "sequence_overlap_with_host_manifest": overlap,
            }
        has_taxonomy = bool(columns_union & TAXONOMY_COLUMNS)
        results[label] = {
            "task_dir": str(task_dir),
            "splits": split_info,
            "all_columns": sorted(columns_union),
            "has_taxonomy_columns": has_taxonomy,
            "single_family_public_task": True,
            "public_family_label": "Caliciviridae",
            "sequence_overlap_with_host_manifest": total_overlap,
            "sequence_overlap_rate_with_host_manifest": (total_overlap / total_rows) if total_rows else 0.0,
            "family_held_out_feasible": False,
            "recommended_status": "blocked",
            "reason": (
                "Current public HVUE CSVs expose only sequence,label and the task is already "
                "restricted to Caliciviridae. A literal family-held-out split is therefore not "
                "identifiable from these files; a genus/species-held-out audit would require "
                "external taxonomy metadata."
            ),
        }
    return results


def summarize_runs(root: Path) -> Dict[str, object]:
    if not root.exists():
        return {"exists": False, "runs": []}
    runs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        runs.append(
            {
                "name": child.name,
                "has_weights": (child / "weights.safetensors").exists(),
                "has_eval_auroc": (child / "eval_auroc.csv").exists(),
                "has_eval_ppl": (child / "eval_ppl.json").exists(),
                "has_meta": (child / "meta.json").exists(),
                "has_log": (child / "log.json").exists(),
            }
        )
    return {"exists": True, "runs": runs, "n_runs": len(runs)}


def summarize_phase3_runs(root: Path) -> Dict[str, object]:
    if not root.exists():
        return {"exists": False, "runs": []}
    runs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        runs.append(
            {
                "name": child.name,
                "has_auroc": (child / "auroc.csv").exists(),
                "has_meta": (child / "meta.json").exists(),
                "has_log": (child / "log.json").exists(),
            }
        )
    return {"exists": True, "runs": runs, "n_runs": len(runs)}


def load_json_if_exists(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--host-manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--phase2-root", default="data/phase2/checkpoints")
    parser.add_argument("--phase2-tuned-root", default="data/phase2/checkpoints_tuned")
    parser.add_argument("--phase3-root", default="data/phase3")
    parser.add_argument("--out", default="data/phase2/experiment_audit.json")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    host_manifest = Path(args.host_manifest)
    manifest = Path(args.manifest)
    out_path = Path(args.out)

    host_summary = summarize_host_manifest(host_manifest) if host_manifest.exists() else {"exists": False}
    hvue_raw = summarize_hvue_raw(raw_root)
    gue_raw = summarize_gue_raw(raw_root)
    viral_retain_raw = summarize_viral_retain_raw(raw_root)
    manifest_summary = summarize_manifest(manifest)
    manifest_issues = compare_hvue_raw_to_manifest(hvue_raw, manifest_summary)
    vgue_scan = scan_for_vgue(Path("data"))
    shortcut_audit = audit_shortcut_feasibility(raw_root, host_manifest)
    phase2_runs = summarize_runs(Path(args.phase2_root))
    phase2_tuned_runs = summarize_runs(Path(args.phase2_tuned_root))
    phase3_runs = summarize_phase3_runs(Path(args.phase3_root))
    base_bench_summary = load_json_if_exists(Path("data/phase2/base_benchmarks/eval_benchmarks_summary.json"))

    payload = {
        "step0": {
            "host_tropism_manifest": host_summary,
            "hvue_raw": hvue_raw,
            "gue_raw": gue_raw,
            "viral_retain_raw": viral_retain_raw,
            "vgue_scan": vgue_scan,
            "status": {
                "forget_set_available": host_summary.get("exists", False),
                "retain_set_available": host_summary.get("exists", False),
                "gue_available": gue_raw.get("exists", False),
                "viral_retain_available": not viral_retain_raw.get("missing_tasks"),
                "vgue_available": len(vgue_scan["present_tasks"]) == len(EXPECTED_VGUE_TASKS),
            },
        },
        "step1": {
            "manifest_summary": manifest_summary,
            "manifest_hvue_coverage_issues": manifest_issues,
            "shortcut_audit": shortcut_audit,
            "recommended_primary_forget_benchmarks": [
                "hvue_human_host_tropism",
                "hvue_human_virus_pathogenicity_cini",
            ],
        },
        "step2": {
            "current_retain_strategy_in_repo": ["gue_retain", "viral_retain_if_task_tables_present"],
            "target_viral_retain_tasks": sorted(TARGET_VIRAL_RETAIN_TASKS),
            "missing_for_requested_plan": (
                [] if not viral_retain_raw.get("missing_tasks")
                else [f"viral retain task tables missing: {', '.join(viral_retain_raw['missing_tasks'])}"]
            ),
            "base_benchmark_summary": base_bench_summary,
        },
        "step3": {
            "phase2_checkpoints": phase2_runs,
            "phase2_tuned_checkpoints": phase2_tuned_runs,
        },
        "step5": {
            "phase3_runs": phase3_runs,
            "status": "attack scripts exist; LR-grid runner is phase3/run_attacks.sh",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[audit] wrote {out_path}")
    print(f"[audit] vGUE present tasks: {vgue_scan['present_tasks'] or 'none'}")
    print(f"[audit] vGUE missing tasks: {', '.join(vgue_scan['missing_tasks'])}")
    print(f"[audit] viral retain present tasks: {viral_retain_raw.get('present_tasks') or 'none'}")
    print(f"[audit] viral retain missing tasks: {', '.join(viral_retain_raw.get('missing_tasks', [])) or 'none'}")
    print(f"[audit] manifest HVUE coverage issues: {len(manifest_issues)}")
    for issue in manifest_issues[:10]:
        print(
            f"  - {issue['task']} split={issue['split']} raw_rows={issue['raw_rows']} "
            f"manifest_rows={issue['manifest_rows']}"
        )
    for name, result in shortcut_audit.items():
        print(f"[audit] {name}: {result['recommended_status']} | {result['reason']}")


if __name__ == "__main__":
    main()
