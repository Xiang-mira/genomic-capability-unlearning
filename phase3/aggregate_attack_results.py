import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def load_localized_layers(path: str) -> List[int]:
    if not os.path.exists(path):
        return list(range(3, 10))
    with open(path) as f:
        payload = json.load(f)
    return sorted(int(layer) for layer in payload.get("layers", list(range(3, 10))))


def mean_for_layers(df: pd.DataFrame, layers: List[int], col: str) -> float:
    sub = df[df["layer"].isin(layers)]
    if sub.empty:
        return float("nan")
    return float(sub[col].mean())


def best_attack_score(phase3_dir: str, run_name: str, attack: str, layers: List[int]) -> Dict[str, object]:
    pattern = os.path.join(phase3_dir, f"{run_name}_{attack}_lr*", "auroc.csv")
    candidates = []
    for auroc_path in glob.glob(pattern):
        df = pd.read_csv(auroc_path)
        score = mean_for_layers(df, layers, "auroc_after")
        run_dir = os.path.dirname(auroc_path)
        meta_path = os.path.join(run_dir, "meta.json")
        lr = ""
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                lr = json.load(f).get("lr", "")
        candidates.append({"score": score, "path": run_dir, "lr": lr})
    if not candidates:
        return {"score": float("nan"), "path": "", "lr": ""}
    return max(candidates, key=lambda row: row["score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 attack runs into a method x attack matrix.")
    parser.add_argument("--phase2-ckpt-root", default="data/phase2/checkpoints")
    parser.add_argument("--phase3-dir", default="data/phase3")
    parser.add_argument("--base-probe-metrics", default="data/family_targets/coronaviridae/probes/probe_metrics_by_layer.csv")
    parser.add_argument("--localized-layers-path", default="data/family_targets/coronaviridae/localized_layers.json")
    parser.add_argument("--out", default="data/phase3/phase3_method_attack_matrix.csv")
    args = parser.parse_args()

    localized_layers = load_localized_layers(args.localized_layers_path)
    base_probe_df = pd.read_csv(args.base_probe_metrics)
    baseline_score = mean_for_layers(base_probe_df, localized_layers, "test_auroc")

    rows = [
        {
            "run": "base",
            "no_attack": baseline_score,
            "best_sft": baseline_score,
            "best_sft_lr": "",
            "best_lora": baseline_score,
            "best_lora_lr": "",
        }
    ]

    for run_dir in sorted(glob.glob(os.path.join(args.phase2_ckpt_root, "*/"))):
        run_name = os.path.basename(os.path.normpath(run_dir))
        eval_path = os.path.join(run_dir, "eval_auroc.csv")
        if not os.path.exists(eval_path):
            continue
        eval_df = pd.read_csv(eval_path)
        no_attack = mean_for_layers(eval_df, localized_layers, "test_auroc")
        best_sft = best_attack_score(args.phase3_dir, run_name, "sft", localized_layers)
        best_lora = best_attack_score(args.phase3_dir, run_name, "lora", localized_layers)
        rows.append(
            {
                "run": run_name,
                "no_attack": no_attack,
                "best_sft": best_sft["score"],
                "best_sft_lr": best_sft["lr"],
                "best_lora": best_lora["score"],
                "best_lora_lr": best_lora["lr"],
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "no_attack", "best_sft", "best_sft_lr", "best_lora", "best_lora_lr"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote matrix to {args.out}")


if __name__ == "__main__":
    main()
