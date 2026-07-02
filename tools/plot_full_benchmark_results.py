#!/usr/bin/env python3
"""Create the three report figures from the completed 44-task benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RANKINGS = (
    ROOT
    / "data"
    / "phase2"
    / "full_benchmarks_lora_optimized_s600"
    / "full_rankings.csv"
)
OUT = ROOT / "figures"

LABELS = {
    "lora_gd_full_ar3_s200": "GD strong",
    "lora_gd_full_ar5_s500": "GD moderate",
    "lora_rmu_full_sc50_s200": "RMU low steer",
    "lora_rmu_full_sc200_s200": "RMU high steer",
}
ORDER = list(LABELS)


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(RANKINGS).set_index("run").loc[ORDER].copy()
    frame["label"] = [LABELS[name] for name in frame.index]
    frame["gue_retain_drop"] = -frame["gue_retain_delta"]
    return frame


def save_target_and_retain(frame: pd.DataFrame) -> None:
    ax = frame.plot.bar(
        x="label",
        y=["hvue_forget_drop", "gue_retain_drop"],
        color=["#3973e6", "#e34b4f"],
        figsize=(10, 5),
    )
    ax.set_title("Full benchmark: target forgetting vs. GUE retain cost")
    ax.set_xlabel("")
    ax.set_ylabel("Change from base")
    ax.legend(["Target-task drop (forget)", "GUE retain drop"])
    ax.tick_params(axis="x", rotation=0)
    ax.axhline(0, color="#222", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(OUT / "full_benchmark_target_vs_retain.png", dpi=180)
    plt.close()


def save_tradeoff(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3973e6" if method == "gd" else "#ff7f0e" for method in frame["method"]]
    ax.scatter(
        frame["gue_retain_drop"],
        frame["balanced_forget_score"],
        c=colors,
        s=110,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            row["label"],
            (row["gue_retain_drop"], row["balanced_forget_score"]),
            xytext=(8, 7),
            textcoords="offset points",
        )
    ax.set_title("Forget-retain trade-off: ideal region is upper-left")
    ax.set_xlabel("GUE retain drop (lower is better)")
    ax.set_ylabel("Balanced forget score (higher is better)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "full_benchmark_tradeoff.png", dpi=180)
    plt.close()


def save_selection(frame: pd.DataFrame) -> None:
    colors = ["#2aaa5b" if score > 0 else "#9ca3af" for score in frame["selection_score"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(frame["label"], frame["selection_score"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("Selection score = balanced forget - retain penalty")
    ax.set_ylabel("Selection score")
    plt.tight_layout()
    plt.savefig(OUT / "full_benchmark_selection_score.png", dpi=180)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_results()
    save_target_and_retain(frame)
    save_tradeoff(frame)
    save_selection(frame)


if __name__ == "__main__":
    main()
