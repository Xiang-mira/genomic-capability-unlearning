import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = ROOT / "data" / "family_targets" / "coronaviridae"
PHASE2_ROOT = ROOT / "data" / "phase2"
OUT_DIR = ROOT / "figures"


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def style_axes(ax):
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_phase1() -> Path:
    probe = pd.read_csv(TARGET_ROOT / "probes" / "probe_metrics_by_layer.csv").sort_values("layer")
    patch = pd.read_csv(TARGET_ROOT / "activation_patching" / "patching_layer_summary.csv")
    localized = read_json(TARGET_ROOT / "localized_layers.json")
    summary = read_json(TARGET_ROOT / "summary.json")

    layers = [int(x) for x in localized["layers"]]
    primary_layer = int(localized["primary_target_layer"])
    effect_col = localized["effect_column"]
    patch = patch.sort_values("layer")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.3),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    fig.suptitle(
        "Phase 1: RefSeq Coronaviridae target is most readable in Evo early/mid layers",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.axvspan(min(layers) - 0.45, max(layers) + 0.45, color="#ffcf5a", alpha=0.26, label="selected L5-9")
    ax.axvline(primary_layer, color="#d23b3b", linewidth=1.7, linestyle="--", label="primary L6")
    ax.plot(
        probe["layer"],
        probe["val_auroc"],
        marker="o",
        markersize=4.6,
        linewidth=1.8,
        color="#3f6fb5",
        label="val AUROC",
    )
    ax.plot(
        probe["layer"],
        probe["test_auroc"],
        marker="o",
        markersize=4.6,
        linewidth=1.8,
        color="#1f9e78",
        label="test AUROC",
    )
    ax.axhline(0.5, color="#777777", linewidth=1.0, linestyle=":", label="chance")
    ax.set_title("Layer-wise linear probe", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Evo layer")
    ax.set_ylabel("AUROC")
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0.45, 1.03)
    ax.set_xticks(range(0, 32, 2))
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower left")
    style_axes(ax)

    ax = axes[1]
    colors = ["#d23b3b" if int(layer) == primary_layer else "#f0a83a" if int(layer) in layers else "#9aa6b2" for layer in patch["layer"]]
    ax.bar(patch["layer"], patch[effect_col], color=colors, width=0.78)
    ax.axhline(localized["threshold"], color="#555555", linestyle="--", linewidth=1.2, label="selection threshold")
    ax.set_title("Activation patching effect", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("|delta target probability|")
    ax.set_xlim(-0.5, 31.5)
    ax.set_xticks(range(0, 32, 2))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(ax)

    split_counts = summary["split_label_counts"]
    note = (
        f"Manifest: train {split_counts['train|1']}+{split_counts['train|0']}, "
        f"val {split_counts['val|1']}+{split_counts['val|0']}, "
        f"test {split_counts['test|1']}+{split_counts['test|0']} windows\n"
        f"Selected layers: {layers}; primary layer: {primary_layer}"
    )
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=9.2, color="#333333")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "meeting_phase1_refseq_target.png"
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def mean_l5_9_auroc(run: str) -> float:
    path = PHASE2_ROOT / "checkpoints_tuned" / run / "eval_auroc.csv"
    df = pd.read_csv(path)
    return float(df[df["layer"].between(5, 9)]["test_auroc"].mean())


def plot_phase2() -> Path:
    runs = pd.read_csv(PHASE2_ROOT / "results" / "task2_runs.csv")
    key_runs = [
        ("refseq_gd_full_ar5_s200", "GD full\nar5 s200", "#2f6fbb"),
        ("refseq_rmu_full_sc50_s200", "RMU full\nsc50 s200", "#6f4fa3"),
        ("refseq_gd_loc_af1_ar3_s200", "GD localized\nar3 s200", "#d7872d"),
        ("refseq_gd_loc_ar5_s500", "GD localized\ns500", "#c85c34"),
        ("refseq_gd_loc_ar5_s1000", "GD localized\ns1000", "#a8412f"),
        ("refseq_rmu_loc_sc50_s200", "RMU localized\nsc50 s200", "#2b9c8a"),
        ("refseq_rmu_random_sc50_s1000", "Random ctrl\nRMU s1000", "#87919c"),
    ]

    rows = []
    run_index = runs.set_index("run")
    for run, label, color in key_runs:
        if run not in run_index.index:
            continue
        item = run_index.loc[run]
        rows.append(
            {
                "run": run,
                "label": label,
                "color": color,
                "condition": item["condition"],
                "method": item["method"],
                "auroc": mean_l5_9_auroc(run),
                "retain_ppl": float(item["retain_ppl"]),
                "forget_ppl": float(item["forget_ppl"]),
            }
        )
    df = pd.DataFrame(rows)
    random_auroc = float(df.loc[df["run"] == "refseq_rmu_random_sc50_s1000", "auroc"].iloc[0])
    df["drop_vs_random"] = random_auroc - df["auroc"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.5),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.suptitle(
        "Phase 2: Full-model unlearning reduces the probe most; localized runs trade off weak forgetting vs PPL damage",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    x = np.arange(len(df))
    bars = ax.bar(x, df["auroc"], color=df["color"], width=0.68)
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.1, label="chance")
    ax.axhline(random_auroc, color="#777777", linestyle="--", linewidth=1.1, label=f"random control {random_auroc:.3f}")
    ax.set_title("Residual target signal after unlearning", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean test AUROC across L5-9 (lower is more forgetting)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0.45, 1.04)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(ax)
    for bar, auroc, drop in zip(bars, df["auroc"], df["drop_vs_random"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{auroc:.3f}\n(drop {drop:.3f})",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    ax = axes[1]
    marker_map = {"full": "o", "localized": "s", "random": "^"}
    for _, row in df.iterrows():
        ax.scatter(
            row["retain_ppl"],
            row["auroc"],
            s=125,
            marker=marker_map.get(row["condition"], "o"),
            color=row["color"],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            row["label"].replace("\n", " "),
            (row["retain_ppl"], row["auroc"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.4,
        )
    ax.axhline(random_auroc, color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0)
    ax.axvline(4.2, color="#777777", linestyle=":", linewidth=1.0, label="base retain PPL approx 4.2")
    ax.set_xscale("log")
    ax.set_title("Forgetting vs retain cost", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Retain PPL (log scale; lower is better)")
    ax.set_ylabel("Mean test AUROC across L5-9")
    ax.set_xlim(3.6, max(1500, df["retain_ppl"].max() * 1.2))
    ax.set_ylim(0.45, 1.04)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax)

    note = (
        "Key readout: full GD reaches AUROC ~0.547 with retain PPL ~4.13; "
        "localized GD only becomes stronger when retain PPL explodes."
    )
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=9.3, color="#333333")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "meeting_phase2_refseq_sweep.png"
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    phase1_path = plot_phase1()
    phase2_path = plot_phase2()
    print(f"Wrote {phase1_path}")
    print(f"Wrote {phase2_path}")


if __name__ == "__main__":
    main()
