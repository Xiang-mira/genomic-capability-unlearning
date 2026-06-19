"""
Visualize activation patching results alongside probe AUROC.

Produces a 3-panel figure:
  Top:    layer-wise probe AUROC (val + test)
  Middle: |delta target prob| per layer, both patching directions
  Bottom: delta_perplexity per layer, both patching directions
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_localized_layers(path: str):
    if not os.path.exists(path):
        return list(range(3, 10))
    with open(path) as f:
        payload = json.load(f)
    return sorted(int(layer) for layer in payload.get("layers", list(range(3, 10))))


def load_data(probe_csv: str, patching_csv: str):
    probes = pd.read_csv(probe_csv).sort_values("layer")
    patching = pd.read_csv(patching_csv)
    retain_to_target = patching[
        patching["direction"].isin(["retain_to_target", "nonhuman_to_human"])
    ].sort_values("layer")
    target_to_retain = patching[
        patching["direction"].isin(["target_to_retain", "human_to_nonhuman"])
    ].sort_values("layer")

    for df in [retain_to_target, target_to_retain]:
        df["delta_perplexity"] = np.exp(df["patched_loss"]) - np.exp(df["clean_loss"])

    merged = retain_to_target[["layer", "abs_delta_human_prob", "delta_perplexity", "activation_l2"]].copy()
    merged = merged.rename(
        columns={
            "abs_delta_human_prob": "abs_delta_prob_retain_to_target",
            "delta_perplexity": "delta_ppl_retain_to_target",
            "activation_l2": "l2_retain_to_target",
        }
    )
    target_to_retain_sub = target_to_retain[
        ["layer", "abs_delta_human_prob", "delta_perplexity"]
    ].rename(
        columns={
            "abs_delta_human_prob": "abs_delta_prob_target_to_retain",
            "delta_perplexity": "delta_ppl_target_to_retain",
        }
    )
    merged = merged.merge(target_to_retain_sub, on="layer")
    merged["mean_abs_delta_prob"] = (
        merged["abs_delta_prob_retain_to_target"] + merged["abs_delta_prob_target_to_retain"]
    ) / 2
    return probes, retain_to_target, target_to_retain, merged


def shade_localized(ax, localized_layers):
    ax.axvspan(
        min(localized_layers) - 0.5,
        max(localized_layers) + 0.5,
        alpha=0.10,
        color="steelblue",
        zorder=0,
        label=f"Localized region ({min(localized_layers)}-{max(localized_layers)})",
    )


def mark_unstable(ax, threshold_l2, merged):
    unstable = merged[merged["l2_retain_to_target"] > threshold_l2]["layer"].values
    if len(unstable):
        ax.axvspan(
            unstable[0] - 0.5,
            merged["layer"].max() + 0.5,
            alpha=0.07,
            color="red",
            zorder=0,
            label=f"Numerically unstable (L2 > {threshold_l2:.0e})",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-csv", default="data/family_targets/coronaviridae/probes/probe_metrics_by_layer.csv")
    parser.add_argument(
        "--patching-csv",
        default="data/family_targets/coronaviridae/activation_patching/patching_by_layer.csv",
    )
    parser.add_argument("--out-dir", default="data/family_targets/coronaviridae/activation_patching")
    parser.add_argument(
        "--localized-layers-path",
        default="data/family_targets/coronaviridae/localized_layers.json",
    )
    args = parser.parse_args()

    localized_layers = load_localized_layers(args.localized_layers_path)
    probes, retain_to_target, target_to_retain, merged = load_data(args.probe_csv, args.patching_csv)
    layers = probes["layer"].values
    unstable_threshold = 1e4

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Activation Patching Analysis — Evo-1-8k-base\nCoronaviridae target-family localization",
        fontsize=13,
        y=0.98,
    )

    ax = axes[0]
    shade_localized(ax, localized_layers)
    mark_unstable(ax, unstable_threshold, merged)
    ax.plot(layers, probes["val_auroc"], "o-", color="royalblue", ms=4, lw=1.5, label="Val AUROC")
    ax.plot(layers, probes["test_auroc"], "s--", color="darkorange", ms=4, lw=1.5, label="Test AUROC")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Probe AUROC", fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) Layer-wise logistic probe accuracy", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    peak_idx = probes["test_auroc"].idxmax()
    peak_layer = probes.loc[peak_idx, "layer"]
    peak_val = probes.loc[peak_idx, "test_auroc"]
    ax.annotate(
        f"peak L{peak_layer}\n{peak_val:.3f}",
        xy=(peak_layer, peak_val),
        xytext=(peak_layer + 1.5, peak_val - 0.04),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    ax = axes[1]
    shade_localized(ax, localized_layers)
    mark_unstable(ax, unstable_threshold, merged)
    valid_layers = probes[probes["test_auroc"] > 0.65]["layer"].values
    merged_valid = merged[merged["layer"].isin(valid_layers)]
    ax.bar(
        merged_valid["layer"] - 0.2,
        merged_valid["abs_delta_prob_retain_to_target"],
        width=0.35,
        color="steelblue",
        alpha=0.75,
        label="retain -> target",
    )
    ax.bar(
        merged_valid["layer"] + 0.2,
        merged_valid["abs_delta_prob_target_to_retain"],
        width=0.35,
        color="tomato",
        alpha=0.75,
        label="target -> retain",
    )
    ax.plot(
        merged_valid["layer"],
        merged_valid["mean_abs_delta_prob"],
        "k^-",
        ms=5,
        lw=1.2,
        label="Mean |Delta prob|",
        zorder=5,
    )
    ax.set_ylabel("|Delta probe prob|", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) Causal effect of patching on probe prediction", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    top3 = merged_valid.nlargest(3, "mean_abs_delta_prob")
    for _, row in top3.iterrows():
        ax.annotate(
            f"L{int(row['layer'])}\n{row['mean_abs_delta_prob']:.3f}",
            xy=(row["layer"], row["mean_abs_delta_prob"]),
            xytext=(row["layer"] + 0.5, row["mean_abs_delta_prob"] + 0.02),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    ax = axes[2]
    shade_localized(ax, localized_layers)
    mark_unstable(ax, unstable_threshold, merged)
    retain_valid = retain_to_target[retain_to_target["layer"].isin(valid_layers)]
    target_valid = target_to_retain[target_to_retain["layer"].isin(valid_layers)]
    ax.plot(
        retain_valid["layer"],
        retain_valid["delta_perplexity"],
        "o-",
        color="steelblue",
        ms=4,
        lw=1.5,
        label="retain -> target",
    )
    ax.plot(
        target_valid["layer"],
        target_valid["delta_perplexity"],
        "s--",
        color="tomato",
        ms=4,
        lw=1.5,
        label="target -> retain",
    )
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Delta Perplexity", fontsize=10)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) Perplexity change after patching", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    axes[2].set_xticks(range(0, len(layers), 2))
    axes[2].set_xlim(-0.5, len(layers) - 0.5)

    loc_patch = mpatches.Patch(
        color="steelblue",
        alpha=0.15,
        label=f"Localized region (layers {min(localized_layers)}-{max(localized_layers)})",
    )
    unstable_patch = mpatches.Patch(color="red", alpha=0.12, label="Numerically unstable")
    fig.legend(handles=[loc_patch, unstable_patch], loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "patching_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
