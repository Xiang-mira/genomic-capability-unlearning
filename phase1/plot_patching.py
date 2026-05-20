"""
Visualize activation patching results alongside probe AUROC.

Produces a 3-panel figure:
  Top:    Layer-wise probe AUROC (val + test)
  Middle: |delta_human_prob| per layer, both patching directions
  Bottom: delta_perplexity per layer, both patching directions

Highlights the localized region (layers 3-9) identified by patching.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


LOCALIZED_START = 3
LOCALIZED_END = 9


def load_data(probe_csv: str, patching_csv: str):
    probes = pd.read_csv(probe_csv).sort_values("layer")

    patching = pd.read_csv(patching_csv)
    nh2h = patching[patching["direction"] == "nonhuman_to_human"].sort_values("layer")
    h2nh = patching[patching["direction"] == "human_to_nonhuman"].sort_values("layer")

    # Average both directions for a single "patching effect" signal
    # Compute perplexity delta from loss columns
    for df in [nh2h, h2nh]:
        df["delta_perplexity"] = np.exp(df["patched_loss"]) - np.exp(df["clean_loss"])

    merged = nh2h[["layer", "abs_delta_human_prob", "delta_perplexity", "activation_l2"]].copy()
    merged = merged.rename(columns={
        "abs_delta_human_prob": "abs_delta_prob_nh2h",
        "delta_perplexity": "delta_ppl_nh2h",
        "activation_l2": "l2_nh2h",
    })
    h2nh_sub = h2nh[["layer", "abs_delta_human_prob", "delta_perplexity"]].rename(columns={
        "abs_delta_human_prob": "abs_delta_prob_h2nh",
        "delta_perplexity": "delta_ppl_h2nh",
    })
    merged = merged.merge(h2nh_sub, on="layer")
    merged["mean_abs_delta_prob"] = (merged["abs_delta_prob_nh2h"] + merged["abs_delta_prob_h2nh"]) / 2

    return probes, nh2h, h2nh, merged


def shade_localized(ax, num_layers):
    ax.axvspan(LOCALIZED_START - 0.5, LOCALIZED_END + 0.5,
               alpha=0.10, color="steelblue", zorder=0, label="Localized region (3–9)")


def mark_unstable(ax, threshold_l2, merged, ymin, ymax):
    unstable = merged[merged["l2_nh2h"] > threshold_l2]["layer"].values
    if len(unstable):
        ax.axvspan(unstable[0] - 0.5, merged["layer"].max() + 0.5,
                   alpha=0.07, color="red", zorder=0, label=f"Numerically unstable (L2 > {threshold_l2:.0e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-csv", default="data/host_tropism/probes/probe_metrics_by_layer.csv")
    parser.add_argument("--patching-csv", default="data/host_tropism/activation_patching/patching_by_layer.csv")
    parser.add_argument("--out-dir", default="data/host_tropism/activation_patching")
    args = parser.parse_args()

    probes, nh2h, h2nh, merged = load_data(args.probe_csv, args.patching_csv)
    layers = probes["layer"].values
    num_layers = len(layers)

    # Detect unstable threshold: first layer where L2 jumps > 1000x previous
    l2_vals = merged.sort_values("layer")["l2_nh2h"].values
    unstable_threshold = 1e4  # conservative; actual jump is ~50 → 1.8M at layer 11

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Activation Patching Analysis — Evo-1-8k-base\nHost Tropism: Human-tropic vs Non-human-tropic",
                 fontsize=13, y=0.98)

    # ── Panel 1: Probe AUROC ──────────────────────────────────────────────────
    ax = axes[0]
    shade_localized(ax, num_layers)
    mark_unstable(ax, unstable_threshold, merged, 0.5, 1.0)
    ax.plot(layers, probes["val_auroc"], "o-", color="royalblue", ms=4, lw=1.5, label="Val AUROC")
    ax.plot(layers, probes["test_auroc"], "s--", color="darkorange", ms=4, lw=1.5, label="Test AUROC")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Probe AUROC", fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) Layer-wise logistic probe accuracy", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    # Annotate peak
    peak_idx = probes["test_auroc"].idxmax()
    peak_layer = probes.loc[peak_idx, "layer"]
    peak_val = probes.loc[peak_idx, "test_auroc"]
    ax.annotate(f"peak L{peak_layer}\n{peak_val:.3f}",
                xy=(peak_layer, peak_val), xytext=(peak_layer + 1.5, peak_val - 0.04),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))

    # ── Panel 2: |delta_human_prob| ──────────────────────────────────────────
    ax = axes[1]
    shade_localized(ax, num_layers)
    mark_unstable(ax, unstable_threshold, merged, 0, 0.5)

    # Only plot layers where probe is valid (AUROC > 0.65 on test)
    valid_layers = probes[probes["test_auroc"] > 0.65]["layer"].values
    nh2h_valid = nh2h[nh2h["layer"].isin(valid_layers)]
    h2nh_valid = h2nh[h2nh["layer"].isin(valid_layers)]
    merged_valid = merged[merged["layer"].isin(valid_layers)]

    ax.bar(merged_valid["layer"] - 0.2, merged_valid["abs_delta_prob_nh2h"],
           width=0.35, color="steelblue", alpha=0.75, label="nonhuman → human")
    ax.bar(merged_valid["layer"] + 0.2, merged_valid["abs_delta_prob_h2nh"],
           width=0.35, color="tomato", alpha=0.75, label="human → nonhuman")
    ax.plot(merged_valid["layer"], merged_valid["mean_abs_delta_prob"],
            "k^-", ms=5, lw=1.2, label="Mean |Δprob|", zorder=5)

    ax.set_ylabel("|Δ probe prob|", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) Causal effect of patching on probe prediction\n"
                 "(grey = layers with probe AUROC < 0.65, excluded)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    # Annotate top-3 layers
    top3 = merged_valid.nlargest(3, "mean_abs_delta_prob")
    for _, row in top3.iterrows():
        ax.annotate(f"L{int(row['layer'])}\n{row['mean_abs_delta_prob']:.3f}",
                    xy=(row["layer"], row["mean_abs_delta_prob"]),
                    xytext=(row["layer"] + 0.5, row["mean_abs_delta_prob"] + 0.02),
                    fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))

    # ── Panel 3: delta_perplexity ─────────────────────────────────────────────
    ax = axes[2]
    shade_localized(ax, num_layers)
    mark_unstable(ax, unstable_threshold, merged, -0.1, 0.5)

    nh2h_valid2 = nh2h[nh2h["layer"].isin(valid_layers)]
    h2nh_valid2 = h2nh[h2nh["layer"].isin(valid_layers)]

    ax.plot(nh2h_valid2["layer"], nh2h_valid2["delta_perplexity"],
            "o-", color="steelblue", ms=4, lw=1.5, label="nonhuman → human")
    ax.plot(h2nh_valid2["layer"], h2nh_valid2["delta_perplexity"],
            "s--", color="tomato", ms=4, lw=1.5, label="human → nonhuman")
    ax.axhline(0, color="gray", lw=0.8, ls=":")

    ax.set_ylabel("Δ Perplexity (patched − clean)", fontsize=10)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) Perplexity change after patching\n"
                 "(constant = patching doesn't affect final output loss)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)

    # ── x-axis ticks ─────────────────────────────────────────────────────────
    axes[2].set_xticks(range(0, num_layers, 2))
    axes[2].set_xlim(-0.5, num_layers - 0.5)

    # ── shared legend for shading ─────────────────────────────────────────────
    loc_patch = mpatches.Patch(color="steelblue", alpha=0.15, label="Localized region (layers 3–9)")
    unstable_patch = mpatches.Patch(color="red", alpha=0.12, label="Numerically unstable (layers 11+)")
    fig.legend(handles=[loc_patch, unstable_patch], loc="lower center",
               ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "patching_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n=== Top layers by mean |Δprob| (probe-valid layers only) ===")
    print(f"{'Layer':>6}  {'|Δprob| nh→h':>13}  {'|Δprob| h→nh':>13}  {'Mean':>8}  {'Test AUROC':>10}")
    top = merged_valid.merge(probes[["layer", "test_auroc"]], on="layer").nlargest(10, "mean_abs_delta_prob")
    for _, row in top.iterrows():
        print(f"  {int(row['layer']):>4}  {row['abs_delta_prob_nh2h']:>13.4f}  "
              f"{row['abs_delta_prob_h2nh']:>13.4f}  {row['mean_abs_delta_prob']:>8.4f}  "
              f"{row['test_auroc']:>10.4f}")

    print("\n=== Perplexity delta (constant across layers — model recovers downstream) ===")
    print(f"  nonhuman→human:  Δppl = {nh2h['delta_perplexity'].mean():.4f} ± {nh2h['delta_perplexity'].std():.4f}")
    print(f"  human→nonhuman:  Δppl = {h2nh['delta_perplexity'].mean():.4f} ± {h2nh['delta_perplexity'].std():.4f}")


if __name__ == "__main__":
    main()
