import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot layer-wise probe metrics.")
    parser.add_argument("--metrics", default="data/family_targets/coronaviridae/probes/probe_metrics_by_layer.csv")
    parser.add_argument("--out-dir", default="data/family_targets/coronaviridae/probes")
    parser.add_argument("--title", default="Layer-wise Coronaviridae target-family probe")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics).sort_values("layer")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, "probe_metrics.png")

    plt.figure(figsize=(10, 4))
    plt.plot(df["layer"], df["val_auroc"], label="val_auroc", marker="o")
    plt.plot(df["layer"], df["test_auroc"], label="test_auroc", marker="o")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
