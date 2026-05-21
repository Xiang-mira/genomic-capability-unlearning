import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot layer-wise probe metrics.")
<<<<<<< HEAD
    parser.add_argument("--metrics", default="data/host_tropism/probes/probe_metrics_by_layer.csv")
    parser.add_argument("--out-dir", default="data/host_tropism/probes")
=======
    parser.add_argument("--metrics", default="data/phase1/probes/probe_metrics_by_layer.csv")
    parser.add_argument("--out-dir", default="data/phase1/probes")
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    df = df.sort_values("layer")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, "probe_metrics.png")

    plt.figure(figsize=(10, 4))
    plt.plot(df["layer"], df["val_auroc"], label="val_auroc", marker="o")
    plt.plot(df["layer"], df["test_auroc"], label="test_auroc", marker="o")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Layer-wise viral vs non-viral probe")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
