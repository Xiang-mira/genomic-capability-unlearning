"""Export and plot GD/RMU convergence diagnostics from training logs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict | list:
    with path.open() as f:
        return json.load(f)


def iter_run_dirs(roots: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and (child / "log.json").exists() and (child / "meta.json").exists():
                dirs.append(child)
    return dirs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_group(rows: list[dict], keys: list[str], out_path: Path, title: str) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = sorted({row["run"] for row in rows})
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, max(2.4 * len(keys), 3)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        for run in runs:
            sub = sorted([row for row in rows if row["run"] == run and row.get(key) != ""], key=lambda r: int(r["step"]))
            if not sub:
                continue
            ax.plot([int(row["step"]) for row in sub], [float(row[key]) for row in sub], marker="o", label=run)
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_title(title)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-roots", nargs="+", default=["data/phase2/checkpoints_tuned", "data/phase2/checkpoints"])
    parser.add_argument("--out-dir", default="data/phase2/virobench_diagnostics")
    args = parser.parse_args()

    gd_rows = []
    rmu_rows = []
    for run_dir in iter_run_dirs(args.checkpoint_roots):
        meta = load_json(run_dir / "meta.json")
        log_rows = load_json(run_dir / "log.json")
        method = meta.get("method", "")
        target = gd_rows if method == "gradient_difference" else rmu_rows if method == "rmu" else None
        if target is None:
            continue
        for row in log_rows:
            target.append({
                "run": run_dir.name,
                "method": method,
                "condition": meta.get("condition", ""),
                **row,
            })

    out_dir = Path(args.out_dir)
    gd_fields = [
        "run", "method", "condition", "step", "L_forget", "L_retain",
        "weighted_forget_term", "weighted_retain_term", "loss",
    ]
    rmu_fields = [
        "run", "method", "condition", "step", "L_forget_mse", "L_retain_mse",
        "forget_to_target_mse", "forget_to_original_mse", "retain_rep_mse",
        "forget_original_modified_cosine", "weighted_forget_term",
        "weighted_retain_term", "loss", "target_norm", "target_variance",
    ]
    write_csv(out_dir / "gd_convergence_diagnostics.csv", gd_rows, gd_fields)
    write_csv(out_dir / "rmu_convergence_diagnostics.csv", rmu_rows, rmu_fields)
    plot_group(
        gd_rows,
        ["L_forget", "L_retain", "weighted_forget_term", "weighted_retain_term", "loss"],
        out_dir / "gd_convergence_diagnostics.png",
        "GD convergence diagnostics",
    )
    plot_group(
        rmu_rows,
        [
            "forget_to_target_mse", "forget_to_original_mse", "retain_rep_mse",
            "forget_original_modified_cosine", "target_norm", "target_variance",
        ],
        out_dir / "rmu_convergence_diagnostics.png",
        "RMU convergence diagnostics",
    )
    print(f"[convergence] wrote {out_dir}")


if __name__ == "__main__":
    main()
