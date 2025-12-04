from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Use a non-interactive backend so the script works headlessly.
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load optimizer metrics from exp/run_*/*_metrics.csv and save comparison plots."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing run_*/ subfolders (default: directory of this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: same as --base-dir).",
    )
    return parser.parse_args()


def load_metrics(base_dir: Path) -> pd.DataFrame:
    metrics_files = sorted(base_dir.glob("run_*/*_metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics CSV files found under {base_dir}/run_*/")

    frames = []
    for csv_path in metrics_files:
        optimizer_name = csv_path.stem.replace("_metrics", "")
        df = pd.read_csv(csv_path)
        missing = {
            "iteration",
            "wall_time_sec",
            "eval_accuracy",
            "eval_loss",
        } - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        df["optimizer"] = optimizer_name
        df["metrics_path"] = csv_path.as_posix()
        frames.append(df)

    metrics_df = pd.concat(frames, ignore_index=True)
    metrics_df = metrics_df.sort_values(["optimizer", "iteration"]).reset_index(drop=True)
    return metrics_df


def plot_accuracy_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values("iteration")
        ax.plot(
            sorted_group["iteration"],
            sorted_group["eval_accuracy"],
            marker="o",
            label=optimizer,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Eval accuracy")
    ax.set_title("Iteration vs eval accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()

    output_path = output_dir / "eval_accuracy_vs_iteration.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_accuracy_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values("wall_time_sec")
        ax.plot(
            sorted_group["wall_time_sec"],
            sorted_group["eval_accuracy"],
            marker="o",
            label=optimizer,
        )

    ax.set_xlabel("Wall clock time (s)")
    ax.set_ylabel("Eval accuracy")
    ax.set_title("Wall clock time vs eval accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()

    output_path = output_dir / "eval_accuracy_vs_wall_time.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values("iteration")
        ax.plot(
            sorted_group["iteration"],
            sorted_group["eval_loss"],
            marker="o",
            label=optimizer,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Eval loss")
    ax.set_title("Iteration vs eval loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()

    output_path = output_dir / "eval_loss_vs_iteration.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values("wall_time_sec")
        ax.plot(
            sorted_group["wall_time_sec"],
            sorted_group["eval_loss"],
            marker="o",
            label=optimizer,
        )

    ax.set_xlabel("Wall clock time (s)")
    ax.set_ylabel("Eval loss")
    ax.set_title("Wall clock time vs eval loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()

    output_path = output_dir / "eval_loss_vs_wall_time.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir or base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(base_dir)
    combined_csv = output_dir / "optimizer_metrics_combined.csv"
    metrics_df.to_csv(combined_csv, index=False)

    iter_plot = plot_accuracy_vs_iteration(metrics_df, output_dir)
    wall_time_plot = plot_accuracy_vs_wall_time(metrics_df, output_dir)
    loss_iter_plot = plot_loss_vs_iteration(metrics_df, output_dir)
    loss_wall_time_plot = plot_loss_vs_wall_time(metrics_df, output_dir)

    print(f"Loaded metrics from {base_dir}")
    print(f"Combined CSV written to {combined_csv}")
    print(f"Saved plot: {iter_plot}")
    print(f"Saved plot: {wall_time_plot}")
    print(f"Saved plot: {loss_iter_plot}")
    print(f"Saved plot: {loss_wall_time_plot}")


if __name__ == "__main__":
    main()
