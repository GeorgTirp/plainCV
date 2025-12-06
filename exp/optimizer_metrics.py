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
            "train_accuracy",
            "train_loss",
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


def _plot_metric(
    metrics_df: pd.DataFrame,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values(x_key)
        ax.plot(
            sorted_group[x_key],
            sorted_group[y_key],
            marker="o",
            label=optimizer,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_accuracy_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="iteration",
        y_key="eval_accuracy",
        title="Iteration vs eval accuracy",
        xlabel="Iteration",
        ylabel="Eval accuracy",
        filename="eval_accuracy_vs_iteration.png",
        output_dir=output_dir,
    )


def plot_accuracy_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="wall_time_sec",
        y_key="eval_accuracy",
        title="Wall clock time vs eval accuracy",
        xlabel="Wall clock time (s)",
        ylabel="Eval accuracy",
        filename="eval_accuracy_vs_wall_time.png",
        output_dir=output_dir,
    )


def plot_loss_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="iteration",
        y_key="eval_loss",
        title="Iteration vs eval loss",
        xlabel="Iteration",
        ylabel="Eval loss",
        filename="eval_loss_vs_iteration.png",
        output_dir=output_dir,
    )


def plot_loss_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="wall_time_sec",
        y_key="eval_loss",
        title="Wall clock time vs eval loss",
        xlabel="Wall clock time (s)",
        ylabel="Eval loss",
        filename="eval_loss_vs_wall_time.png",
        output_dir=output_dir,
    )


def plot_train_accuracy_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="iteration",
        y_key="train_accuracy",
        title="Iteration vs train accuracy",
        xlabel="Iteration",
        ylabel="Train accuracy",
        filename="train_accuracy_vs_iteration.png",
        output_dir=output_dir,
    )


def plot_train_accuracy_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="wall_time_sec",
        y_key="train_accuracy",
        title="Wall clock time vs train accuracy",
        xlabel="Wall clock time (s)",
        ylabel="Train accuracy",
        filename="train_accuracy_vs_wall_time.png",
        output_dir=output_dir,
    )


def plot_train_loss_vs_iteration(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="iteration",
        y_key="train_loss",
        title="Iteration vs train loss",
        xlabel="Iteration",
        ylabel="Train loss",
        filename="train_loss_vs_iteration.png",
        output_dir=output_dir,
    )


def plot_train_loss_vs_wall_time(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric(
        metrics_df,
        x_key="wall_time_sec",
        y_key="train_loss",
        title="Wall clock time vs train loss",
        xlabel="Wall clock time (s)",
        ylabel="Train loss",
        filename="train_loss_vs_wall_time.png",
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir or base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(base_dir)
    combined_csv = output_dir / "optimizer_metrics_combined.csv"
    metrics_df.to_csv(combined_csv, index=False)

    plots = [
        plot_accuracy_vs_iteration(metrics_df, output_dir),
        plot_accuracy_vs_wall_time(metrics_df, output_dir),
        plot_loss_vs_iteration(metrics_df, output_dir),
        plot_loss_vs_wall_time(metrics_df, output_dir),
        plot_train_accuracy_vs_iteration(metrics_df, output_dir),
        plot_train_accuracy_vs_wall_time(metrics_df, output_dir),
        plot_train_loss_vs_iteration(metrics_df, output_dir),
        plot_train_loss_vs_wall_time(metrics_df, output_dir),
    ]

    print(f"Loaded metrics from {base_dir}")
    print(f"Combined CSV written to {combined_csv}")
    for path in plots:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
