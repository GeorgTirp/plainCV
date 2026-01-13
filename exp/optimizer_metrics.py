from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Use a non-interactive backend so the script works headlessly.
matplotlib.use("Agg")
sns.set_context("paper")


def _friendly_run_label(run_folder: str) -> str | None:
    """Map run folder names to human-friendly labels; return None to skip plotting."""
    name = run_folder.lower()
    explicit_labels = {
        "run_adam_vit_small": "AdamW",
        "run_muon_vit_small": "Muon",
        "run_pns_eigenadam_vit_small": "PARSEC-H",
        "run_soap_vit_small": "SOAP",
        "run_pns_eigenmuon_vit_small": "PARSEC-M",
    }
    if name in explicit_labels:
        return explicit_labels[name]
    if "pns_eigenadam" in name or "leneidam" in name:
        return "PARSEC-H"
    if "pns_eigenmuon" in name or "leneimuon" in name:
        return "PARSEC-M"
    if "fim" in name and "sqrt" in name:
        return "PARSEC-H FIM sqrt"
    if "sqrt" in name:
        return "PARSEC-H sqrt GGN"
    if "neg" in name or "hessian" in name:
        return "PARSEC-H Hessian"
    # Filter out PARSEC-H GGN runs from plots.
    if "pos" in name or "ggn" in name:
        return None
    return run_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load optimizer metrics from exp/run_ti*/*_metrics.csv and save comparison plots."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing run_ti*/ subfolders (default: directory of this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: same as --base-dir).",
    )
    parser.add_argument(
        "--folders-list",
        type=Path,
        default=None,
        help=(
            "Path to a text file containing one run folder per line. "
            "Used for both optimizer metrics and curvature plotting unless "
            "--curvature-folders-list is provided."
        ),
    )
    parser.add_argument(
        "--curvature-folders-list",
        type=Path,
        default=None,
        help=(
            "Path to a text file containing one run folder per line. "
            "Each folder must contain a curvature_metrics CSV (or curvature.csv)."
        ),
    )
    return parser.parse_args()

def _read_run_folders_list(folders_list_path: Path) -> list[Path]:
    if not folders_list_path.exists():
        raise FileNotFoundError(f"Run folders list not found: {folders_list_path}")

    run_folders: list[Path] = []
    for raw_line in folders_list_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        folder = Path(line)
        if not folder.is_absolute():
            folder = (folders_list_path.parent / folder).resolve()

        if not folder.is_dir():
            raise FileNotFoundError(f"Listed run folder does not exist: {folder}")
        run_folders.append(folder)

    if not run_folders:
        raise ValueError(f"No run folders found in {folders_list_path}")

    return run_folders


def _normalize_run_folders(
    run_folders: list[str | Path], base_dir: Path
) -> list[Path]:
    normalized: list[Path] = []
    for folder in run_folders:
        path = Path(folder)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Listed run folder does not exist: {path}")
        normalized.append(path)
    if not normalized:
        raise ValueError("Run folders list is empty.")
    return normalized


def load_metrics(
    base_dir: Path,
    folders_list_path: Path | None = None,
    run_folders: list[Path] | None = None,
) -> pd.DataFrame:
    if run_folders is not None:
        metrics_files: list[Path] = []
        for folder in run_folders:
            metrics_files.extend(sorted(folder.glob("*_metrics.csv")))
        if not metrics_files:
            raise FileNotFoundError("No metrics CSV files found in the listed run folders.")
    elif folders_list_path is not None:
        run_folders = _read_run_folders_list(folders_list_path)
        metrics_files: list[Path] = []
        for folder in run_folders:
            metrics_files.extend(sorted(folder.glob("*_metrics.csv")))
        if not metrics_files:
            raise FileNotFoundError(
                f"No metrics CSV files found in listed folders from {folders_list_path}"
            )
    else:
        metrics_files = sorted(base_dir.glob("run_*/*_metrics.csv"))
        if not metrics_files:
            raise FileNotFoundError(f"No metrics CSV files found under {base_dir}/run_*/")

    frames = []
    skipped_runs = []
    for csv_path in metrics_files:
        run_folder = csv_path.parent.name
        optimizer_name = _friendly_run_label(run_folder)
        if optimizer_name is None:
            skipped_runs.append(run_folder)
            continue
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

    if not frames:
        skipped_msg = f" Skipped runs: {sorted(set(skipped_runs))}" if skipped_runs else ""
        raise FileNotFoundError(f"No metrics CSV files left to plot under {base_dir}/run_*/.{skipped_msg}")

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
            markersize=3,
            linewidth=1.2,
            label=optimizer,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()
    sns.despine(fig=fig)

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


def plot_train_eval_accuracy(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for optimizer, group in metrics_df.groupby("optimizer"):
        sorted_group = group.sort_values("iteration")
        ax.plot(
            sorted_group["iteration"],
            sorted_group["eval_accuracy"],
            label=f"{optimizer} eval",
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
        ax.plot(
            sorted_group["iteration"],
            sorted_group["train_accuracy"],
            label=f"{optimizer} train",
            marker="s",
            markersize=2.5,
            linewidth=1.0,
            alpha=0.85,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Iteration vs train/eval accuracy")
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Optimizer")
    plt.tight_layout()
    sns.despine(fig=fig)

    output_path = output_dir / "train_eval_accuracy_vs_iteration.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_curvature_runs_from_folders(
    run_folders: list[Path],
) -> list[tuple[str, pd.DataFrame]]:
    runs: list[tuple[str, pd.DataFrame]] = []
    skipped: list[str] = []
    for folder in run_folders:
        maybe_run = _maybe_load_curvature_run(folder)
        if maybe_run is None:
            skipped.append(folder.name)
            continue
        runs.append(maybe_run)
    if skipped:
        print(
            "Warning: skipped runs without curvature CSV: "
            f"{sorted(set(skipped))}"
        )
    return runs


def load_curvature_runs(folders_list_path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Read curvature CSVs for the runs listed in ``folders_list_path``.

    The file should contain one folder per line. Each folder is expected to have
    either ``curvature_metrics.csv`` or ``curvature.csv`` inside. Relative paths
    are resolved relative to the list file location.
    """

    run_folders = _read_run_folders_list(folders_list_path)
    return load_curvature_runs_from_folders(run_folders)


def _extract_eigenvalue_columns(df: pd.DataFrame) -> list[str]:
    eigen_cols = [c for c in df.columns if c.startswith("eig_")]
    if not eigen_cols:
        raise ValueError("Curvature CSV has no eigenvalue columns starting with 'eig_'")

    def _sort_key(col: str) -> tuple[str, int]:
        parts = col.split("_")
        try:
            idx = int(parts[-1])
        except ValueError:
            idx = math.inf
        return ("_".join(parts[:-1]), idx)

    return sorted(eigen_cols, key=_sort_key)


def _sort_eigen_cols(cols: list[str]) -> list[str]:
    def _sort_key(col: str) -> tuple[str, int]:
        parts = col.split("_")
        try:
            idx = int(parts[-1])
        except ValueError:
            idx = math.inf
        return ("_".join(parts[:-1]), idx)

    return sorted(cols, key=_sort_key)


def _pick_x_key(df: pd.DataFrame) -> str:
    for candidate in ("global_step", "step", "iteration", "epoch"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a suitable x-axis column (global_step/step/iteration/epoch)")


def plot_curvature_trajectories(
    curvature_runs: list[tuple[str, pd.DataFrame]], output_dir: Path
) -> Path:
    filtered_runs: list[tuple[str, pd.DataFrame]] = []
    skipped: list[tuple[str, list[str]]] = []
    shared_eigen_cols: set[str] | None = None
    eigen_cols_per_run: list[tuple[str, list[str]]] = []

    for run_name, df in curvature_runs:
        eigen_cols = _extract_eigenvalue_columns(df)
        eigen_cols_per_run.append((run_name, eigen_cols))

        if shared_eigen_cols is None:
            shared_eigen_cols = set(eigen_cols)
            filtered_runs.append((run_name, df))
            continue

        overlap = shared_eigen_cols.intersection(eigen_cols)
        if not overlap:
            skipped.append((run_name, eigen_cols))
            continue

        shared_eigen_cols = overlap
        filtered_runs.append((run_name, df))

    if not shared_eigen_cols:
        details = {name: cols for name, cols in eigen_cols_per_run}
        raise ValueError(
            "No common eigenvalue columns across runs. Found per run: " f"{details}"
        )

    if skipped:
        print(
            "Warning: skipped curvature runs with incompatible eigen columns: "
            f"{ {name: cols for name, cols in skipped} }"
        )

    curvature_runs = filtered_runs
    eigen_cols = _sort_eigen_cols(list(shared_eigen_cols))

    first_df = curvature_runs[0][1]
    x_key = _pick_x_key(first_df)

    total_plots = len(eigen_cols) + 1  # extra panel for rotation_diff_pos
    cols = min(3, total_plots)
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_iter = iter(axes.flat)

    for eig_col in eigen_cols:
        ax = next(axes_iter)
        for run_name, df in curvature_runs:
            ax.plot(
                df[x_key],
                df[eig_col],
                label=run_name,
                marker="o",
                markersize=3,
                linewidth=1.2,
            )
        ax.set_title(eig_col)
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel("eigenvalue")
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    rot_col_candidates = ["rotation_diff_pos", "rotation_diff"]
    rot_col = next((c for c in rot_col_candidates if all(c in df.columns for _, df in curvature_runs)), None)
    if rot_col is None:
        available = {name: [c for c in df.columns if "rotation" in c] for name, df in curvature_runs}
        raise ValueError(
            "No common rotation diff column across runs; looked for rotation_diff_pos/rotation_diff."
            f" Available: {available}"
        )

    ax = next(axes_iter)
    for run_name, df in curvature_runs:
        ax.plot(
            df[x_key],
            df[rot_col],
            label=run_name,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    ax.set_title(rot_col)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel("rotation diff")
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Hide any unused axes
    for ax in axes_iter:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), cols))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    sns.despine(fig=fig)

    output_path = output_dir / "curvature_trajectories.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _maybe_load_curvature_run(folder: Path) -> tuple[str, pd.DataFrame] | None:
    """Return (label, df) for the folder if a curvature CSV exists, else None."""
    candidates = [folder / "curvature_metrics.csv", folder / "curvature.csv"]
    curvature_csv = next((c for c in candidates if c.exists()), None)
    if curvature_csv is None:
        return None

    df = pd.read_csv(curvature_csv)
    run_label = _friendly_run_label(folder.name)
    if run_label is None:
        return None
    df["run"] = run_label
    return run_label, df


def discover_curvature_runs(base_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    """Collect curvature CSVs under run_* subfolders for plotting."""
    runs: list[tuple[str, pd.DataFrame]] = []
    for run_dir in sorted(base_dir.glob("run_*")):
        maybe_run = _maybe_load_curvature_run(run_dir)
        if maybe_run is not None:
            runs.append(maybe_run)
    return runs


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    output_dir = args.output_dir or base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional in-code override for both metrics + curvature selection.
    run_folders_override: list[str] | None = None
    # Example:
    run_folders_override = [
        "run_adam_resnet18",
        "run_muon_resnet18",
        "run_pns_eigenadam_resnet18",
        "run_soap_resnet18",
        "run_pns_eigenmuon_resnet18",
    ]
    run_folders = (
        _normalize_run_folders(run_folders_override, base_dir)
        if run_folders_override is not None
        else None
    )

    metrics_df = load_metrics(base_dir, args.folders_list, run_folders)
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
        plot_train_eval_accuracy(metrics_df, output_dir),
    ]

    curvature_plot = None
    curvature_runs: list[tuple[str, pd.DataFrame]] = []
    curvature_list = args.curvature_folders_list or args.folders_list
    if run_folders is not None:
        curvature_runs = load_curvature_runs_from_folders(run_folders)
    elif curvature_list is not None:
        curvature_runs = load_curvature_runs(curvature_list)
    else:
        curvature_runs = discover_curvature_runs(base_dir)

    if curvature_runs:
        curvature_plot = plot_curvature_trajectories(curvature_runs, output_dir)

    print(f"Loaded metrics from {base_dir}")
    print(f"Combined CSV written to {combined_csv}")
    for path in plots:
        print(f"Saved plot: {path}")
    if curvature_plot is not None:
        print(f"Saved plot: {curvature_plot}")


if __name__ == "__main__":
    main()
