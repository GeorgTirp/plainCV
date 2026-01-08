from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_context("paper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-layer eigenvalue trajectories from a gradient_eigenvalues folder."
        )
    )
    parser.add_argument(
        "--eigs-dir",
        type=Path,
        required=True,
        help="Path to a gradient_eigenvalues folder containing per-layer CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: same as --eigs-dir).",
    )
    return parser.parse_args()


def _pick_x_key(df: pd.DataFrame) -> str:
    for candidate in ("global_step", "step", "iteration", "epoch"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a suitable x-axis column (global_step/step/iteration/epoch)"
    )


def _eigen_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("eig_")]
    if not cols:
        raise ValueError("No eigenvalue columns starting with 'eig_' found.")
    return cols


def _format_title(stem: str) -> str:
    title = stem
    if title.startswith("params_"):
        title = title[len("params_") :]
    for suffix in ("_kernel", "_bias"):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
            break
    return title


def _classify_layer(stem: str) -> str | None:
    name = stem.lower()
    if "selfattention" in name:
        return "attention"
    if "mlpblock" in name or "dense" in name:
        return "mlp_dense"
    return None


def _plot_layer_tiles(
    layer_files: list[Path], output_path: Path, page_title: str
) -> Path:
    if not layer_files:
        raise ValueError(f"No layer files matched for page: {page_title}")

    total_plots = len(layer_files)
    cols = min(3, total_plots)
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_iter = iter(axes.flat)

    for csv_path in layer_files:
        ax = next(axes_iter)
        df = pd.read_csv(csv_path)
        x_key = _pick_x_key(df)
        eig_cols = _eigen_cols(df)
        for eig_col in eig_cols:
            ax.plot(
                df[x_key],
                df[eig_col],
                linewidth=1.0,
                alpha=0.85,
            )
        ax.set_title(_format_title(csv_path.stem))
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel("eigenvalue")
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    for ax in axes_iter:
        ax.axis("off")

    fig.suptitle(page_title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    sns.despine(fig=fig)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_muon_eigenvalues_pages(eigs_dir: Path, output_dir: Path) -> list[Path]:
    if not eigs_dir.exists():
        raise FileNotFoundError(f"Eigenvalue folder not found: {eigs_dir}")

    csv_files = sorted(eigs_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {eigs_dir}")

    mlp_dense_files: list[Path] = []
    attention_files: list[Path] = []
    for csv_path in csv_files:
        category = _classify_layer(csv_path.stem)
        if category == "attention":
            attention_files.append(csv_path)
        elif category == "mlp_dense":
            mlp_dense_files.append(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if mlp_dense_files:
        outputs.append(
            _plot_layer_tiles(
                mlp_dense_files,
                output_dir / "eigenmuon_mlp_dense_layers.png",
                "MLP Blocks + Dense Layer Eigenvalue Trajectories",
            )
        )
    if attention_files:
        outputs.append(
            _plot_layer_tiles(
                attention_files,
                output_dir / "eigenmuon_attention_layers.png",
                "Self-Attention Layer Eigenvalue Trajectories",
            )
        )

    if not outputs:
        raise ValueError(
            "No layer CSVs matched MLP/Dense or SelfAttention patterns. "
            "Check layer naming in gradient_eigenvalues."
        )
    return outputs


def main() -> None:
    args = parse_args()
    eigs_dir = args.eigs_dir
    output_dir = args.output_dir or eigs_dir
    output_paths = plot_muon_eigenvalues_pages(eigs_dir, output_dir)
    for path in output_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
