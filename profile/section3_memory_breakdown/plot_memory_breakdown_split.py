#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_STYLES = [
    ("metadata_original_bytes", "Raw metadata", "#ed7d31"),
    ("metadata_lz4_bytes", "LZ4", "#a5a5a5"),
    ("metadata_deflate_bytes", "Deflate", "#264478"),
]


def find_dataset(datasets: list[dict], name: str) -> dict:
    for ds in datasets:
        if str(ds.get("name", "")).lower() == name.lower():
            return ds
    raise RuntimeError(f"Dataset '{name}' not found in profiling JSON.")


def scaled_values(ds: dict, mode: str, billion_rows: float) -> tuple[np.ndarray, str]:
    rows = float(ds["rows"])
    if rows <= 0.0:
        raise RuntimeError(f"Dataset '{ds.get('name', 'unknown')}' has non-positive rows.")

    raw = {k: float(v) for k, v in ds["raw_sizes"].items()}
    vals_bytes = np.array([raw[key] for key, _, _ in CATEGORY_STYLES], dtype=np.float64)

    if mode == "original":
        scale = 1.0
        denom = 1024.0 * 1024.0
        unit = "MB"
    elif mode == "billion":
        scale = billion_rows / rows
        denom = 1024.0 * 1024.0 * 1024.0
        unit = "GB"
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

    return (vals_bytes * scale) / denom, unit


def draw_panel(
    ax,
    values: np.ndarray,
    title: str,
    unit: str,
    *,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(CATEGORY_STYLES), dtype=np.float64)
    bar_w = 0.58

    for i, (_, _, color) in enumerate(CATEGORY_STYLES):
        ax.bar(
            x[i],
            values[i],
            width=bar_w,
            color=color,
            edgecolor="black",
            linewidth=0.25,
        )

    ymax = max(1e-9, float(values.max()))
    ypad = max(0.02 * ymax, 0.02)
    for i, v in enumerate(values):
        ax.text(
            x[i],
            v + ypad,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw", "LZ4", "Deflate"], fontsize=8, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(f"Size ({unit})", fontsize=9, fontweight="bold")
    ax.set_ylim(0.0, ymax * 1.22 + ypad)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def render_split_plot(data: dict, output_path: Path, billion_rows: float) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    datasets = data["datasets"]
    panel_specs = [
        ("laion", "original"),
        ("laion", "billion"),
        ("hnm", "original"),
        ("hnm", "billion"),
    ]
    panel_titles = {
        ("laion", "original"): "LAION (Original)",
        ("laion", "billion"): "LAION (1B)",
        ("hnm", "original"): "H&M (Original)",
        ("hnm", "billion"): "H&M (1B)",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.0 / 2.54, 5.0 / 2.54),
        sharey="col",
    )

    # Required placement:
    # left-top  : LAION original
    # left-bottom: H&M original
    # right-top : LAION billion
    # right-bottom: H&M billion
    ordered = {
        (0, 0): panel_specs[0],
        (0, 1): panel_specs[1],
        (1, 0): panel_specs[2],
        (1, 1): panel_specs[3],
    }

    for (r, c), (ds_name, mode) in ordered.items():
        ds = find_dataset(datasets, ds_name)
        vals, unit = scaled_values(ds, mode, billion_rows)
        draw_panel(
            axes[r, c],
            vals,
            panel_titles[(ds_name, mode)],
            unit,
            show_ylabel=True,
        )

    # Shared legend for categories.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.25)
        for _, _, color in CATEGORY_STYLES
    ]
    legend_labels = [label for _, label, _ in CATEGORY_STYLES]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9,
        prop={"family": "Arial", "size": 9, "weight": "bold"},
        handlelength=1.2,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.80, wspace=0.14, hspace=0.38)
    fig.savefig(output_path, format="png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot section3 memory breakdown split by dataset and scale."
    )
    parser.add_argument("--input", type=Path, default=Path("profiling.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--billion-rows", type=float, default=1_000_000_000.0)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not data.get("datasets"):
        raise RuntimeError("No datasets found in JSON.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "hnm_laion_memory_breakdown_split.png"
    render_split_plot(data, out_path, args.billion_rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
