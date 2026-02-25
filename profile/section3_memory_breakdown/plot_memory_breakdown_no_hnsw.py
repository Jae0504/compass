#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_STYLES = [
    ("baseline", "Raw metadata", "#ed7d31"),             # orange  (B=1 color)
    ("lz4", "LZ4", "#a5a5a5"),                # gray    (B=4 color)
    ("deflate", "Deflate", "#264478"),        # navy    (B=16 color)
    ("acorn_minus_hnsw", "Extra structure of ACORN", "#548235"),  # green   (B=64 color)
]


def find_dataset(datasets: list[dict], name: str) -> dict:
    for ds in datasets:
        if str(ds.get("name", "")).lower() == name.lower():
            return ds
    raise RuntimeError(f"Dataset '{name}' not found in profiling JSON.")


def extract_no_hnsw_values(raw_sizes: dict[str, float]) -> dict[str, float]:
    baseline = float(raw_sizes["metadata_original_bytes"])
    lz4 = float(raw_sizes["metadata_lz4_bytes"])
    deflate = float(raw_sizes["metadata_deflate_bytes"])
    acorn = float(raw_sizes["acorn_graph_bytes"])
    hnsw = float(raw_sizes["hnsw_graph_bytes"])
    return {
        "baseline": baseline,
        "lz4": lz4,
        "deflate": deflate,
        "acorn_minus_hnsw": max(0.0, acorn - hnsw),
    }


def collect_group_data(data: dict, billion_rows: float) -> tuple[np.ndarray, np.ndarray]:
    datasets = data["datasets"]
    group_ratios: list[np.ndarray] = []
    group_size_text: list[list[str]] = []

    for ds_name in ["hnm", "laion"]:
        ds = find_dataset(datasets, ds_name)
        rows = float(ds["rows"])
        if rows <= 0.0:
            raise RuntimeError(f"Dataset '{ds_name}' has non-positive rows.")
        raw = {k: float(v) for k, v in ds["raw_sizes"].items()}
        vals_map = extract_no_hnsw_values(raw)
        scales = [
            ("million", 1.0, "MB"),
            ("billion", billion_rows / rows, "GB"),
        ]

        for _, scale, unit in scales:
            vals = np.array([vals_map[key] * scale for key, _, _ in CATEGORY_STYLES], dtype=np.float64)
            ratios = vals / vals[0]  # Baseline is 1
            group_ratios.append(ratios)

            if unit == "MB":
                denom = 1024.0 * 1024.0
            else:
                denom = 1024.0 * 1024.0 * 1024.0
            group_size_text.append([f"{(v / denom):.1f}" for v in vals])

    return np.array(group_ratios, dtype=np.float64), np.array(group_size_text, dtype=object)


def render_combined_plot(data: dict, output_path: Path, billion_rows: float) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    ratio_mat, size_text = collect_group_data(data, billion_rows)  # shape: (4 groups, 4 categories)

    # Group order: HNM million, HNM billion, LAION million, LAION billion.
    # Keep bars adjacent within each group.
    centers = np.arange(ratio_mat.shape[0], dtype=np.float64)
    bar_w = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * bar_w

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 5.0 / 2.54))

    ymax = float(ratio_mat.max())
    ypad = max(0.015, ymax * 0.022)

    for j, (_, legend_label, color) in enumerate(CATEGORY_STYLES):
        xs = centers + offsets[j]
        ys = ratio_mat[:, j]
        ax.bar(xs, ys, width=bar_w, color=color, edgecolor="black", linewidth=0.25, label=legend_label)
        for i, (x, y) in enumerate(zip(xs, ys)):
            label_y = y + ypad * (1.0 + 0.10 * j)
            ax.text(x, label_y, str(size_text[i, j]), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(
        "Normalized extra data size \n for filtering",
        fontsize=10,
        fontweight="bold",
        fontfamily="Arial",
    )
    ax.yaxis.set_label_coords(-0.052, 0.5)

    ax.set_xlabel("Benchmark", labelpad=-1, fontsize=10, fontweight="bold", fontfamily="Arial")
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [
            "H&M (Original)",
            "H&M (1B)",
            "LAION (Original)",
            "LAION (1B)",
        ]
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    left = centers[0] - 0.60
    right = centers[-1] + 0.60
    ax.set_xlim(left, right)
    ax.set_ylim(0.0, max(1.12, ymax * 1.14 + ypad))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=10,
        prop={"family": "Arial", "size": 10, "weight": "bold"},
        handlelength=1.2,
        columnspacing=0.85,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.17, top=0.87)
    ax.tick_params(axis="x", pad=1)

    # No title requested.
    fig.savefig(output_path, format="png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot section3 memory breakdown without HNSW-added metadata."
    )
    parser.add_argument("--input", type=Path, default=Path("profiling.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--billion-rows", type=float, default=1_000_000_000.0)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not data.get("datasets"):
        raise RuntimeError("No datasets found in JSON.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "hnm_laion_memory_breakdown_no_hnsw.png"
    render_combined_plot(data, out_path, args.billion_rows)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
