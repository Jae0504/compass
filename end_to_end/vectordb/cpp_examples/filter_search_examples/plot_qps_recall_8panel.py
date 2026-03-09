#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

METHOD_LABELS = {
    "post_filter_hnsw": "Post-filter HNSW",
    "in_search_filter_hnsw": "HNSW filter",
    "acorn": "ACORN",
    "compass_lz4": "COMPASS-LZ4",
    "compass_lz4_grouping": "COMPASS-LZ4-group",
    "compass_iaa": "COMPASS-IAA",
    "compass_iaa_grouping": "COMPASS-IAA-group",
    "compass_iaa_1": "COMPASS-IAA_1",
    "compass_iaa_2": "COMPASS-IAA_2",
    "compass_iaa_4": "COMPASS-IAA_4",
    "compass_iaa_8": "COMPASS-IAA_8",
}

METHOD_ORDER = [
    "post_filter_hnsw",
    "in_search_filter_hnsw",
    "acorn",
    "compass_lz4",
    "compass_lz4_grouping",
    "compass_iaa_1",
    "compass_iaa_2",
    "compass_iaa_4",
    "compass_iaa_8",
    "compass_iaa",
    "compass_iaa_grouping",
]

METHOD_COLORS = {
    "post_filter_hnsw": "#8c564b",
    "in_search_filter_hnsw": "#9467bd",
    "acorn": "#2ca02c",
    "compass_lz4": "#ff7f0e",
    "compass_lz4_grouping": "#ffbb78",
    "compass_iaa": "#1f77b4",
    "compass_iaa_grouping": "#17becf",
    "compass_iaa_1": "#9ecae1",
    "compass_iaa_2": "#6baed6",
    "compass_iaa_4": "#3182bd",
    "compass_iaa_8": "#08519c",
}

METHOD_MARKERS = {
    "post_filter_hnsw": "D",
    "in_search_filter_hnsw": "s",
    "acorn": "P",
    "compass_lz4": "o",
    "compass_lz4_grouping": "X",
    "compass_iaa": "^",
    "compass_iaa_grouping": "h",
    "compass_iaa_1": "v",
    "compass_iaa_2": ">",
    "compass_iaa_4": "<",
    "compass_iaa_8": "^",
}

SELECTIVITY_CONFIG = [("1", "1%"), ("10", "10%")]
DATASET_LAYOUT = [
    [("sift1m", "(a) SIFT1M"), ("sift1b", "(b) SIFT1B")],
    [("laion", "(c) LAION"), ("hnm", "(d) H&M")],
]
FIG_W_CM = 33.0
FIG_H_CM = 24.0
# 254 DPI = 100 px/cm exactly, so 33x24 cm maps to exactly 3300x2400 pixels.
SAVE_DPI = 254


def is_finite_number(s: str) -> bool:
    if s is None or s == "":
        return False
    try:
        v = float(s)
    except ValueError:
        return False
    return math.isfinite(v)


def resolve_csv_path(out_dir: str, file_name: str):
    if not out_dir:
        return None
    candidates = [
        os.path.join(out_dir, file_name),
        os.path.join(out_dir, "filter_method_compare", file_name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_one_csv(csv_path: str, force_sel: str):
    rows = []
    if not csv_path:
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"method", "recall", "qps", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_path}: missing required columns {sorted(missing)}")
        for row in reader:
            if row.get("status", "").strip() != "OK":
                continue
            if not is_finite_number(row.get("recall", "")):
                continue
            if not is_finite_number(row.get("qps", "")):
                continue
            recall = float(row["recall"])
            qps = float(row["qps"])
            if qps <= 0.0:
                continue
            rows.append({
                "selectivity_pct": force_sel,
                "method": row["method"].strip(),
                "recall": recall,
                "qps": qps,
            })
    return rows


def load_dataset_rows(out_dir: str):
    rows = []
    p1 = resolve_csv_path(out_dir, "results_1pct.csv")
    p10 = resolve_csv_path(out_dir, "results_10pct.csv")
    rows.extend(load_one_csv(p1, "1"))
    rows.extend(load_one_csv(p10, "10"))
    return rows


def build_series(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sel = row["selectivity_pct"]
        method = row["method"]
        grouped[sel][method].append((row["recall"], row["qps"]))
    for sel in grouped:
        for method in grouped[sel]:
            grouped[sel][method].sort(key=lambda x: x[0])
    return grouped


def apply_axis_style(ax):
    ax.set_facecolor("#ffffff")
    ax.patch.set_alpha(1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.6, 1.01)
    # Keep y-axis on the left only.
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", which="both", left=True, right=False, labelright=False)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.9, alpha=0.35, color="#c7c7c7")
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.6, alpha=0.22, color="#dddddd")


def plot_panel(ax, series_by_sel, sel_key, sel_label):
    sel_series = series_by_sel.get(sel_key, {})
    has_data = False
    for method in METHOD_ORDER:
        pts = sel_series.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            color=METHOD_COLORS.get(method, "#444444"),
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            markersize=6,
            label=METHOD_LABELS.get(method, method),
        )
        has_data = True

    apply_axis_style(ax)
    ax.set_title(f"selectivity: {sel_label}", fontweight="bold")
    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


def add_pair_labels(fig, axes):
    top_left = axes[0][0].get_position()
    bot_left = axes[1][0].get_position()
    bot_right = axes[1][3].get_position()

    top_pair1_x = (axes[0][0].get_position().x0 + axes[0][1].get_position().x1) * 0.5
    top_pair2_x = (axes[0][2].get_position().x0 + axes[0][3].get_position().x1) * 0.5
    bot_pair1_x = (axes[1][0].get_position().x0 + axes[1][1].get_position().x1) * 0.5
    bot_pair2_x = (axes[1][2].get_position().x0 + axes[1][3].get_position().x1) * 0.5

    # Pair-level Recall and dataset captions (a/b/c/d).
    row_gap_center_y = (top_left.y0 + bot_left.y1) * 0.5
    top_recall_y = row_gap_center_y + 0.015
    top_caption_y = row_gap_center_y - 0.015

    bottom_caption_y = max(0.02, bot_right.y0 - 0.055)
    bottom_recall_y = bottom_caption_y + 0.022

    fig.text(top_pair1_x, top_recall_y, "Recall", ha="center", va="center", fontweight="bold")
    fig.text(top_pair2_x, top_recall_y, "Recall", ha="center", va="center", fontweight="bold")
    fig.text(bot_pair1_x, bottom_recall_y, "Recall", ha="center", va="center", fontweight="bold")
    fig.text(bot_pair2_x, bottom_recall_y, "Recall", ha="center", va="center", fontweight="bold")

    fig.text(top_pair1_x, top_caption_y, "(a) SIFT1M", ha="center", va="center", fontweight="bold")
    fig.text(top_pair2_x, top_caption_y, "(b) SIFT1B", ha="center", va="center", fontweight="bold")
    fig.text(bot_pair1_x, bottom_caption_y, "(c) LAION", ha="center", va="center", fontweight="bold")
    fig.text(bot_pair2_x, bottom_caption_y, "(d) H&M", ha="center", va="center", fontweight="bold")


def build_legend(fig, all_series):
    methods_with_data = set()
    for series_by_sel in all_series.values():
        for sel_map in series_by_sel.values():
            methods_with_data.update(sel_map.keys())

    legend_methods = [m for m in METHOD_ORDER if m in methods_with_data]
    if not legend_methods:
        legend_methods = METHOD_ORDER

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS.get(m, "#444444"),
            marker=METHOD_MARKERS.get(m, "o"),
            linewidth=2.0,
            markersize=6,
            label=METHOD_LABELS.get(m, m),
        )
        for m in legend_methods
    ]
    labels = [METHOD_LABELS.get(m, m) for m in legend_methods]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(1, min(len(labels), 6)),
        frameon=False,
    )


def plot_figure(dataset_series_map, output_path):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54),
        sharey="row",
    )
    fig.patch.set_facecolor("#ffffff")

    for row_idx, row_cfg in enumerate(DATASET_LAYOUT):
        for pair_idx, (dataset_name, _caption) in enumerate(row_cfg):
            base_col = pair_idx * 2
            series = dataset_series_map.get(dataset_name, {})
            for sel_offset, (sel_key, sel_label) in enumerate(SELECTIVITY_CONFIG):
                ax = axes[row_idx][base_col + sel_offset]
                plot_panel(ax, series, sel_key, sel_label)
                # Only keep y tick labels on the left-most subplot per row.
                if base_col + sel_offset != 0:
                    ax.tick_params(axis="y", labelleft=False)

    # One y-axis label per row.
    axes[0][0].set_ylabel("Queries Per Second (QPS)", fontweight="bold")
    axes[1][0].set_ylabel("Queries Per Second (QPS)", fontweight="bold")
    for ax in axes.flat:
        ax.set_xlabel("")

    # Keep layout as large as possible while reserving top legend and bottom captions.
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.105, wspace=0.12, hspace=0.34)

    build_legend(fig, dataset_series_map)
    add_pair_labels(fig, axes)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, facecolor="#ffffff", edgecolor="#ffffff")

    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".png":
        fig.savefig(base + ".svg")


def main():
    parser = argparse.ArgumentParser(description="Plot 8 QPS-recall panels across 4 datasets (1%/10%).")
    parser.add_argument("--sift1m-out", default="", help="SIFT1M out directory")
    parser.add_argument("--sift1b-out", default="", help="SIFT1B out directory")
    parser.add_argument("--laion-out", default="", help="LAION out directory")
    parser.add_argument("--hnm-out", default="", help="H&M out directory")
    parser.add_argument(
        "--output",
        default="qps_recall_8panel.png",
        help="Output figure path (.png recommended)",
    )
    args = parser.parse_args()

    dataset_outs = {
        "sift1m": args.sift1m_out.strip(),
        "sift1b": args.sift1b_out.strip(),
        "laion": args.laion_out.strip(),
        "hnm": args.hnm_out.strip(),
    }

    dataset_series_map = {}
    for dataset_name, out_dir in dataset_outs.items():
        rows = load_dataset_rows(out_dir)
        dataset_series_map[dataset_name] = build_series(rows)

    plot_figure(dataset_series_map, args.output)


if __name__ == "__main__":
    main()
