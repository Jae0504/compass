#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

METHOD_LABELS = {
    "post_filter_hnsw": "Post-filter HNSW",
    "in_search_filter_hnsw": "HNSW filter",
    "acorn": "ACORN",
    "compass_lz4": "COMPASS-LZ4",
    "compass_iaa": "COMPASS-IAA",
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
    "compass_iaa_1",
    "compass_iaa_2",
    "compass_iaa_4",
    "compass_iaa_8",
    "compass_iaa",
]

METHOD_COLORS = {
    "post_filter_hnsw": "#8c564b",
    "in_search_filter_hnsw": "#9467bd",
    "acorn": "#2ca02c",
    "compass_lz4": "#ff7f0e",
    "compass_iaa": "#1f77b4",
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
    "compass_iaa": "^",
    "compass_iaa_1": "v",
    "compass_iaa_2": ">",
    "compass_iaa_4": "<",
    "compass_iaa_8": "^",
}


def parse_selectivities(raw: str):
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(item)
    if not vals:
        raise ValueError("--selectivities must include at least one value")
    return vals


def is_finite_number(s: str) -> bool:
    if s is None or s == "":
        return False
    try:
        v = float(s)
    except ValueError:
        return False
    return math.isfinite(v)


def load_rows(input_csv: str):
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "selectivity_pct",
            "method",
            "recall",
            "qps",
            "status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            if row.get("status", "").strip() != "OK":
                continue
            if not is_finite_number(row.get("recall", "")):
                continue
            if not is_finite_number(row.get("qps", "")):
                continue
            rows.append(row)
    return rows


def build_series(rows, selectivities):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sel = row["selectivity_pct"].strip()
        if sel not in selectivities:
            continue
        method = row["method"].strip()
        recall = float(row["recall"])
        qps = float(row["qps"])
        if qps <= 0.0:
            continue
        grouped[sel][method].append((recall, qps))

    for sel in grouped:
        for method in grouped[sel]:
            grouped[sel][method].sort(key=lambda x: x[0])
    return grouped


def make_plot(series, selectivities, output_path):
    n = len(selectivities)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    legend_entries = {}
    for idx, sel in enumerate(selectivities):
        ax = axes[idx]
        sel_series = series.get(sel, {})

        for method in METHOD_ORDER:
            pts = sel_series.get(method, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=1.8,
                markersize=5,
            )
            legend_entries[method] = METHOD_LABELS.get(method, method)

        ax.set_title(f"Selectivity: {sel}%")
        if idx == 0:
            ax.set_ylabel("QPS")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    handles = []
    labels = []
    for method in METHOD_ORDER:
        label = legend_entries.get(method)
        if not label:
            continue
        handles.append(plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS.get(method),
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.8,
            markersize=5,
            label=label,
        ))
        labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

    # Show a single global x-axis label centered under all subplots.
    if hasattr(fig, "supxlabel"):
        fig.supxlabel("Recall")
    else:
        fig.text(0.5, 0.02, "Recall", ha="center")

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".png":
        fig.savefig(base + ".svg")


def main():
    parser = argparse.ArgumentParser(description="Plot QPS vs Recall by selectivity")
    parser.add_argument("--input-csv", required=True, help="Path to merged results CSV")
    parser.add_argument("--selectivities", default="1,10", help="Comma-separated selectivity list, e.g. 1,10")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    selectivities = parse_selectivities(args.selectivities)
    rows = load_rows(args.input_csv)
    series = build_series(rows, selectivities)
    make_plot(series, selectivities, args.output)


if __name__ == "__main__":
    main()
