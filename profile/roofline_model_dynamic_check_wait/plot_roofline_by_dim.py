#!/usr/bin/env python3
"""
Roofline plot — 3 subplots (one per chunk size: 64KB, 256KB, 1MB).

Each subplot shows, for a single user-selected dimension:
  - IAA wait latency (submit excluded) for each engine count (1, 2, 4, 8)
  - Distance calculation latency  (optional; same line in every subplot)
  - LZ4 decompression latency     (engine-count independent)

Usage:
    python3 plot_roofline_by_dim.py --dim <dim> [--out-dir <path>]
                                    [--dist-csv <path>]

<dim> is the vector dimension: 128, 512, or 2048.

Reads:
    <out-dir>/decomp_latency.csv
    <dist-csv>  (optional) results_dist.csv from the distance profiler

Outputs:
    <out-dir>/roofline_dim<dim>.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Font / style ───────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "Arial",
    "font.size":        12,
    "font.weight":      "bold",
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.labelweight": "bold",
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
})

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZES   = [64 * 1024, 256 * 1024, 1 * 1024 * 1024]
N_TICKS       = [1, 2, 4, 8, 16, 32]
ENGINE_COUNTS = [1, 2, 4, 8]

DIM_LABELS    = {128: "128-D (SIFT)", 512: "512-D (LAION)", 2048: "2048-D (HNM)"}

# IAA lines: light → dark blue as engine count increases
IAA_COLORS  = {1: "#9ecae1", 2: "#4292c6", 4: "#2171b5", 8: "#084594"}
IAA_MARKERS = {1: "^",       2: "v",       4: "D",       8: "s"}

COLOR_DIST = "tab:green"
COLOR_LZ4  = "tab:red"

# Shaded region colours
COLOR_ABOVE_LZ4  = "#ffcccc"  # light red   — above LZ4 line
COLOR_BETWEEN    = "#e8e8e8"  # light grey  — between LZ4 and dist-calc
COLOR_BELOW_DIST = "#ccffcc"  # light green — below dist-calc line
REGION_ALPHA     = 0.55

# Figure size in cm → inches
FIG_W_CM, FIG_H_CM = 16, 7
FIG_W_IN = FIG_W_CM / 2.54
FIG_H_IN = FIG_H_CM / 2.54


# ── Helpers ────────────────────────────────────────────────────────────────────
def human_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)} MB"
    return f"{b // 1024} KB"


X_FILL_MIN = 1
X_FILL_MAX = 32


def _fill_regions(ax, lz4_x, lz4_y, dist_x, dist_y):
    """Draw the three background shading regions, clipped to [X_FILL_MIN, X_FILL_MAX]."""
    BIG  = 1e15
    TINY = 1e-3

    if lz4_y is not None and dist_y is not None:
        merged = pd.merge(
            pd.DataFrame({"n": lz4_x,  "lz4":  lz4_y}),
            pd.DataFrame({"n": dist_x, "dist": dist_y}),
            on="n", how="inner"
        )
        mask = (merged["n"] >= X_FILL_MIN) & (merged["n"] <= X_FILL_MAX)
        merged = merged[mask]
        if not merged.empty:
            rx   = merged["n"].to_numpy()
            rlz4 = merged["lz4"].to_numpy()
            rdst = merged["dist"].to_numpy()
            ax.fill_between(rx, rlz4, BIG,  color=COLOR_ABOVE_LZ4,  alpha=REGION_ALPHA, zorder=0, linewidth=0)
            ax.fill_between(rx, rdst, rlz4, color=COLOR_BETWEEN,     alpha=REGION_ALPHA, zorder=0, linewidth=0)
            ax.fill_between(rx, TINY, rdst, color=COLOR_BELOW_DIST,  alpha=REGION_ALPHA, zorder=0, linewidth=0)
            return
    # Fallback: only one boundary available
    if lz4_y is not None:
        mask = (lz4_x >= X_FILL_MIN) & (lz4_x <= X_FILL_MAX)
        ax.fill_between(lz4_x[mask], lz4_y[mask], BIG,  color=COLOR_ABOVE_LZ4,  alpha=REGION_ALPHA, zorder=0, linewidth=0)
        ax.fill_between(lz4_x[mask], TINY, lz4_y[mask], color=COLOR_BELOW_DIST, alpha=REGION_ALPHA, zorder=0, linewidth=0)
    elif dist_y is not None:
        mask = (dist_x >= X_FILL_MIN) & (dist_x <= X_FILL_MAX)
        ax.fill_between(dist_x[mask], dist_y[mask], BIG,  color=COLOR_ABOVE_LZ4,  alpha=REGION_ALPHA, zorder=0, linewidth=0)
        ax.fill_between(dist_x[mask], TINY, dist_y[mask], color=COLOR_BELOW_DIST, alpha=REGION_ALPHA, zorder=0, linewidth=0)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  default=".",
                        help="Directory containing decomp_latency.csv and for output PNG")
    parser.add_argument("--dim",      type=int, required=True,
                        help="Vector dimension to plot: 128, 512, or 2048")
    parser.add_argument("--dist-csv", default=None,
                        help="Path to results_dist.csv from the distance profiler (optional)")
    args = parser.parse_args()

    out_dir  = args.out_dir
    dim      = args.dim
    dist_csv = args.dist_csv

    # ── Load decomp CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "decomp_latency.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if dim not in df["dim"].values:
        avail = sorted(df["dim"].unique())
        print(f"ERROR: dim={dim} not in CSV. Available: {avail}", file=sys.stderr)
        sys.exit(1)

    df_dim = df[df["dim"] == dim]
    avail_engines   = sorted(df_dim["engine_count"].unique())
    engines_to_plot = [e for e in ENGINE_COUNTS if e in avail_engines]

    # ── Load distance CSV (optional) ───────────────────────────────────────────
    df_dist = None
    if dist_csv is not None:
        if not os.path.exists(dist_csv):
            print(f"ERROR: dist-csv not found: {dist_csv}", file=sys.stderr)
            sys.exit(1)
        df_dist = pd.read_csv(dist_csv)
        df_dist = df_dist[df_dist["dim"] == dim].sort_values("n_nodes")

    Y_MIN = 1e2

    # ── Build figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))
    # No suptitle

    legend_handles = []
    legend_labels  = []

    for ax_idx, (ax, chunk_size) in enumerate(zip(axes, CHUNK_SIZES)):
        is_leftmost = (ax_idx == 0)

        # Compute LZ4 (and optionally LZ4+Dist) line data for this chunk size
        df_lz4_cs = df_dim[
            (df_dim["engine_count"] == avail_engines[0]) &
            (df_dim["chunk_size_bytes"] == chunk_size)
        ].sort_values("n_nodes")

        lz4_x = lz4_y = None
        lz4_label = "LZ4"

        if not df_lz4_cs.empty and df_dist is not None and not df_dist.empty:
            merged = pd.merge(
                df_lz4_cs[["n_nodes", "lz4_decompress_ns_median"]],
                df_dist[["n_nodes", "latency_ns_median"]],
                on="n_nodes", how="inner"
            )
            if not merged.empty:
                lz4_x     = merged["n_nodes"].to_numpy()
                lz4_y     = (merged["lz4_decompress_ns_median"]
                             + merged["latency_ns_median"]).to_numpy()
                lz4_label = "LZ4 + Dist calc"
        elif not df_lz4_cs.empty:
            lz4_x = df_lz4_cs["n_nodes"].to_numpy()
            lz4_y = df_lz4_cs["lz4_decompress_ns_median"].to_numpy()

        dist_x = dist_y = None
        if df_dist is not None and not df_dist.empty:
            dist_x = df_dist["n_nodes"].to_numpy()
            dist_y = df_dist["latency_ns_median"].to_numpy()

        # ── Per-subplot y-axis: min/max across all data for this chunk size ──────
        df_cs_all = df_dim[df_dim["chunk_size_bytes"] == chunk_size]
        all_y_vals = []
        if not df_cs_all.empty:
            all_y_vals += df_cs_all["iaa_wait_ns_median"].dropna().tolist()
            all_y_vals += df_cs_all["lz4_decompress_ns_median"].dropna().tolist()
        if lz4_y is not None:
            all_y_vals += lz4_y.tolist()
        if dist_y is not None:
            all_y_vals += dist_y.tolist()
        all_y_vals = [v for v in all_y_vals if v > 0]
        y_min = min(all_y_vals) if all_y_vals else Y_MIN
        y_max = max(all_y_vals) if all_y_vals else Y_MIN * 10

        # ── Shaded background regions (before lines so lines render on top) ────
        _fill_regions(ax, lz4_x, lz4_y, dist_x, dist_y)

        # ── IAA wait lines ─────────────────────────────────────────────────────
        for eng in engines_to_plot:
            df_cs = df_dim[
                (df_dim["engine_count"] == eng) &
                (df_dim["chunk_size_bytes"] == chunk_size)
            ].sort_values("n_nodes")

            if df_cs.empty:
                continue

            x      = df_cs["n_nodes"].to_numpy()
            y_wait = df_cs["iaa_wait_ns_median"].to_numpy()

            h, = ax.plot(x, y_wait,
                         color=IAA_COLORS[eng], linewidth=1.5,
                         marker=IAA_MARKERS[eng], markersize=5,
                         label=f"IAA wait e{eng}", zorder=4)
            if is_leftmost:
                legend_handles.append(h)
                legend_labels.append(f"IAA wait e{eng}")

        # ── LZ4 line ───────────────────────────────────────────────────────────
        if lz4_x is not None and lz4_y is not None:
            h, = ax.plot(lz4_x, lz4_y,
                         color=COLOR_LZ4, linewidth=1.5, marker="o", markersize=5,
                         label=lz4_label, zorder=5)
            if is_leftmost:
                legend_handles.append(h)
                legend_labels.append(lz4_label)

        # ── Distance-calc line ─────────────────────────────────────────────────
        if dist_x is not None and dist_y is not None:
            h, = ax.plot(dist_x, dist_y,
                         color=COLOR_DIST, linewidth=1.5, marker="P", markersize=5,
                         label="Dist calc", zorder=5)
            if is_leftmost:
                legend_handles.append(h)
                legend_labels.append("Dist calc")

        # ── Axis formatting ────────────────────────────────────────────────────
        has_data = not df_dim[df_dim["chunk_size_bytes"] == chunk_size].empty
        if has_data:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(N_TICKS)
            ax.set_xticklabels([str(n) for n in N_TICKS])
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("")
            if is_leftmost:
                ax.set_ylabel("Latency (ns)")
            else:
                ax.set_ylabel("")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")

        # ── Chunk-size label: top-left, semi-transparent background ────────────
        ax.text(0.03, 0.97, human_bytes(chunk_size),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", alpha=0.70,
                          edgecolor="none"))

        # Remove per-subplot legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # ── Single shared legend — top centre of figure ────────────────────────────
    fig.legend(legend_handles, legend_labels,
               loc="upper center",
               ncol=len(legend_handles),
               framealpha=0.85,
               edgecolor="none",
               bbox_to_anchor=(0.5, 1.04))

    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.12, wspace=0.05)
    fig.text(0.5, 0.01, "Number of nodes / jobs",
             ha="center", va="bottom",
             fontsize=12, fontweight="bold")
    out_name = f"roofline_dim{dim}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
