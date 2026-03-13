#!/usr/bin/env python3
"""
Roofline plot — 3 subplots (one per chunk size: 4KB, 64KB, 1MB).

Each subplot shows, for a single user-selected dimension:
  - IAA total latency (submit + wait) for each engine count (1, 2, 4, 8)
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


# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZES    = [4 * 1024, 64 * 1024, 1 * 1024 * 1024]
N_TICKS        = [1, 2, 4, 8, 16, 32, 64]
ENGINE_COUNTS  = [1, 2, 4, 8]

DIM_LABELS     = {128: "128-D (SIFT)", 512: "512-D (LAION)", 2048: "2048-D (HNM)"}

# IAA lines: light → dark red as engine count increases
IAA_COLORS     = {1: "#f4a582", 2: "#d6604d", 4: "#b2182b", 8: "#67001f"}
IAA_MARKERS    = {1: "^", 2: "v", 4: "D", 8: "s"}

COLOR_DIST     = "tab:blue"
COLOR_LZ4      = "tab:orange"


# ── Helpers ───────────────────────────────────────────────────────────────────

def human_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)}MB"
    return f"{b // 1024}KB"


# ── Main ──────────────────────────────────────────────────────────────────────

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

    # ── Load decomp CSV ───────────────────────────────────────────────────────
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

    # Engine counts actually present in data
    avail_engines = sorted(df_dim["engine_count"].unique())
    engines_to_plot = [e for e in ENGINE_COUNTS if e in avail_engines]

    # ── Load distance CSV (optional) ──────────────────────────────────────────
    df_dist = None
    if dist_csv is not None:
        if not os.path.exists(dist_csv):
            print(f"ERROR: dist-csv not found: {dist_csv}", file=sys.stderr)
            sys.exit(1)
        df_dist = pd.read_csv(dist_csv)
        df_dist = df_dist[df_dist["dim"] == dim].sort_values("n_nodes")

    # ── Build figure ──────────────────────────────────────────────────────────
    dim_label = DIM_LABELS.get(dim, f"{dim}-D")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Roofline — {dim_label}", fontsize=14)

    for ax, chunk_size in zip(axes, CHUNK_SIZES):

        # ── IAA lines: one per engine count ───────────────────────────────────
        for eng in engines_to_plot:
            df_cs = df_dim[
                (df_dim["engine_count"] == eng) &
                (df_dim["chunk_size_bytes"] == chunk_size)
            ].sort_values("n_nodes")

            if df_cs.empty:
                continue

            x      = df_cs["n_nodes"].to_numpy()
            y_wait = (df_cs["iaa_submit_ns_median"] + df_cs["iaa_wait_ns_median"]).to_numpy()

            ax.plot(x, y_wait,
                    color=IAA_COLORS[eng], linewidth=2,
                    marker=IAA_MARKERS[eng], markersize=6,
                    label=f"IAA e{eng}", zorder=4)

        # ── LZ4+dist line (engine-count independent — use first available engine) ──
        df_lz4 = df_dim[
            (df_dim["engine_count"] == avail_engines[0]) &
            (df_dim["chunk_size_bytes"] == chunk_size)
        ].sort_values("n_nodes")

        if not df_lz4.empty and df_dist is not None and not df_dist.empty:
            # Align on n_nodes then sum
            merged = pd.merge(
                df_lz4[["n_nodes", "lz4_decompress_ns_median"]],
                df_dist[["n_nodes", "latency_ns_median"]],
                on="n_nodes", how="inner"
            )
            if not merged.empty:
                ax.plot(merged["n_nodes"].to_numpy(),
                        (merged["lz4_decompress_ns_median"] + merged["latency_ns_median"]).to_numpy(),
                        color=COLOR_LZ4, linewidth=2, marker="o", markersize=6,
                        label="LZ4 + Dist calc", zorder=4)
        elif not df_lz4.empty:
            ax.plot(df_lz4["n_nodes"].to_numpy(),
                    df_lz4["lz4_decompress_ns_median"].to_numpy(),
                    color=COLOR_LZ4, linewidth=2, marker="o", markersize=6,
                    label="LZ4", zorder=4)

        # ── Distance line (same across all chunk sizes) ────────────────────────
        if df_dist is not None and not df_dist.empty:
            ax.plot(df_dist["n_nodes"].to_numpy(),
                    df_dist["latency_ns_median"].to_numpy(),
                    color=COLOR_DIST, linewidth=2, marker="P", markersize=6,
                    label="Dist calc", zorder=4)

        if not df_dim[df_dim["chunk_size_bytes"] == chunk_size].empty:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(N_TICKS)
            ax.set_xticklabels([str(n) for n in N_TICKS])
            ax.set_xlabel("Number of nodes / jobs", fontsize=11)
            ax.set_ylabel("Latency (ns)", fontsize=11)
            ax.set_title(human_bytes(chunk_size), fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        else:
            ax.set_title(f"{human_bytes(chunk_size)}\n(no data)")

    fig.tight_layout()
    out_name = f"roofline_dim{dim}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
