#!/usr/bin/env python3
"""
Roofline plot — 3 subplots (one per dimension: 128D, 512D, 2048D).

Each subplot shows, for a single user-selected chunk size:
  - IAA latency line
  - Distance calculation latency line
  - Distance calculation + LZ4 latency line  (element-wise sum at each N)
  - Shaded fill between the two distance lines  (= LZ4 overhead region)

Usage:
    python3 plot_roofline.py --chunk-size <size> [--out-dir <path>]

<size> can be bytes (65536), KB (64KB), or MB (1MB).

Reads:
    <out-dir>/results_dist.csv
    <out-dir>/results_iaa.csv
    <out-dir>/results_lz4.csv

Outputs:
    <out-dir>/roofline_<chunk_size>.png
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

DIMS       = [128, 512, 2048]
DIM_LABELS = {128: "128-D (SIFT)", 512: "512-D (LAION)", 2048: "2048-D (HNM)"}
N_TICKS    = [1, 2, 4, 8, 16, 32]

COLOR_IAA      = "tab:red"
COLOR_DIST     = "tab:blue"
COLOR_DIST_LZ4 = "tab:orange"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_chunk_size(s: str) -> int:
    s = s.strip()
    if s.upper().endswith("MB"):
        return int(s[:-2]) * 1024 * 1024
    if s.upper().endswith("KB"):
        return int(s[:-2]) * 1024
    return int(s)


def human_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)}MB"
    return f"{b // 1024}KB"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",    default=".",
                        help="Directory containing CSVs and for output PNG")
    parser.add_argument("--chunk-size", required=True,
                        help="Chunk size to plot, e.g. 65536, 64KB, or 1MB")
    args = parser.parse_args()

    out_dir    = args.out_dir
    chunk_size = parse_chunk_size(args.chunk_size)

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    for name in ["results_dist.csv", "results_iaa.csv", "results_lz4.csv"]:
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    df_dist = pd.read_csv(os.path.join(out_dir, "results_dist.csv"))
    df_iaa  = pd.read_csv(os.path.join(out_dir, "results_iaa.csv"))
    df_lz4  = pd.read_csv(os.path.join(out_dir, "results_lz4.csv"))

    # Validate chunk size exists in IAA results
    if chunk_size not in df_iaa["chunk_size_bytes"].values:
        avail = sorted(df_iaa["chunk_size_bytes"].unique())
        print(f"ERROR: chunk_size={chunk_size} not in results_iaa.csv. "
              f"Available: {[human_bytes(x) for x in avail]}", file=sys.stderr)
        sys.exit(1)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Roofline — chunk size: {human_bytes(chunk_size)}", fontsize=14)

    for ax, dim in zip(axes, DIMS):
        df_d = df_dist[df_dist["dim"] == dim].sort_values("n_nodes")
        if df_d.empty:
            ax.set_title(DIM_LABELS[dim] + "\n(no data)")
            continue

        # Filter IAA and LZ4 to this dim + chunk_size
        df_iaa_cs = df_iaa[
            (df_iaa["dim"] == dim) & (df_iaa["chunk_size_bytes"] == chunk_size)
        ].sort_values("n_jobs")
        df_lz4_cs = df_lz4[
            (df_lz4["dim"] == dim) & (df_lz4["chunk_size_bytes"] == chunk_size)
        ].sort_values("n_jobs")

        x_dist = df_d["n_nodes"].to_numpy()
        y_dist = df_d["latency_ns_median"].to_numpy()

        x_lz4 = df_lz4_cs["n_jobs"].to_numpy()
        y_lz4 = df_lz4_cs["latency_ns_median"].to_numpy()

        # Build dist + LZ4 line on the shared N values
        n_common   = sorted(set(x_dist.tolist()) & set(x_lz4.tolist()))
        x_c        = np.array(n_common)
        dist_map   = dict(zip(x_dist.tolist(), y_dist.tolist()))
        lz4_map    = dict(zip(x_lz4.tolist(),  y_lz4.tolist()))
        y_dist_c   = np.array([dist_map[n] for n in n_common])
        y_lz4_c    = np.array([lz4_map[n]  for n in n_common])
        y_dist_lz4 = y_dist_c + y_lz4_c

        # IAA line (per-dim data for this chunk_size)
        x_iaa = df_iaa_cs["n_jobs"].to_numpy()
        y_iaa = df_iaa_cs["latency_ns_median"].to_numpy()
        ax.plot(x_iaa, y_iaa,
                color=COLOR_IAA, linewidth=2, marker="^", markersize=6,
                label=f"IAA ({human_bytes(chunk_size)})", zorder=4)

        # Distance calculation line
        ax.plot(x_c, y_dist_c,
                color=COLOR_DIST, linewidth=2, marker="o", markersize=6,
                label="Dist calc", zorder=4)

        # Distance calculation + LZ4 line
        ax.plot(x_c, y_dist_lz4,
                color=COLOR_DIST_LZ4, linewidth=2, marker="s", markersize=6,
                linestyle="--", label="Dist calc + LZ4", zorder=4)

        # Shaded region = LZ4 overhead on top of distance calculation
        ax.fill_between(x_c, y_dist_c, y_dist_lz4,
                        alpha=0.25, color=COLOR_DIST_LZ4, zorder=3)

        # Axes
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(N_TICKS)
        ax.set_xticklabels([str(n) for n in N_TICKS])
        ax.set_xlabel("Number of nodes / jobs", fontsize=11)
        ax.set_ylabel("Latency (ns)", fontsize=11)
        ax.set_title(DIM_LABELS[dim], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    out_name = f"roofline_{human_bytes(chunk_size)}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
