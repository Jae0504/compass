#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

N_BINS = [1, 2, 4, 8, 16, 32, 64]
DIMS = [128, 512, 2048]
DIM_LABEL = {
    128: "128D (sift1m)",
    512: "512D (laion)",
    2048: "2048D (hnm)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot dynamic wait/check roofline + zone views")
    p.add_argument(
        "--expansion-glob",
        default="./out/traversal/*.csv",
        help="Glob for expansion metrics CSV files",
    )
    p.add_argument(
        "--decomp-csv",
        default="./out/decomp_latency.csv",
        help="Decompression benchmark CSV",
    )
    p.add_argument(
        "--out-dir",
        default="./out",
        help="Output directory",
    )
    return p.parse_args()


def fmt_chunk(c: int) -> str:
    if c >= 1024 * 1024:
        return f"{c // (1024 * 1024)}MB"
    return f"{c // 1024}KB"


def bin_pow2_ceil(v: int) -> int:
    if v <= 1:
        return 1
    b = 1
    while b < v and b < N_BINS[-1]:
        b <<= 1
    return min(b, N_BINS[-1])


def read_expansion_dist(glob_pattern: str) -> Dict[int, Dict[int, float]]:
    per_dim_bin_vals: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No expansion CSV files found: {glob_pattern}")

    for path in files:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dim = int(row["dim"])
                    n_tb1 = int(row["n_tb1_nodes"])
                    dist_ns = int(row["distance_ns"])
                except Exception:
                    continue
                if dim not in DIMS:
                    continue
                b = bin_pow2_ceil(n_tb1)
                per_dim_bin_vals[dim][b].append(float(dist_ns))

    out: Dict[int, Dict[int, float]] = {}
    for dim in DIMS:
        bucket = per_dim_bin_vals.get(dim, {})
        med_map: Dict[int, float] = {}
        for n in N_BINS:
            vals = bucket.get(n, [])
            if vals:
                med_map[n] = float(median(vals))

        if not med_map:
            raise RuntimeError(f"No distance points for dim={dim}")

        overall = float(median([v for vals in bucket.values() for v in vals]))
        for n in N_BINS:
            if n in med_map:
                continue
            smaller = [k for k in med_map.keys() if k < n]
            larger = [k for k in med_map.keys() if k > n]
            if smaller and larger:
                lo = max(smaller)
                hi = min(larger)
                med_map[n] = med_map[lo] + (med_map[hi] - med_map[lo]) * ((n - lo) / (hi - lo))
            elif smaller:
                med_map[n] = med_map[max(smaller)]
            elif larger:
                med_map[n] = med_map[min(larger)]
            else:
                med_map[n] = overall

        out[dim] = med_map
    return out


def read_decomp_csv(
    path: str,
) -> Tuple[
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int, int], float],
    List[int],
    List[int],
]:
    lz4_vals: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    iaa_vals: Dict[Tuple[int, int, int, int], List[float]] = defaultdict(list)
    engines = set()
    chunks = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim = int(row["dim"])
            engine = int(row["engine_count"])
            chunk = int(row["chunk_size_bytes"])
            n = int(row["n_nodes"])
            wait_ns = float(row["iaa_wait_ns_median"])
            lz4_ns = float(row["lz4_decompress_ns_median"])
            if dim not in DIMS:
                continue
            engines.add(engine)
            chunks.add(chunk)
            lz4_vals[(dim, chunk, n)].append(lz4_ns)
            iaa_vals[(dim, engine, chunk, n)].append(wait_ns)

    if not iaa_vals:
        raise RuntimeError(f"No rows in decomp csv: {path}")

    lz4_med: Dict[Tuple[int, int, int], float] = {
        k: float(median(v)) for k, v in lz4_vals.items()
    }
    iaa_med: Dict[Tuple[int, int, int, int], float] = {
        k: float(median(v)) for k, v in iaa_vals.items()
    }
    return lz4_med, iaa_med, sorted(engines), sorted(chunks)


def pick_base_engine(engines: List[int], iaa_med: Dict[Tuple[int, int, int, int], float], dim: int) -> int:
    if 8 in engines:
        has_dim = any(k[0] == dim and k[1] == 8 for k in iaa_med.keys())
        if has_dim:
            return 8
    counts = defaultdict(int)
    for d, e, _, _ in iaa_med.keys():
        if d == dim:
            counts[e] += 1
    if not counts:
        return engines[-1]
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def high_quantile(vals: List[float], q: float = 95.0) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.array(vals, dtype=np.float64), q))


def zone_code(wait_ns: float, dist_ns: float, lz4_ns: float) -> int:
    if wait_ns <= dist_ns:
        return 0
    if wait_ns <= dist_ns + lz4_ns:
        return 1
    return 2


def build_zone_matrix(
    dim: int,
    engine: int,
    chunks: List[int],
    dist_map: Dict[int, Dict[int, float]],
    lz4_map: Dict[Tuple[int, int, int], float],
    iaa_map: Dict[Tuple[int, int, int, int], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zones = np.full((len(chunks), len(N_BINS)), np.nan)
    wait_mat = np.full((len(chunks), len(N_BINS)), np.nan)
    dist_mat = np.full((len(chunks), len(N_BINS)), np.nan)
    dist_lz4_mat = np.full((len(chunks), len(N_BINS)), np.nan)

    for yi, chunk in enumerate(chunks):
        for xi, n in enumerate(N_BINS):
            lz4_key = (dim, chunk, n)
            iaa_key = (dim, engine, chunk, n)
            if lz4_key not in lz4_map or iaa_key not in iaa_map:
                continue
            dist_ns = dist_map[dim].get(bin_pow2_ceil(n))
            if dist_ns is None:
                continue
            lz4_ns = lz4_map[lz4_key]
            wait_ns = iaa_map[iaa_key]
            zones[yi, xi] = zone_code(wait_ns, dist_ns, lz4_ns)
            wait_mat[yi, xi] = wait_ns
            dist_mat[yi, xi] = dist_ns
            dist_lz4_mat[yi, xi] = dist_ns + lz4_ns

    return zones, wait_mat, dist_mat, dist_lz4_mat


def plot_zone_map_multi_engine(
    dim: int,
    engines: List[int],
    chunks: List[int],
    dist_map: Dict[int, Dict[int, float]],
    lz4_map: Dict[Tuple[int, int, int], float],
    iaa_map: Dict[Tuple[int, int, int, int], float],
    out_dir: Path,
) -> None:
    if not chunks or not engines:
        return

    ncols = 2 if len(engines) > 1 else 1
    nrows = int(math.ceil(len(engines) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.5 * nrows), constrained_layout=True)
    axes_list = np.atleast_1d(axes).ravel()

    zone_cmap = ListedColormap(["#67c587", "#f5c04a", "#e56b6f"])
    zone_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], zone_cmap.N)

    for ax in axes_list:
        ax.set_visible(False)

    for idx, eng in enumerate(engines):
        ax = axes_list[idx]
        ax.set_visible(True)
        zones, _, _, _ = build_zone_matrix(dim, eng, chunks, dist_map, lz4_map, iaa_map)

        im = ax.imshow(zones, origin="lower", aspect="auto", cmap=zone_cmap, norm=zone_norm)
        ax.set_title(f"engine {eng}")
        ax.set_xlabel("FID jobs in-flight (n_nodes)")
        ax.set_ylabel("FID chunk size")
        ax.set_xticks(np.arange(len(N_BINS)))
        ax.set_xticklabels([str(n) for n in N_BINS])
        ax.set_yticks(np.arange(len(chunks)))
        ax.set_yticklabels([fmt_chunk(c) for c in chunks])

        for yi in range(len(chunks)):
            for xi in range(len(N_BINS)):
                if np.isnan(zones[yi, xi]):
                    continue
                z = int(zones[yi, xi])
                txt = ["S", "T", "O"][z]
                ax.text(xi, yi, txt, ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=axes_list.tolist(), ticks=[0, 1, 2], shrink=0.92)
    cbar.ax.set_yticklabels(["safe", "threshold", "out"])
    fig.suptitle(
        f"Zone Map by Engine - {DIM_LABEL[dim]}\n"
        "safe: wait<=dist, threshold: dist<wait<=dist+lz4, out: wait>dist+lz4",
        fontsize=12,
    )

    out_png = out_dir / f"zone_map_{dim}d.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")


def pick_chunk_triplet(chunks: List[int]) -> List[int]:
    if not chunks:
        return []
    uniq = sorted(set(chunks))
    if len(uniq) <= 3:
        return uniq
    return [uniq[0], uniq[len(uniq) // 2], uniq[-1]]


def pick_engine_quartet(engines: List[int]) -> List[int]:
    preferred = [1, 2, 4, 8]
    chosen = [e for e in preferred if e in engines]
    for e in engines:
        if len(chosen) >= 4:
            break
        if e not in chosen:
            chosen.append(e)
    return chosen[:4]


def plot_zone_curves(
    dim: int,
    engine: int,
    chunks: List[int],
    dist_map: Dict[int, Dict[int, float]],
    lz4_map: Dict[Tuple[int, int, int], float],
    iaa_map: Dict[Tuple[int, int, int, int], float],
    out_dir: Path,
) -> None:
    selected = pick_chunk_triplet(chunks)
    if not selected:
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(5.0 * len(selected), 4.2), constrained_layout=True)
    axes_list = np.atleast_1d(axes).ravel()

    for ax, chunk in zip(axes_list, selected):
        xs = []
        dist = []
        dist_lz4 = []
        wait = []
        zone_colors = []

        for n in N_BINS:
            lz4_key = (dim, chunk, n)
            iaa_key = (dim, engine, chunk, n)
            if lz4_key not in lz4_map or iaa_key not in iaa_map:
                continue
            d = dist_map[dim].get(bin_pow2_ceil(n))
            if d is None:
                continue
            l = lz4_map[lz4_key]
            w = iaa_map[iaa_key]
            xs.append(n)
            dist.append(d)
            dist_lz4.append(d + l)
            wait.append(w)
            z = zone_code(w, d, l)
            zone_colors.append(["#67c587", "#f5c04a", "#e56b6f"][z])

        if not xs:
            ax.set_visible(False)
            continue

        x_idx = np.arange(len(xs))
        ax.fill_between(x_idx, 1e-3, dist, color="#67c587", alpha=0.16)
        ax.fill_between(x_idx, dist, dist_lz4, color="#f5c04a", alpha=0.18)
        ax.fill_between(x_idx, dist_lz4, np.array(dist_lz4) * 3.0, color="#e56b6f", alpha=0.12)
        ax.plot(x_idx, dist, color="#2e7d32", linewidth=1.7, label="dist")
        ax.plot(x_idx, dist_lz4, color="#ef6c00", linewidth=1.7, linestyle="--", label="dist + lz4")
        ax.plot(x_idx, wait, color="#1565c0", linewidth=1.9, marker="o", label="iaa wait")
        ax.scatter(x_idx, wait, c=zone_colors, s=35, zorder=5, edgecolors="black", linewidths=0.3)

        ax.set_yscale("log")
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel("FID jobs in-flight (n_nodes)")
        ax.set_ylabel("Time (ns, log)")
        ax.set_title(f"chunk={fmt_chunk(chunk)}")
        ax.grid(True, which="both", alpha=0.25, linestyle=":")

    handles, labels = axes_list[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, fontsize=9)
    fig.suptitle(f"Per-chunk Decision Curves - {DIM_LABEL[dim]} (engine {engine})", fontsize=12)

    out_png = out_dir / f"zone_curves_{dim}d_engine{engine}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")


def plot_zone_bar_example(
    dim: int,
    base_engine: int,
    engines: List[int],
    chunks: List[int],
    dist_map: Dict[int, Dict[int, float]],
    lz4_map: Dict[Tuple[int, int, int], float],
    iaa_map: Dict[Tuple[int, int, int, int], float],
    out_dir: Path,
) -> None:
    selected_chunks = pick_chunk_triplet(chunks)
    if not selected_chunks:
        return

    line_engines = pick_engine_quartet(engines)
    if not line_engines:
        return

    fig, axes = plt.subplots(
        1,
        len(selected_chunks),
        figsize=(5.2 * len(selected_chunks), 4.8),
        constrained_layout=True,
    )
    axes_list = np.atleast_1d(axes).ravel()

    zone_palette = {
        0: "#67c587",  # safe
        1: "#f5c04a",  # threshold
        2: "#e56b6f",  # skip/out
    }
    line_palette = plt.get_cmap("tab10")

    for ax, chunk in zip(axes_list, selected_chunks):
        xs = []
        dist = []
        dist_lz4 = []
        waits_by_engine: Dict[int, List[float]] = {e: [] for e in line_engines}

        for n in N_BINS:
            lz4_key = (dim, chunk, n)
            if lz4_key not in lz4_map:
                continue
            d = dist_map[dim].get(bin_pow2_ceil(n))
            if d is None:
                continue
            xs.append(n)
            dist.append(d)
            dist_lz4.append(d + lz4_map[lz4_key])
            for eng in line_engines:
                waits_by_engine[eng].append(iaa_map.get((dim, eng, chunk, n), np.nan))

        if not xs:
            ax.set_visible(False)
            continue

        x_idx = np.arange(len(xs))
        bar_engine = base_engine if base_engine in line_engines else line_engines[0]
        bar_vals = np.array(waits_by_engine[bar_engine], dtype=float)
        bar_colors = []
        for i in range(len(xs)):
            if not np.isfinite(bar_vals[i]):
                bar_colors.append("#d9d9d9")
                continue
            lz4_only = dist_lz4[i] - dist[i]
            z = zone_code(float(bar_vals[i]), float(dist[i]), float(lz4_only))
            bar_colors.append(zone_palette[z])

        ax.bar(
            x_idx,
            np.where(np.isfinite(bar_vals), bar_vals, np.nan),
            width=0.72,
            alpha=0.35,
            color=bar_colors,
            label=f"IAA bar (engine {bar_engine})",
            zorder=1,
        )

        ax.plot(
            x_idx,
            dist,
            color="#2e7d32",
            linewidth=2.0,
            linestyle="-",
            label="distance_calculation",
            zorder=3,
        )
        ax.plot(
            x_idx,
            dist_lz4,
            color="#ef6c00",
            linewidth=2.0,
            linestyle="-",
            label="distance_calculation + lz4",
            zorder=3,
        )

        for i, eng in enumerate(line_engines):
            y = np.array(waits_by_engine[eng], dtype=float)
            if np.isfinite(y).sum() == 0:
                continue
            ax.plot(
                x_idx,
                y,
                color=line_palette(i % 10),
                linewidth=1.8,
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"IAA wait (engine {eng})",
                zorder=4,
            )

        ax.set_yscale("log")
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel("Number of nodes (n_nodes)")
        ax.set_ylabel("Latency (ns, log)")
        ax.set_title(f"chunk={fmt_chunk(chunk)}")
        ax.grid(True, which="both", alpha=0.24, linestyle=":")

    handles, labels = axes_list[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    zone_patches = [
        Patch(facecolor=zone_palette[0], alpha=0.35, label="safe zone bar"),
        Patch(facecolor=zone_palette[1], alpha=0.35, label="threshold zone bar"),
        Patch(facecolor=zone_palette[2], alpha=0.35, label="skip zone bar"),
    ]
    for p in zone_patches:
        if p.get_label() not in uniq:
            uniq[p.get_label()] = p

    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        f"Bar Example - {DIM_LABEL[dim]}\n"
        "safe: wait<=dist, threshold: dist<wait<=dist+lz4, skip: wait>dist+lz4",
        fontsize=12,
    )

    out_png = out_dir / f"zone_bar_example_{dim}d.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")


def plot_classic_roofline(
    dim: int,
    base_engine: int,
    dist_map: Dict[int, Dict[int, float]],
    lz4_map: Dict[Tuple[int, int, int], float],
    iaa_map: Dict[Tuple[int, int, int, int], float],
    engines: List[int],
    out_dir: Path,
) -> None:
    lz4_points = []
    iaa_points_by_engine: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    bw_lz4_samples = []
    bw_iaa_samples_by_engine: Dict[int, List[float]] = defaultdict(list)
    x_vals = []
    y_vals = []

    for (d, chunk, n), lz4_ns in lz4_map.items():
        if d != dim:
            continue
        n_bin = bin_pow2_ceil(n)
        dist_ns = dist_map[dim].get(n_bin)
        if dist_ns is None or dist_ns <= 0:
            continue
        ops = 2.0 * float(dim) * float(n)
        bytes_in = float(chunk) * float(n)
        if bytes_in <= 0:
            continue
        intensity = ops / bytes_in
        total_lz4 = dist_ns + lz4_ns
        if total_lz4 <= 0:
            continue
        perf_lz4 = ops / total_lz4
        lz4_points.append((intensity, perf_lz4))
        x_vals.append(intensity)
        y_vals.append(perf_lz4)
        if lz4_ns > 0:
            bw_lz4_samples.append(bytes_in / lz4_ns)

    for (d, eng, chunk, n), wait_ns in iaa_map.items():
        if d != dim:
            continue
        n_bin = bin_pow2_ceil(n)
        dist_ns = dist_map[dim].get(n_bin)
        if dist_ns is None or dist_ns <= 0:
            continue
        ops = 2.0 * float(dim) * float(n)
        bytes_in = float(chunk) * float(n)
        if bytes_in <= 0:
            continue
        intensity = ops / bytes_in
        total_iaa = max(dist_ns, wait_ns)
        if total_iaa <= 0:
            continue
        perf_iaa = ops / total_iaa
        iaa_points_by_engine[eng].append((intensity, perf_iaa))
        x_vals.append(intensity)
        y_vals.append(perf_iaa)
        if wait_ns > 0:
            bw_iaa_samples_by_engine[eng].append(bytes_in / wait_ns)

    if not x_vals or not lz4_points:
        print(f"[WARN] skip roofline dim={dim} due to insufficient data")
        return

    bw_lz4 = high_quantile(bw_lz4_samples, 95.0)
    bw_iaa = high_quantile(bw_iaa_samples_by_engine.get(base_engine, []), 95.0)
    if bw_lz4 <= 0 or bw_iaa <= 0:
        print(f"[WARN] skip roofline dim={dim} due to invalid BW estimate")
        return

    p_peak = 0.0
    for n, dist_ns in dist_map[dim].items():
        if dist_ns <= 0:
            continue
        ops = 2.0 * float(dim) * float(n)
        p_peak = max(p_peak, ops / dist_ns)
    if p_peak <= 0:
        print(f"[WARN] skip roofline dim={dim} due to invalid peak estimate")
        return

    x_crit_lz4 = p_peak / bw_lz4
    x_crit_iaa = p_peak / bw_iaa
    finite_x = [v for v in x_vals if math.isfinite(v) and v > 0] + [x_crit_lz4, x_crit_iaa]
    finite_y = [v for v in y_vals if math.isfinite(v) and v > 0] + [p_peak]
    x_min = max(min(finite_x) * 0.6, 1e-6)
    x_max = max(finite_x) * 1.8
    y_min = max(min(finite_y) * 0.6, 1e-6)
    y_max = max(finite_y) * 1.8

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    left = min(x_crit_lz4, x_crit_iaa)
    right = max(x_crit_lz4, x_crit_iaa)
    ax.axvspan(x_min, left, color="#e8c9cc", alpha=0.55)
    ax.axvspan(left, right, color="#f2e6be", alpha=0.65)
    ax.axvspan(right, x_max, color="#d9ead8", alpha=0.6)

    x_line = np.logspace(math.log10(x_min), math.log10(x_max), 512)
    y_lz4 = np.minimum(p_peak, bw_lz4 * x_line)
    y_iaa = np.minimum(p_peak, bw_iaa * x_line)
    ax.plot(x_line, y_lz4, color="#56b4e9", lw=2.0, label="LZ4 roof")
    ax.plot(x_line, y_iaa, color="#d81b60", lw=2.0, label=f"IAA roof (engine {base_engine})")
    ax.hlines(p_peak, x_min, x_max, color="#444444", lw=1.8, linestyles=":", label="Compute peak")
    ax.axvline(x_crit_lz4, color="#56b4e9", lw=1.5, linestyle="--")
    ax.axvline(x_crit_iaa, color="#d81b60", lw=1.5, linestyle="--")

    lz4_x = [p[0] for p in lz4_points]
    lz4_y = [p[1] for p in lz4_points]
    ax.scatter(lz4_x, lz4_y, s=30, marker="x", linewidths=1.2, color="#1f77b4", alpha=0.75, label="LZ4 measured")

    tab10 = plt.get_cmap("tab10")
    for i, eng in enumerate(engines):
        pts = iaa_points_by_engine.get(eng, [])
        if not pts:
            continue
        xx = [p[0] for p in pts]
        yy = [p[1] for p in pts]
        marker = "^" if eng == base_engine else "o"
        ax.scatter(xx, yy, s=26, marker=marker, color=tab10(i % 10), alpha=0.55, label=f"IAA measured (engine {eng})")

    ax.text(math.sqrt(x_min * left), y_min * 1.15, "Bandwidth-bound", fontsize=10, ha="center")
    ax.text(math.sqrt(left * right), y_min * 1.15, "Transition", fontsize=10, ha="center")
    ax.text(math.sqrt(right * x_max), y_min * 1.15, "Compute-bound", fontsize=10, ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic intensity proxy (ops/byte)")
    ax.set_ylabel("Realized compute (ops/ns, log)")
    ax.set_title(f"Classic Roofline View - {DIM_LABEL[dim]}")
    ax.grid(True, which="both", alpha=0.22, linestyle=":")

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)

    out_png = out_dir / f"roofline_wait_check_{dim}d.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_map = read_expansion_dist(args.expansion_glob)
    lz4_map, iaa_map, engines, chunks = read_decomp_csv(args.decomp_csv)

    for dim in DIMS:
        base_engine = pick_base_engine(engines, iaa_map, dim)
        plot_classic_roofline(dim, base_engine, dist_map, lz4_map, iaa_map, engines, out_dir)
        plot_zone_map_multi_engine(dim, engines, chunks, dist_map, lz4_map, iaa_map, out_dir)
        plot_zone_curves(dim, base_engine, chunks, dist_map, lz4_map, iaa_map, out_dir)
        plot_zone_bar_example(dim, base_engine, engines, chunks, dist_map, lz4_map, iaa_map, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
