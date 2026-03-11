#!/usr/bin/env python3
"""Bar chart: metadata size — original scale vs 1B-vector projection.

Solid bars = original scale; hatched (\\) bars stacked on top = 1B projection.
Log-scale y-axis with MB/GB tick labels.

Usage:
    python3 plot_billion_scale.py [--output billion_scale.png]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------------------
# Data — original in MB, 1B projection converted from GB to MB
# ---------------------------------------------------------------------------
DATASETS = ["LAION", "H&M"]
METHODS  = ["Raw", "LZ4", "Deflate"]

DATA_ORIG_MB = {
    "LAION": {"Raw":  2.8,  "LZ4":  1.4,  "Deflate":  0.8},
    "H&M":   {"Raw": 17.0,  "LZ4":  3.3,  "Deflate":  1.7},
}

# Original DATA_GB × 1000
DATA_1B_MB = {
    "LAION": {"Raw":  27200, "LZ4": 14100, "Deflate":  8100},
    "H&M":   {"Raw": 157900, "LZ4": 30900, "Deflate": 16200},
}

METHOD_COLORS = {
    "Raw":     "#bbbbbb",
    "LZ4":     "#FF0000",
    "Deflate": "#1E8FFF",
}

FIG_W_CM = 7.5
FIG_H_CM = 6.0


def _size_fmt(val, pos):
    """Tick formatter: shows 'X MB' below 1000, 'X GB' above."""
    if val >= 1000:
        v = val / 1000
        return f"{int(round(v))} GB" if v == round(v) else f"{v:.1f} GB"
    return f"{int(round(val))} MB" if val >= 1 else f"{val:.1f} MB"


def plot(output_path):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size":   12,
    })

    fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    n_methods = len(METHODS)
    bar_w     = 0.22
    group_gap = 0.30
    xs        = np.arange(len(DATASETS)) * (n_methods * bar_w + group_gap)

    for m_idx, method in enumerate(METHODS):
        offsets   = xs + (m_idx - (n_methods - 1) / 2) * bar_w
        orig_vals = [DATA_ORIG_MB[ds][method] for ds in DATASETS]
        proj_vals = [DATA_1B_MB[ds][method] - DATA_ORIG_MB[ds][method]
                     for ds in DATASETS]

        # Original — solid fill with black border
        ax.bar(offsets, orig_vals, bar_w,
               color=METHOD_COLORS[method],
               edgecolor="black", linewidth=0.6,
               zorder=3)

        # 1B projection — same color, \\ hatch, black border
        ax.bar(offsets, proj_vals, bar_w,
               bottom=orig_vals,
               color=METHOD_COLORS[method],
               hatch="\\\\",
               edgecolor="black", linewidth=0.6,
               zorder=3)

    # --- Small white gap on every bar at the log break ---
    BREAK_Y = 316.0
    BRK_LO  = BREAK_Y / 1.08
    BRK_HI  = BREAK_Y * 1.08

    for m_idx, method in enumerate(METHODS):
        offsets = xs + (m_idx - (n_methods - 1) / 2) * bar_w
        for x_pos in offsets:
            xl = x_pos - bar_w / 2 + 0.005
            xr = x_pos + bar_w / 2 - 0.005
            ax.fill([xl, xr, xr, xl],
                    [BRK_LO, BRK_LO, BRK_HI, BRK_HI],
                    color="white", zorder=5, linewidth=0)

    ax.set_yscale("log")
    ax.set_ylim(0.5, 4e5)
    ax.set_yticks([1, 10, 10_000, 100_000])
    ax.yaxis.set_major_formatter(FuncFormatter(_size_fmt))
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Break marks on y-axis (geometric midpoint of the log gap)
    FACTOR  = 1.25   # ±25 % in log space ≈ small visual tick
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    mark_kw = dict(transform=trans, color="k", clip_on=False,
                   linewidth=1.2, zorder=10)
    for x_c in [-0.01, 0.025]:   # two parallel slashes
        ax.plot([x_c - 0.025, x_c + 0.025],
                [BREAK_Y / FACTOR, BREAK_Y * FACTOR],
                **mark_kw)

    ax.set_xticks(xs)
    ax.set_xticklabels(DATASETS)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Metadata Size")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8,
            alpha=0.4, color="#c7c7c7", zorder=0)
    ax.set_axisbelow(True)

    all_handles = [
        Patch(facecolor=METHOD_COLORS[m], label=m) for m in METHODS
    ] + [
        Patch(facecolor="none", edgecolor="black", label="Orig."),
        Patch(facecolor="none", edgecolor="black", hatch="\\\\", label="1B"),
    ]
    fig.legend(all_handles, [h.get_label() for h in all_handles],
               frameon=False, fontsize=12, loc="upper center",
               bbox_to_anchor=(0.52, 1.05), ncol=5,
               handlelength=0.7, handleheight=0.7,
               columnspacing=0.3, handletextpad=0.3,
               prop={"family": "Arial", "size": 12, "weight": "normal"})

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.92])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="#ffffff")
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".png":
        fig.savefig(base + ".svg")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Metadata size bar chart: original (solid) + 1B projection (hatched).")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "billion_scale.png"),
    )
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
