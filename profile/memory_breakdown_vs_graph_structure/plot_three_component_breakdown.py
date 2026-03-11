#!/usr/bin/env python3
"""Normalized memory breakdown: Embedding + Graph + Metadata = 100%.

Assumptions:
  - 128-D float32 embedding  → 128 × 4 = 512 B / vector
  - HNSW M=32
      layer 0  : 2M × 4 + 2 = 258 B / vector
      layer 1+ : (M × 4 + 2) / (M-1) ≈ 4.19 B / vector
      total    ≈ 262.19 B / vector
  - Metadata: num_attrs × 4 B / vector  (one 4-byte attribute each)

X-axis shows number of attributes: 4, 8, 16, 64.

Usage:
    python3 plot_three_component_breakdown.py [--output three_component_breakdown.png]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ATTR_COUNTS = [4, 8, 16, 64]

M   = 32
DIM = 128

EMBED_B_PER_VEC = DIM * 4                                     # 512 B
GRAPH_B_PER_VEC = (2 * M * 4 + 2) + (M * 4 + 2) / (M - 1)   # ≈ 262.19 B

METADATA_B_PER_ATTR = 4   # 4 bytes per attribute per vector

# Colors — Embedding: green, Graph: blue (same as breakdown.py), Metadata: orange
COLOR_EMBED = "#444444"
COLOR_GRAPH = "#888888"
COLOR_META  = "#bbbbbb"

FIG_W_CM = 7.5
FIG_H_CM = 6.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(output_path):
    # Per-vector sizes in bytes
    embed_b = EMBED_B_PER_VEC
    graph_b = GRAPH_B_PER_VEC
    meta_bs = [k * METADATA_B_PER_ATTR for k in ATTR_COUNTS]

    totals    = [embed_b + graph_b + mb for mb in meta_bs]
    embed_pct = [embed_b / t * 100 for t in totals]
    graph_pct = [graph_b / t * 100 for t in totals]
    meta_pct  = [mb      / t * 100 for mb, t in zip(meta_bs, totals)]

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size":   12,
    })

    fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    xs    = np.arange(len(ATTR_COUNTS))
    bar_w = 0.5

    ax.bar(xs, embed_pct, bar_w, color=COLOR_EMBED, label="Embedding")
    ax.bar(xs, graph_pct, bar_w, bottom=embed_pct,  color=COLOR_GRAPH, label="Graph")
    meta_bottom = [e + g for e, g in zip(embed_pct, graph_pct)]
    ax.bar(xs, meta_pct,  bar_w, bottom=meta_bottom, color=COLOR_META,  label="Metadata")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ATTR_COUNTS])
    ax.set_xlabel("Number of Attributes", labelpad=1)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("Memory Breakdown (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, color="#c7c7c7")
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.92])
    fig.legend(frameon=False, fontsize=12, loc="upper left",
               bbox_to_anchor=(-0.02, 1.05), ncol=3,
               handlelength=0.7, handleheight=0.7,
               columnspacing=0.8, handletextpad=0.4)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="#ffffff")
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".png":
        fig.savefig(base + ".svg")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalized 3-component breakdown: Embedding + Graph + Metadata."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "three_component_breakdown.png"),
    )
    args = parser.parse_args()

    print(f"Per-vector sizes (M={M}, dim={DIM}):")
    print(f"  Embedding : {EMBED_B_PER_VEC:.1f} B")
    print(f"  Graph     : {GRAPH_B_PER_VEC:.2f} B")
    print(f"  Metadata  : num_attrs × {METADATA_B_PER_ATTR} B")
    print()
    for k, t, ep, gp, mp in zip(
        ATTR_COUNTS, [embed_b + GRAPH_B_PER_VEC + k * METADATA_B_PER_ATTR
                      for k in ATTR_COUNTS],
        [EMBED_B_PER_VEC / (EMBED_B_PER_VEC + GRAPH_B_PER_VEC + k * METADATA_B_PER_ATTR) * 100
         for k in ATTR_COUNTS],
        [GRAPH_B_PER_VEC / (EMBED_B_PER_VEC + GRAPH_B_PER_VEC + k * METADATA_B_PER_ATTR) * 100
         for k in ATTR_COUNTS],
        [k * METADATA_B_PER_ATTR / (EMBED_B_PER_VEC + GRAPH_B_PER_VEC + k * METADATA_B_PER_ATTR) * 100
         for k in ATTR_COUNTS],
    ):
        print(f"  {k:2d} attrs  total={t:.1f} B  "
              f"embed={ep:.1f}%  graph={gp:.1f}%  meta={mp:.1f}%")

    plot(args.output)


# fix undefined embed_b in main's comprehension
embed_b = EMBED_B_PER_VEC

if __name__ == "__main__":
    main()
