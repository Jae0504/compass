#!/usr/bin/env python3
"""Profile memory breakdown: HNSW graph structure vs raw metadata (H&M dataset).

Each bar is normalized to 100%: graph + metadata = 100%.
X-axis shows the number of attributes included (4, 8, 16, 22).

Raw metadata bytes per column = sum over records of (len(str(value)) + 1),
matching the text_bytes accounting in metadata_column_profile.cpp.

Usage:
    python3 profile_memory_breakdown.py \\
        [--payloads /storage/jykang5/payloads/hnm_payloads.jsonl] \\
        [--n 105100] [--m 32] [--output breakdown.png]
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

EXCLUDE_COLS   = {"detail_desc"}
ATTR_COUNTS    = [4, 8, 16, 20]
N_DROP_SMALLEST = 2   # drop this many attributes with the smallest text_bytes


# ---------------------------------------------------------------------------
# Data loading / column classification
# ---------------------------------------------------------------------------

def load_payloads(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_columns(records):
    """Return includable columns in JSONL order with their raw text_bytes totals.

    Inclusion rules (same as metadata_column_profile.cpp):
      integer : always included (range-mapped to 0-255 if unique > 256)
      string  : included only if unique values <= 256

    text_bytes per value = len(str(value)) + 1  (digits/chars + separator).
    """
    first_rec  = records[0]
    ordered    = [c for c in first_rec.keys() if c not in EXCLUDE_COLS]

    col_type       = {}          # col -> "int" | "float" | "str"
    col_unique     = {c: set() for c in ordered}
    col_text_bytes = {c: 0     for c in ordered}

    for rec in records:
        for col in ordered:
            val = rec.get(col)
            if val is None:
                continue
            if isinstance(val, bool):
                col_type.setdefault(col, "str")
                col_unique[col].add(str(val))
                col_text_bytes[col] += len(str(val)) + 1
            elif isinstance(val, int):
                col_type[col] = "int"
                col_unique[col].add(val)
                col_text_bytes[col] += len(str(val)) + 1
            elif isinstance(val, float):
                col_type[col] = "float"
                col_unique[col].add(val)
                col_text_bytes[col] += len(str(val)) + 1
            elif isinstance(val, str):
                col_type.setdefault(col, "str")
                col_unique[col].add(val)
                col_text_bytes[col] += len(val) + 1

    includable = OrderedDict()
    excluded   = []
    for col in ordered:
        t        = col_type.get(col, "str")
        n_unique = len(col_unique[col])
        tb       = col_text_bytes[col]
        if t in ("int", "float"):
            includable[col] = {"type": t, "n_unique": n_unique, "text_bytes": tb}
        elif t == "str" and n_unique <= 256:
            includable[col] = {"type": t, "n_unique": n_unique, "text_bytes": tb}
        else:
            excluded.append((col, t, n_unique))

    return includable, excluded


def select_columns(includable_cols, n_drop_smallest=N_DROP_SMALLEST):
    """Drop the n_drop_smallest columns by text_bytes, then reorder:
    string/char columns first (JSONL order), integer columns second (JSONL order).
    """
    by_size   = sorted(includable_cols.items(), key=lambda x: x[1]["text_bytes"])
    drop      = {col for col, _ in by_size[:n_drop_smallest]}
    remaining = [(col, info) for col, info in includable_cols.items() if col not in drop]

    strings = [(col, info) for col, info in remaining if info["type"] == "str"]
    ints    = [(col, info) for col, info in remaining if info["type"] != "str"]
    return OrderedDict(strings + ints), drop


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def hnsw_graph_bytes(n, m):
    """hnswlib link storage (bytes).

    Layer 0  : N nodes × (2M neighbors × 4 B + 2 B count)
    Layer 1+ : N/(M-1) nodes total × (M neighbors × 4 B + 2 B count)
    """
    layer0  = n * (2 * m * 4 + 2)
    higher  = (n / (m - 1)) * (m * 4 + 2)
    return layer0 + higher


def cumulative_text_bytes(includable_cols, num_attrs):
    """Sum of text_bytes for the first `num_attrs` includable columns."""
    cols = list(includable_cols.values())[:num_attrs]
    return sum(info["text_bytes"] for info in cols)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_breakdown(n, m, includable_cols, output_path):
    graph_b = hnsw_graph_bytes(n, m)

    meta_b = [cumulative_text_bytes(includable_cols, k) for k in ATTR_COUNTS]

    # Normalize each bar to 100 %.
    totals       = [graph_b + mb for mb in meta_b]
    graph_pct    = [graph_b / t * 100 for t in totals]
    meta_pct     = [mb     / t * 100 for mb, t in zip(meta_b, totals)]

    plt.rcParams.update({
        "font.family" : "Arial",
        "font.size"   : 12,
    })

    fig, ax = plt.subplots(figsize=(7.5 / 2.54, 6.0 / 2.54))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    xs    = np.arange(len(ATTR_COUNTS))
    bar_w = 0.5

    p_graph = ax.bar(xs, graph_pct, bar_w, color="#08519c", label="Graph Structure")
    p_meta  = ax.bar(xs, meta_pct,  bar_w, bottom=graph_pct, color="#fd8d3c", label="Metadata")

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
    # Place legend at the very top edge of the figure with square handles.
    fig.legend(frameon=False, fontsize=12, loc="upper center",
               bbox_to_anchor=(0.55, 1.05), ncol=2,
               handlelength=0.7, handleheight=0.7,
               columnspacing=0.8, handletextpad=0.4)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="#ffffff")
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".png":
        fig.savefig(base + ".svg")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalized memory breakdown: HNSW graph vs raw metadata."
    )
    parser.add_argument(
        "--payloads", default="/storage/jykang5/payloads/hnm_payloads.jsonl",
    )
    parser.add_argument("--n", type=int, default=105100)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "breakdown.png"),
    )
    args = parser.parse_args()

    print(f"Loading {args.payloads} …")
    records = load_payloads(args.payloads)
    print(f"  {len(records):,} records")

    includable, excluded = classify_columns(records)
    selected, dropped   = select_columns(includable)
    print(f"\nDropped (2 smallest text_bytes): {sorted(dropped)}")
    print(f"\nSelected attributes ({len(selected)}) — strings first, then ints:")
    for col, info in selected.items():
        print(f"  {col:40s}  {info['type']:5s}  unique={info['n_unique']:4d}"
              f"  text_bytes={info['text_bytes']:>10,}")
    print(f"\nExcluded by inclusion rules ({len(excluded)}): {[c for c,*_ in excluded]}")

    graph_mb = hnsw_graph_bytes(args.n, args.m) / 1024 ** 2
    print(f"\nHNSW graph (M={args.m}, N={args.n:,}): {graph_mb:.2f} MB")
    print(f"\nNormalized breakdown (graph + metadata = 100%):")
    for k in ATTR_COUNTS:
        mb    = cumulative_text_bytes(selected, k) / 1024 ** 2
        total = graph_mb + mb
        print(f"  {k:2d} attrs  meta={mb:.2f} MB  graph={graph_mb:.2f} MB  "
              f"meta%={mb/total*100:.1f}%  graph%={graph_mb/total*100:.1f}%")

    plot_breakdown(args.n, args.m, selected, args.output)


if __name__ == "__main__":
    main()
