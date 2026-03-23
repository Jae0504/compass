#!/usr/bin/env python3
"""Plot FID/TB compression results for SIFT1M, HNM, and LAION.

One figure with three dataset groups separated by vertical dividers.
Each group has 5 bars (stacked FID + TB):
  1. Original Metadata  (4 × N bytes, orange)
  2. REAP-Overlay       (FID + TB uncompressed, blue stacked)
  3. LZ4(C)
  4. Deflate(C)
  5. Deflate(I)         (IAA hardware)
"""
import argparse
import csv
from pathlib import Path

# ── Colors ─────────────────────────────────────────────────────────────────────
COLOR_FID      = "#2171b5"   # blue   — FID segment
COLOR_TB       = "#6baed6"   # light blue — TB segment
COLOR_METADATA = "#fd8d3c"   # orange — Original Metadata bar

BYTES_PER_MB = 1024 ** 2

DATASET_ORDER  = ["sift1m", "hnm", "laion"]
DATASET_LABELS = {"sift1m": "SIFT1M", "hnm": "H&M", "laion": "LAION"}

BAR_DEFS = [
    # (bar_label, algo_key)   algo_key None → metadata / uncompressed
    ("Orig. Meta",   "metadata"),
    ("REAP-Overlay", "uncompressed"),
    ("LZ4(C)",       "LZ4"),
    ("Deflate(C)",   "Deflate"),
    ("Deflate(I)",   "IAA"),
]


# ── CSV parsing ────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    """
    Returns nested dict:
      data[dataset][data_type][algorithm] = {original_bytes, compressed_bytes, n_elements}
    data_type: "fid" | "tb"
    """
    data: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds   = row["dataset"]
            dt   = row["data_type"]
            algo = row["algorithm"]
            data.setdefault(ds, {}).setdefault(dt, {})[algo] = {
                "original_bytes":   int(row["original_bytes"]),
                "compressed_bytes": int(row["compressed_bytes"]),
                "n_elements":       int(row["n_elements"]),
            }
    return data


def get_bars(data: dict, dataset: str):
    """
    Returns (fid_mb_list, tb_mb_list) each of length 5, one per BAR_DEFS entry.
    """
    ds = data.get(dataset, {})
    fid_d  = ds.get("fid",      {})
    tb_d   = ds.get("tb",       {})
    meta_d = ds.get("metadata", {})

    # Original (uncompressed) sizes — same for all algos
    fid_orig = next(iter(fid_d.values()))["original_bytes"] if fid_d else 0
    tb_orig  = next(iter(tb_d.values()))["original_bytes"]  if tb_d  else 0

    # Actual metadata bytes from CSV "metadata" row; fallback to n_elements * 4
    if meta_d:
        meta_bytes = next(iter(meta_d.values()))["original_bytes"]
    else:
        n_elements = 0
        for v in fid_d.values():
            n_elements = v["n_elements"]
            break
        meta_bytes = n_elements * 4

    fid_mb_list = []
    tb_mb_list  = []

    for _, algo_key in BAR_DEFS:
        if algo_key == "metadata":
            fid_mb_list.append(meta_bytes / BYTES_PER_MB)
            tb_mb_list.append(0.0)
        elif algo_key == "uncompressed":
            fid_mb_list.append(fid_orig / BYTES_PER_MB)
            tb_mb_list.append(tb_orig  / BYTES_PER_MB)
        else:
            fid_row = fid_d.get(algo_key, {})
            tb_row  = tb_d.get(algo_key,  {})
            fid_mb_list.append(fid_row.get("compressed_bytes", 0) / BYTES_PER_MB)
            tb_mb_list.append(tb_row.get("compressed_bytes",  0) / BYTES_PER_MB)

    return fid_mb_list, tb_mb_list


# ── Matplotlib plot ────────────────────────────────────────────────────────────

def _try_plot_matplotlib(data: dict, output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ModuleNotFoundError:
        return False

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11, "font.weight": "bold",
        "axes.labelweight": "bold",
    })

    n_datasets = len(DATASET_ORDER)
    n_bars     = len(BAR_DEFS)
    bar_w      = 0.55
    group_gap  = 2.5

    # Build x positions: 5 bars per group, group_gap between groups
    positions = []
    x = 0.0
    group_centers = []
    for di in range(n_datasets):
        start = x
        for bi in range(n_bars):
            positions.append(x)
            x += 1.0
        group_centers.append((start + x - 1.0) / 2.0)
        x += group_gap

    fig, ax = plt.subplots(figsize=(20.0 / 2.54, 7.0 / 2.54))

    all_totals = []
    for di, ds_key in enumerate(DATASET_ORDER):
        fid_mb, tb_mb = get_bars(data, ds_key)
        for bi, ((label, _), fid, tb) in enumerate(zip(BAR_DEFS, fid_mb, tb_mb)):
            xi = positions[di * n_bars + bi]
            if label.startswith("Orig"):
                ax.bar(xi, fid, width=bar_w, color=COLOR_METADATA,
                       edgecolor="#222", linewidth=0.6)
                all_totals.append(fid)
            else:
                ax.bar(xi, fid, width=bar_w, color=COLOR_FID,
                       edgecolor="#222", linewidth=0.6)
                ax.bar(xi, tb,  width=bar_w, bottom=fid, color=COLOR_TB,
                       edgecolor="#222", linewidth=0.6)
                all_totals.append(fid + tb)

    ymax = max(all_totals) * 1.22 if all_totals else 1.0
    ax.set_ylim(0, ymax)

    # Value annotations
    for di, ds_key in enumerate(DATASET_ORDER):
        fid_mb, tb_mb = get_bars(data, ds_key)
        for bi, (fid, tb) in enumerate(zip(fid_mb, tb_mb)):
            xi = positions[di * n_bars + bi]
            total = fid + tb
            ax.text(xi, total + ymax * 0.01, f"{total:.1f}",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", fontfamily="Arial")

    # ── X-axis: minor ticks for bar labels, major ticks for dataset names ──
    bar_label_xs = positions[:]
    bar_labels = [label for _ in range(n_datasets) for label, _ in BAR_DEFS]
    ax.set_xticks(bar_label_xs, minor=True)
    ax.set_xticklabels(bar_labels, minor=True, fontsize=8,
                       rotation=45, ha="right")
    ax.tick_params(axis="x", which="minor", length=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS[k] for k in DATASET_ORDER], fontsize=10)
    ax.tick_params(axis="x", which="major", top=False, bottom=False,
                   direction="in", pad=30)

    for tick in ax.get_xticklabels() + ax.get_xticklabels(minor=True) + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ── Dividers below x-axis (following generate_decompress.py style) ──
    left_x  = positions[0]    - bar_w
    right_x = positions[-1]   + bar_w
    # Between-group dividers
    for di in range(n_datasets - 1):
        last_x  = positions[(di + 1) * n_bars - 1]
        first_x = positions[(di + 1) * n_bars]
        div_x   = (last_x + first_x) / 2.0
        ax.axvline(div_x, color="black", linewidth=1.0,
                   ymin=-0.6, ymax=-0.0004, clip_on=False)
    # Left and right outer borders
    ax.axvline(left_x,  color="black", linewidth=1.0,
               ymin=-0.6, ymax=-0.0004, clip_on=False)
    ax.axvline(right_x, color="black", linewidth=1.0,
               ymin=-0.6, ymax=-0.0004, clip_on=False)
    ax.set_xlim(left_x, right_x)

    ax.set_ylabel("Size (MB)", fontsize=11, fontweight="bold", fontfamily="Arial")
    ax.grid(axis="y", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        mpatches.Patch(color=COLOR_FID,      label="FID"),
        mpatches.Patch(color=COLOR_TB,       label="TB (bucket-major)"),
        mpatches.Patch(color=COLOR_METADATA, label="Original Metadata"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, prop={"family": "Arial", "size": 10, "weight": "bold"},
               frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=0.8, borderaxespad=0.0)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.86)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200)
    print(f"Saved: {out}")
    return True


# ── SVG fallback ──────────────────────────────────────────────────────────────

def _plot_svg(data: dict, output_path: str) -> None:
    import math

    n_datasets = len(DATASET_ORDER)
    n_bars     = len(BAR_DEFS)

    width  = 1400
    height = 560
    ml, mr, mt, mb = 90, 30, 90, 130
    pw = width - ml - mr
    ph = height - mt - mb

    # Collect all totals for ymax
    all_totals = []
    for ds_key in DATASET_ORDER:
        fid_mb, tb_mb = get_bars(data, ds_key)
        for f, t in zip(fid_mb, tb_mb):
            all_totals.append(f + t)
    ymax = max(all_totals) * 1.22 if all_totals else 1.0

    def ypx(v: float) -> float:
        return mt + ph * (1.0 - v / ymax)

    # x layout: n_datasets groups, each n_bars bars, group_gap between
    total_slots = n_datasets * n_bars + (n_datasets - 1) * 2  # 2 extra slots gap
    gw = pw / total_slots
    bw = min(50.0, gw * 0.72)

    def bar_cx(di: int, bi: int) -> float:
        slot = di * (n_bars + 2) + bi
        return ml + gw * (slot + 0.5)

    def _esc(t: str) -> str:
        return (t.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    # Grid
    y_base = ypx(0)
    for ytick in range(0, int(math.ceil(ymax)) + 5, 5):
        if ytick > ymax:
            continue
        yp = ypx(ytick)
        lw = "1.0" if ytick == 0 else "0.5"
        alpha = "0.6" if ytick == 0 else "0.25"
        parts.append(
            f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+pw}" y2="{yp:.1f}" '
            f'stroke="#888" stroke-width="{lw}" opacity="{alpha}"/>')
        if ytick > 0:
            parts.append(
                f'<text x="{ml-6}" y="{yp:.1f}" font-size="11" font-family="Arial" '
                f'font-weight="bold" text-anchor="end" dominant-baseline="middle">'
                f'{ytick}</text>')

    # Axes
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{y_base:.1f}" '
                 f'stroke="#111" stroke-width="1.5"/>')
    parts.append(f'<line x1="{ml}" y1="{y_base:.1f}" x2="{ml+pw}" y2="{y_base:.1f}" '
                 f'stroke="#111" stroke-width="1.5"/>')

    for di, ds_key in enumerate(DATASET_ORDER):
        fid_mb, tb_mb = get_bars(data, ds_key)

        for bi, ((label, _), fid, tb) in enumerate(zip(BAR_DEFS, fid_mb, tb_mb)):
            cx = bar_cx(di, bi)
            bx = cx - bw / 2

            if label.startswith("Orig"):
                bar_top = ypx(fid)
                bar_h   = y_base - bar_top
                if bar_h > 0.5:
                    parts.append(
                        f'<rect x="{bx:.1f}" y="{bar_top:.1f}" width="{bw:.1f}" '
                        f'height="{bar_h:.1f}" fill="{COLOR_METADATA}" '
                        f'stroke="#222" stroke-width="0.5"/>')
                total = fid
            else:
                if fid > 0:
                    fid_top = ypx(fid)
                    fid_h   = y_base - fid_top
                    if fid_h > 0.5:
                        parts.append(
                            f'<rect x="{bx:.1f}" y="{fid_top:.1f}" width="{bw:.1f}" '
                            f'height="{fid_h:.1f}" fill="{COLOR_FID}" '
                            f'stroke="#222" stroke-width="0.5"/>')
                if tb > 0:
                    tb_top = ypx(fid + tb)
                    tb_h   = ypx(fid) - tb_top
                    if tb_h > 0.5:
                        parts.append(
                            f'<rect x="{bx:.1f}" y="{tb_top:.1f}" width="{bw:.1f}" '
                            f'height="{tb_h:.1f}" fill="{COLOR_TB}" '
                            f'stroke="#222" stroke-width="0.5"/>')
                total = fid + tb

            # Value annotation
            ann_y = ypx(total) - 5
            parts.append(
                f'<text x="{cx:.1f}" y="{ann_y:.1f}" font-size="10" font-family="Arial" '
                f'font-weight="bold" text-anchor="middle">{total:.1f}</text>')

            # Bar label (x-axis)
            lines = label.split("\n")
            for li, line in enumerate(lines):
                parts.append(
                    f'<text x="{cx:.1f}" y="{y_base + 16 + li * 13:.1f}" '
                    f'font-size="10" font-family="Arial" font-weight="bold" '
                    f'text-anchor="middle">{_esc(line)}</text>')

        # Dataset label
        cx_group = (bar_cx(di, 0) + bar_cx(di, n_bars - 1)) / 2
        parts.append(
            f'<text x="{cx_group:.1f}" y="{y_base + 58}" '
            f'font-size="12" font-family="Arial" font-weight="bold" '
            f'text-anchor="middle">{_esc(DATASET_LABELS[ds_key])}</text>')

        # Vertical divider after group (except last)
        if di < n_datasets - 1:
            div_x = (bar_cx(di, n_bars - 1) + bar_cx(di + 1, 0)) / 2
            parts.append(
                f'<line x1="{div_x:.1f}" y1="{mt}" x2="{div_x:.1f}" y2="{y_base:.1f}" '
                f'stroke="#333" stroke-width="1.5"/>')

    # Y-axis label
    parts.append(
        f'<text x="22" y="{mt + ph/2:.1f}" font-size="12" font-family="Arial" '
        f'font-weight="bold" text-anchor="middle" '
        f'transform="rotate(-90 22 {mt + ph/2:.1f})">Size (MB)</text>')

    # Legend
    legend_items = [
        ("FID",               COLOR_FID),
        ("TB (bucket-major)", COLOR_TB),
        ("Original Metadata", COLOR_METADATA),
    ]
    cell_w = 210
    lx = ml + max(0.0, (pw - len(legend_items) * cell_w) / 2.0)
    ly = 32
    for idx, (lbl, color) in enumerate(legend_items):
        xx = lx + idx * cell_w
        parts.append(
            f'<rect x="{xx}" y="{ly - 10}" width="14" height="14" fill="{color}" '
            f'stroke="#222" stroke-width="0.5"/>')
        parts.append(
            f'<text x="{xx + 20}" y="{ly + 2}" font-size="12" font-family="Arial" '
            f'font-weight="bold">{_esc(lbl)}</text>')

    parts.append("</svg>")
    out = Path(output_path)
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved (SVG): {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",    default="out/fid_tb_compression.csv")
    p.add_argument("--output", default="out/fid_tb_compression.png")
    return p


def main() -> None:
    args = build_parser().parse_args()
    data = load_csv(args.csv)

    # Print summary table
    print(f"\n{'Dataset':<10} {'Bar':<22} {'FID(MB)':>10} {'TB(MB)':>10} {'Total(MB)':>12}")
    print("-" * 68)
    for ds_key in DATASET_ORDER:
        fid_mb, tb_mb = get_bars(data, ds_key)
        for (label, _), fid, tb in zip(BAR_DEFS, fid_mb, tb_mb):
            lbl = label
            print(f"{DATASET_LABELS[ds_key]:<10} {lbl:<22} {fid:>10.3f} {tb:>10.3f} {fid+tb:>12.3f}")
        print()

    plotted = _try_plot_matplotlib(data, args.output)
    if not plotted:
        _plot_svg(data, args.output)


if __name__ == "__main__":
    main()
