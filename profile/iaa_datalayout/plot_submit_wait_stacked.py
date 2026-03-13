#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

CM_TO_INCH = 1.0 / 2.54


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot stacked submit/wait latency breakdown for aligned vs unaligned layouts."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/alignment_microbench_1mb_chunk.csv"),
        help="Input CSV from decompress_scan_alignment_bench.run",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/submit_wait_stacked.png"),
        help="Output PNG path",
    )
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("CSV has no rows")
    return rows


def derive_size_bytes(row: dict) -> int:
    if "size_bytes" in row and row["size_bytes"]:
        return int(float(row["size_bytes"]))
    if "size_mb" in row and row["size_mb"]:
        return int(round(float(row["size_mb"]) * 1024.0 * 1024.0))
    raise RuntimeError("Row is missing both size_bytes and size_mb")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)

    aligned = {}
    legacy = {}
    for r in rows:
        layout = r["layout"]
        size_bytes = derive_size_bytes(r)
        wait_us = float(r["avg_wait_us"]) if r.get("avg_wait_us") else 0.0
        submit_us = float(r["avg_submit_us"]) if r.get("avg_submit_us") else 0.0
        jobs = int(r["jobs_per_query"]) if r.get("jobs_per_query") else 0
        entry = {
            "wait_us": wait_us,
            "submit_us": submit_us,
            "jobs": jobs,
        }
        if layout == "aligned_4byte":
            aligned[size_bytes] = entry
        elif layout == "legacy_unaligned_4shift":
            legacy[size_bytes] = entry

    sizes = sorted(set(aligned.keys()) & set(legacy.keys()))
    if not sizes:
        raise RuntimeError("No common payload sizes found between aligned and legacy rows")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
        ) from e
    from matplotlib.ticker import AutoMinorLocator, NullLocator
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    x = np.arange(len(sizes))
    width = 0.36

    a_submit = np.array([aligned[s]["submit_us"] for s in sizes], dtype=float)
    a_wait = np.array([aligned[s]["wait_us"] for s in sizes], dtype=float)
    a_total = a_submit + a_wait

    l_submit = np.array([legacy[s]["submit_us"] for s in sizes], dtype=float)
    l_wait = np.array([legacy[s]["wait_us"] for s in sizes], dtype=float)
    l_total = l_submit + l_wait
    max_total = float(max(np.max(a_total), np.max(l_total)))
    text_offset = max(2.0, 0.03 * max_total)

    fig, ax = plt.subplots(figsize=(16.0 * CM_TO_INCH, 6.0 * CM_TO_INCH))

    ax.bar(
        x - width / 2,
        a_submit,
        width,
        color="#1E8FFF",
        hatch="///",
        edgecolor="black",
        label="Submit (A)",
    )
    ax.bar(
        x - width / 2,
        a_wait,
        width,
        bottom=a_submit,
        color="#9FD0FF",
        label="Wait (A)",
    )

    ax.bar(
        x + width / 2,
        l_submit,
        width,
        color="#FF0000",
        hatch="///",
        edgecolor="black",
        label="Submit (N-A)",
    )
    ax.bar(
        x + width / 2,
        l_wait,
        width,
        bottom=l_submit,
        color="#FF9999",
        label="Wait (N-A)",
    )

    for i in range(len(sizes)):
        ax.text(
            x[i] - width / 2,
            a_total[i] + text_offset,
            f"{a_total[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x[i] + width / 2,
            l_total[i] + text_offset,
            f"{l_total[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fixed_job_labels = ["1/4", "2/8", "4/16", "8/32"]
    if len(sizes) == 4:
        x_labels = fixed_job_labels
    else:
        x_labels = [f"{aligned[s]['jobs']}/{legacy[s]['jobs']}" for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(
        "Number of IAA jobs for Scan (Aligned (A) / Non-Aligned (N-A))",
        labelpad=-1,
        fontsize=12,
        fontweight="bold",
        fontfamily="Arial",
    )
    ax.set_ylabel(
        "Latency(us)",
        labelpad=2,
        fontsize=12,
        fontweight="bold",
        fontfamily="Arial",
    )
    ax.set_ylim(0.0, max_total + 8.0 * text_offset)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", left=False, right=False, length=0)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.9, alpha=0.45)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.7, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    non_overlappable_patch = Patch(
        facecolor="gray",
        hatch="///",
        edgecolor="black",
        label="Non-overlappable",
    )
    handles.append(non_overlappable_patch)
    labels.append("Non-overlappable")
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
        prop={"family": "Arial", "size": 12, "weight": "bold"},
        handlelength=0.6,
        handleheight=0.6,
        columnspacing=0.5,
        handletextpad=0.4,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.18, top=0.87)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {args.output_png}")


if __name__ == "__main__":
    main()
