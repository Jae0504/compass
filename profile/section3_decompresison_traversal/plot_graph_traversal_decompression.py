#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


KEY_PATTERNS = {
    "common_lz4": r"\(2\)\s+common_lz4_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "common_deflate": r"\(2\)\s+common_deflate_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "graph_traversal": r"\(3\)\s+graph_traversal_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "distance_calc": r"\(4\)\s+distance_calculation_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "candidate_update": r"\(5\)\s+candidate_update_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
}
NS_TO_MS = 1.0 / 1_000_000.0


def parse_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    out = {}
    for key, pat in KEY_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Missing '{key}' in log: {log_path}")
        out[key] = float(m.group(1))
    return out


def resolve_input_log_path(input_dir: Path, log_arg: str) -> Path:
    p = Path(log_arg)
    return p if p.is_absolute() else input_dir / p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot graph traversal and decompression per-node latency in milliseconds."
    )
    p.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing benchmark logs.",
    )
    p.add_argument("--sift1m-log", default="sift1m_profile.log")
    p.add_argument("--sift1b-log", default="sift1b_profile.log")
    p.add_argument("--laion-log", default="laion_profile.log")
    p.add_argument("--hnm-log", default="hnm_profile.log")
    p.add_argument(
        "--output",
        default="graph_traversal_decompression_bars.png",
        help="Output image path.",
    )
    return p


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _try_plot_matplotlib(
    bench_names: list,
    common_lz4: list,
    common_deflate: list,
    trav: list,
    dist: list,
    cand: list,
    out_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    x = list(range(len(bench_names)))
    w = 0.18

    colors = {
        "common_lz4": "#FF0000",
        "common_deflate": "#1E8FFF",
        "distance": "#111111",
        "traversal": "#6D6D6D",
        "candidate": "#BFBFBF",
    }

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 5.0 / 2.54))
    # Keep stacked bar first in each benchmark group.
    stack_x = [v - 1 * w for v in x]
    ax.bar(stack_x, dist, width=w, label="Dist. Calc.", color=colors["distance"])
    ax.bar(
        stack_x,
        trav,
        width=w,
        bottom=dist,
        label="Graph Trav.",
        color=colors["traversal"],
    )
    ax.bar(
        stack_x,
        cand,
        width=w,
        bottom=[dist[i] + trav[i] for i in range(len(dist))],
        label="Cand. Update",
        color=colors["candidate"],
    )

    ax.bar(
        [v + 0 * w for v in x],
        common_lz4,
        width=w,
        label="(D) LZ4",
        color=colors["common_lz4"],
    )
    ax.bar(
        [v + 1 * w for v in x],
        common_deflate,
        width=w,
        label="(D) Deflate",
        color=colors["common_deflate"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bench_names)
    ax.set_xlabel("Benchmark", labelpad=-1, fontsize=10, fontweight="bold", fontfamily="Arial")
    ax.set_ylabel(
        "Latency taken per-node (ms)",
        fontsize=10,
        fontweight="bold",
        fontfamily="Arial",
    )
    max_top = max(
        max([dist[i] + trav[i] + cand[i] for i in range(len(dist))]),
        max(common_lz4),
        max(common_deflate),
    )
    ax.set_ylim(0.0, max_top * 1.1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    legend_order = [
        "Graph Trav.",
        "Dist. Calc.",
        "Cand. Update",
        "(D) LZ4",
        "(D) Deflate",
    ]
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [label for label in legend_order if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=10,
        prop={"family": "Arial", "size": 10, "weight": "bold"},
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.17, top=0.87)
    ax.tick_params(axis="x", pad=1)
    fig.savefig(out_path, dpi=200)
    return True


def _plot_svg(
    bench_names: list,
    common_lz4: list,
    common_deflate: list,
    trav: list,
    dist: list,
    cand: list,
    out_path: Path,
) -> None:
    width = 1380
    height = 760
    margin_left = 100
    margin_right = 40
    margin_top = 120
    margin_bottom = 160
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_values = common_lz4 + common_deflate
    all_values += [dist[i] + trav[i] + cand[i] for i in range(len(dist))]
    ymax = max(1.0, max(all_values)) * 1.1

    def y_to_px(v: float) -> float:
        return margin_top + plot_h * (1.0 - (v / ymax))

    group_count = len(bench_names)
    group_w = plot_w / max(1, group_count)
    bar_w = min(34.0, group_w * 0.16)
    bar_gap = 8.0
    offsets = [
        -1 * (bar_w + bar_gap),
        0.0,
        1 * (bar_w + bar_gap),
    ]

    colors = {
        "common_lz4": "#FF0000",
        "common_deflate": "#1E8FFF",
        "distance": "#111111",
        "traversal": "#6D6D6D",
        "candidate": "#BFBFBF",
    }

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # Grid + y-axis ticks.
    n_ticks = 6
    for i in range(n_ticks + 1):
        v = ymax * i / n_ticks
        y = y_to_px(v)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="10" font-family="Arial" font-weight="bold" '
            'text-anchor="end" dominant-baseline="middle">{t}</text>'.format(
                x=margin_left - 8,
                y=f"{y:.2f}",
                t=f"{v:.3f}",
            )
        )

    y0 = y_to_px(0.0)
    parts.append(
        f'<line x1="{margin_left}" y1="{y0:.2f}" x2="{margin_left + plot_w}" y2="{y0:.2f}" '
        'stroke="#111111" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{y0:.2f}" '
        'stroke="#111111" stroke-width="1.5"/>'
    )

    # Bars.
    for i in range(group_count):
        cx = margin_left + group_w * (i + 0.5)

        # Stacked bar order stays: distance (bottom), traversal (middle), candidate (top).
        bx = cx + offsets[0] - bar_w / 2
        d_h = y0 - y_to_px(dist[i])
        t_h = y0 - y_to_px(trav[i])
        c_h = y0 - y_to_px(cand[i])

        by_dist = y0 - d_h
        by_trav = by_dist - t_h
        by_cand = by_trav - c_h

        parts.append(
            f'<rect x="{bx:.2f}" y="{by_dist:.2f}" width="{bar_w:.2f}" height="{d_h:.2f}" '
            f'fill="{colors["distance"]}" stroke="#222222" stroke-width="0.5"/>'
        )
        parts.append(
            f'<rect x="{bx:.2f}" y="{by_trav:.2f}" width="{bar_w:.2f}" height="{t_h:.2f}" '
            f'fill="{colors["traversal"]}" stroke="#222222" stroke-width="0.5"/>'
        )
        parts.append(
            f'<rect x="{bx:.2f}" y="{by_cand:.2f}" width="{bar_w:.2f}" height="{c_h:.2f}" '
            f'fill="{colors["candidate"]}" stroke="#222222" stroke-width="0.5"/>'
        )

        simple_vals = [common_lz4[i], common_deflate[i]]
        simple_colors = [colors["common_lz4"], colors["common_deflate"]]
        for j, val in enumerate(simple_vals):
            bx = cx + offsets[j + 1] - bar_w / 2
            by = y_to_px(val)
            bh = y0 - by
            parts.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
                f'fill="{simple_colors[j]}" stroke="#222222" stroke-width="0.5"/>'
            )

        parts.append(
            '<text x="{x}" y="{y}" font-size="10" font-family="Arial" font-weight="bold" text-anchor="middle">'
            "{t}</text>".format(
                x=f"{cx:.2f}",
                y=f"{(y0 + 28):.2f}",
                t=_svg_escape(bench_names[i]),
            )
        )

    # Axis labels.
    parts.append(
        '<text x="{x}" y="{y}" font-size="10" font-family="Arial" font-weight="bold" text-anchor="middle">'
        "Benchmark</text>".format(x=margin_left + plot_w / 2, y=height - 70)
    )
    parts.append(
        '<text x="{x}" y="{y}" font-size="10" font-family="Arial" font-weight="bold" text-anchor="middle" '
        'transform="rotate(-90 {x} {y})">{t}</text>'.format(
            x=30,
            y=margin_top + plot_h / 2,
            t=_svg_escape("Latency taken per-node (ms)"),
        )
    )

    # Legend order requested: Graph Trav., Dist. Calc., Cand. Update, (D) LZ4, (D) Deflate.
    legend_items = [
        ("Graph Trav.", colors["traversal"]),
        ("Dist. Calc.", colors["distance"]),
        ("Cand. Update", colors["candidate"]),
        ("(D) LZ4", colors["common_lz4"]),
        ("(D) Deflate", colors["common_deflate"]),
    ]
    legend_cols = 5
    cell_w = 245
    lx = margin_left
    ly = 28
    for i, (label, color) in enumerate(legend_items):
        row = i // legend_cols
        col = i % legend_cols
        xx = lx + col * cell_w
        yy = ly + row * 24
        parts.append(
            f'<rect x="{xx}" y="{yy - 11}" width="15" height="15" fill="{color}" stroke="#222222" stroke-width="0.5"/>'
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="10" font-family="Arial" font-weight="bold" dominant-baseline="middle">{t}</text>'.format(
                x=xx + 22,
                y=yy - 3,
                t=_svg_escape(label),
            )
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()

    logs_in_order = [
        ("SIFT1M", resolve_input_log_path(input_dir, args.sift1m_log)),
        ("SIFT1B", resolve_input_log_path(input_dir, args.sift1b_log)),
        ("LAION", resolve_input_log_path(input_dir, args.laion_log)),
        ("H&M", resolve_input_log_path(input_dir, args.hnm_log)),
    ]

    for name, path in logs_in_order:
        if not path.exists():
            raise FileNotFoundError(f"{name} log not found: {path}")

    parsed = {name: parse_metrics(path) for name, path in logs_in_order}

    bench_names = [name for name, _ in logs_in_order]
    common_lz4 = [parsed[b]["common_lz4"] * NS_TO_MS for b in bench_names]
    common_deflate = [parsed[b]["common_deflate"] * NS_TO_MS for b in bench_names]
    trav = [parsed[b]["graph_traversal"] * NS_TO_MS for b in bench_names]
    dist = [parsed[b]["distance_calc"] * NS_TO_MS for b in bench_names]
    cand = [parsed[b]["candidate_update"] * NS_TO_MS for b in bench_names]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = input_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotted = _try_plot_matplotlib(
        bench_names,
        common_lz4,
        common_deflate,
        trav,
        dist,
        cand,
        out_path,
    )

    if plotted:
        print(f"Saved plot to: {out_path}")
        return

    # Fallback path when matplotlib is unavailable in this environment.
    fallback_path = out_path if out_path.suffix.lower() == ".svg" else out_path.with_suffix(".svg")
    _plot_svg(
        bench_names,
        common_lz4,
        common_deflate,
        trav,
        dist,
        cand,
        fallback_path,
    )
    if fallback_path == out_path:
        print(f"Saved plot to: {fallback_path} (SVG fallback, matplotlib not installed)")
    else:
        print(
            f"matplotlib not installed; saved SVG fallback to: {fallback_path}"
        )


if __name__ == "__main__":
    main()
