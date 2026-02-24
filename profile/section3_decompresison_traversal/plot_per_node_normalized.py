#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


KEY_PATTERNS = {
    "naive_lz4": r"\(1\)\s+naive_lz4_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "naive_deflate": r"\(1\)\s+naive_deflate_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "common_lz4": r"\(2\)\s+common_lz4_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "common_deflate": r"\(2\)\s+common_deflate_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "graph_traversal": r"\(3\)\s+graph_traversal_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "distance_calc": r"\(4\)\s+distance_calculation_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
    "candidate_update": r"\(5\)\s+candidate_update_ns_per_node:\s*([0-9]+(?:\.[0-9]+)?)",
}


def parse_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    out = {}
    for key, pat in KEY_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Missing '{key}' in log: {log_path}")
        out[key] = float(m.group(1))

    denom = out["graph_traversal"] + out["distance_calc"] + out["candidate_update"]
    if denom <= 0:
        raise ValueError(f"Invalid denominator (3+4+5) in log: {log_path}")

    out["denom"] = denom
    out["n_naive_lz4"] = out["naive_lz4"] / denom
    out["n_naive_deflate"] = out["naive_deflate"] / denom
    out["n_common_lz4"] = out["common_lz4"] / denom
    out["n_common_deflate"] = out["common_deflate"] / denom
    out["n_graph_traversal"] = out["graph_traversal"] / denom
    out["n_distance_calc"] = out["distance_calc"] / denom
    out["n_candidate_update"] = out["candidate_update"] / denom
    return out


def resolve_laion_log(input_dir: Path, laion_arg: str) -> Path:
    if laion_arg:
        p = Path(laion_arg)
        return p if p.is_absolute() else input_dir / p

    preferred = input_dir / "laion_proflie.log"  # Typo kept intentionally for compatibility.
    fallback = input_dir / "laion_profile.log"
    if preferred.exists():
        return preferred
    return fallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot normalized per-node access averages from section3 profile logs."
    )
    p.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing benchmark logs.",
    )
    p.add_argument("--sift-log", default="sift_profile.log")
    p.add_argument("--hnm-log", default="hnm_profile.log")
    p.add_argument(
        "--laion-log",
        default="",
        help="Optional explicit LAION log path. If omitted, tries laion_proflie.log then laion_profile.log.",
    )
    p.add_argument(
        "--output",
        default="per_node_normalized_bars.png",
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
    naive_lz4: list,
    naive_deflate: list,
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
    plt.rcParams["font.size"] = 8
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    x = list(range(len(bench_names)))
    w = 0.14

    colors = {
        "naive_lz4": "#C00000",
        "naive_deflate": "#003AFF",
        "common_lz4": "#FF0000",
        "common_deflate": "#1E8FFF",
        "distance": "#111111",
        "traversal": "#6D6D6D",
        "candidate": "#BFBFBF",
    }

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 4.0 / 2.54))
    # Stacked bar is placed as the first bar in each benchmark group.
    stack_x = [v - 2 * w for v in x]
    ax.bar(stack_x, dist, width=w, label="Dist. Calc.", color=colors["distance"])
    ax.bar(stack_x, trav, width=w, bottom=dist, label="Graph Trav.", color=colors["traversal"])
    ax.bar(
        stack_x,
        cand,
        width=w,
        bottom=[dist[i] + trav[i] for i in range(len(dist))],
        label="Cand. Update",
        color=colors["candidate"],
    )

    ax.bar([v - 1 * w for v in x], naive_lz4, width=w, label="(N) LZ4", color=colors["naive_lz4"])
    ax.bar(
        [v + 0 * w for v in x],
        naive_deflate,
        width=w,
        label="(N) Deflate",
        color=colors["naive_deflate"],
    )
    ax.bar([v + 1 * w for v in x], common_lz4, width=w, label="(G) LZ4", color=colors["common_lz4"])
    ax.bar(
        [v + 2 * w for v in x],
        common_deflate,
        width=w,
        label="(G) Deflate",
        color=colors["common_deflate"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bench_names)
    ax.set_xlabel("Benchmark", labelpad=1)
    ax.set_ylabel("Normalized \n per-node latency")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        fontsize=8,
        prop={"family": "Arial", "size": 8, "weight": "bold"},
        frameon=False,
        handlelength=1.2,
        columnspacing=0.85,
    )
    # Expand drawable axis area to make bars larger within the fixed PNG canvas.
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.17, top=0.80)
    ax.tick_params(axis="x", pad=1)
    fig.savefig(out_path, dpi=200)
    return True


def _plot_svg(
    bench_names: list,
    naive_lz4: list,
    naive_deflate: list,
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

    all_values = naive_lz4 + naive_deflate + common_lz4 + common_deflate
    all_values += [dist[i] + trav[i] + cand[i] for i in range(len(dist))]
    ymax = max(1.0, max(all_values)) * 1.1

    def y_to_px(v: float) -> float:
        return margin_top + plot_h * (1.0 - (v / ymax))

    group_count = len(bench_names)
    group_w = plot_w / max(1, group_count)
    bar_w = min(34.0, group_w * 0.12)
    bar_gap = 8.0
    offsets = [
        -2 * (bar_w + bar_gap),
        -1 * (bar_w + bar_gap),
        0.0,
        1 * (bar_w + bar_gap),
        2 * (bar_w + bar_gap),
    ]

    colors = {
        "naive_lz4": "#C00000",
        "naive_deflate": "#003AFF",
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
            '<text x="{x}" y="{y}" font-size="8" font-family="Arial" font-weight="bold" '
            'text-anchor="end" dominant-baseline="middle">{t}</text>'.format(
                x=margin_left - 8,
                y=f"{y:.2f}",
                t=f"{v:.2f}",
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

        # Stacked baseline bar first: distance (bottom), traversal (middle), candidate (top)
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

        simple_vals = [naive_lz4[i], naive_deflate[i], common_lz4[i], common_deflate[i]]
        simple_colors = [
            colors["naive_lz4"],
            colors["naive_deflate"],
            colors["common_lz4"],
            colors["common_deflate"],
        ]
        for j, val in enumerate(simple_vals):
            bx = cx + offsets[j + 1] - bar_w / 2
            by = y_to_px(val)
            bh = y0 - by
            parts.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
                f'fill="{simple_colors[j]}" stroke="#222222" stroke-width="0.5"/>'
            )

        parts.append(
            '<text x="{x}" y="{y}" font-size="8" font-family="Arial" font-weight="bold" text-anchor="middle">'
            "{t}</text>".format(
                x=f"{cx:.2f}",
                y=f"{(y0 + 28):.2f}",
                t=_svg_escape(bench_names[i]),
            )
        )

    # Axis labels.
    parts.append(
        '<text x="{x}" y="{y}" font-size="8" font-family="Arial" font-weight="bold" text-anchor="middle">'
        "Benchmark</text>".format(x=margin_left + plot_w / 2, y=height - 70)
    )
    parts.append(
        '<text x="{x}" y="{y}" font-size="8" font-family="Arial" font-weight="bold" text-anchor="middle" '
        'transform="rotate(-90 {x} {y})">{t}</text>'.format(
            x=30,
            y=margin_top + plot_h / 2,
            t=_svg_escape("Normalized per-node latency"),
        )
    )

    # Legend.
    legend_items = [
        ("Dist. Calc.", colors["distance"]),
        ("Graph Trav.", colors["traversal"]),
        ("Cand. Update", colors["candidate"]),
        ("(N) LZ4", colors["naive_lz4"]),
        ("(N) Deflate", colors["naive_deflate"]),
        ("(G) LZ4", colors["common_lz4"]),
        ("(G) Deflate", colors["common_deflate"]),
    ]
    legend_cols = 4
    cell_w = 290
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
            '<text x="{x}" y="{y}" font-size="8" font-family="Arial" font-weight="bold" dominant-baseline="middle">{t}</text>'.format(
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

    sift_log = Path(args.sift_log)
    if not sift_log.is_absolute():
        sift_log = input_dir / sift_log

    hnm_log = Path(args.hnm_log)
    if not hnm_log.is_absolute():
        hnm_log = input_dir / hnm_log

    laion_log = resolve_laion_log(input_dir, args.laion_log)

    logs = {
        "SIFT": sift_log,
        "HNM": hnm_log,
        "LAION": laion_log,
    }

    for name, path in logs.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} log not found: {path}")

    parsed = {name: parse_metrics(path) for name, path in logs.items()}

    bench_names = list(parsed.keys())
    naive_lz4 = [parsed[b]["n_naive_lz4"] for b in bench_names]
    naive_deflate = [parsed[b]["n_naive_deflate"] for b in bench_names]
    common_lz4 = [parsed[b]["n_common_lz4"] for b in bench_names]
    common_deflate = [parsed[b]["n_common_deflate"] for b in bench_names]
    trav = [parsed[b]["n_graph_traversal"] for b in bench_names]
    dist = [parsed[b]["n_distance_calc"] for b in bench_names]
    cand = [parsed[b]["n_candidate_update"] for b in bench_names]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = input_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotted = _try_plot_matplotlib(
        bench_names,
        naive_lz4,
        naive_deflate,
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
        naive_lz4,
        naive_deflate,
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
