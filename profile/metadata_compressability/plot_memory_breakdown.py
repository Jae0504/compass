#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot memory breakdown from profiling.json")
    parser.add_argument("--input", type=Path, default=Path("profiling.json"))
    parser.add_argument("--output", type=Path, default=Path("profiling_breakdown.png"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    datasets = data["datasets"]

    reps = ["raw", "binary", "compressed_raw", "compressed_after_binary"]
    rep_labels = ["Raw", "Binary", "Compressed(raw)", "Compressed(after binary)"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(reps))
        graph = np.array([ds["sizes"][r]["graph"] for r in reps], dtype=np.float64)
        embedding = np.array([ds["sizes"][r]["embedding"] for r in reps], dtype=np.float64)
        metadata = np.array([ds["sizes"][r]["metadata"] for r in reps], dtype=np.float64)

        ax.bar(x, graph, label="HNSW graph")
        ax.bar(x, embedding, bottom=graph, label="Embedding")
        ax.bar(x, metadata, bottom=graph + embedding, label="Metadata")

        ax.set_title(ds["name"])
        ax.set_xticks(x)
        ax.set_xticklabels(rep_labels, rotation=20, ha="right")
        ax.set_ylabel("Bytes")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Memory Breakdown by Representation", y=1.02)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
