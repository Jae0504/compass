#!/usr/bin/env bash
# run_section3_profile.sh — run section3 decompression/traversal profiler
#                            for all available datasets, then plot.
#
# Usage:
#   ./run_section3_profile.sh [--block-size 65536] [--synthetic] [--skip-plot]
#                              [--k 10] [--num-queries 100]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/profile_section3_decompression"
PLOT_PY="$SCRIPT_DIR/plot_graph_traversal_decompression.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
BLOCK_SIZE=65536
SYNTHETIC=0
SKIP_PLOT=0
K=10
NUM_QUERIES=100

DATA_DIR="/storage/jykang5"
GRAPH_DIR="$DATA_DIR/compass_graphs"
QUERY_DIR="$DATA_DIR/compass_base_query"

# ── Datasets: name, graph, query ──────────────────────────────────────────────
declare -A GRAPHS=(
  [sift1m]="$GRAPH_DIR/sift_m128_efc200.bin"
  [sift1b]="$GRAPH_DIR/sift1b_m128_efc200.bin"
  [laion]="$GRAPH_DIR/laion_m128_efc200.bin"
  [hnm]="$GRAPH_DIR/hnm_m128_efc200.bin"
)
declare -A QUERIES=(
  [sift1m]="$QUERY_DIR/sift1m_query.fvecs"
  [sift1b]="$QUERY_DIR/sift1b_query.fvecs"
  [laion]="$QUERY_DIR/laion_query.fvecs"
  [hnm]="$QUERY_DIR/hnm_query.fvecs"
)
DATASET_ORDER=(sift1m sift1b laion hnm)

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --block-size)   BLOCK_SIZE="$2";   shift 2 ;;
    --synthetic)    SYNTHETIC=1;       shift   ;;
    --skip-plot)    SKIP_PLOT=1;       shift   ;;
    --k)            K="$2";            shift 2 ;;
    --num-queries)  NUM_QUERIES="$2";  shift 2 ;;
    --data-dir)     DATA_DIR="$2";     shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN" >&2
  echo "Run 'make' first." >&2
  exit 1
fi

synthetic_opt=()
if [[ "$SYNTHETIC" -eq 1 ]]; then
  synthetic_opt=(--synthetic)
fi

# ── Run profiler for each dataset ────────────────────────────────────────────
echo "=== Section 3 Profiler (block_size=$BLOCK_SIZE, synthetic=$SYNTHETIC) ==="
for ds in "${DATASET_ORDER[@]}"; do
  graph="${GRAPHS[$ds]}"
  query="${QUERIES[$ds]}"
  log_out="$SCRIPT_DIR/${ds}_profile.log"

  if [[ ! -f "$graph" ]]; then
    echo "  [$ds] SKIPPED — graph not found: $graph"
    continue
  fi
  if [[ ! -f "$query" ]]; then
    echo "  [$ds] SKIPPED — query not found: $query"
    continue
  fi

  echo "  [$ds] running..."
  "$BIN" "$graph" "$query" "$K" "$NUM_QUERIES" \
    --block-size "$BLOCK_SIZE" \
    "${synthetic_opt[@]+"${synthetic_opt[@]}"}" \
    --log-out "$log_out"
  echo "  [$ds] done → $log_out"
done

# ── Plot ─────────────────────────────────────────────────────────────────────
if [[ "$SKIP_PLOT" -eq 1 ]]; then
  echo "Skipping plot (--skip-plot)"
else
  echo "Plotting..."
  python3 "$PLOT_PY"
  echo "Done."
fi
