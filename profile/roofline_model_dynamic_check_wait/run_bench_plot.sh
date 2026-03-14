#!/usr/bin/env bash
# run_bench_plot.sh — measure IAA/LZ4 decompression + distance, then plot roofline.
#
# Produces: out/roofline_dim<DIM>.png
#
# Steps:
#   1. Run roofline_dynamic_wait_bench (IAA + LZ4) for each engine count
#   2. Run profile_roofline (distance calculation)  [skippable with --skip-dist]
#   3. Call plot_roofline_by_dim.py
#
# Usage:
#   ./run_bench_plot.sh [--dim 128|512|2048] [--engines "1 2 4 8"]
#                       [--skip-bench] [--skip-dist]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$(cd "$SCRIPT_DIR/../roofline_distance_calcualtion_decompresison" && pwd)"
CFG_DIR="$ROOT_DIR/scripts/iaa"

# ── Defaults ──────────────────────────────────────────────────────────────────
ENGINES=(1 2 4 8)
CHUNK_128="$((1024*128))"
CHUNK_512="$((1024*256))"
CHUNK_2048="$((1024*256))"
CPU_CORE=8
NUMA_NODE=0

OUT_DIR="$SCRIPT_DIR/out"
DIST_OUT_DIR="$DIST_DIR/out"

SKIP_BENCH=0
SKIP_DIST=0

WARMUP=10
ITERS=50
POOL_SIZE=128
CHUNK_SIZES_CSV="65536,262144,1048576"
N_LIST_CSV="1,2,4,8,16,32"

BASE_DIR="/storage/jykang5/compass_base_query"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engines)     read -ra ENGINES <<< "$2"; shift 2 ;;
    --chunk-128)   CHUNK_128="$2";            shift 2 ;;
    --chunk-512)   CHUNK_512="$2";            shift 2 ;;
    --chunk-2048)  CHUNK_2048="$2";           shift 2 ;;
    --skip-bench)  SKIP_BENCH=1;              shift   ;;
    --skip-dist)   SKIP_DIST=1;               shift   ;;
    --out-dir)     OUT_DIR="$2";              shift 2 ;;
    --base-dir)    BASE_DIR="$2";             shift 2 ;;
    --cpu-core)    CPU_CORE="$2";             shift 2 ;;
    --numa-node)   NUMA_NODE="$2";            shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

BENCH_BIN="$SCRIPT_DIR/roofline_dynamic_wait_bench"
DIST_BIN="$DIST_DIR/profile_roofline"
PLOT_PY="$SCRIPT_DIR/plot_roofline_by_dim.py"
DECOMP_CSV="$OUT_DIR/decomp_latency.csv"
DIST_CSV="$DIST_OUT_DIR/results_dist.csv"

# ── numactl prefix ────────────────────────────────────────────────────────────
RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
  RUN_PREFIX=(numactl --physcpubind="$CPU_CORE" --membind="$NUMA_NODE")
elif command -v taskset >/dev/null 2>&1; then
  RUN_PREFIX=(taskset -c "$CPU_CORE")
fi

run() { "${RUN_PREFIX[@]+"${RUN_PREFIX[@]}"}" "$@"; }

# ── Step 1: decomp bench ──────────────────────────────────────────────────────
if [[ "$SKIP_BENCH" -eq 1 ]]; then
  echo "[1/3] Skipping bench (--skip-bench)"
  if [[ ! -f "$DECOMP_CSV" ]]; then
    echo "ERROR: --skip-bench set but $DECOMP_CSV not found" >&2; exit 1
  fi
else
  echo "[1/3] Running decompression bench (engines: ${ENGINES[*]})"
  mkdir -p "$OUT_DIR"
  rm -f "$DECOMP_CSV"
  append_opt=()
  for engine in "${ENGINES[@]}"; do
    echo "  [engine=$engine] configuring IAA"
    cfg="$CFG_DIR/configure_iaa_user_${engine}.sh"
    if [[ ! -x "$cfg" ]]; then
      echo "ERROR: missing config script: $cfg" >&2; exit 1
    fi
    sudo "$cfg"
    if [[ -e /dev/iax ]]; then
      sudo chown -R jykang5 /dev/iax
    fi

    echo "  [engine=$engine] running bench"
    run "$BENCH_BIN" \
      --out-csv    "$DECOMP_CSV" \
      --engine-count "$engine" \
      "${append_opt[@]}" \
      --warmup     "$WARMUP" \
      --iters      "$ITERS" \
      --pool-size  "$POOL_SIZE" \
      --cpu-core   "$CPU_CORE" \
      --numa-node  "$NUMA_NODE" \
      --chunk-sizes "$CHUNK_SIZES_CSV" \
      --n-list     "$N_LIST_CSV"
    append_opt=(--append)
  done
fi

# ── Step 2: distance profiler ─────────────────────────────────────────────────
if [[ "$SKIP_DIST" -eq 1 ]]; then
  echo "[2/3] Skipping distance profiler (--skip-dist)"
  if [[ ! -f "$DIST_CSV" ]]; then
    echo "ERROR: --skip-dist set but $DIST_CSV not found" >&2; exit 1
  fi
else
  echo "[2/3] Running distance profiler"
  mkdir -p "$DIST_OUT_DIR"
  run "$DIST_BIN" \
    --base-dir  "$BASE_DIR" \
    --out-dir   "$DIST_OUT_DIR" \
    --cpu-core  "$CPU_CORE" \
    --numa-node "$NUMA_NODE"
fi

# ── Step 3: plot ──────────────────────────────────────────────────────────────
echo "[3/3] Plotting roofline"
python3 "$PLOT_PY" \
  --out-dir    "$OUT_DIR" \
  --dist-csv   "$DIST_CSV" \
  --chunk-128  "$CHUNK_128" \
  --chunk-512  "$CHUNK_512" \
  --chunk-2048 "$CHUNK_2048"

echo "Done. Output: $OUT_DIR/roofline_dims.png"
