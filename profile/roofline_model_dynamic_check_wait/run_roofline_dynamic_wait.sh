#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =======================
# User-editable defaults
# =======================
OUT_DIR="$SCRIPT_DIR/out"
NUM_QUERIES=100
K=10

USE_CURRENT_CONFIG=0
CURRENT_ENGINE_COUNT=8
ENGINE_COUNTS=(1 2 4 8)
CHOWN_DEV_IAX=1

CPU_CORE=8
NUMA_NODE=0

WARMUP_ITERS=10
MEASURE_ITERS=50
POOL_SIZE=128
CHUNK_SIZES_CSV="4096,8192,16384,32768,65536,131072,262144,524288,1048576"
N_LIST_CSV="1,2,4,8,16,32,64"

EXACT_GRAPH="/home/jykang5/compass/dataset2/compass_graphs/sift_m128_efc200.bin"
EXACT_QUERY="/home/jykang5/compass/dataset2/compass_base_query/sift1m_query.fvecs"
EXACT_MANIFEST_1PCT="/home/jykang5/compass/dataset2/fid_tb/n_filter_100/sift1m/manifest.json"
EXACT_MANIFEST_10PCT="/home/jykang5/compass/dataset2/fid_tb/n_filter_10/sift1m/manifest.json"
EXACT_FILTER_1PCT="synthetic_id_bucket == 99"
EXACT_FILTER_10PCT="synthetic_id_bucket == 9"
EXACT_EF_1PCT=160
EXACT_EF_10PCT=1800

RANGE_GRAPH="/home/jykang5/compass/dataset2/compass_graphs/laion_m128_efc200.bin"
RANGE_QUERY="/home/jykang5/compass/dataset2/compass_base_query/laion_query.fvecs"
RANGE_MANIFEST="/home/jykang5/compass/dataset2/fid_tb/laion/manifest.json"
RANGE_PAYLOAD="/home/jykang5/compass/dataset2/payloads/laion_payloads.jsonl"
RANGE_FILTER_1PCT="original_width >= 958 AND original_width <= 965"
RANGE_FILTER_10PCT="original_width >= 598 AND original_width <= 769"
RANGE_EF_1PCT=200
RANGE_EF_10PCT=512

MULTI_GRAPH="/home/jykang5/compass/dataset2/compass_graphs/hnm_m128_efc200.bin"
MULTI_QUERY="/home/jykang5/compass/dataset2/compass_base_query/hnm_query.fvecs"
MULTI_MANIFEST="/home/jykang5/compass/dataset2/fid_tb/hnm/manifest.json"
MULTI_PAYLOAD="/home/jykang5/compass/dataset2/payloads/hnm_payloads.jsonl"
MULTI_FILTER_1PCT="department_name == 'Baby Toys/Acc' AND garment_group_name == 'Accessories'"
MULTI_FILTER_10PCT="department_name == 'Blouse' OR garment_group_name == 'Jersey Basic'"
MULTI_EF_1PCT=16
MULTI_EF_10PCT=96

CFG_DIR="$ROOT_DIR/scripts/iaa"

EXACT_BIN="$SCRIPT_DIR/roofline_exact_iaa.run"
RANGE_BIN="$SCRIPT_DIR/roofline_range_iaa.run"
MULTI_BIN="$SCRIPT_DIR/roofline_multi_iaa.run"
BENCH_BIN="$SCRIPT_DIR/roofline_dynamic_wait_bench"
PLOT_PY="$SCRIPT_DIR/plot_dynamic_wait_roofline.py"

run_with_affinity() {
  if [[ ${#RUN_PREFIX[@]} -gt 0 ]]; then
    "${RUN_PREFIX[@]}" "$@"
  else
    "$@"
  fi
}

if [[ "$USE_CURRENT_CONFIG" -eq 1 ]]; then
  ENGINE_COUNTS=("$CURRENT_ENGINE_COUNT")
fi

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
  RUN_PREFIX=(numactl --physcpubind="$CPU_CORE" --membind="$NUMA_NODE")
elif command -v taskset >/dev/null 2>&1; then
  RUN_PREFIX=(taskset -c "$CPU_CORE")
fi

echo "[1/5] Build roofline-local binaries"
make -C "$SCRIPT_DIR" clean all

echo "[2/5] Prepare output directory (single overwrite mode)"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/traversal" "$OUT_DIR/summaries"

DECOMP_CSV="$OUT_DIR/decomp_latency.csv"
append_flag=""

run_traversal_suite() {
  local engine="$1"

  echo "  [traversal] exact_match engine=$engine"
  run_with_affinity "$EXACT_BIN" \
    --dataset-type sift1m \
    --graph "$EXACT_GRAPH" \
    --query "$EXACT_QUERY" \
    --k "$K" \
    --ef "$EXACT_EF_1PCT" \
    --filter "$EXACT_FILTER_1PCT" \
    --fidtb-manifest "$EXACT_MANIFEST_1PCT" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "1pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/exact_1pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/exact_1pct_e${engine}.summary.txt"

  run_with_affinity "$EXACT_BIN" \
    --dataset-type sift1m \
    --graph "$EXACT_GRAPH" \
    --query "$EXACT_QUERY" \
    --k "$K" \
    --ef "$EXACT_EF_10PCT" \
    --filter "$EXACT_FILTER_10PCT" \
    --fidtb-manifest "$EXACT_MANIFEST_10PCT" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "10pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/exact_10pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/exact_10pct_e${engine}.summary.txt"

  echo "  [traversal] range_search engine=$engine"
  run_with_affinity "$RANGE_BIN" \
    --dataset-type laion \
    --graph "$RANGE_GRAPH" \
    --query "$RANGE_QUERY" \
    --payload-jsonl "$RANGE_PAYLOAD" \
    --k "$K" \
    --ef "$RANGE_EF_1PCT" \
    --filter "$RANGE_FILTER_1PCT" \
    --fidtb-manifest "$RANGE_MANIFEST" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "1pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/range_1pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/range_1pct_e${engine}.summary.txt"

  run_with_affinity "$RANGE_BIN" \
    --dataset-type laion \
    --graph "$RANGE_GRAPH" \
    --query "$RANGE_QUERY" \
    --payload-jsonl "$RANGE_PAYLOAD" \
    --k "$K" \
    --ef "$RANGE_EF_10PCT" \
    --filter "$RANGE_FILTER_10PCT" \
    --fidtb-manifest "$RANGE_MANIFEST" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "10pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/range_10pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/range_10pct_e${engine}.summary.txt"

  echo "  [traversal] multiple_exact_match engine=$engine"
  run_with_affinity "$MULTI_BIN" \
    --dataset-type hnm \
    --graph "$MULTI_GRAPH" \
    --query "$MULTI_QUERY" \
    --payload-jsonl "$MULTI_PAYLOAD" \
    --k "$K" \
    --ef "$MULTI_EF_1PCT" \
    --filter "$MULTI_FILTER_1PCT" \
    --fidtb-manifest "$MULTI_MANIFEST" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "1pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/multi_1pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/multi_1pct_e${engine}.summary.txt"

  run_with_affinity "$MULTI_BIN" \
    --dataset-type hnm \
    --graph "$MULTI_GRAPH" \
    --query "$MULTI_QUERY" \
    --payload-jsonl "$MULTI_PAYLOAD" \
    --k "$K" \
    --ef "$MULTI_EF_10PCT" \
    --filter "$MULTI_FILTER_10PCT" \
    --fidtb-manifest "$MULTI_MANIFEST" \
    --num-queries "$NUM_QUERIES" \
    --scenario-tag "10pct_e${engine}" \
    --expansion-metrics-out "$OUT_DIR/traversal/multi_10pct_e${engine}.csv" \
    --summary-out "$OUT_DIR/summaries/multi_10pct_e${engine}.summary.txt"
}

echo "[3/5] Run traversal logging + decomp benchmark"
for engine in "${ENGINE_COUNTS[@]}"; do
  if [[ "$USE_CURRENT_CONFIG" -eq 0 ]]; then
    cfg="$CFG_DIR/configure_iaa_user_${engine}.sh"
    if [[ ! -x "$cfg" ]]; then
      echo "ERROR: missing config script: $cfg" >&2
      exit 1
    fi
    echo "  [engine=$engine] configuring IAA profile via sudo"
    sudo "$cfg"
    if [[ "$CHOWN_DEV_IAX" -eq 1 ]]; then
      if [[ -e /dev/iax ]]; then
        echo "  [engine=$engine] chown /dev/iax to $USER"
        sudo chown -R "$USER:$USER" /dev/iax || true
      else
        echo "  [engine=$engine] /dev/iax not found, skip chown"
      fi
    fi
    sudo chown -R "$USER:$USER" "$OUT_DIR" || true
  else
    echo "  [engine=$engine] using current IAA configuration (USE_CURRENT_CONFIG=1)"
  fi

  run_traversal_suite "$engine"

  echo "  [engine=$engine] running decomp benchmark"
  if [[ -n "$append_flag" ]]; then
    append_opt=(--append)
  else
    append_opt=()
  fi

  run_with_affinity "$BENCH_BIN" \
    --out-csv "$DECOMP_CSV" \
    --engine-count "$engine" \
    "${append_opt[@]}" \
    --warmup "$WARMUP_ITERS" \
    --iters "$MEASURE_ITERS" \
    --pool-size "$POOL_SIZE" \
    --cpu-core "$CPU_CORE" \
    --numa-node "$NUMA_NODE" \
    --chunk-sizes "$CHUNK_SIZES_CSV" \
    --n-list "$N_LIST_CSV"

  append_flag=1
done

echo "[4/5] Plot roofline zones"
python3 "$PLOT_PY" \
  --expansion-glob "$OUT_DIR/traversal/*.csv" \
  --decomp-csv "$DECOMP_CSV" \
  --out-dir "$OUT_DIR"

echo "[5/5] Done"
echo "  Output directory: $OUT_DIR"
echo "  Figures:"
ls -1 "$OUT_DIR"/roofline_wait_check_*d.png
