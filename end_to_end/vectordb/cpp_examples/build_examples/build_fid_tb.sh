#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  build_fid_tb.sh [--base <path>] [--threads <int>] [--dry-run]

Description:
  Builds FID/TB for SIFT1M with:
    - nfilters=256
    - steiner_factor=4
    - ep_factor=4
    - isolated_connection_factor=5

Defaults:
  runner      /home/jykang5/compass/end_to_end/vectordb/cpp_examples/build_examples/build_FID_TB.run
  graph       /home/jykang5/compass/end_to_end/vectordb/dataset/hnsw_graph/sift1m/sift_m128_efc200.bin
  out-dir     /home/jykang5/compass/end_to_end/vectordb/dataset/fid_tb
  benchmark   sift1m

Notes:
  - Final outputs are written under: <out-dir>/<benchmark>
  - For default settings, this is:
    /home/jykang5/compass/end_to_end/vectordb/dataset/fid_tb/sift1m
USAGE
}

RUNNER="/home/jykang5/compass/end_to_end/vectordb/cpp_examples/build_examples/build_FID_TB.run"
GRAPH="/home/jykang5/compass/end_to_end/vectordb/dataset/hnsw_graph/sift1m/sift_m128_efc200.bin"
OUT_DIR="/home/jykang5/compass/end_to_end/vectordb/dataset/fid_tb"
BENCHMARK="sift1m"
DATASET_TYPE="sift"
NFILTERS=256
STEINER_FACTOR=4
EP_FACTOR=4
ISOLATED_CONNECTION_FACTOR=5
THREADS=64
BASE_PATH=""
DRY_RUN=0

detect_base_path() {
  local candidates=(
    "/home/jykang5/compass/end_to_end/vectordb/dataset/sift1m/base.fvecs"
    "/home/jykang5/compass/end_to_end/vectordb/dataset/sift1m/sift_base.fvecs"
    "/fast-lab-share/benchmarks/VectorDB/ANN/sift1m/base.fvecs"
    "/fast-lab-share/benchmarks/VectorDB/ANN/sift1m/sift_base.fvecs"
  )

  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_PATH="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_PATH" ]]; then
  if ! BASE_PATH="$(detect_base_path)"; then
    echo "Could not auto-detect SIFT base vectors (.fvecs)." >&2
    echo "Provide the base path explicitly, for example:" >&2
    echo "  ./build_fid_tb.sh --base /path/to/base.fvecs" >&2
    exit 1
  fi
fi

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found or not executable: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$GRAPH" ]]; then
  echo "Graph file not found: $GRAPH" >&2
  exit 1
fi
if [[ ! -f "$BASE_PATH" ]]; then
  echo "Base vectors file not found: $BASE_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

CMD=(
  "$RUNNER"
  --dataset-type "$DATASET_TYPE"
  --benchmark "$BENCHMARK"
  --graph "$GRAPH"
  --base "$BASE_PATH"
  --out-dir "$OUT_DIR"
  --threads "$THREADS"
  --nfilters "$NFILTERS"
  --steiner-factor "$STEINER_FACTOR"
  --ep-factor "$EP_FACTOR"
  --isolated-connection-factor "$ISOLATED_CONNECTION_FACTOR"
)

echo "Running build_FID_TB with:"
echo "  graph: $GRAPH"
echo "  base: $BASE_PATH"
echo "  out:  $OUT_DIR/$BENCHMARK"
echo "  nfilters=$NFILTERS steiner=$STEINER_FACTOR ep=$EP_FACTOR isolated=$ISOLATED_CONNECTION_FACTOR threads=$THREADS"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry run command:\n'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

echo "Done. Outputs under: $OUT_DIR/$BENCHMARK"

# /home/jykang5/compass/end_to_end/vectordb/cpp_examples/build_examples/build_FID_TB.run \
#   --dataset-type laion \
#   --benchmark laion \
#   --graph /home/jykang5/compass/end_to_end/vectordb/dataset/hnsw_graph/laion/laion_m128_efc200.bin \
#   --base /fast-lab-share/benchmarks/VectorDB/FILTER/LAION/base.fvecs \
#   --payload /fast-lab-share/benchmarks/VectorDB/FILTER/LAION/payloads.jsonl\
#   --out-dir /home/jykang5/compass/end_to_end/vectordb/dataset/fid_tb \
#   --threads 64 \
#   --nfilters 256 \
#   --steiner-factor 4 \
#   --ep-factor 4 \
#   --isolated-connection-factor 5

# /home/jykang5/compass/end_to_end/vectordb/cpp_examples/build_examples/build_FID_TB.run \
#   --dataset-type hnm \
#   --benchmark hnm \
#   --graph /home/jykang5/compass/end_to_end/vectordb/dataset/hnsw_graph/hnm/hnm_m128_efc200.bin \
#   --base /fast-lab-share/benchmarks/VectorDB/FILTER/HnM/base.fvecs\
#   --payload /fast-lab-share/benchmarks/VectorDB/FILTER/HnM/payloads.jsonl \
#   --out-dir /home/jykang5/compass/end_to_end/vectordb/dataset/fid_tb \
#   --threads 64 \
#   --nfilters 256 \
#   --steiner-factor 4 \
#   --ep-factor 4 \
#   --isolated-connection-factor 5
