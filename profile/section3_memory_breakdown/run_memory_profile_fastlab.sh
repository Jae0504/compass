#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HNM_FVECS="/fast-lab-share/benchmarks/VectorDB/FILTER/HnM/base.fvecs"
HNM_JSONL="/fast-lab-share/benchmarks/VectorDB/FILTER/HnM/payloads.jsonl"
LAION_FVECS="/fast-lab-share/benchmarks/VectorDB/FILTER/LAION/base.fvecs"
LAION_JSONL="/fast-lab-share/benchmarks/VectorDB/FILTER/LAION/payloads.jsonl"

usage() {
  cat <<'EOF'
Run section3 memory profiling using FAST-LAB FILTER datasets.

Fixed inputs:
  HnM   fvecs: /fast-lab-share/benchmarks/VectorDB/FILTER/HnM/base.fvecs
  HnM   jsonl: /fast-lab-share/benchmarks/VectorDB/FILTER/HnM/payloads.jsonl
  LAION fvecs: /fast-lab-share/benchmarks/VectorDB/FILTER/LAION/base.fvecs
  LAION jsonl: /fast-lab-share/benchmarks/VectorDB/FILTER/LAION/payloads.jsonl

Usage:
  run_memory_profile_fastlab.sh [extra options]

Extra options are forwarded to run_memory_profile.sh, e.g.:
  --max-rows 1000000
  --out-dir /path/to/out
  --out-plot-dir /path/to/plots
  --m 32 --ef-construct 128 --acorn-gamma 12 --acorn-mbeta 64
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for p in "$HNM_FVECS" "$HNM_JSONL" "$LAION_FVECS" "$LAION_JSONL"; do
  if [[ ! -f "$p" ]]; then
    echo "Input file not found: $p" >&2
    exit 1
  fi
done

"$SCRIPT_DIR/run_memory_profile.sh" \
  --hnm-fvecs "$HNM_FVECS" \
  --hnm-jsonl "$HNM_JSONL" \
  --laion-fvecs "$LAION_FVECS" \
  --laion-jsonl "$LAION_JSONL" \
  "$@"

