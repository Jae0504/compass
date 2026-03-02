#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  build_acorn.sh [options]

Description:
  Builds ACORN indices for SIFT1M, LAION, and H&M using build_acorn.run.

  - SIFT1M metadata is synthetic:
    if nfilters=10, first 1/10 vectors -> 0, next 1/10 -> 1, ...
  - LAION metadata comes from payload JSON/JSONL, using selected keys.
  - H&M metadata comes from payload JSON/JSONL, excluding detail_desc and prod_name.

Options:
  --sift-base <path>       Path to SIFT1M base vectors (.fvecs/.bvecs)
  --laion-base <path>      Path to LAION base vectors (.fvecs/.bvecs)
  --laion-payload <path>   Path to LAION payload (.json/.jsonl)
  --hnm-base <path>        Path to H&M base vectors (.fvecs/.bvecs)
  --hnm-payload <path>     Path to H&M payload (.json/.jsonl)
  --out-dir <dir>          Output directory for ACORN indices
  --nfilters <int>         Number of filter groups (default: 10)
  --m <int>                ACORN M (default: 32)
  --mbeta <int>            ACORN M_beta (default: 64)
  --ef-search <int>        ACORN efSearch (default: 64)
  --ef-construction <int>  ACORN efConstruction, -1 keeps default (default: -1)
  --threads <int>          OpenMP threads (default: 64)
  --skip-build             Skip 'make build_acorn.run'
  --dry-run                Print commands only
  -h, --help               Show this help

Examples:
  ./build_acorn.sh \
    --sift-base /path/sift/base.fvecs \
    --laion-base /path/laion/base.fvecs --laion-payload /path/laion/payloads.jsonl \
    --hnm-base /path/hnm/base.fvecs --hnm-payload /path/hnm/payloads.jsonl
USAGE
}

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${THIS_DIR}/build_acorn.run"
OUT_DIR="/home/jykang5/compass/end_to_end/vectordb/dataset/acorn_graph"

NFILTERS=10
M=32
MBETA=64
EF_SEARCH=64
EF_CONSTRUCTION=-1
THREADS=64
SKIP_BUILD=0
DRY_RUN=0

SIFT_BASE=""
LAION_BASE=""
LAION_PAYLOAD=""
HNM_BASE=""
HNM_PAYLOAD=""

detect_sift_base() {
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
    --sift-base) SIFT_BASE="$2"; shift 2 ;;
    --laion-base) LAION_BASE="$2"; shift 2 ;;
    --laion-payload) LAION_PAYLOAD="$2"; shift 2 ;;
    --hnm-base) HNM_BASE="$2"; shift 2 ;;
    --hnm-payload) HNM_PAYLOAD="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --nfilters) NFILTERS="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --mbeta) MBETA="$2"; shift 2 ;;
    --ef-search) EF_SEARCH="$2"; shift 2 ;;
    --ef-construction) EF_CONSTRUCTION="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SIFT_BASE" ]]; then
  if SIFT_BASE="$(detect_sift_base)"; then
    echo "Auto-detected SIFT1M base: $SIFT_BASE"
  else
    echo "SIFT1M base was not auto-detected. Provide --sift-base to enable SIFT build." >&2
  fi
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: make -C \"$THIS_DIR\" build_acorn.run"
  else
    make -C "$THIS_DIR" build_acorn.run
  fi
fi

if [[ "$DRY_RUN" -eq 0 && ! -x "$RUNNER" ]]; then
  echo "Runner not found or not executable: $RUNNER" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"/{sift1m,laion,hnm}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Dry run command:\n'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ -n "$SIFT_BASE" ]]; then
  if [[ "$DRY_RUN" -eq 0 && ! -f "$SIFT_BASE" ]]; then
    echo "SIFT base not found: $SIFT_BASE" >&2
    exit 1
  fi
  run_cmd "$RUNNER" \
    --dataset-type sift1m \
    --base "$SIFT_BASE" \
    --out "$OUT_DIR/sift1m/sift1m_acorn_m${M}_nf${NFILTERS}.index" \
    --nfilters "$NFILTERS" \
    --m "$M" \
    --mbeta "$MBETA" \
    --ef-search "$EF_SEARCH" \
    --ef-construction "$EF_CONSTRUCTION" \
    --threads "$THREADS"
fi

if [[ -n "$LAION_BASE" || -n "$LAION_PAYLOAD" ]]; then
  if [[ -z "$LAION_BASE" || -z "$LAION_PAYLOAD" ]]; then
    echo "For LAION, both --laion-base and --laion-payload are required." >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 0 && ! -f "$LAION_BASE" ]]; then
    echo "LAION base not found: $LAION_BASE" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 0 && ! -f "$LAION_PAYLOAD" ]]; then
    echo "LAION payload not found: $LAION_PAYLOAD" >&2
    exit 1
  fi
  run_cmd "$RUNNER" \
    --dataset-type laion \
    --base "$LAION_BASE" \
    --payload "$LAION_PAYLOAD" \
    --out "$OUT_DIR/laion/laion_acorn_m${M}_nf${NFILTERS}.index" \
    --nfilters "$NFILTERS" \
    --m "$M" \
    --mbeta "$MBETA" \
    --ef-search "$EF_SEARCH" \
    --ef-construction "$EF_CONSTRUCTION" \
    --threads "$THREADS"
fi

if [[ -n "$HNM_BASE" || -n "$HNM_PAYLOAD" ]]; then
  if [[ -z "$HNM_BASE" || -z "$HNM_PAYLOAD" ]]; then
    echo "For H&M, both --hnm-base and --hnm-payload are required." >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 0 && ! -f "$HNM_BASE" ]]; then
    echo "H&M base not found: $HNM_BASE" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 0 && ! -f "$HNM_PAYLOAD" ]]; then
    echo "H&M payload not found: $HNM_PAYLOAD" >&2
    exit 1
  fi
  run_cmd "$RUNNER" \
    --dataset-type hnm \
    --base "$HNM_BASE" \
    --payload "$HNM_PAYLOAD" \
    --out "$OUT_DIR/hnm/hnm_acorn_m${M}_nf${NFILTERS}.index" \
    --nfilters "$NFILTERS" \
    --m "$M" \
    --mbeta "$MBETA" \
    --ef-search "$EF_SEARCH" \
    --ef-construction "$EF_CONSTRUCTION" \
    --threads "$THREADS"
fi

echo "Done."
