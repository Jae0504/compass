#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run section3 memory profiling (hnm + laion) and generate one combined PNG plot.

Usage:
  run_memory_profile.sh \
    --hnm-fvecs /path/to/hnm_base.fvecs \
    --hnm-jsonl /path/to/hnm_payloads.jsonl \
    --laion-fvecs /path/to/laion_base.fvecs \
    --laion-jsonl /path/to/laion_payloads.jsonl \
    [--acorn-dir /path/to/ACORN] [--acorn-build-dir /path/to/ACORN/build] \
    [--m 32] [--ef-construct 128] \
    [--acorn-gamma 12] [--acorn-mbeta 64] [--max-rows 0] \
    [--out-dir .] [--out-txt profiling.txt] [--out-json profiling.json] \
    [--out-plot-dir .] [--billion-rows 1000000000]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="$SCRIPT_DIR/../metadata_compressability/dataset"

HNM_FVECS="$DEFAULT_DATA_DIR/hnm_base.fvecs"
HNM_JSONL="$DEFAULT_DATA_DIR/hnm_payloads.jsonl"
LAION_FVECS="$DEFAULT_DATA_DIR/laion_base.fvecs"
LAION_JSONL="$DEFAULT_DATA_DIR/laion_payloads.jsonl"

M=32
EF_CONSTRUCT=128
ACORN_GAMMA=12
ACORN_M_BETA=64
MAX_ROWS=0
ACORN_DIR_ARG=""
ACORN_BUILD_DIR_ARG=""

OUT_DIR="$SCRIPT_DIR"
OUT_TXT="profiling.txt"
OUT_JSON="profiling.json"
OUT_PLOT_DIR="$SCRIPT_DIR"
BILLION_ROWS=1000000000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hnm-fvecs) HNM_FVECS="$2"; shift 2 ;;
    --hnm-jsonl) HNM_JSONL="$2"; shift 2 ;;
    --laion-fvecs) LAION_FVECS="$2"; shift 2 ;;
    --laion-jsonl) LAION_JSONL="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --ef-construct) EF_CONSTRUCT="$2"; shift 2 ;;
    --acorn-gamma) ACORN_GAMMA="$2"; shift 2 ;;
    --acorn-mbeta) ACORN_M_BETA="$2"; shift 2 ;;
    --acorn-dir) ACORN_DIR_ARG="$2"; shift 2 ;;
    --acorn-build-dir) ACORN_BUILD_DIR_ARG="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --out-txt) OUT_TXT="$2"; shift 2 ;;
    --out-json) OUT_JSON="$2"; shift 2 ;;
    --out-plot-dir) OUT_PLOT_DIR="$2"; shift 2 ;;
    --billion-rows) BILLION_ROWS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

for p in "$HNM_FVECS" "$HNM_JSONL" "$LAION_FVECS" "$LAION_JSONL"; do
  if [[ ! -f "$p" ]]; then
    echo "Input file not found: $p" >&2
    exit 1
  fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

missing = []
for mod in ["numpy", "matplotlib"]:
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)

if missing:
    print("Missing Python modules: " + ", ".join(missing), file=sys.stderr)
    print("Install with: pip install numpy matplotlib", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_PLOT_DIR"

resolve_default_acorn_dir() {
  local candidates=(
    "/home/jykang5/compass/ACORN"
    "/home/jykang5/ACORN"
    "/home/jykang5/acorn_build/ACORN"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c/CMakeLists.txt" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

RESOLVED_ACORN_DIR="$ACORN_DIR_ARG"
if [[ -z "$RESOLVED_ACORN_DIR" ]]; then
  if ! RESOLVED_ACORN_DIR="$(resolve_default_acorn_dir)"; then
    echo "Unable to resolve ACORN source. Pass --acorn-dir explicitly." >&2
    exit 1
  fi
fi
RESOLVED_ACORN_BUILD_DIR="$ACORN_BUILD_DIR_ARG"
if [[ -z "$RESOLVED_ACORN_BUILD_DIR" ]]; then
  RESOLVED_ACORN_BUILD_DIR="/home/jykang5/compass/ACORN/build"
fi

echo "[1/3] Bootstrapping ACORN and building C++ profiler..."
"$SCRIPT_DIR/build_acorn.sh" \
  --acorn-dir "$RESOLVED_ACORN_DIR" \
  --acorn-build-dir "$RESOLVED_ACORN_BUILD_DIR"

make -C "$SCRIPT_DIR" \
  ACORN_DIR="$RESOLVED_ACORN_DIR" \
  ACORN_BUILD_DIR="$RESOLVED_ACORN_BUILD_DIR"

TXT_PATH="$OUT_DIR/$OUT_TXT"
JSON_PATH="$OUT_DIR/$OUT_JSON"

echo "[2/3] Running C++ profiling for hnm + laion..."
"$SCRIPT_DIR/profile_section3" \
  --hnm-fvecs "$HNM_FVECS" \
  --hnm-jsonl "$HNM_JSONL" \
  --laion-fvecs "$LAION_FVECS" \
  --laion-jsonl "$LAION_JSONL" \
  --m "$M" \
  --ef-construct "$EF_CONSTRUCT" \
  --acorn-gamma "$ACORN_GAMMA" \
  --acorn-mbeta "$ACORN_M_BETA" \
  --max-rows "$MAX_ROWS" \
  --out-txt "$TXT_PATH" \
  --out-json "$JSON_PATH"

echo "[3/3] Generating combined PNG plot with python3..."
python3 "$SCRIPT_DIR/plot_memory_breakdown_no_hnsw.py" \
  --input "$JSON_PATH" \
  --output-dir "$OUT_PLOT_DIR" \
  --billion-rows "$BILLION_ROWS"

cat <<EOF
Done.
- Report: $TXT_PATH
- JSON:   $JSON_PATH
- Plots:
  - $OUT_PLOT_DIR/hnm_laion_memory_breakdown_no_hnsw.png
EOF
