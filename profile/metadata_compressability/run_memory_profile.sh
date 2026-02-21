#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run full memory breakdown profiling and plotting in one command.

Usage:
  run_memory_profile.sh \
    --hnm-fvecs /path/to/hnm_base.fvecs \
    --hnm-jsonl /path/to/hnm_payloads.jsonl \
    --laion-fvecs /path/to/laion_base.fvecs \
    --laion-jsonl /path/to/laion_payloads.jsonl \
    [--m 32] [--ef-construct 128] \
    [--out-dir .] [--out-txt profiling.txt] [--out-json profiling.json] [--out-plot profiling_breakdown.png]

Notes:
- Requires Python packages: faiss-cpu, numpy, matplotlib
- Produces a combined profiling report for both datasets.
EOF
}

HNM_FVECS=""
HNM_JSONL=""
LAION_FVECS=""
LAION_JSONL=""
M=32
EF_CONSTRUCT=128
OUT_DIR="."
OUT_TXT="profiling.txt"
OUT_JSON="profiling.json"
OUT_PLOT="profiling_breakdown.png"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hnm-fvecs)
      HNM_FVECS="$2"; shift 2 ;;
    --hnm-jsonl)
      HNM_JSONL="$2"; shift 2 ;;
    --laion-fvecs)
      LAION_FVECS="$2"; shift 2 ;;
    --laion-jsonl)
      LAION_JSONL="$2"; shift 2 ;;
    --m)
      M="$2"; shift 2 ;;
    --ef-construct)
      EF_CONSTRUCT="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --out-txt)
      OUT_TXT="$2"; shift 2 ;;
    --out-json)
      OUT_JSON="$2"; shift 2 ;;
    --out-plot)
      OUT_PLOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$HNM_FVECS" || -z "$HNM_JSONL" || -z "$LAION_FVECS" || -z "$LAION_JSONL" ]]; then
  echo "Missing required input paths." >&2
  usage
  exit 1
fi

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="$SCRIPT_DIR/profile_memory_breakdown.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot_memory_breakdown.py"

if [[ ! -f "$PROFILE_SCRIPT" || ! -f "$PLOT_SCRIPT" ]]; then
  echo "Required scripts not found in: $SCRIPT_DIR" >&2
  exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

missing = []
for mod in ["faiss", "numpy", "matplotlib"]:
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)

if missing:
    print("Missing Python modules: " + ", ".join(missing), file=sys.stderr)
    print("Install with: pip install faiss-cpu numpy matplotlib", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$OUT_DIR"
TXT_PATH="$OUT_DIR/$OUT_TXT"
JSON_PATH="$OUT_DIR/$OUT_JSON"
PLOT_PATH="$OUT_DIR/$OUT_PLOT"

echo "[1/2] Running memory profiling..."
python3 "$PROFILE_SCRIPT" \
  --hnm-fvecs "$HNM_FVECS" \
  --hnm-jsonl "$HNM_JSONL" \
  --laion-fvecs "$LAION_FVECS" \
  --laion-jsonl "$LAION_JSONL" \
  --m "$M" \
  --ef-construct "$EF_CONSTRUCT" \
  --out-txt "$TXT_PATH" \
  --out-json "$JSON_PATH"

echo "[2/2] Generating plot..."
python3 "$PLOT_SCRIPT" \
  --input "$JSON_PATH" \
  --output "$PLOT_PATH"

echo "Done."
echo "- Report: $TXT_PATH"
echo "- Data:   $JSON_PATH"
echo "- Plot:   $PLOT_PATH"
