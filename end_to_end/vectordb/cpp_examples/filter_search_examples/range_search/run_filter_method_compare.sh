#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_PY="$(cd "$SCRIPT_DIR/.." && pwd)/plot_qps_recall.py"

usage() {
  cat <<'USAGE'
Usage:
  run_filter_method_compare.sh \
    [--dataset laion] \
    [--k <int>] \
    [--ef-list <comma-separated>] \
    [--ef-list-1pct <comma-separated>] \
    [--ef-list-10pct <comma-separated>] \
    [--num-queries <int>] \
    [--out-dir <path>] \
    [--postfilter-max-candidates <int>] \
    [--postfilter-max-candidates-list <comma-separated>] \
    [--iaa-engine-profiles <comma-separated 1|2|4|8>] \
    [--skip-iaa-config] \
    [--build|--no-build] \
    [--plot|--no-plot]

Core defaults:
  --dataset                     laion
  --k                           10
  --ef-list                     64,96,128,160,200
  --ef-list-1pct                (inherits --ef-list by default)
  --ef-list-10pct               (inherits --ef-list by default)
  --num-queries                 20
  --out-dir                     <this_dir>/out/filter_method_compare
  --clean-out-dir               enabled (keeps only current run files under out-dir)
  --postfilter-max-candidates   3000 (in_search_filter_hnsw)
  --postfilter-max-candidates-list 500,1000,1500,2000,2500 (post_filter_hnsw sweep)
  --fid-block-size-bytes        65536
  --tb-block-size-bytes         1048576
  --manifest                    /storage/jykang5/fid_tb/laion/manifest.json
  --iaa-engine-profiles         1,2,4,8
  --skip-iaa-config             disabled
  --build                       enabled
  --plot                        enabled

Range defaults:
  --filter-expr-1pct            "230 <= original_height <= 236"
  --filter-expr-10pct           "290 <= original_height <= 329"

Advanced overrides:
  --graph <path>
  --query <path>
  --manifest <path>
  --manifest-1pct <path>
  --manifest-10pct <path>
  --acorn-index <path>
  --acorn-index-1pct <path>
  --acorn-index-10pct <path>
  --payload-jsonl <path>
  --iaa-engine-profiles <csv 1|2|4|8>
  --skip-iaa-config
  --iaa-config-dir <path>
  --iaa-device-owner <user>
  --fid-block-size-bytes <int>
  --tb-block-size-bytes <int>

Output files:
  <out-dir>/results_1pct.csv
  <out-dir>/results_10pct.csv
  <out-dir>/results_merged.csv
  <out-dir>/qps_vs_recall.png   (when --plot)
USAGE
}

is_numeric() {
  local v="$1"
  [[ "$v" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]
}

extract_summary_value() {
  local key="$1"
  local summary_path="$2"
  awk -F': ' -v k="$key" '$1 == k {print $2; exit}' "$summary_path"
}

sanitize_csv_field() {
  local v="$1"
  v="${v//$'\n'/ }"
  v="${v//$'\r'/ }"
  v="${v//,/;}"
  printf '%s' "$v"
}

ensure_readable_file() {
  local p="$1"
  local msg="$2"
  if [[ ! -f "$p" ]]; then
    echo "Error: $msg ($p)" >&2
    exit 1
  fi
}

ensure_executable_file() {
  local p="$1"
  local msg="$2"
  if [[ ! -x "$p" ]]; then
    echo "Error: $msg ($p)" >&2
    exit 1
  fi
}

first_existing_path_or_empty() {
  local p=""
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      printf '%s' "$p"
      return 0
    fi
  done
  printf ''
}

normalize_range_expr() {
  local expr="$1"
  python3 - "$expr" <<'PY'
import re
import sys

expr = sys.argv[1].strip()

pat = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(<=|<)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*$'
)
m = pat.match(expr)
if m:
    lo, lo_op, field, hi_op, hi = m.groups()
    lo_cmp = '>=' if lo_op == '<=' else '>'
    hi_cmp = '<=' if hi_op == '<=' else '<'
    print(f"{field} {lo_cmp} {lo} AND {field} {hi_cmp} {hi}")
else:
    print(expr)
PY
}

compute_range_bucket_mapping() {
  local manifest_path="$1"
  local normalized_expr="$2"
  python3 - "$manifest_path" "$normalized_expr" <<'PY'
import json
import math
import os
import re
import struct
import sys

manifest_path = sys.argv[1]
expr = sys.argv[2].strip()


def fail(msg: str) -> None:
    print("MAP_STATUS=ERROR")
    print(f"MAP_DETAIL={msg}")
    sys.exit(1)


def parse_expr(s: str):
    m = re.fullmatch(
        r'\s*([A-Za-z_][A-Za-z0-9_]*)\s+BETWEEN\s+([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s+AND\s+([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*',
        s,
        re.IGNORECASE,
    )
    if m:
        field, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
        return {
            "field": field,
            "low": lo,
            "high": hi,
            "low_inclusive": True,
            "high_inclusive": True,
        }

    parts = re.split(r'\bAND\b', s, flags=re.IGNORECASE)
    if len(parts) != 2:
        fail("expression must be BETWEEN or AND-combined range compares")

    cmp_pat = re.compile(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*$'
    )
    parsed = []
    for p in parts:
        m2 = cmp_pat.fullmatch(p.strip())
        if not m2:
            fail(f"cannot parse compare clause: {p.strip()}")
        parsed.append((m2.group(1), m2.group(2), float(m2.group(3))))

    if parsed[0][0] != parsed[1][0]:
        fail("range compares must use the same field")

    field = parsed[0][0]
    low = -math.inf
    high = math.inf
    low_inclusive = True
    high_inclusive = True

    for _, op, v in parsed:
        if op == '>':
            if v > low:
                low = v
                low_inclusive = False
            elif v == low:
                low_inclusive = False
        elif op == '>=':
            if v > low:
                low = v
                low_inclusive = True
            elif v == low:
                low_inclusive = low_inclusive and True
        elif op == '<':
            if v < high:
                high = v
                high_inclusive = False
            elif v == high:
                high_inclusive = False
        elif op == '<=':
            if v < high:
                high = v
                high_inclusive = True
            elif v == high:
                high_inclusive = high_inclusive and True

    return {
        "field": field,
        "low": low,
        "high": high,
        "low_inclusive": low_inclusive,
        "high_inclusive": high_inclusive,
    }


def bucket_from_builder(usable_bins: int, min_v: float, max_v: float, x: float) -> int:
    if usable_bins <= 1 or max_v <= min_v:
        return 0
    normalized = (x - min_v) / (max_v - min_v)
    idx = int(normalized * usable_bins)
    if idx < 0:
        idx = 0
    if idx >= usable_bins:
        idx = usable_bins - 1
    return idx

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

spec = parse_expr(expr)
field = spec["field"]

attrs = manifest.get("attributes", [])
attr = next((a for a in attrs if a.get("key") == field), None)
if attr is None:
    fail(f"field not found in manifest: {field}")
if not attr.get("numeric", False) or attr.get("encoding") != "numeric_minmax_quantized":
    fail(f"field is not numeric_minmax_quantized: {field}")

nfilters = int(manifest.get("nfilters", 0))
if nfilters <= 0:
    fail(f"invalid nfilters: {nfilters}")
missing_bucket = nfilters - 1
usable_bins = max(1, missing_bucket)

min_value = float(attr.get("min_value", 0.0))
max_value = float(attr.get("max_value", 0.0))
bucket_size = 0.0
if usable_bins > 0 and max_value > min_value:
    bucket_size = (max_value - min_value) / float(usable_bins)

bucket_low = 0
bucket_high = usable_bins - 1
if math.isfinite(spec["low"]):
    b = bucket_from_builder(usable_bins, min_value, max_value, spec["low"])
    bucket_low = b if spec["low_inclusive"] else min(usable_bins, b + 1)
if math.isfinite(spec["high"]):
    b = bucket_from_builder(usable_bins, min_value, max_value, spec["high"])
    bucket_high = b if spec["high_inclusive"] else max(-1, b - 1)

empty_result = bucket_low > bucket_high
selected_bucket_count = 0 if empty_result else (bucket_high - bucket_low + 1)

fid_file = attr.get("fid_file", "")
if not os.path.isabs(fid_file):
    fid_file = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), fid_file)

fid_bucket_match_count = -1
fid_missing_count = -1
fid_elements = int(manifest.get("n_elements", 0))
if fid_file and os.path.isfile(fid_file):
    with open(fid_file, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            fail(f"failed to read FID header: {fid_file}")
        count = struct.unpack("Q", hdr)[0]
        payload = f.read(count)
        if len(payload) != count:
            fail(f"failed to read full FID payload: {fid_file}")
    fid_elements = count
    if empty_result:
        fid_bucket_match_count = 0
    else:
        fid_bucket_match_count = sum(1 for v in payload if bucket_low <= v <= bucket_high)
    fid_missing_count = sum(1 for v in payload if v == missing_bucket)

print("MAP_STATUS=OK")
print(f"MAP_FIELD={field}")
print(f"MAP_NFILTERS={nfilters}")
print(f"MAP_MISSING_BUCKET={missing_bucket}")
print(f"MAP_USABLE_BINS={usable_bins}")
print(f"MAP_MIN_VALUE={min_value}")
print(f"MAP_MAX_VALUE={max_value}")
print(f"MAP_BUCKET_SIZE={bucket_size}")
print(f"MAP_LOW={spec['low'] if math.isfinite(spec['low']) else ''}")
print(f"MAP_HIGH={spec['high'] if math.isfinite(spec['high']) else ''}")
print(f"MAP_LOW_INCLUSIVE={1 if spec['low_inclusive'] else 0}")
print(f"MAP_HIGH_INCLUSIVE={1 if spec['high_inclusive'] else 0}")
print(f"MAP_BUCKET_LOW={bucket_low}")
print(f"MAP_BUCKET_HIGH={bucket_high}")
print(f"MAP_SELECTED_BUCKET_COUNT={selected_bucket_count}")
print(f"MAP_EMPTY_RESULT={1 if empty_result else 0}")
print(f"MAP_FID_ELEMENTS={fid_elements}")
print(f"MAP_FID_BUCKET_MATCH_COUNT={fid_bucket_match_count}")
print(f"MAP_FID_MISSING_COUNT={fid_missing_count}")
print(f"MAP_FID_FILE={fid_file}")
PY
}

parse_map_output() {
  local map_out="$1"
  local prefix="$2"
  local k=""
  local v=""

  while IFS='=' read -r k v; do
    [[ -z "$k" ]] && continue
    case "$k" in
      MAP_STATUS|MAP_DETAIL|MAP_FIELD|MAP_NFILTERS|MAP_MISSING_BUCKET|MAP_USABLE_BINS|MAP_MIN_VALUE|MAP_MAX_VALUE|MAP_BUCKET_SIZE|MAP_LOW|MAP_HIGH|MAP_LOW_INCLUSIVE|MAP_HIGH_INCLUSIVE|MAP_BUCKET_LOW|MAP_BUCKET_HIGH|MAP_SELECTED_BUCKET_COUNT|MAP_EMPTY_RESULT|MAP_FID_ELEMENTS|MAP_FID_BUCKET_MATCH_COUNT|MAP_FID_MISSING_COUNT|MAP_FID_FILE)
        printf -v "${prefix}_${k#MAP_}" '%s' "$v"
        ;;
    esac
  done <<< "$map_out"
}

parse_positive_int_list() {
  local raw_list="$1"
  local out_array_name="$2"
  local field_name="$3"

  local -n out_ref="$out_array_name"
  out_ref=()

  local -a raw_values=()
  IFS=',' read -r -a raw_values <<< "$raw_list"

  local item=""
  for item in "${raw_values[@]}"; do
    item="$(echo "$item" | tr -d '[:space:]')"
    [[ -z "$item" ]] && continue
    if ! [[ "$item" =~ ^[0-9]+$ ]] || [[ "$item" -le 0 ]]; then
      echo "Error: invalid value '$item' in $field_name" >&2
      exit 1
    fi
    out_ref+=("$item")
  done

  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    echo "Error: $field_name produced no valid values" >&2
    exit 1
  fi
}

parse_iaa_engine_profiles() {
  local raw_list="$1"
  local out_array_name="$2"
  local field_name="$3"

  local -n out_ref="$out_array_name"
  out_ref=()

  local -a raw_values=()
  IFS=',' read -r -a raw_values <<< "$raw_list"

  local item=""
  for item in "${raw_values[@]}"; do
    item="$(echo "$item" | tr -d '[:space:]')"
    [[ -z "$item" ]] && continue
    case "$item" in
      1|2|4|8)
        ;;
      *)
        echo "Error: invalid engine profile '$item' in $field_name (allowed: 1,2,4,8)" >&2
        exit 1
        ;;
    esac
    if [[ ! " ${out_ref[*]} " =~ [[:space:]]${item}[[:space:]] ]]; then
      out_ref+=("$item")
    fi
  done

  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    echo "Error: $field_name produced no valid engine profiles" >&2
    exit 1
  fi
}

# Methods to run.
# Comment out entries here for quick manual method selection.
METHODS=(
  "post_filter_hnsw"
  "in_search_filter_hnsw"
  "acorn"
  "compass_lz4"
  "compass_iaa_1"
  "compass_iaa_2"
  "compass_iaa_4"
  "compass_iaa_8"
)

method_requires_manifest() {
  local method="$1"
  case "$method" in
    compass_lz4|compass_iaa|compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

method_requires_iaa_profile() {
  local method="$1"
  case "$method" in
    compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

DATASET="laion"
K=10
EF_LIST="64,96,128,160,200"
EF_LIST_1PCT=""
EF_LIST_10PCT=""
EF_LIST_SET_BY_USER=0
NUM_QUERIES=10
OUT_DIR="$SCRIPT_DIR/out/filter_method_compare"
CLEAN_OUT_DIR=1
POSTFILTER_MAX_CANDIDATES=3000
POSTFILTER_MAX_CANDIDATES_SET_BY_USER=0
POSTFILTER_MAX_CANDIDATES_LIST="500,1000,1500,2000,2500"
POSTFILTER_MAX_CANDIDATES_LIST_SET_BY_USER=0
DO_BUILD=1
DO_PLOT=1

FILTER_EXPR_1PCT='958 <= original_width <= 965'
FILTER_EXPR_10PCT='598 <= original_width <= 769'

GRAPH_PATH="/storage/jykang5/compass_graphs/laion_m128_efc200.bin"
QUERY_PATH="/storage/jykang5/compass_base_query/laion_query.fvecs"
MANIFEST="/storage/jykang5/fid_tb/laion/manifest.json"
MANIFEST_1PCT=""
MANIFEST_10PCT=""
ACORN_INDEX=""
ACORN_INDEX_1PCT=""
ACORN_INDEX_10PCT=""
PAYLOAD_JSONL="/fast-lab-share/benchmarks/VectorDB/FILTER/LAION/payloads.jsonl"
FID_BLOCK_SIZE_BYTES="$((1024*128))"
TB_BLOCK_SIZE_BYTES="$((1024*256))"
IAA_ENGINE_PROFILES="1,2,4,8"
SKIP_IAA_CONFIG=0
IAA_CONFIG_DIR="/home/jykang5/compass/scripts/iaa"
IAA_DEVICE_OWNER="jykang5"
CURRENT_IAA_ENGINE_PROFILE=""
SUDO_KEEPALIVE_PID=""
IAA_ENGINE_VALUES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --ef-list)
      EF_LIST="$2"
      EF_LIST_SET_BY_USER=1
      shift 2
      ;;
    --ef-list-1pct)
      EF_LIST_1PCT="$2"
      shift 2
      ;;
    --ef-list-10pct)
      EF_LIST_10PCT="$2"
      shift 2
      ;;
    --num-queries)
      NUM_QUERIES="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --clean-out-dir)
      CLEAN_OUT_DIR=1
      shift
      ;;
    --no-clean-out-dir)
      CLEAN_OUT_DIR=0
      shift
      ;;
    --postfilter-max-candidates)
      POSTFILTER_MAX_CANDIDATES="$2"
      POSTFILTER_MAX_CANDIDATES_SET_BY_USER=1
      shift 2
      ;;
    --postfilter-max-candidates-list)
      POSTFILTER_MAX_CANDIDATES_LIST="$2"
      POSTFILTER_MAX_CANDIDATES_LIST_SET_BY_USER=1
      shift 2
      ;;
    --iaa-engine-profiles)
      IAA_ENGINE_PROFILES="$2"
      shift 2
      ;;
    --skip-iaa-config)
      SKIP_IAA_CONFIG=1
      shift
      ;;
    --no-skip-iaa-config)
      SKIP_IAA_CONFIG=0
      shift
      ;;
    --build)
      DO_BUILD=1
      shift
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --plot)
      DO_PLOT=1
      shift
      ;;
    --no-plot)
      DO_PLOT=0
      shift
      ;;
    --filter-expr-1pct)
      FILTER_EXPR_1PCT="$2"
      shift 2
      ;;
    --filter-expr-10pct)
      FILTER_EXPR_10PCT="$2"
      shift 2
      ;;
    --graph)
      GRAPH_PATH="$2"
      shift 2
      ;;
    --query)
      QUERY_PATH="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --manifest-1pct)
      MANIFEST_1PCT="$2"
      shift 2
      ;;
    --manifest-10pct)
      MANIFEST_10PCT="$2"
      shift 2
      ;;
    --acorn-index)
      ACORN_INDEX="$2"
      shift 2
      ;;
    --acorn-index-1pct)
      ACORN_INDEX_1PCT="$2"
      shift 2
      ;;
    --acorn-index-10pct)
      ACORN_INDEX_10PCT="$2"
      shift 2
      ;;
    --payload-jsonl)
      PAYLOAD_JSONL="$2"
      shift 2
      ;;
    --iaa-config-dir)
      IAA_CONFIG_DIR="$2"
      shift 2
      ;;
    --iaa-device-owner)
      IAA_DEVICE_OWNER="$2"
      shift 2
      ;;
    --fid-block-size-bytes)
      FID_BLOCK_SIZE_BYTES="$2"
      shift 2
      ;;
    --tb-block-size-bytes)
      TB_BLOCK_SIZE_BYTES="$2"
      shift 2
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

if [[ "$DATASET" != "laion" ]]; then
  echo "Error: this script supports only --dataset laion" >&2
  exit 1
fi
if ! [[ "$K" =~ ^[0-9]+$ ]] || [[ "$K" -le 0 ]]; then
  echo "Error: --k must be a positive integer" >&2
  exit 1
fi
if ! [[ "$NUM_QUERIES" =~ ^[0-9]+$ ]] || [[ "$NUM_QUERIES" -le 0 ]]; then
  echo "Error: --num-queries must be a positive integer" >&2
  exit 1
fi
if ! [[ "$POSTFILTER_MAX_CANDIDATES" =~ ^[0-9]+$ ]] || [[ "$POSTFILTER_MAX_CANDIDATES" -le 0 ]]; then
  echo "Error: --postfilter-max-candidates must be a positive integer" >&2
  exit 1
fi
if [[ "$POSTFILTER_MAX_CANDIDATES" -lt "$K" ]]; then
  echo "Error: --postfilter-max-candidates must be >= --k" >&2
  exit 1
fi
if [[ "$POSTFILTER_MAX_CANDIDATES_SET_BY_USER" -eq 1 && "$POSTFILTER_MAX_CANDIDATES_LIST_SET_BY_USER" -eq 0 ]]; then
  POSTFILTER_MAX_CANDIDATES_LIST="$POSTFILTER_MAX_CANDIDATES"
fi
if ! [[ "$FID_BLOCK_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$FID_BLOCK_SIZE_BYTES" -le 0 ]]; then
  echo "Error: --fid-block-size-bytes must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TB_BLOCK_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$TB_BLOCK_SIZE_BYTES" -le 0 ]]; then
  echo "Error: --tb-block-size-bytes must be a positive integer" >&2
  exit 1
fi
if [[ "$SKIP_IAA_CONFIG" -ne 0 && "$SKIP_IAA_CONFIG" -ne 1 ]]; then
  echo "Error: invalid --skip-iaa-config state" >&2
  exit 1
fi
if [[ "$CLEAN_OUT_DIR" -ne 0 && "$CLEAN_OUT_DIR" -ne 1 ]]; then
  echo "Error: invalid --clean-out-dir state" >&2
  exit 1
fi

parse_iaa_engine_profiles "$IAA_ENGINE_PROFILES" IAA_ENGINE_VALUES "--iaa-engine-profiles"

# Optional profile filter: keep non-IAA methods as-is, and keep only selected
# IAA engine variants among explicitly listed METHODS above.
FILTERED_METHODS=()
for method in "${METHODS[@]}"; do
  case "$method" in
    compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      method_profile="${method##compass_iaa_}"
      if [[ " ${IAA_ENGINE_VALUES[*]} " =~ (^|[[:space:]])${method_profile}([[:space:]]|$) ]]; then
        FILTERED_METHODS+=("$method")
      fi
      ;;
    *)
      FILTERED_METHODS+=("$method")
      ;;
  esac
done
METHODS=("${FILTERED_METHODS[@]}")

REQUIRE_MANIFEST=0
REQUIRE_IAA_PROFILE=0
for method in "${METHODS[@]}"; do
  if method_requires_manifest "$method"; then
    REQUIRE_MANIFEST=1
  fi
  if method_requires_iaa_profile "$method"; then
    REQUIRE_IAA_PROFILE=1
  fi
done

if [[ -z "$MANIFEST_1PCT" ]]; then
  MANIFEST_1PCT="$MANIFEST"
fi
if [[ -z "$MANIFEST_10PCT" ]]; then
  MANIFEST_10PCT="$MANIFEST"
fi
if [[ -z "$MANIFEST_1PCT" ]]; then
  MANIFEST_1PCT="$(first_existing_path_or_empty \
    /storage/jykang5/fid_tb/n_filter_100/laion/manifest.json \
    /fast-lab-share/jykang5/fid_tb/n_filter_100/laion/manifest.json)"
fi
if [[ -z "$MANIFEST_10PCT" ]]; then
  MANIFEST_10PCT="$(first_existing_path_or_empty \
    /storage/jykang5/fid_tb/n_filter_10/laion/manifest.json \
    /fast-lab-share/jykang5/fid_tb/n_filter_10/laion/manifest.json)"
fi
if [[ "$REQUIRE_MANIFEST" -eq 1 && ( -z "$MANIFEST_1PCT" || -z "$MANIFEST_10PCT" ) ]]; then
  echo "Error: active methods require manifests; provide --manifest or --manifest-1pct/--manifest-10pct" >&2
  exit 1
fi
if [[ -z "$ACORN_INDEX_1PCT" ]]; then
  ACORN_INDEX_1PCT="$ACORN_INDEX"
fi
if [[ -z "$ACORN_INDEX_10PCT" ]]; then
  ACORN_INDEX_10PCT="$ACORN_INDEX"
fi
if [[ -z "$ACORN_INDEX_10PCT" || ! -f "$ACORN_INDEX_10PCT" ]]; then
  ACORN_INDEX_10PCT="$(first_existing_path_or_empty \
    /storage/jykang5/compass_graphs/acorn/laion_acorn_m64_nf10.index \
    /storage/jykang5/compass_graphs/acorn/laion_acorn_m32_nf10.index \
    /fast-lab-share/jykang5/compass_graphs/acorn/laion_acorn_m64_nf10.index \
    /fast-lab-share/jykang5/compass_graphs/acorn/laion_acorn_m32_nf10.index)"
fi
if [[ -z "$ACORN_INDEX_1PCT" || ! -f "$ACORN_INDEX_1PCT" ]]; then
  ACORN_INDEX_1PCT="$(first_existing_path_or_empty \
    /storage/jykang5/compass_graphs/acorn/laion_acorn_m64_nf100.index \
    /storage/jykang5/compass_graphs/acorn/laion_acorn_m32_nf100.index \
    /fast-lab-share/jykang5/compass_graphs/acorn/laion_acorn_m64_nf100.index \
    /fast-lab-share/jykang5/compass_graphs/acorn/laion_acorn_m32_nf100.index)"
fi
if [[ -z "$ACORN_INDEX_10PCT" ]]; then
  ACORN_INDEX_10PCT="$ACORN_INDEX_1PCT"
fi

if [[ -z "$EF_LIST_1PCT" ]]; then
  EF_LIST_1PCT="$EF_LIST"
fi
if [[ -z "$EF_LIST_10PCT" ]]; then
  EF_LIST_10PCT="$EF_LIST"
fi

parse_positive_int_list "$EF_LIST_1PCT" EF_VALUES_1PCT "--ef-list-1pct"
parse_positive_int_list "$EF_LIST_10PCT" EF_VALUES_10PCT "--ef-list-10pct"
parse_positive_int_list "$POSTFILTER_MAX_CANDIDATES_LIST" POSTFILTER_MAX_VALUES "--postfilter-max-candidates-list"

for max_candidates in "${POSTFILTER_MAX_VALUES[@]}"; do
  if [[ "$max_candidates" -lt "$K" ]]; then
    echo "Error: every --postfilter-max-candidates-list value must be >= --k (found: $max_candidates)" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUT_DIR" -eq 1 ]]; then
  rm -rf "$OUT_DIR/summaries"
  rm -f "$OUT_DIR/results_1pct.csv" \
        "$OUT_DIR/results_10pct.csv" \
        "$OUT_DIR/results_merged.csv" \
        "$OUT_DIR/qps_vs_recall.png" \
        "$OUT_DIR/qps_vs_recall.svg"
fi

HNSW_RUN="$SCRIPT_DIR/hnswlib_filter_search.run"
LZ4_RUN="$SCRIPT_DIR/compass_search_w_lz4_only_tb.run"
IAA_RUN="$SCRIPT_DIR/compass_search_w_iaa_only_tb.run"
ACORN_RUN="$SCRIPT_DIR/acorn_search.run"

if [[ "$DO_BUILD" -eq 1 ]]; then
  MAKE_TARGETS=()
  for method in "${METHODS[@]}"; do
    case "$method" in
      post_filter_hnsw|in_search_filter_hnsw)
        MAKE_TARGETS+=("hnswlib_filter_search.run")
        ;;
      compass_lz4)
        MAKE_TARGETS+=("compass_search_w_lz4_only_tb.run")
        ;;
      compass_iaa|compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
        MAKE_TARGETS+=("compass_search_w_iaa_only_tb.run")
        ;;
      acorn)
        MAKE_TARGETS+=("acorn_search.run")
        ;;
      *)
        echo "Error: unknown method '$method' in METHODS list" >&2
        exit 1
        ;;
    esac
  done
  # shellcheck disable=SC2207
  MAKE_TARGETS=($(printf '%s\n' "${MAKE_TARGETS[@]}" | awk '!seen[$0]++'))
  make -B -C "$SCRIPT_DIR" "${MAKE_TARGETS[@]}"
fi

for method in "${METHODS[@]}"; do
  case "$method" in
    post_filter_hnsw|in_search_filter_hnsw)
      ensure_executable_file "$HNSW_RUN" "hnswlib_filter_search.run is missing or not executable"
      ;;
    compass_lz4)
      ensure_executable_file "$LZ4_RUN" "compass_search_w_lz4_only_tb.run is missing or not executable"
      ;;
    compass_iaa|compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      ensure_executable_file "$IAA_RUN" "compass_search_w_iaa_only_tb.run is missing or not executable"
      ;;
    acorn)
      ensure_executable_file "$ACORN_RUN" "acorn_search.run is missing or not executable"
      ;;
    *)
      echo "Error: unknown method '$method' in METHODS list" >&2
      exit 1
      ;;
  esac
done
ensure_readable_file "$GRAPH_PATH" "graph file not found"
ensure_readable_file "$QUERY_PATH" "query file not found"
if [[ "$REQUIRE_MANIFEST" -eq 1 ]]; then
  ensure_readable_file "$MANIFEST_1PCT" "1% manifest file not found"
  ensure_readable_file "$MANIFEST_10PCT" "10% manifest file not found"
fi
ensure_readable_file "$ACORN_INDEX_1PCT" "ACORN 1% index file not found"
ensure_readable_file "$ACORN_INDEX_10PCT" "ACORN 10% index file not found"
ensure_readable_file "$PAYLOAD_JSONL" "payload JSONL file not found"

if [[ "$REQUIRE_IAA_PROFILE" -eq 1 && "$SKIP_IAA_CONFIG" -eq 0 ]]; then
  for iaa_engines in 1 2 4 8; do
    ensure_readable_file \
      "$IAA_CONFIG_DIR/configure_iaa_user_${iaa_engines}.sh" \
      "IAA config script not found for ${iaa_engines} engine(s)"
  done
fi

configure_iaa_profile() {
  local engine_count="$1"
  if [[ "$SKIP_IAA_CONFIG" -eq 1 ]]; then
    return
  fi
  local cfg_script="$IAA_CONFIG_DIR/configure_iaa_user_${engine_count}.sh"
  if [[ "$CURRENT_IAA_ENGINE_PROFILE" == "$engine_count" ]]; then
    return
  fi

  echo "  [IAA] configure ${engine_count} engine(s): $cfg_script"
  if ! sudo bash "$cfg_script"; then
    echo "Error: failed to run IAA config script: $cfg_script" >&2
    exit 1
  fi
  if ! sudo chown -R "$IAA_DEVICE_OWNER" /dev/iax; then
    echo "Error: failed to change /dev/iax ownership to '$IAA_DEVICE_OWNER'" >&2
    exit 1
  fi
  CURRENT_IAA_ENGINE_PROFILE="$engine_count"
}

start_sudo_keepalive() {
  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    return
  fi
  sudo -v
  (
    while true; do
      sudo -n true
      sleep 50
    done
  ) >/dev/null 2>&1 &
  SUDO_KEEPALIVE_PID=$!
}

stop_sudo_keepalive() {
  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
    SUDO_KEEPALIVE_PID=""
  fi
}

trap stop_sudo_keepalive EXIT
if [[ "$REQUIRE_IAA_PROFILE" -eq 1 && "$SKIP_IAA_CONFIG" -eq 0 ]]; then
  start_sudo_keepalive
fi

NORM_FILTER_1PCT="$(normalize_range_expr "$FILTER_EXPR_1PCT")"
NORM_FILTER_10PCT="$(normalize_range_expr "$FILTER_EXPR_10PCT")"

MAP_OUT_1PCT=""
MAP_OUT_10PCT=""
if [[ -n "$MANIFEST_1PCT" && -f "$MANIFEST_1PCT" && -n "$MANIFEST_10PCT" && -f "$MANIFEST_10PCT" ]]; then
  MAP_OUT_1PCT="$(compute_range_bucket_mapping "$MANIFEST_1PCT" "$NORM_FILTER_1PCT")"
  MAP_OUT_10PCT="$(compute_range_bucket_mapping "$MANIFEST_10PCT" "$NORM_FILTER_10PCT")"

  parse_map_output "$MAP_OUT_1PCT" "MAP1"
  parse_map_output "$MAP_OUT_10PCT" "MAP10"

  if [[ "${MAP1_STATUS:-}" != "OK" ]]; then
    echo "Error: failed to map 1% range expression to buckets" >&2
    echo "$MAP_OUT_1PCT" >&2
    exit 1
  fi
  if [[ "${MAP10_STATUS:-}" != "OK" ]]; then
    echo "Error: failed to map 10% range expression to buckets" >&2
    echo "$MAP_OUT_10PCT" >&2
    exit 1
  fi

  echo "[LAION_RANGE_MAP] 1%"
  echo "$MAP_OUT_1PCT"
  echo "[LAION_RANGE_MAP] 10%"
  echo "$MAP_OUT_10PCT"
else
  MAP1_FIELD=""
  MAP1_BUCKET_LOW=""
  MAP1_BUCKET_HIGH=""
  MAP1_BUCKET_SIZE=""
  MAP1_SELECTED_BUCKET_COUNT=""
  MAP1_FID_BUCKET_MATCH_COUNT=""
  MAP1_FID_MISSING_COUNT=""
  MAP10_FIELD=""
  MAP10_BUCKET_LOW=""
  MAP10_BUCKET_HIGH=""
  MAP10_BUCKET_SIZE=""
  MAP10_SELECTED_BUCKET_COUNT=""
  MAP10_FID_BUCKET_MATCH_COUNT=""
  MAP10_FID_MISSING_COUNT=""
  echo "[LAION_RANGE_MAP] Skipped: manifests not available; proceeding with expression-only filtering"
fi

RESULTS_1PCT="$OUT_DIR/results_1pct.csv"
RESULTS_10PCT="$OUT_DIR/results_10pct.csv"
RESULTS_MERGED="$OUT_DIR/results_merged.csv"

printf 'dataset,selectivity_pct,method,ef,postfilter_max_candidates,k,queries_executed,recall,qps,fid_comp_ratio,tb_comp_ratio,total_comp_ratio,summary_path,status,error\n' > "$RESULTS_1PCT"
printf 'dataset,selectivity_pct,method,ef,postfilter_max_candidates,k,queries_executed,recall,qps,fid_comp_ratio,tb_comp_ratio,total_comp_ratio,summary_path,status,error\n' > "$RESULTS_10PCT"

run_single() {
  local method="$1"
  local selectivity_pct="$2"
  local ef="$3"
  local manifest_path="$4"
  local acorn_index_path="$5"
  local out_csv="$6"
  local filter_expr="$7"
  local postfilter_max_candidates="$8"

  local summary_dir="$OUT_DIR/summaries/${selectivity_pct}pct/${method}"
  mkdir -p "$summary_dir"

  local summary_stem="ef_${ef}"
  if [[ "$method" == "post_filter_hnsw" ]]; then
    summary_stem="ef_${ef}_pfc_${postfilter_max_candidates}"
  fi
  local summary_path="$summary_dir/${summary_stem}.summary.txt"
  local log_path="$summary_dir/${summary_stem}.log"
  local iaa_engine_profile=""
  case "$method" in
    compass_iaa_1) iaa_engine_profile="1" ;;
    compass_iaa_2) iaa_engine_profile="2" ;;
    compass_iaa_4) iaa_engine_profile="4" ;;
    compass_iaa_8) iaa_engine_profile="8" ;;
  esac
  if [[ -n "$iaa_engine_profile" ]]; then
    configure_iaa_profile "$iaa_engine_profile"
  fi

  local -a cmd
  case "$method" in
    post_filter_hnsw)
      cmd=(
        "$HNSW_RUN"
        --dataset-type laion
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --search-mode post_filter_iterative
        --postfilter-max-candidates "$postfilter_max_candidates"
        --payload-jsonl "$PAYLOAD_JSONL"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    in_search_filter_hnsw)
      cmd=(
        "$HNSW_RUN"
        --dataset-type laion
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --search-mode in_search_filter
        --postfilter-max-candidates "$POSTFILTER_MAX_CANDIDATES"
        --payload-jsonl "$PAYLOAD_JSONL"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    compass_lz4)
      cmd=(
        "$LZ4_RUN"
        --dataset-type laion
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --payload-jsonl "$PAYLOAD_JSONL"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      if [[ -n "$FID_BLOCK_SIZE_BYTES" ]]; then
        cmd+=(--fid-block-size-bytes "$FID_BLOCK_SIZE_BYTES")
      fi
      if [[ -n "$TB_BLOCK_SIZE_BYTES" ]]; then
        cmd+=(--tb-block-size-bytes "$TB_BLOCK_SIZE_BYTES")
      fi
      ;;
    compass_iaa|compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      cmd=(
        "$IAA_RUN"
        --dataset-type laion
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --payload-jsonl "$PAYLOAD_JSONL"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      if [[ -n "$FID_BLOCK_SIZE_BYTES" ]]; then
        cmd+=(--fid-block-size-bytes "$FID_BLOCK_SIZE_BYTES")
      fi
      if [[ -n "$TB_BLOCK_SIZE_BYTES" ]]; then
        cmd+=(--tb-block-size-bytes "$TB_BLOCK_SIZE_BYTES")
      fi
      ;;
    acorn)
      cmd=(
        "$ACORN_RUN"
        --dataset-type laion
        --graph "$acorn_index_path"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --payload-jsonl "$PAYLOAD_JSONL"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    *)
      echo "Error: unknown method '$method'" >&2
      exit 1
      ;;
  esac

  local status="OK"
  local error_text=""
  local queries_executed=""
  local recall=""
  local qps=""
  local fid_comp_ratio=""
  local tb_comp_ratio=""
  local total_comp_ratio=""
  local postfilter_value_out=""
  if [[ "$method" == "post_filter_hnsw" ]]; then
    postfilter_value_out="$postfilter_max_candidates"
  fi

  if "${cmd[@]}" > "$log_path" 2>&1; then
    queries_executed="$(extract_summary_value "queries_executed" "$summary_path" || true)"
    recall="$(extract_summary_value "average_recall_at_k" "$summary_path" || true)"
    qps="$(extract_summary_value "qps" "$summary_path" || true)"
    fid_comp_ratio="$(extract_summary_value "fid_compression_ratio_raw_over_compressed" "$summary_path" || true)"
    tb_comp_ratio="$(extract_summary_value "tb_compression_ratio_raw_over_compressed" "$summary_path" || true)"
    total_comp_ratio="$(extract_summary_value "filter_payload_compression_ratio_raw_over_compressed" "$summary_path" || true)"

    if [[ -z "$qps" ]] || ! is_numeric "$qps"; then
      local loop_ms
      loop_ms="$(extract_summary_value "search_loop_time_ms" "$summary_path" || true)"
      if is_numeric "$queries_executed" && is_numeric "$loop_ms"; then
        qps="$(awk -v q="$queries_executed" -v ms="$loop_ms" 'BEGIN { if (ms > 0) printf "%.6f", (q * 1000.0) / ms; else print "0" }')"
      fi
    fi

    if [[ -z "$queries_executed" ]] || [[ -z "$recall" ]] || [[ -z "$qps" ]] || \
       ! is_numeric "$queries_executed" || ! is_numeric "$recall" || ! is_numeric "$qps"; then
      status="PARSE_FAIL"
      error_text="failed to parse one or more summary metrics"
    fi
  else
    status="RUN_FAIL"
    error_text="$(tail -n 5 "$log_path" || true)"
    error_text="$(sanitize_csv_field "$error_text")"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATASET" \
    "$selectivity_pct" \
    "$method" \
    "$ef" \
    "$postfilter_value_out" \
    "$K" \
    "$queries_executed" \
    "$recall" \
    "$qps" \
    "$fid_comp_ratio" \
    "$tb_comp_ratio" \
    "$total_comp_ratio" \
    "$summary_path" \
    "$status" \
    "$error_text" >> "$out_csv"
}

run_selectivity() {
  local selectivity_pct="$1"
  local manifest_path="$2"
  local acorn_index_path="$3"
  local out_csv="$4"
  local filter_expr="$5"
  local ef_values_name="$6"
  local map_field="$7"
  local map_bucket_low="$8"
  local map_bucket_high="$9"
  local map_bucket_size="${10}"
  local map_bucket_count="${11}"
  local map_fid_match="${12}"
  local map_fid_missing="${13}"

  local -n ef_values_ref="$ef_values_name"

  local ef=""
  local method=""
  for ef in "${ef_values_ref[@]}"; do
    for method in "${METHODS[@]}"; do
      echo "[RANGE_EXPECT] sel=${selectivity_pct}% method=${method} ef=${ef} field=${map_field} buckets=${map_bucket_low}..${map_bucket_high} bucket_size=${map_bucket_size} count=${map_bucket_count} fid_matches=${map_fid_match} fid_missing=${map_fid_missing}"
      if [[ "$method" == "post_filter_hnsw" ]]; then
        local postfilter_max_candidates=""
        for postfilter_max_candidates in "${POSTFILTER_MAX_VALUES[@]}"; do
          run_single \
            "$method" \
            "$selectivity_pct" \
            "$ef" \
            "$manifest_path" \
            "$acorn_index_path" \
            "$out_csv" \
            "$filter_expr" \
            "$postfilter_max_candidates"
        done
      else
        run_single \
          "$method" \
          "$selectivity_pct" \
          "$ef" \
          "$manifest_path" \
          "$acorn_index_path" \
          "$out_csv" \
          "$filter_expr" \
          "$POSTFILTER_MAX_CANDIDATES"
      fi
    done
  done
}

echo "Running filter benchmark comparison"
echo "  dataset: $DATASET"
echo "  graph: $GRAPH_PATH"
echo "  query: $QUERY_PATH"
echo "  manifests:"
echo "    1%  -> $MANIFEST_1PCT"
echo "    10% -> $MANIFEST_10PCT"
echo "  acorn indexes:"
echo "    1%  -> $ACORN_INDEX_1PCT"
echo "    10% -> $ACORN_INDEX_10PCT"
echo "  k: $K"
echo "  ef list (1%): ${EF_VALUES_1PCT[*]}"
echo "  ef list (10%): ${EF_VALUES_10PCT[*]}"
echo "  postfilter max-candidates list: ${POSTFILTER_MAX_VALUES[*]}"
echo "  in_search_filter_hnsw postfilter-max-candidates: $POSTFILTER_MAX_CANDIDATES"
echo "  num-queries: $NUM_QUERIES"
echo "  filter input (1%): $FILTER_EXPR_1PCT"
echo "  filter normalized (1%): $NORM_FILTER_1PCT"
echo "  filter input (10%): $FILTER_EXPR_10PCT"
echo "  filter normalized (10%): $NORM_FILTER_10PCT"
echo "  methods: ${METHODS[*]}"
if [[ "$REQUIRE_IAA_PROFILE" -eq 1 ]]; then
  echo "  iaa engine profiles: ${IAA_ENGINE_VALUES[*]}"
  echo "  skip iaa config: $SKIP_IAA_CONFIG"
fi
if [[ "$REQUIRE_IAA_PROFILE" -eq 1 && "$SKIP_IAA_CONFIG" -eq 0 ]]; then
  echo "  iaa config dir: $IAA_CONFIG_DIR"
  echo "  iaa device owner: $IAA_DEVICE_OWNER"
fi
echo "  out-dir: $OUT_DIR"
echo "  clean out-dir: $CLEAN_OUT_DIR"

run_selectivity \
  "1" \
  "$MANIFEST_1PCT" \
  "$ACORN_INDEX_1PCT" \
  "$RESULTS_1PCT" \
  "$NORM_FILTER_1PCT" \
  EF_VALUES_1PCT \
  "$MAP1_FIELD" \
  "$MAP1_BUCKET_LOW" \
  "$MAP1_BUCKET_HIGH" \
  "$MAP1_BUCKET_SIZE" \
  "$MAP1_SELECTED_BUCKET_COUNT" \
  "$MAP1_FID_BUCKET_MATCH_COUNT" \
  "$MAP1_FID_MISSING_COUNT"

run_selectivity \
  "10" \
  "$MANIFEST_10PCT" \
  "$ACORN_INDEX_10PCT" \
  "$RESULTS_10PCT" \
  "$NORM_FILTER_10PCT" \
  EF_VALUES_10PCT \
  "$MAP10_FIELD" \
  "$MAP10_BUCKET_LOW" \
  "$MAP10_BUCKET_HIGH" \
  "$MAP10_BUCKET_SIZE" \
  "$MAP10_SELECTED_BUCKET_COUNT" \
  "$MAP10_FID_BUCKET_MATCH_COUNT" \
  "$MAP10_FID_MISSING_COUNT"

cat "$RESULTS_1PCT" > "$RESULTS_MERGED"
tail -n +2 "$RESULTS_10PCT" >> "$RESULTS_MERGED"

echo "Done benchmarking"
echo "  $RESULTS_1PCT"
echo "  $RESULTS_10PCT"
echo "  $RESULTS_MERGED"

if [[ "$DO_PLOT" -eq 1 ]]; then
  ensure_readable_file "$PLOT_PY" "plot script not found"
  python3 "$PLOT_PY" \
    --input-csv "$RESULTS_MERGED" \
    --selectivities "1,10" \
    --output "$OUT_DIR/qps_vs_recall.png"
  echo "  $OUT_DIR/qps_vs_recall.png"
fi
