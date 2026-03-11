#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_PY="$(cd "$SCRIPT_DIR/.." && pwd)/plot_qps_recall.py"

usage() {
  cat <<'USAGE'
Usage:
  run_filter_method_compare.sh \
    --filter-expr-and "<field1 == value1 AND field2 == value2>" \
    --filter-expr-or  "<field1 == value1 OR field2 == value2>" \
    [--dataset hnm] \
    [--k <int>] \
    [--ef-list <comma-separated>] \
    [--ef-list-1pct <comma-separated>] \
    [--ef-list-10pct <comma-separated>] \
    [--num-queries <int>] \
    [--out-dir <path>] \
    [--postfilter-max-candidates <int>] \
    [--postfilter-max-candidates-list <comma-separated>] \
    [--postfilter-max-candidates-list-1pct <comma-separated>] \
    [--postfilter-max-candidates-list-10pct <comma-separated>] \
    [--postfilter-ef-list-1pct <comma-separated>] \
    [--postfilter-ef-list-10pct <comma-separated>] \
    [--build|--no-build] \
    [--plot|--no-plot]

Core defaults:
  --dataset                     hnm
  --k                           10
  --ef-list                     64,96,128,160,200
  --ef-list-1pct                (inherits --ef-list)
  --ef-list-10pct               (inherits --ef-list)
  --num-queries                 20
  --out-dir                     <this_dir>/out/filter_method_compare
  --postfilter-max-candidates   3000
  --postfilter-max-candidates-list 3000
  --postfilter-max-candidates-list-1pct (inherits --postfilter-max-candidates-list)
  --postfilter-max-candidates-list-10pct (inherits --postfilter-max-candidates-list)
  --postfilter-ef-list-1pct     8
  --postfilter-ef-list-10pct    8
  --fid-block-size-bytes        65536
  --tb-block-size-bytes         1048576
  --build                       enabled
  --plot                        enabled

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
  --fid-block-size-bytes <int>
  --tb-block-size-bytes <int>

Scenario mapping:
  selectivity_pct=1  -> AND expression
  selectivity_pct=10 -> OR expression

Output files:
  <out-dir>/results_1pct.csv
  <out-dir>/results_10pct.csv
  <out-dir>/results_merged.csv
  <out-dir>/qps_vs_recall.png (when --plot)
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

normalize_expr_input() {
  local expr="$1"
  expr="${expr#<}"
  expr="${expr%>}"
  printf '%s' "$expr"
}

validate_two_leaf_expr() {
  local expr="$1"
  local expected_op="$2"
  local manifest_a="$3"
  local manifest_b="$4"
  local payload_jsonl="$5"

  python3 - "$expr" "$expected_op" "$manifest_a" "$manifest_b" "$payload_jsonl" <<'PY'
import json
import re
import sys

expr = sys.argv[1].strip()
expected_op = sys.argv[2].strip().upper()
manifest_a = sys.argv[3]
manifest_b = sys.argv[4]
payload_jsonl = sys.argv[5]

pat = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+?)\s+(AND|OR)\s+([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+?)\s*$',
    re.IGNORECASE,
)
m = pat.match(expr)
if not m:
    raise SystemExit(
        "expression must be exactly '<field1 == value1 AND/OR field2 == value2>'"
    )

field1 = m.group(1)
op = m.group(3).upper()
field2 = m.group(4)
if op != expected_op:
    raise SystemExit(f"expression operator must be {expected_op}, got {op}")
if field1 == field2:
    raise SystemExit("expression must use two different fields")

manifest_fields = set()
for path in (manifest_a, manifest_b):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    attrs = manifest.get("attributes", [])
    for attr in attrs:
        key = attr.get("key")
        if isinstance(key, str) and key:
            manifest_fields.add(key)

missing_manifest = [f for f in (field1, field2) if f not in manifest_fields]
if missing_manifest:
    raise SystemExit(
        "fields missing in manifest: " + ", ".join(missing_manifest)
    )

payload_keys = None
with open(payload_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            payload_keys = set(obj.keys())
            break

if payload_keys is None:
    raise SystemExit("payload JSONL has no valid object row")

missing_payload = [f for f in (field1, field2) if f not in payload_keys]
if missing_payload:
    raise SystemExit(
        "fields missing in payload JSONL schema: " + ", ".join(missing_payload)
    )
PY
}

parse_ef_list_into_array() {
  local ef_list="$1"
  local ef_label="$2"
  local out_var_name="$3"

  IFS=',' read -r -a RAW_EFS <<< "$ef_list"
  local -a parsed_values=()
  local raw_ef=""
  local ef=""
  for raw_ef in "${RAW_EFS[@]}"; do
    ef="$(echo "$raw_ef" | tr -d '[:space:]')"
    if [[ -z "$ef" ]]; then
      continue
    fi
    if ! [[ "$ef" =~ ^[0-9]+$ ]] || [[ "$ef" -le 0 ]]; then
      echo "Error: invalid ef value '$ef' in --$ef_label" >&2
      exit 1
    fi
    parsed_values+=("$ef")
  done

  if [[ "${#parsed_values[@]}" -eq 0 ]]; then
    echo "Error: --$ef_label produced empty ef list" >&2
    exit 1
  fi

  printf -v "$out_var_name" '%s ' "${parsed_values[@]}"
}

# Methods to run.
# Comment out entries here for quick manual method selection.
METHODS=(
  # "post_filter_hnsw"
  # "in_search_filter_hnsw"
  "acorn"
  # "compass_lz4"
  # "compass_iaa_1"
  # "compass_iaa_2"
  # "compass_iaa_4"
  # "compass_iaa_8"
)

DATASET="hnm"
K=10
EF_LIST="64,96,128,160,200"
EF_LIST_1PCT="32, 48, 64, 96"
EF_LIST_10PCT="32, 48, 64, 96, 128, 160, 200"
IN_SEARCH_EF_LIST_1PCT="4,8,16,32,64"
IN_SEARCH_EF_LIST_10PCT="4,8,16,32,64"
ACORN_EF_LIST_1PCT="44,52,64,80,96,112,128,144,160,320"
ACORN_EF_LIST_10PCT="80,96,600, 800"
NUM_QUERIES=100
OUT_DIR="$SCRIPT_DIR/out/filter_method_compare"
POSTFILTER_MAX_CANDIDATES=3000
POSTFILTER_MAX_CANDIDATES_LIST="3000"
POSTFILTER_MAX_CANDIDATES_LIST_1PCT="21000,22000,23000,24000,25000"
POSTFILTER_MAX_CANDIDATES_LIST_10PCT="3200,3400,3800,4500,6000"
POSTFILTER_EF_LIST_1PCT="8"
POSTFILTER_EF_LIST_10PCT="8"
DO_BUILD=1
DO_PLOT=1

FILTER_EXPR_AND="department_name == 'Baby Toys/Acc' AND garment_group_name == 'Accessories'"
FILTER_EXPR_OR="department_name == 'Blouse' OR garment_group_name == 'Jersey Basic'"

GRAPH_PATH=""
QUERY_PATH=""
MANIFEST=""
MANIFEST_1PCT=""
MANIFEST_10PCT=""
ACORN_INDEX=""
ACORN_INDEX_1PCT=""
ACORN_INDEX_10PCT=""
PAYLOAD_JSONL=""
FID_BLOCK_SIZE_BYTES="$((1024 * 8))"
TB_BLOCK_SIZE_BYTES="$((1024 * 128))"
CPU_PIN_CORE="0"

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
    --postfilter-max-candidates)
      POSTFILTER_MAX_CANDIDATES="$2"
      shift 2
      ;;
    --postfilter-max-candidates-list)
      POSTFILTER_MAX_CANDIDATES_LIST="$2"
      shift 2
      ;;
    --postfilter-max-candidates-list-1pct)
      POSTFILTER_MAX_CANDIDATES_LIST_1PCT="$2"
      shift 2
      ;;
    --postfilter-max-candidates-list-10pct)
      POSTFILTER_MAX_CANDIDATES_LIST_10PCT="$2"
      shift 2
      ;;
    --postfilter-ef-list-1pct)
      POSTFILTER_EF_LIST_1PCT="$2"
      shift 2
      ;;
    --postfilter-ef-list-10pct)
      POSTFILTER_EF_LIST_10PCT="$2"
      shift 2
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
    --filter-expr-and)
      FILTER_EXPR_AND="$2"
      shift 2
      ;;
    --filter-expr-or)
      FILTER_EXPR_OR="$2"
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

if [[ "$DATASET" != "hnm" ]]; then
  echo "Error: this runner currently supports only --dataset hnm" >&2
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
if [[ -z "$POSTFILTER_MAX_CANDIDATES_LIST_1PCT" ]]; then
  POSTFILTER_MAX_CANDIDATES_LIST_1PCT="$POSTFILTER_MAX_CANDIDATES_LIST"
fi
if [[ -z "$POSTFILTER_MAX_CANDIDATES_LIST_10PCT" ]]; then
  POSTFILTER_MAX_CANDIDATES_LIST_10PCT="$POSTFILTER_MAX_CANDIDATES_LIST"
fi
if ! [[ "$FID_BLOCK_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$FID_BLOCK_SIZE_BYTES" -le 0 ]]; then
  echo "Error: --fid-block-size-bytes must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TB_BLOCK_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$TB_BLOCK_SIZE_BYTES" -le 0 ]]; then
  echo "Error: --tb-block-size-bytes must be a positive integer" >&2
  exit 1
fi
if [[ -z "$FILTER_EXPR_AND" ]]; then
  echo "Error: missing required --filter-expr-and" >&2
  exit 1
fi
if [[ -z "$FILTER_EXPR_OR" ]]; then
  echo "Error: missing required --filter-expr-or" >&2
  exit 1
fi

if [[ -n "$MANIFEST" ]]; then
  if [[ -z "$MANIFEST_1PCT" ]]; then
    MANIFEST_1PCT="$MANIFEST"
  fi
  if [[ -z "$MANIFEST_10PCT" ]]; then
    MANIFEST_10PCT="$MANIFEST"
  fi
fi
if [[ -n "$ACORN_INDEX" ]]; then
  if [[ -z "$ACORN_INDEX_1PCT" ]]; then
    ACORN_INDEX_1PCT="$ACORN_INDEX"
  fi
  if [[ -z "$ACORN_INDEX_10PCT" ]]; then
    ACORN_INDEX_10PCT="$ACORN_INDEX"
  fi
fi

: "${GRAPH_PATH:=/storage/jykang5/compass_graphs/hnm_m128_efc200.bin}"
: "${QUERY_PATH:=/storage/jykang5/compass_base_query/hnm_query.fvecs}"
: "${MANIFEST_1PCT:=/storage/jykang5/fid_tb/hnm/manifest.json}"
: "${MANIFEST_10PCT:=/storage/jykang5/fid_tb/hnm/manifest.json}"
: "${ACORN_INDEX_1PCT:=/storage/jykang5/compass_graphs/acorn/hnm_acorn_m64_nf256.index}"
: "${ACORN_INDEX_10PCT:=/storage/jykang5/compass_graphs/acorn/hnm_acorn_m64_nf256.index}"
: "${PAYLOAD_JSONL:=/storage/jykang5/payloads/hnm_payloads.jsonl}"

HNSW_RUN="$SCRIPT_DIR/hnswlib_filter_search.run"
LZ4_RUN="$SCRIPT_DIR/compass_search_w_lz4.run"
IAA_RUN="$SCRIPT_DIR/compass_search_w_iaa.run"
ACORN_RUN="$SCRIPT_DIR/acorn_search.run"

mkdir -p "$OUT_DIR"

if [[ "$DO_BUILD" -eq 1 ]]; then
  make -B -C "$SCRIPT_DIR"
fi

ensure_executable_file "$HNSW_RUN" "hnswlib_filter_search.run is missing or not executable"
ensure_executable_file "$LZ4_RUN" "compass_search_w_lz4.run is missing or not executable"
ensure_executable_file "$IAA_RUN" "compass_search_w_iaa.run is missing or not executable"
ensure_executable_file "$ACORN_RUN" "acorn_search.run is missing or not executable"
ensure_readable_file "$GRAPH_PATH" "graph file not found"
ensure_readable_file "$QUERY_PATH" "query file not found"
ensure_readable_file "$MANIFEST_1PCT" "1% manifest file not found"
ensure_readable_file "$MANIFEST_10PCT" "10% manifest file not found"
ensure_readable_file "$ACORN_INDEX_1PCT" "ACORN 1% index file not found"
ensure_readable_file "$ACORN_INDEX_10PCT" "ACORN 10% index file not found"
ensure_readable_file "$PAYLOAD_JSONL" "payload JSONL not found"

FILTER_EXPR_AND="$(normalize_expr_input "$FILTER_EXPR_AND")"
FILTER_EXPR_OR="$(normalize_expr_input "$FILTER_EXPR_OR")"

validate_two_leaf_expr "$FILTER_EXPR_AND" "AND" "$MANIFEST_1PCT" "$MANIFEST_10PCT" "$PAYLOAD_JSONL"
validate_two_leaf_expr "$FILTER_EXPR_OR" "OR" "$MANIFEST_1PCT" "$MANIFEST_10PCT" "$PAYLOAD_JSONL"

if [[ -z "$EF_LIST_1PCT" ]]; then
  EF_LIST_1PCT="$EF_LIST"
fi
if [[ -z "$EF_LIST_10PCT" ]]; then
  EF_LIST_10PCT="$EF_LIST"
fi

parse_ef_list_into_array "$EF_LIST_1PCT" "ef-list-1pct" EF_VALUES_1PCT_STR
parse_ef_list_into_array "$EF_LIST_10PCT" "ef-list-10pct" EF_VALUES_10PCT_STR
parse_ef_list_into_array "$IN_SEARCH_EF_LIST_1PCT" "in-search-ef-list-1pct" IN_SEARCH_EF_VALUES_1PCT_STR
parse_ef_list_into_array "$IN_SEARCH_EF_LIST_10PCT" "in-search-ef-list-10pct" IN_SEARCH_EF_VALUES_10PCT_STR
parse_ef_list_into_array "$ACORN_EF_LIST_1PCT" "acorn-ef-list-1pct" ACORN_EF_VALUES_1PCT_STR
parse_ef_list_into_array "$ACORN_EF_LIST_10PCT" "acorn-ef-list-10pct" ACORN_EF_VALUES_10PCT_STR
parse_ef_list_into_array "$POSTFILTER_EF_LIST_1PCT" "postfilter-ef-list-1pct" POSTFILTER_EF_VALUES_1PCT_STR
parse_ef_list_into_array "$POSTFILTER_EF_LIST_10PCT" "postfilter-ef-list-10pct" POSTFILTER_EF_VALUES_10PCT_STR
parse_ef_list_into_array "$POSTFILTER_MAX_CANDIDATES_LIST_1PCT" "postfilter-max-candidates-list-1pct" POSTFILTER_MAX_VALUES_1PCT_STR
parse_ef_list_into_array "$POSTFILTER_MAX_CANDIDATES_LIST_10PCT" "postfilter-max-candidates-list-10pct" POSTFILTER_MAX_VALUES_10PCT_STR
IFS=' ' read -r -a EF_VALUES_1PCT <<< "$EF_VALUES_1PCT_STR"
IFS=' ' read -r -a EF_VALUES_10PCT <<< "$EF_VALUES_10PCT_STR"
IFS=' ' read -r -a IN_SEARCH_EF_VALUES_1PCT <<< "$IN_SEARCH_EF_VALUES_1PCT_STR"
IFS=' ' read -r -a IN_SEARCH_EF_VALUES_10PCT <<< "$IN_SEARCH_EF_VALUES_10PCT_STR"
IFS=' ' read -r -a ACORN_EF_VALUES_1PCT <<< "$ACORN_EF_VALUES_1PCT_STR"
IFS=' ' read -r -a ACORN_EF_VALUES_10PCT <<< "$ACORN_EF_VALUES_10PCT_STR"
IFS=' ' read -r -a POSTFILTER_EF_VALUES_1PCT <<< "$POSTFILTER_EF_VALUES_1PCT_STR"
IFS=' ' read -r -a POSTFILTER_EF_VALUES_10PCT <<< "$POSTFILTER_EF_VALUES_10PCT_STR"
IFS=' ' read -r -a POSTFILTER_MAX_VALUES_1PCT <<< "$POSTFILTER_MAX_VALUES_1PCT_STR"
IFS=' ' read -r -a POSTFILTER_MAX_VALUES_10PCT <<< "$POSTFILTER_MAX_VALUES_10PCT_STR"

for max_candidates in "${POSTFILTER_MAX_VALUES_1PCT[@]}" "${POSTFILTER_MAX_VALUES_10PCT[@]}"; do
  if [[ "$max_candidates" -lt "$K" ]]; then
    echo "Error: every postfilter max-candidates list value must be >= --k (found: $max_candidates)" >&2
    exit 1
  fi
done

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
  local postfilter_max_candidates="${8:-$POSTFILTER_MAX_CANDIDATES}"

  local summary_dir="$OUT_DIR/summaries/${selectivity_pct}pct/${method}"
  mkdir -p "$summary_dir"
  local summary_stem="ef_${ef}"
  if [[ "$method" == "post_filter_hnsw" ]]; then
    summary_stem="ef_${ef}_pfc_${postfilter_max_candidates}"
  fi
  local summary_path="$summary_dir/${summary_stem}.summary.txt"
  local log_path="$summary_dir/${summary_stem}.log"

  local -a cmd
  case "$method" in
    post_filter_hnsw)
      cmd=(
        "$HNSW_RUN"
        --dataset-type hnm
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
        --dataset-type hnm
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
        --dataset-type hnm
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --payload-jsonl "$PAYLOAD_JSONL"
        --fid-block-size-bytes "$FID_BLOCK_SIZE_BYTES"
        --tb-block-size-bytes "$TB_BLOCK_SIZE_BYTES"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    compass_iaa|compass_iaa_1|compass_iaa_2|compass_iaa_4|compass_iaa_8)
      cmd=(
        "$IAA_RUN"
        --dataset-type hnm
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --payload-jsonl "$PAYLOAD_JSONL"
        --fid-block-size-bytes "$FID_BLOCK_SIZE_BYTES"
        --tb-block-size-bytes "$TB_BLOCK_SIZE_BYTES"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    acorn)
      cmd=(
        "$ACORN_RUN"
        --dataset-type hnm
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
  if [[ "$method" == "in_search_filter_hnsw" ]]; then
    postfilter_value_out="$POSTFILTER_MAX_CANDIDATES"
  elif [[ "$method" == "post_filter_hnsw" ]]; then
    postfilter_value_out="$postfilter_max_candidates"
  fi

  local -a run_cmd
  if [[ -n "$CPU_PIN_CORE" ]]; then
    run_cmd=(taskset -c "$CPU_PIN_CORE" "${cmd[@]}")
  else
    run_cmd=("${cmd[@]}")
  fi

  if "${run_cmd[@]}" > "$log_path" 2>&1; then
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

run_scenario() {
  local selectivity_pct="$1"
  local manifest_path="$2"
  local acorn_index_path="$3"
  local out_csv="$4"
  local filter_expr="$5"
  local ef_values_name="$6"
  local -n ef_values_ref="$ef_values_name"

  for method in "${METHODS[@]}"; do
    local -a method_ef_values=("${ef_values_ref[@]}")
    local -a postfilter_values=()
    if [[ "$method" == "in_search_filter_hnsw" ]]; then
      if [[ "$selectivity_pct" == "1" ]]; then
        method_ef_values=("${IN_SEARCH_EF_VALUES_1PCT[@]}")
      else
        method_ef_values=("${IN_SEARCH_EF_VALUES_10PCT[@]}")
      fi
    elif [[ "$method" == "post_filter_hnsw" ]]; then
      if [[ "$selectivity_pct" == "1" ]]; then
        method_ef_values=("${POSTFILTER_EF_VALUES_1PCT[@]}")
        postfilter_values=("${POSTFILTER_MAX_VALUES_1PCT[@]}")
      else
        method_ef_values=("${POSTFILTER_EF_VALUES_10PCT[@]}")
        postfilter_values=("${POSTFILTER_MAX_VALUES_10PCT[@]}")
      fi
    elif [[ "$method" == "acorn" ]]; then
      if [[ "$selectivity_pct" == "1" ]]; then
        method_ef_values=("${ACORN_EF_VALUES_1PCT[@]}")
      else
        method_ef_values=("${ACORN_EF_VALUES_10PCT[@]}")
      fi
    fi
    if [[ "$method" == "post_filter_hnsw" ]]; then
      for ef in "${method_ef_values[@]}"; do
        for postfilter_max_candidates in "${postfilter_values[@]}"; do
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
      done
      continue
    fi
    for ef in "${method_ef_values[@]}"; do
      run_single \
        "$method" \
        "$selectivity_pct" \
        "$ef" \
        "$manifest_path" \
        "$acorn_index_path" \
        "$out_csv" \
        "$filter_expr"
    done
  done
}

echo "Running multiple_exact_match benchmark comparison"
echo "  dataset: $DATASET"
echo "  graph: $GRAPH_PATH"
echo "  query: $QUERY_PATH"
echo "  manifests:"
echo "    1%  -> $MANIFEST_1PCT"
echo "    10% -> $MANIFEST_10PCT"
echo "  acorn indexes:"
echo "    1%  -> $ACORN_INDEX_1PCT"
echo "    10% -> $ACORN_INDEX_10PCT"
echo "  payload-jsonl: $PAYLOAD_JSONL"
echo "  k: $K"
echo "  ef list (1%): ${EF_VALUES_1PCT[*]}"
echo "  ef list (10%): ${EF_VALUES_10PCT[*]}"
echo "  in_search ef list (1%): ${IN_SEARCH_EF_VALUES_1PCT[*]}"
echo "  in_search ef list (10%): ${IN_SEARCH_EF_VALUES_10PCT[*]}"
echo "  post_filter ef list (1%): ${POSTFILTER_EF_VALUES_1PCT[*]}"
echo "  post_filter ef list (10%): ${POSTFILTER_EF_VALUES_10PCT[*]}"
echo "  post_filter max-candidates list (1%): ${POSTFILTER_MAX_VALUES_1PCT[*]}"
echo "  post_filter max-candidates list (10%): ${POSTFILTER_MAX_VALUES_10PCT[*]}"
echo "  acorn ef list (1%): ${ACORN_EF_VALUES_1PCT[*]}"
echo "  acorn ef list (10%): ${ACORN_EF_VALUES_10PCT[*]}"
echo "  postfilter-max-candidates: $POSTFILTER_MAX_CANDIDATES"
echo "  num-queries: $NUM_QUERIES"
echo "  filter (1%, AND): $FILTER_EXPR_AND"
echo "  filter (10%, OR): $FILTER_EXPR_OR"
echo "  cpu pin core: $CPU_PIN_CORE"
echo "  out-dir: $OUT_DIR"

run_scenario "1" "$MANIFEST_1PCT" "$ACORN_INDEX_1PCT" "$RESULTS_1PCT" "$FILTER_EXPR_AND" EF_VALUES_1PCT
run_scenario "10" "$MANIFEST_10PCT" "$ACORN_INDEX_10PCT" "$RESULTS_10PCT" "$FILTER_EXPR_OR" EF_VALUES_10PCT

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
