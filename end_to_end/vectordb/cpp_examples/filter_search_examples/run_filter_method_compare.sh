#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  run_filter_method_compare.sh \
    [--dataset <sift1m|sift1b|laion|hnm>] \
    [--k <int>] \
    [--ef-list <comma-separated>] \
    [--num-queries <int>] \
    [--out-dir <path>] \
    [--postfilter-max-candidates <int>] \
    [--build|--no-build] \
    [--plot|--no-plot]

Core defaults:
  --dataset                     sift1m
  --k                           10
  --ef-list                     32,64,96,128,160,200,256,320,400
  --num-queries                 100
  --out-dir                     <this_dir>/out/filter_method_compare
  --postfilter-max-candidates   5000
  --build                       enabled
  --plot                        enabled

Advanced overrides:
  --filter-expr <expr>
  --filter-expr-1pct <expr>
  --filter-expr-10pct <expr>
  --graph <path>
  --query <path>
  --manifest-1pct <path>
  --manifest-10pct <path>
  --acorn-index-1pct <path>
  --acorn-index-10pct <path>
  --payload-jsonl <path>

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

DATASET="sift1m"
K=10
EF_LIST="32,64,96,128,160,200"
NUM_QUERIES=100
OUT_DIR="$SCRIPT_DIR/out/filter_method_compare"
POSTFILTER_MAX_CANDIDATES=50
DO_BUILD=1
DO_PLOT=1
FILTER_EXPR='synthetic_id_bucket == 255'
FILTER_EXPR_SET_BY_USER=0
FILTER_EXPR_1PCT=""
FILTER_EXPR_10PCT=""

GRAPH_PATH=""
QUERY_PATH=""
MANIFEST_1PCT=""
MANIFEST_10PCT=""
ACORN_INDEX_1PCT=""
ACORN_INDEX_10PCT=""
PAYLOAD_JSONL=""

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
    --filter-expr)
      FILTER_EXPR="$2"
      FILTER_EXPR_SET_BY_USER=1
      shift 2
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
    --manifest-1pct)
      MANIFEST_1PCT="$2"
      shift 2
      ;;
    --manifest-10pct)
      MANIFEST_10PCT="$2"
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

HNSW_RUN="$SCRIPT_DIR/hnswlib_filter_search.run"
LZ4_RUN="$SCRIPT_DIR/compass_search_w_lz4.run"
IAA_RUN="$SCRIPT_DIR/compass_search_w_iaa_async.run"
ACORN_RUN="$SCRIPT_DIR/acorn_search.run"
PLOT_PY="$SCRIPT_DIR/plot_qps_recall.py"

HNSW_DATASET_TYPE=""
LZ4_DATASET_TYPE=""
IAA_DATASET_TYPE=""

set_dataset_defaults() {
  case "$DATASET" in
    sift1m)
      HNSW_DATASET_TYPE="sift"
      LZ4_DATASET_TYPE="sift"
      IAA_DATASET_TYPE="sift1m"
      : "${GRAPH_PATH:=/storage/jykang5/compass_graphs/sift_m128_efc200.bin}"
      : "${QUERY_PATH:=/storage/jykang5/compass_base_query/sift1m_query.fvecs}"
      : "${MANIFEST_1PCT:=/storage/jykang5/fid_tb/n_filter_100/sift1m/manifest.json}"
      : "${MANIFEST_10PCT:=/storage/jykang5/fid_tb/n_filter_10/sift1m/manifest.json}"
      : "${ACORN_INDEX_1PCT:=/storage/jykang5/compass_graphs/acorn/sift1m_acorn_m32_nf100.index}"
      : "${ACORN_INDEX_10PCT:=/storage/jykang5/compass_graphs/acorn/sift1m_acorn_m32_nf10.index}"
      ;;
    sift1b)
      HNSW_DATASET_TYPE="sift"
      LZ4_DATASET_TYPE="sift"
      IAA_DATASET_TYPE="sift1b"
      : "${GRAPH_PATH:=/storage/jykang5/compass_graphs/sift1b_m128_efc200.bin}"
      : "${QUERY_PATH:=/storage/jykang5/compass_base_query/sift1b_query.fvecs}"
      : "${MANIFEST_1PCT:=/storage/jykang5/fid_tb/n_filter_100/sift1b/manifest.json}"
      : "${MANIFEST_10PCT:=/storage/jykang5/fid_tb/n_filter_10/sift1b/manifest.json}"
      : "${ACORN_INDEX_1PCT:=/home/jykang5/compass/end_to_end/vectordb/dataset/acorn_graph/sift1b/sift1b_acorn_m32_nf100.index}"
      : "${ACORN_INDEX_10PCT:=/home/jykang5/compass/end_to_end/vectordb/dataset/acorn_graph/sift1b/sift1b_acorn_m32_nf10.index}"
      ;;
    laion)
      HNSW_DATASET_TYPE="laion"
      LZ4_DATASET_TYPE="laion"
      IAA_DATASET_TYPE="laion"
      : "${GRAPH_PATH:=/storage/jykang5/compass_graphs/laion_m128_efc200.bin}"
      : "${QUERY_PATH:=/storage/jykang5/compass_base_query/laion_query.fvecs}"
      : "${MANIFEST_1PCT:=/storage/jykang5/fid_tb/n_filter_100/laion/manifest.json}"
      : "${MANIFEST_10PCT:=/storage/jykang5/fid_tb/n_filter_10/laion/manifest.json}"
      : "${ACORN_INDEX_1PCT:=/storage/jykang5/compass_graphs/acorn/laion_acorn_m32_nf100.index}"
      : "${ACORN_INDEX_10PCT:=/storage/jykang5/compass_graphs/acorn/laion_acorn_m32_nf10.index}"
      : "${PAYLOAD_JSONL:=/fast-lab-share/benchmarks/VectorDB/FILTER/LAION/payloads.jsonl}"
      ;;
    hnm)
      HNSW_DATASET_TYPE="hnm"
      LZ4_DATASET_TYPE="hnm"
      IAA_DATASET_TYPE="hnm"
      : "${GRAPH_PATH:=/storage/jykang5/compass_graphs/hnm_m128_efc200.bin}"
      : "${QUERY_PATH:=/storage/jykang5/compass_base_query/hnm_query.fvecs}"
      : "${MANIFEST_1PCT:=/storage/jykang5/fid_tb/n_filter_100/hnm/manifest.json}"
      : "${MANIFEST_10PCT:=/storage/jykang5/fid_tb/n_filter_10/hnm/manifest.json}"
      : "${ACORN_INDEX_1PCT:=/storage/jykang5/compass_graphs/acorn/hnm_acorn_m32_nf100.index}"
      : "${ACORN_INDEX_10PCT:=/storage/jykang5/compass_graphs/acorn/hnm_acorn_m32_nf10.index}"
      : "${PAYLOAD_JSONL:=/fast-lab-share/benchmarks/VectorDB/FILTER/HnM/payloads.jsonl}"
      ;;
    *)
      echo "Error: --dataset must be one of: sift1m, sift1b, laion, hnm" >&2
      exit 1
      ;;
  esac
}

set_dataset_defaults

mkdir -p "$OUT_DIR"

if [[ "$DO_BUILD" -eq 1 ]]; then
  make -C "$SCRIPT_DIR"
fi

ensure_executable_file "$HNSW_RUN" "hnswlib_filter_search.run is missing or not executable"
ensure_executable_file "$LZ4_RUN" "compass_search_w_lz4.run is missing or not executable"
ensure_executable_file "$IAA_RUN" "compass_search_w_iaa_async.run is missing or not executable"
ensure_executable_file "$ACORN_RUN" "acorn_search.run is missing or not executable"
ensure_readable_file "$GRAPH_PATH" "graph file not found"
ensure_readable_file "$QUERY_PATH" "query file not found"
ensure_readable_file "$MANIFEST_1PCT" "1% manifest file not found"
ensure_readable_file "$MANIFEST_10PCT" "10% manifest file not found"
ensure_readable_file "$ACORN_INDEX_1PCT" "ACORN 1% index file not found"
ensure_readable_file "$ACORN_INDEX_10PCT" "ACORN 10% index file not found"
if [[ "$HNSW_DATASET_TYPE" != "sift" ]]; then
  ensure_readable_file "$PAYLOAD_JSONL" "payload JSONL is required for non-sift hnsw baseline"
fi

IFS=',' read -r -a RAW_EFS <<< "$EF_LIST"
EF_VALUES=()
for raw_ef in "${RAW_EFS[@]}"; do
  ef="$(echo "$raw_ef" | tr -d '[:space:]')"
  if [[ -z "$ef" ]]; then
    continue
  fi
  if ! [[ "$ef" =~ ^[0-9]+$ ]] || [[ "$ef" -le 0 ]]; then
    echo "Error: invalid ef in --ef-list: $ef" >&2
    exit 1
  fi
  EF_VALUES+=("$ef")
done
if [[ "${#EF_VALUES[@]}" -eq 0 ]]; then
  echo "Error: --ef-list produced no valid ef values" >&2
  exit 1
fi

generate_sift_metadata_csv() {
  local manifest_path="$1"
  local out_csv="$2"
  python3 - "$manifest_path" "$out_csv" <<'PY'
import csv
import json
import math
import sys

manifest_path, out_csv = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

n_elements = int(manifest["n_elements"])
nfilters = int(manifest["nfilters"])
if n_elements <= 0:
    raise RuntimeError(f"invalid n_elements={n_elements}")
if nfilters <= 0:
    raise RuntimeError(f"invalid nfilters={nfilters}")

max_elements_per_group = int(math.ceil(float(n_elements) / float(nfilters)))

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "synthetic_id_bucket"])
    for i in range(n_elements):
        gid = i // max_elements_per_group
        if gid >= nfilters:
            gid = nfilters - 1
        w.writerow([i, gid])
PY
}

validate_sift_metadata_csv() {
  local manifest_path="$1"
  local csv_path="$2"
  python3 - "$manifest_path" "$csv_path" <<'PY'
import csv
import json
import sys

manifest_path, csv_path = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
expected_n = int(manifest["n_elements"])
nfilters = int(manifest["nfilters"])
expected_max = nfilters - 1

count = 0
min_gid = None
max_gid = None
with open(csv_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    if r.fieldnames != ["id", "synthetic_id_bucket"]:
        raise RuntimeError(f"unexpected CSV header: {r.fieldnames}")
    for row in r:
        count += 1
        gid = int(row["synthetic_id_bucket"])
        min_gid = gid if min_gid is None else min(min_gid, gid)
        max_gid = gid if max_gid is None else max(max_gid, gid)

if count != expected_n:
    raise RuntimeError(f"CSV row count mismatch: got {count}, expected {expected_n}")
if min_gid is None or max_gid is None:
    raise RuntimeError("CSV has no rows")
if min_gid < 0 or max_gid > expected_max:
    raise RuntimeError(
        f"synthetic_id_bucket out of range: min={min_gid}, max={max_gid}, expected [0,{expected_max}]"
    )
PY
}

derive_tail_bucket_filter_expr() {
  local manifest_path="$1"
  python3 - "$manifest_path" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

nfilters = int(manifest["nfilters"])
if nfilters <= 0:
    raise RuntimeError(f"invalid nfilters={nfilters}")

print(f"synthetic_id_bucket == {nfilters - 1}")
PY
}

RESULTS_1PCT="$OUT_DIR/results_1pct.csv"
RESULTS_10PCT="$OUT_DIR/results_10pct.csv"
RESULTS_MERGED="$OUT_DIR/results_merged.csv"

printf 'dataset,selectivity_pct,method,ef,k,queries_executed,recall,qps,summary_path,status,error\n' > "$RESULTS_1PCT"
printf 'dataset,selectivity_pct,method,ef,k,queries_executed,recall,qps,summary_path,status,error\n' > "$RESULTS_10PCT"

GENERATED_META_DIR="$OUT_DIR/generated_metadata"
mkdir -p "$GENERATED_META_DIR"
META_1PCT=""
META_10PCT=""

if [[ "$HNSW_DATASET_TYPE" == "sift" ]]; then
  META_1PCT="$GENERATED_META_DIR/${DATASET}_meta_1pct.csv"
  META_10PCT="$GENERATED_META_DIR/${DATASET}_meta_10pct.csv"
  generate_sift_metadata_csv "$MANIFEST_1PCT" "$META_1PCT"
  generate_sift_metadata_csv "$MANIFEST_10PCT" "$META_10PCT"
  validate_sift_metadata_csv "$MANIFEST_1PCT" "$META_1PCT"
  validate_sift_metadata_csv "$MANIFEST_10PCT" "$META_10PCT"
fi

EFFECTIVE_FILTER_EXPR_1PCT="$FILTER_EXPR"
EFFECTIVE_FILTER_EXPR_10PCT="$FILTER_EXPR"

if [[ -n "$FILTER_EXPR_1PCT" ]]; then
  EFFECTIVE_FILTER_EXPR_1PCT="$FILTER_EXPR_1PCT"
elif [[ "$DATASET" == "sift1m" && "$FILTER_EXPR_SET_BY_USER" -eq 0 ]]; then
  EFFECTIVE_FILTER_EXPR_1PCT="$(derive_tail_bucket_filter_expr "$MANIFEST_1PCT")"
fi

if [[ -n "$FILTER_EXPR_10PCT" ]]; then
  EFFECTIVE_FILTER_EXPR_10PCT="$FILTER_EXPR_10PCT"
elif [[ "$DATASET" == "sift1m" && "$FILTER_EXPR_SET_BY_USER" -eq 0 ]]; then
  EFFECTIVE_FILTER_EXPR_10PCT="$(derive_tail_bucket_filter_expr "$MANIFEST_10PCT")"
fi

run_single() {
  local method="$1"
  local selectivity_pct="$2"
  local ef="$3"
  local manifest_path="$4"
  local metadata_csv="$5"
  local acorn_index_path="$6"
  local out_csv="$7"
  local filter_expr="$8"

  local summary_dir="$OUT_DIR/summaries/${selectivity_pct}pct/${method}"
  mkdir -p "$summary_dir"

  local summary_path="$summary_dir/ef_${ef}.summary.txt"
  local log_path="$summary_dir/ef_${ef}.log"

  local -a cmd
  case "$method" in
    post_filter_hnsw)
      cmd=(
        "$HNSW_RUN"
        --dataset-type "$HNSW_DATASET_TYPE"
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --search-mode post_filter_iterative
        --postfilter-max-candidates "$POSTFILTER_MAX_CANDIDATES"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      if [[ "$HNSW_DATASET_TYPE" == "sift" ]]; then
        cmd+=(--metadata-csv "$metadata_csv" --id-column id)
      else
        cmd+=(--payload-jsonl "$PAYLOAD_JSONL")
      fi
      ;;
    in_search_filter_hnsw)
      cmd=(
        "$HNSW_RUN"
        --dataset-type "$HNSW_DATASET_TYPE"
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --search-mode in_search_filter
        --postfilter-max-candidates "$POSTFILTER_MAX_CANDIDATES"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      if [[ "$HNSW_DATASET_TYPE" == "sift" ]]; then
        cmd+=(--metadata-csv "$metadata_csv" --id-column id)
      else
        cmd+=(--payload-jsonl "$PAYLOAD_JSONL")
      fi
      ;;
    compass_lz4)
      cmd=(
        "$LZ4_RUN"
        --dataset-type "$LZ4_DATASET_TYPE"
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    compass_iaa)
      cmd=(
        "$IAA_RUN"
        --dataset-type "$IAA_DATASET_TYPE"
        --graph "$GRAPH_PATH"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --fidtb-manifest "$manifest_path"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      ;;
    acorn)
      cmd=(
        "$ACORN_RUN"
        --dataset-type "$DATASET"
        --graph "$acorn_index_path"
        --query "$QUERY_PATH"
        --k "$K"
        --ef "$ef"
        --filter "$filter_expr"
        --num-queries "$NUM_QUERIES"
        --summary-out "$summary_path"
      )
      if [[ "$HNSW_DATASET_TYPE" == "sift" ]]; then
        cmd+=(--metadata-csv "$metadata_csv" --id-column id)
      else
        cmd+=(--payload-jsonl "$PAYLOAD_JSONL")
      fi
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

  if "${cmd[@]}" > "$log_path" 2>&1; then
    queries_executed="$(extract_summary_value "queries_executed" "$summary_path" || true)"
    recall="$(extract_summary_value "average_recall_at_k" "$summary_path" || true)"
    qps="$(extract_summary_value "qps" "$summary_path" || true)"

    if [[ -z "$qps" ]] || ! is_numeric "$qps"; then
      local loop_ms
      loop_ms="$(extract_summary_value "search_loop_time_ms" "$summary_path" || true)"
      if is_numeric "$queries_executed" && is_numeric "$loop_ms"; then
        qps="$(awk -v q="$queries_executed" -v ms="$loop_ms" 'BEGIN { if (ms > 0) printf "%.6f", (q * 100.0) / ms; else print "0" }')"
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

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATASET" \
    "$selectivity_pct" \
    "$method" \
    "$ef" \
    "$K" \
    "$queries_executed" \
    "$recall" \
    "$qps" \
    "$summary_path" \
    "$status" \
    "$error_text" >> "$out_csv"
}

run_selectivity() {
  local selectivity_pct="$1"
  local manifest_path="$2"
  local metadata_csv="$3"
  local acorn_index_path="$4"
  local out_csv="$5"
  local filter_expr="$6"

  local methods=(
    # "post_filter_hnsw"
    # "in_search_filter_hnsw"
    "acorn"
    "compass_lz4"
    "compass_iaa"
  )

  for ef in "${EF_VALUES[@]}"; do
    for method in "${methods[@]}"; do
      run_single "$method" "$selectivity_pct" "$ef" "$manifest_path" "$metadata_csv" "$acorn_index_path" "$out_csv" "$filter_expr"
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
echo "  ef list: ${EF_VALUES[*]}"
echo "  num-queries: $NUM_QUERIES"
echo "  filter (1%): $EFFECTIVE_FILTER_EXPR_1PCT"
echo "  filter (10%): $EFFECTIVE_FILTER_EXPR_10PCT"
echo "  out-dir: $OUT_DIR"

run_selectivity "1" "$MANIFEST_1PCT" "$META_1PCT" "$ACORN_INDEX_1PCT" "$RESULTS_1PCT" "$EFFECTIVE_FILTER_EXPR_1PCT"
run_selectivity "10" "$MANIFEST_10PCT" "$META_10PCT" "$ACORN_INDEX_10PCT" "$RESULTS_10PCT" "$EFFECTIVE_FILTER_EXPR_10PCT"

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
