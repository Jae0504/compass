#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_bucket_recall_sweep.sh \
    --graph <path> \
    --query <path> \
    --metadata-csv <path> \
    [--runner <path>] \
    [--dataset-type <sift|laion|hnm>] \
    [--field <name>] \
    [--k <int>] \
    [--ef <int>] \
    [--num-queries <int>] \
    [--start <int>] \
    [--end <int>] \
    [--filter-template <expr>] \
    [--out-csv <path>] \
    [--workdir <path>]

Defaults:
  --runner          ./hnswlib_filter_search.run
  --dataset-type    sift
  --field           synthetic_id_bucket
  --k               10
  --ef              32
  --num-queries     100
  --start           0
  --end             255
  --filter-template "{field} == {value}"
  --out-csv         bucket_recall.csv
USAGE
}

is_numeric() {
  local v="$1"
  [[ "$v" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]
}

render_filter() {
  local tmpl="$1"
  local fld="$2"
  local val="$3"
  local out="$tmpl"
  out="${out//\{field\}/$fld}"
  out="${out//\{value\}/$val}"
  printf '%s' "$out"
}

extract_summary_value() {
  local key="$1"
  local summary_path="$2"
  awk -F': ' -v k="$key" '$1 == k {print $2; exit}' "$summary_path"
}

graph_path=""
query_path=""
metadata_csv=""
runner="./hnswlib_filter_search.run"
dataset_type="sift"
field="synthetic_id_bucket"
k=10
ef=32
num_queries=100
start_bucket=0
end_bucket=255
filter_template="{field} == {value}"
out_csv="bucket_recall.csv"
workdir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)
      graph_path="$2"
      shift 2
      ;;
    --query)
      query_path="$2"
      shift 2
      ;;
    --metadata-csv)
      metadata_csv="$2"
      shift 2
      ;;
    --runner)
      runner="$2"
      shift 2
      ;;
    --dataset-type)
      dataset_type="$2"
      shift 2
      ;;
    --field)
      field="$2"
      shift 2
      ;;
    --k)
      k="$2"
      shift 2
      ;;
    --ef)
      ef="$2"
      shift 2
      ;;
    --num-queries)
      num_queries="$2"
      shift 2
      ;;
    --start)
      start_bucket="$2"
      shift 2
      ;;
    --end)
      end_bucket="$2"
      shift 2
      ;;
    --filter-template)
      filter_template="$2"
      shift 2
      ;;
    --out-csv)
      out_csv="$2"
      shift 2
      ;;
    --workdir)
      workdir="$2"
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

if [[ -z "$graph_path" || -z "$query_path" || -z "$metadata_csv" ]]; then
  echo "Missing required arguments. --graph, --query, and --metadata-csv are required." >&2
  usage
  exit 1
fi

if [[ ! -x "$runner" ]]; then
  echo "Runner not found or not executable: $runner" >&2
  exit 1
fi
if [[ ! -f "$graph_path" ]]; then
  echo "Graph file not found: $graph_path" >&2
  exit 1
fi
if [[ ! -f "$query_path" ]]; then
  echo "Query file not found: $query_path" >&2
  exit 1
fi
if [[ ! -f "$metadata_csv" ]]; then
  echo "Metadata CSV not found: $metadata_csv" >&2
  exit 1
fi
if [[ "$start_bucket" -gt "$end_bucket" ]]; then
  echo "Invalid range: --start must be <= --end" >&2
  exit 1
fi

if [[ -n "$workdir" ]]; then
  if [[ ! -d "$workdir" ]]; then
    echo "workdir does not exist: $workdir" >&2
    exit 1
  fi
  tmp_dir="$(mktemp -d "${workdir%/}/bucket_sweep.XXXXXX")"
else
  tmp_dir="$(mktemp -d)"
fi
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$(dirname "$out_csv")"
printf 'bucket,recall,selectivity_count,selectivity_ratio,status\n' > "$out_csv"

buckets_total=0
buckets_success=0
buckets_failed=0
recall_numeric_count=0
recall_na_count=0
sum_recall="0"

for bucket in $(seq "$start_bucket" "$end_bucket"); do
  buckets_total=$((buckets_total + 1))

  filter_expr="$(render_filter "$filter_template" "$field" "$bucket")"
  summary_file="$tmp_dir/summary_${bucket}.txt"
  run_log="$tmp_dir/run_${bucket}.log"

  status="OK"
  recall="N/A"
  selectivity_count="N/A"
  selectivity_ratio="N/A"

  if "$runner" \
      --dataset-type "$dataset_type" \
      --graph "$graph_path" \
      --query "$query_path" \
      --k "$k" \
      --ef "$ef" \
      --filter "$filter_expr" \
      --metadata-csv "$metadata_csv" \
      --num-queries "$num_queries" \
      --summary-out "$summary_file" \
      > "$run_log" 2>&1; then
    buckets_success=$((buckets_success + 1))

    if [[ -f "$summary_file" ]]; then
      parsed_recall="$(extract_summary_value "average_recall_at_k" "$summary_file" || true)"
      parsed_sel_count="$(extract_summary_value "selectivity_count" "$summary_file" || true)"
      parsed_sel_ratio="$(extract_summary_value "selectivity_ratio" "$summary_file" || true)"

      if [[ -n "$parsed_sel_count" ]]; then
        selectivity_count="$parsed_sel_count"
      fi
      if [[ -n "$parsed_sel_ratio" ]]; then
        selectivity_ratio="$parsed_sel_ratio"
      fi

      if [[ -n "$parsed_recall" ]] && is_numeric "$parsed_recall"; then
        recall="$parsed_recall"
        recall_numeric_count=$((recall_numeric_count + 1))
        sum_recall="$(awk -v a="$sum_recall" -v b="$recall" 'BEGIN {printf "%.12f", a + b}')"
      else
        recall_na_count=$((recall_na_count + 1))
        status="RECALL_NA"
      fi
    else
      recall_na_count=$((recall_na_count + 1))
      status="SUMMARY_MISSING"
    fi
  else
    buckets_failed=$((buckets_failed + 1))
    recall_na_count=$((recall_na_count + 1))
    status="RUN_FAIL"
  fi

  printf '%s,%s,%s,%s,%s\n' "$bucket" "$recall" "$selectivity_count" "$selectivity_ratio" "$status" >> "$out_csv"
done

if [[ "$recall_numeric_count" -gt 0 ]]; then
  average_recall="$(awk -v s="$sum_recall" -v n="$recall_numeric_count" 'BEGIN {printf "%.6f", s / n}')"
else
  average_recall="N/A"
fi

echo "Bucket recall sweep finished"
echo "out_csv: $out_csv"
echo "buckets_total: $buckets_total"
echo "buckets_success: $buckets_success"
echo "buckets_failed: $buckets_failed"
echo "recall_numeric_count: $recall_numeric_count"
echo "recall_na_count: $recall_na_count"
echo "average_recall_excluding_na: $average_recall"
