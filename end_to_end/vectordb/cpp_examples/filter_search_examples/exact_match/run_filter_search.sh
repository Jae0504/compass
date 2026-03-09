#!/usr/bin/env bash
set -euo pipefail

# Build binary:
#   make -C .
# Note:
#   ENNS+recall computation is always enabled and can be expensive when selectivity is high.

# Example 1: SIFT (.fvecs/.bvecs) with sidecar CSV metadata containing explicit id column.
# ./hnswlib_filter_search.run \
#   --dataset-type sift \
#   --graph /path/to/index.bin \
#   --query /path/to/query.fvecs \
#   --k 10 \
#   --ef 200 \
#   --filter "country IN [\"US\",\"KR\"] AND score BETWEEN 10 AND 20" \
#   --metadata-csv /path/to/sift_metadata.csv \
#   --id-column id \
#   --max-queries 1000 \
#   --topk-out ./out/sift_topk.txt \
#   --per-query-out ./out/sift_per_query.csv \
#   --summary-out ./out/sift_summary.txt

# Example 2: LAION/HNM with JSONL payload (row index == label id).
# ./hnswlib_filter_search.run \
#   --dataset-type laion \
#   --graph /path/to/index.bin \
#   --query /path/to/query.fvecs \
#   --k 10 \
#   --filter "(NSFW == \"UNLIKELY\" OR NSFW == \"UNSURE\") AND similarity >= 0.25" \
#   --payload-jsonl /path/to/payloads.jsonl \
#   --max-queries 1000 \
#   --topk-out ./out/laion_topk.txt \
#   --per-query-out ./out/laion_per_query.csv \
#   --summary-out ./out/laion_summary.txt

echo "Uncomment and edit one of the examples in $(basename "$0"), then run it."
