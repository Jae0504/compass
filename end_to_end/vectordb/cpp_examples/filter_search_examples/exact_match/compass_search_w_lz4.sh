#!/usr/bin/env bash
set -euo pipefail

# Build binary:
#   make -C . compass_search_w_lz4.run

# Example 1: SIFT with manifest-generated FID/TB.
# ./compass_search_w_lz4.run \
#   --dataset-type sift \
#   --graph /path/to/index.bin \
#   --query /path/to/query.fvecs \
#   --k 10 \
#   --ef 200 \
#   --num-queries 1000 \
#   --fidtb-manifest /path/to/fid_tb/sift1m/manifest.json \
#   --filter "synthetic_id_bucket == 0" \
#   --topk-out ./out/sift_topk.txt \
#   --per-query-out ./out/sift_per_query.csv \
#   --summary-out ./out/sift_summary.txt

# Example 2: Multi-attribute expression with AND/OR from manifest attributes.
# ./compass_search_w_lz4.run \
#   --dataset-type laion \
#   --graph /path/to/index.index \
#   --query /path/to/query.fvecs \
#   --k 10 \
#   --ef 200 \
#   --num-queries 500 \
#   --fidtb-manifest /path/to/fid_tb/laion/manifest.json \
#   --filter "(NSFW IN [\"UNLIKELY\",\"UNSURE\"]) AND (similarity >= 0.25 OR original_width >= 1024)" \
#   --summary-out ./out/laion_summary.txt

# Example 3: Bvec query/index path.
# ./compass_search_w_lz4.run \
#   --dataset-type hnm \
#   --graph /path/to/index.bin \
#   --query /path/to/query.bvecs \
#   --k 10 \
#   --ef 200 \
#   --num-queries 1000 \
#   --fidtb-manifest /path/to/fid_tb/hnm/manifest.json \
#   --filter "price BETWEEN 10 AND 50 OR category IN [\"home\",\"office\"]" \
#   --summary-out ./out/hnm_summary.txt

echo "Uncomment and edit one of the examples in $(basename "$0"), then run it."
