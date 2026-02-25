# compass_search_w_iaa

## Purpose
`compass_search_w_iaa.cpp` runs HNSW filter search using prebuilt FID/TB files from a manifest and Intel QPL hardware path (IAA) for startup compression and query-time block decompression.

Filtering semantics follow your `graph_filter_exp` logic:
- Traversal gate: `FID match OR TB connector match`
- Final result gate: `FID match only`

## CLI
Required:
- `--dataset-type <sift|sift1m|sift1b|laion|hnm>`
- `--graph <path.bin|path.index>`
- `--query <path.fvecs|path.bvecs>`
- `--k <int>`
- `--filter "<expression>"`
- `--fidtb-manifest <path>`

Optional:
- `--ef <int>`
- `--num-queries <int>` or `--max-queries <int>`
- `--fid-block-size-bytes <int>` (default `8192`)
- `--tb-block-size-bytes <int>` (default `8192`)
- `--topk-out <path>`
- `--per-query-out <path>`
- `--summary-out <path>`

## Key Variables
- `fid_block_size_bytes`: block size used to compress/decompress FID payload bytes.
- `tb_block_size_bytes`: block size used to compress/decompress TB payload bytes.
- `k`: number of final ANN results.
- `ef`: HNSW search breadth at level 0.
- `num_queries` / `max_queries`: cap on executed queries.

## How It Works
1. Parse filter expression (`filter_expr`) and collect referenced fields.
2. Load `manifest.json`, then load only referenced attribute FID/TB files.
3. Compress raw FID/TB payloads into independent blocks with QPL hardware compression.
4. During each query:
   - Use per-query block caches (`attribute_id + block_id`).
   - Decompress a block at most once per query.
   - Evaluate filter recursively over AST (`AND/OR`, comparisons, `IN`, `BETWEEN`).
5. HNSW traversal uses `allow_traversal`; insertion to candidate results uses `allow_result`.
6. Compute ENNS on filtered candidates for recall@k and print summary/QPS.

## Numeric/Categorical Predicate Mapping
- Numeric attributes require `numeric_minmax_quantized` metadata in manifest.
- Bucket-level predicate evaluation is used for `==`, `!=`, range, `IN`, `BETWEEN`.
- Categorical attributes use `category_map` from manifest.

## Metrics
Summary includes:
- `average_recall_at_k`
- `qps`
- `search_loop_time_ms`, `avg_query_time_ms`
- `filter_eval_time_ms`, `avg_filter_eval_ns`
- `iaa_decompress_time_ms`
- `avg_decompress_time_per_query_ms`
- `fid_blocks_decompressed`, `tb_blocks_decompressed`
- `fid_cache_hits`, `tb_cache_hits`
- `fid_bytes_decompressed`, `tb_bytes_decompressed`

For compatibility, summary also emits `lz4_decompress_time_ms` with the same value as `iaa_decompress_time_ms`.

## Failure Modes
- Manifest does not contain a referenced field.
- Manifest metadata is insufficient for strict predicate translation.
- FID/TB payload element counts do not match `n_elements`.
- QPL hardware path is not available (`qpl_init_job` failure).
- Graph/query dimension mismatch or unsupported file extension.
