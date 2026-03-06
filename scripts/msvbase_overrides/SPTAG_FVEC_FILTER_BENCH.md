# SPTAG fvec Filter Benchmark (MSVBASE)

This document explains how to run `MSVBASE/scripts/sptag_fvec_filter_bench.py`, what it implements, and how to reuse it when `MSVBASE` is a submodule in another repo/server.

## 1. What the script does

`MSVBASE/scripts/sptag_fvec_filter_bench.py` automates:

1. Docker build + container start for MSVBASE.
2. Loading `base.fvecs` into PostgreSQL (`id`, `synthetic_id_bucket`, `embedding FLOAT8[]`).
3. Building an SPTAG index with L2 distance.
4. Running filtered top-k SQL queries from `query.fvecs`.
5. Computing exact filtered L2 ground truth and reporting recall + latency/QPS.

## 2. Repo layout assumption (important for submodule use)

The script resolves:

- `MSVBASE_ROOT = <repo_root>/MSVBASE`
- `REPO_ROOT = <repo_root>`

Container mapping is `/vectordb -> <repo_root>`.
Because of that, files used by SQL (`base_for_copy.tsv`, SQL scripts, etc.) must be inside `<repo_root>`.

Recommended layout:

```text
<repo_root>/
  MSVBASE/                    # submodule
  temp/base.fvecs
  temp/query.fvecs
```

Run commands from `<repo_root>`.

## 3. Prerequisites

- Docker available.
- `sudo` access for Docker commands.
- Python 3.8+.
- Optional: `numpy` (faster ground-truth recall computation).

## 4. How to run

### 4.1 Full pipeline (prepare + query)

```bash
cd /path/to/repo_root
python3 MSVBASE/scripts/sptag_fvec_filter_bench.py \
  --mode all \
  --base-fvecs /path/to/repo_root/temp/base.fvecs \
  --query-fvecs /path/to/repo_root/temp/query.fvecs \
  --k 10 \
  --num-queries 100 \
  --nfilters 100
```

### 4.2 Prepare only (build DB table + SPTAG index)

```bash
python3 MSVBASE/scripts/sptag_fvec_filter_bench.py \
  --mode prepare \
  --base-fvecs /path/to/repo_root/temp/base.fvecs \
  --nfilters 100
```

### 4.3 Query only (reuse prepared DB/index)

```bash
python3 MSVBASE/scripts/sptag_fvec_filter_bench.py \
  --mode query \
  --base-fvecs /path/to/repo_root/temp/base.fvecs \
  --query-fvecs /path/to/repo_root/temp/query.fvecs \
  --k 10 \
  --num-queries 100 \
  --nfilters 100 \
  --filter-bucket 99
```

### 4.4 Useful smoke test

```bash
python3 MSVBASE/scripts/sptag_fvec_filter_bench.py \
  --mode all \
  --base-fvecs /path/to/repo_root/temp/base.fvecs \
  --query-fvecs /path/to/repo_root/temp/query.fvecs \
  --base-limit 20000 \
  --num-queries 20 \
  --k 10 \
  --nfilters 100
```

## 5. Script defaults

- `--mode all`
- `--k 10`
- `--num-queries 100`
- `--nfilters 100`
- `--filter-bucket nfilters-1`
- `--container-name vbase_open_source`
- `--db-name sptag_bench`
- `--out-dir /home/jykang5/compass/temp/sptag_bench_out`

## 6. Implementation details

### 6.1 Synthetic metadata assignment

For loaded base vector count `N` and filter count `nfilters`:

- `block_size = ceil(N / nfilters)`
- `synthetic_id_bucket = min(floor(id / block_size), nfilters - 1)`

This creates contiguous id ranges per filter bucket.

### 6.2 SQL used in prepare phase

The script generates SQL to:

- create DB if missing,
- create extension `vectordb`,
- create benchmark table,
- bulk load TSV via `\copy`,
- create B-tree index on `synthetic_id_bucket`,
- create SPTAG index:

```sql
CREATE INDEX sptag_bench_embedding_idx ON sptag_bench_vectors
USING sptag(embedding vector_l2_ops)
WITH (distmethod=l2_distance, threads=<sptag_threads>);
```

### 6.3 SQL used in query phase

For each query vector:

```sql
SELECT id, embedding <-> ARRAY[...]::float8[] AS dist
FROM sptag_bench_vectors
WHERE synthetic_id_bucket = <filter_bucket>
ORDER BY dist
LIMIT <k>;
```

The script captures `psql` timing and result IDs per query.

### 6.4 Metrics computed

- `average_recall_at_k` vs exact filtered L2 top-k.
- Latency: avg / p50 / p95 / p99.
- `qps = num_queries / total_latency_seconds`.
- Candidate count and selectivity for the chosen bucket.

## 7. Output artifacts

Under `--out-dir`:

- `base_for_copy.tsv`
- `prepare.sql`
- `prepare_psql_output.txt`
- `prepare_meta.json`
- `query.sql`
- `explain.sql`
- `explain_output.txt`
- `query_psql_output.txt`
- `per_query.csv`
- `summary.txt`

## 8. Where index/data is stored and low-space servers

SPTAG index files are saved under PostgreSQL `DataDir/DatabasePath/<index_name>`.
In this image:

- `PGDATA` is under `/u02/pgdata/13` inside container.

If you do not mount `/u02`, data goes to Docker writable layer (often space-limited).
For large datasets, mount `/u02` to a large host disk:

```bash
sudo docker rm -f vbase_open_source
sudo mkdir -p /mnt/bigdisk/msvbase_u02
sudo chmod 777 /mnt/bigdisk/msvbase_u02

cd /path/to/repo_root
sudo docker run --name=vbase_open_source \
  -e PGPASSWORD=vectordb -e PGUSERNAME=vectordb -e PGDATABASE=vectordb \
  -v /mnt/bigdisk/msvbase_u02:/u02 \
  -v "$PWD":/vectordb \
  vbase_open_source &
```

If needed, update `MSVBASE/scripts/dockerrun.sh` similarly so the script can reuse it unchanged.

## 9. Naming with dataset + nfilter

Current script uses fixed names:

- table: `sptag_bench_vectors`
- vector index: `sptag_bench_embedding_idx`
- filter index: `sptag_bench_bucket_idx`

If you want naming like `sift1m_nf_100`, either:

1. Edit constants in `sptag_fvec_filter_bench.py`, or
2. Add CLI options for dataset/index naming.

## 10. Moving this patch to another server (MSVBASE as submodule)

From source repo:

```bash
git -C MSVBASE diff -- \
  Dockerfile \
  scripts/dockerbuild.sh \
  scripts/sptag_fvec_filter_bench.py \
  scripts/SPTAG_FVEC_FILTER_BENCH.md \
  > /tmp/msvbase_sptag_bench.patch
```

On target repo (same submodule path):

```bash
git -C MSVBASE apply /tmp/msvbase_sptag_bench.patch
```

Then run commands from Section 4 on that server.
