#!/usr/bin/env python3
"""SPTAG filtered-search benchmark runner for MSVBASE.

This script orchestrates:
1) Docker build/run using MSVBASE scripts
2) Loading .fvecs vectors into PostgreSQL float8[] table
3) Building an SPTAG L2 index
4) Running filtered top-k SQL queries
5) Computing exact filtered L2 recall and latency statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


SCRIPT_PATH = Path(__file__).resolve()
MSVBASE_ROOT = SCRIPT_PATH.parent.parent
REPO_ROOT = MSVBASE_ROOT.parent

DEFAULT_BASE_FVECS = Path("/home/jykang5/compass/temp/base.fvecs")
DEFAULT_QUERY_FVECS = Path("/home/jykang5/compass/temp/query.fvecs")
DEFAULT_OUT_DIR = Path("/home/jykang5/compass/temp/sptag_bench_out")

DEFAULT_TABLE_NAME = "sptag_bench_vectors"
DEFAULT_INDEX_NAME = "sptag_bench_embedding_idx"
DEFAULT_FILTER_INDEX_NAME = "sptag_bench_bucket_idx"
DEFAULT_META_JSON = "prepare_meta.json"


@dataclass
class FvecInfo:
    dim: int
    total_vectors: int
    record_bytes: int


@dataclass
class PreparedMetadata:
    dim: int
    loaded_base_count: int
    source_base_count: int
    nfilters: int
    block_size: int
    table_name: str
    db_name: str
    tsv_path: str


def log(msg: str) -> None:
    print(f"[sptag_bench] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPTAG L2 filtered-search benchmark runner (MSVBASE + SQL + fvecs)."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "prepare", "query"),
        default="all",
        help="Execution mode: prepare (load/index), query (search/report), or all.",
    )
    parser.add_argument(
        "--base-fvecs",
        type=Path,
        default=DEFAULT_BASE_FVECS,
        help=f"Path to base .fvecs (default: {DEFAULT_BASE_FVECS})",
    )
    parser.add_argument(
        "--query-fvecs",
        type=Path,
        default=DEFAULT_QUERY_FVECS,
        help=f"Path to query .fvecs (default: {DEFAULT_QUERY_FVECS})",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k value (default: 10).")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to run from query.fvecs (default: 100; first N queries).",
    )
    parser.add_argument(
        "--base-limit",
        type=int,
        default=None,
        help="Optional cap for number of base vectors loaded/indexed (for smoke tests).",
    )
    parser.add_argument(
        "--nfilters",
        type=int,
        default=100,
        help="Number of synthetic metadata buckets (default: 100).",
    )
    parser.add_argument(
        "--filter-bucket",
        type=int,
        default=None,
        help="Synthetic bucket id for WHERE synthetic_id_bucket = <bucket> (default: nfilters-1).",
    )
    parser.add_argument(
        "--sptag-threads",
        type=int,
        default=None,
        help="SPTAG index threads (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="vbase_open_source",
        help="Docker container name for SQL exec (default: vbase_open_source).",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="sptag_bench",
        help="Postgres database name for benchmark objects (default: sptag_bench).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for SQL/results/artifacts (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args()


def validate_identifier(name: str, what: str) -> None:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(
            f"Invalid {what} '{name}'. Use letters/digits/underscore, starting with letter/underscore."
        )


def ensure_file(path: Path, what: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    where = f" (cwd={cwd})" if cwd else ""
    log(f"Running: {pretty}{where}")
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
    )
    if check and proc.returncode != 0:
        msg = [f"Command failed with exit code {proc.returncode}: {pretty}"]
        if proc.stdout:
            msg.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            msg.append(f"stderr:\n{proc.stderr}")
        raise RuntimeError("\n".join(msg))
    return proc


def host_to_container_path(host_path: Path) -> str:
    host_abs = host_path.resolve()
    try:
        rel = host_abs.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Path must be inside repo root {REPO_ROOT} so container can see it: {host_abs}"
        ) from exc
    return str(Path("/vectordb") / rel)


def inspect_fvec(path: Path) -> FvecInfo:
    ensure_file(path, ".fvecs file")
    size = path.stat().st_size
    if size < 4:
        raise ValueError(f"Invalid .fvecs file (too small): {path}")
    with path.open("rb") as f:
        dim_bytes = f.read(4)
    dim = struct.unpack("<i", dim_bytes)[0]
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension in {path}: {dim}")
    record_bytes = 4 + 4 * dim
    if size % record_bytes != 0:
        raise ValueError(
            f"{path} size {size} is not divisible by record size {record_bytes} for dim={dim}"
        )
    total = size // record_bytes
    return FvecInfo(dim=dim, total_vectors=total, record_bytes=record_bytes)


def _read_next_fvec_raw(fh, expected_dim: int) -> bytes:
    dim_bytes = fh.read(4)
    if not dim_bytes:
        raise EOFError("Unexpected EOF while reading dim header")
    if len(dim_bytes) != 4:
        raise EOFError("Truncated dim header")
    dim = struct.unpack("<i", dim_bytes)[0]
    if dim != expected_dim:
        raise ValueError(f"Dimension mismatch in record: got {dim}, expected {expected_dim}")
    vec_bytes = fh.read(expected_dim * 4)
    if len(vec_bytes) != expected_dim * 4:
        raise EOFError("Truncated vector payload")
    return vec_bytes


def _vec_bytes_to_list(vec_bytes: bytes, dim: int) -> List[float]:
    vals = struct.unpack("<" + "f" * dim, vec_bytes)
    return [float(v) for v in vals]


def _vec_bytes_to_np(vec_bytes: bytes, dim: int):
    return np.frombuffer(vec_bytes, dtype="<f4", count=dim).astype(np.float32, copy=True)


def vector_to_pg_array(vec: Sequence[float]) -> str:
    return "{" + ",".join(f"{float(v):.8f}" for v in vec) + "}"


def write_base_tsv(
    base_fvecs: Path,
    tsv_out: Path,
    *,
    nfilters: int,
    base_limit: Optional[int],
) -> PreparedMetadata:
    info = inspect_fvec(base_fvecs)
    loaded = info.total_vectors if base_limit is None else min(base_limit, info.total_vectors)
    if loaded <= 0:
        raise ValueError("No base vectors to load after applying --base-limit")

    block_size = math.ceil(loaded / nfilters)
    progress_every = 10_000

    log(
        f"Converting base fvecs to TSV: dim={info.dim}, source_n={info.total_vectors}, "
        f"loaded_n={loaded}, nfilters={nfilters}, block_size={block_size}"
    )

    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    with base_fvecs.open("rb") as fin, tsv_out.open("w", encoding="utf-8") as fout:
        for idx in range(loaded):
            vec_bytes = _read_next_fvec_raw(fin, info.dim)
            vec = _vec_bytes_to_list(vec_bytes, info.dim)
            gid = idx // block_size
            if gid >= nfilters:
                gid = nfilters - 1
            fout.write(f"{idx}\t{gid}\t{vector_to_pg_array(vec)}\n")
            if (idx + 1) % progress_every == 0:
                log(f"  wrote {idx + 1}/{loaded} rows")

    return PreparedMetadata(
        dim=info.dim,
        loaded_base_count=loaded,
        source_base_count=info.total_vectors,
        nfilters=nfilters,
        block_size=block_size,
        table_name=DEFAULT_TABLE_NAME,
        db_name="",
        tsv_path=str(tsv_out),
    )


def read_first_queries(query_fvecs: Path, expected_dim: int, n: int):
    info = inspect_fvec(query_fvecs)
    if info.dim != expected_dim:
        raise ValueError(
            f"Query dim mismatch: query dim={info.dim}, expected base/index dim={expected_dim}"
        )
    actual = min(n, info.total_vectors)
    if actual <= 0:
        raise ValueError("No query vectors available")
    vectors = []
    with query_fvecs.open("rb") as fin:
        for _ in range(actual):
            vec_bytes = _read_next_fvec_raw(fin, info.dim)
            if HAS_NUMPY:
                vectors.append(_vec_bytes_to_np(vec_bytes, info.dim))
            else:
                vectors.append(_vec_bytes_to_list(vec_bytes, info.dim))

    if HAS_NUMPY:
        return np.vstack(vectors)
    return vectors


def bucket_bounds(total_n: int, nfilters: int, bucket: int) -> Tuple[int, int]:
    block = math.ceil(total_n / nfilters)
    start = bucket * block
    end = min(total_n, (bucket + 1) * block)
    return start, end


def load_base_vectors_in_range(
    base_fvecs: Path, dim: int, start: int, end: int
) -> Tuple[Sequence[int], Sequence[Sequence[float]]]:
    if start >= end:
        if HAS_NUMPY:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, dim), dtype=np.float32)
        return [], []

    info = inspect_fvec(base_fvecs)
    if info.dim != dim:
        raise ValueError(f"Base dim mismatch: file dim={info.dim}, expected={dim}")

    ids: List[int] = []
    vectors = []

    with base_fvecs.open("rb") as fin:
        fin.seek(start * info.record_bytes)
        for idx in range(start, end):
            vec_bytes = _read_next_fvec_raw(fin, dim)
            ids.append(idx)
            if HAS_NUMPY:
                vectors.append(_vec_bytes_to_np(vec_bytes, dim))
            else:
                vectors.append(_vec_bytes_to_list(vec_bytes, dim))

    if HAS_NUMPY:
        return np.asarray(ids, dtype=np.int64), np.vstack(vectors)
    return ids, vectors


def docker_prepare_environment(container_name: str) -> None:
    if container_name != "vbase_open_source":
        raise ValueError(
            "--container-name must be vbase_open_source for prepare/all because "
            "MSVBASE/scripts/dockerrun.sh hardcodes this name."
        )

    run_cmd(["sudo", "./scripts/dockerbuild.sh"], cwd=MSVBASE_ROOT)
    run_cmd(["sudo", "docker", "rm", "-f", "vbase_open_source"], check=False)
    run_cmd(["sudo", "./MSVBASE/scripts/dockerrun.sh"], cwd=REPO_ROOT)


def ensure_container_running(container_name: str) -> None:
    proc = run_cmd(
        ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Docker container '{container_name}' is not available. "
            "Run prepare/all mode to build/start it."
        )
    state = (proc.stdout or "").strip().lower()
    if state != "true":
        raise RuntimeError(f"Docker container '{container_name}' exists but is not running.")


def run_psql_file_in_container(
    container_name: str, db_name: str, sql_file_host: Path
) -> str:
    sql_in_container = host_to_container_path(sql_file_host)
    proc = run_cmd(
        [
            "sudo",
            "docker",
            "exec",
            "-i",
            container_name,
            "psql",
            "-U",
            "vectordb",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            db_name,
            "-f",
            sql_in_container,
        ],
        capture_output=True,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def run_psql_scalar(container_name: str, db_name: str, sql: str) -> str:
    proc = run_cmd(
        [
            "sudo",
            "docker",
            "exec",
            "-i",
            container_name,
            "psql",
            "-U",
            "vectordb",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            db_name,
            "-t",
            "-A",
            "-c",
            sql,
        ],
        capture_output=True,
    )
    return (proc.stdout or "").strip()


def generate_prepare_sql(
    *,
    db_name: str,
    table_name: str,
    copy_tsv_container_path: str,
    sptag_threads: int,
) -> str:
    esc_db = db_name.replace("'", "''")
    esc_tsv = copy_tsv_container_path.replace("'", "''")
    return f"""\\set ON_ERROR_STOP on
SELECT 'CREATE DATABASE {db_name}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{esc_db}')\\gexec
\\c {db_name}
CREATE EXTENSION IF NOT EXISTS vectordb;
DROP TABLE IF EXISTS {table_name};
CREATE TABLE {table_name} (
    id INT PRIMARY KEY,
    synthetic_id_bucket INT NOT NULL,
    embedding FLOAT8[] NOT NULL
);
\\copy {table_name}(id, synthetic_id_bucket, embedding) FROM '{esc_tsv}' WITH (FORMAT csv, DELIMITER E'\\t', QUOTE E'\\x01')
CREATE INDEX {DEFAULT_FILTER_INDEX_NAME} ON {table_name}(synthetic_id_bucket);
CREATE INDEX {DEFAULT_INDEX_NAME} ON {table_name}
USING sptag(embedding vector_l2_ops) WITH (distmethod=l2_distance, threads={sptag_threads});
ANALYZE {table_name};
"""


def _vector_to_sql_array(vec: Sequence[float]) -> str:
    return "ARRAY[" + ",".join(f"{float(v):.8f}" for v in vec) + "]::float8[]"


def generate_query_sql(
    *,
    db_name: str,
    table_name: str,
    filter_bucket: int,
    k: int,
    query_vectors: Sequence[Sequence[float]],
) -> str:
    lines = [
        "\\set ON_ERROR_STOP on",
        f"\\c {db_name}",
        "SET enable_seqscan=off;",
        "SET enable_indexscan=on;",
        "\\pset format unaligned",
        "\\pset tuples_only on",
        "\\timing on",
    ]
    for i, q in enumerate(query_vectors):
        qarr = _vector_to_sql_array(q)
        lines.extend(
            [
                f"\\echo QUERY {i}",
                (
                    "SELECT COALESCE(string_agg(id::text, ',' ORDER BY dist), '') "
                    "FROM ("
                    f"SELECT id, embedding <-> {qarr} AS dist "
                    f"FROM {table_name} "
                    f"WHERE synthetic_id_bucket = {filter_bucket} "
                    "ORDER BY dist "
                    f"LIMIT {k}"
                    ") AS ranked;"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def generate_explain_sql(
    *,
    db_name: str,
    table_name: str,
    filter_bucket: int,
    k: int,
    first_query: Sequence[float],
) -> str:
    qarr = _vector_to_sql_array(first_query)
    return (
        "\\set ON_ERROR_STOP on\n"
        f"\\c {db_name}\n"
        "SET enable_seqscan=off;\n"
        "SET enable_indexscan=on;\n"
        "EXPLAIN (ANALYZE, VERBOSE, BUFFERS)\n"
        "SELECT id\n"
        f"FROM {table_name}\n"
        f"WHERE synthetic_id_bucket = {filter_bucket}\n"
        f"ORDER BY embedding <-> {qarr}\n"
        f"LIMIT {k};\n"
    )


def parse_id_csv(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if not re.match(r"^-?\d+$", token):
            continue
        out.append(int(token))
    return out


def parse_query_output(raw_text: str, expected_queries: int) -> Tuple[List[List[int]], List[float]]:
    results_by_idx = {}
    lat_by_idx = {}

    current_q: Optional[int] = None
    current_res = ""
    time_re = re.compile(r"^Time:\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)$")

    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        mq = re.match(r"^QUERY\s+(\d+)$", s)
        if mq:
            current_q = int(mq.group(1))
            current_res = ""
            continue

        mt = time_re.match(s)
        if mt and current_q is not None:
            val = float(mt.group(1))
            unit = mt.group(2)
            lat_ms = val * 1000.0 if unit == "s" else val
            lat_by_idx[current_q] = lat_ms
            results_by_idx[current_q] = parse_id_csv(current_res)
            current_q = None
            current_res = ""
            continue

        if s.startswith("SET"):
            continue
        if s.startswith("Timing is"):
            continue
        if s.startswith("You are now connected"):
            continue
        if s.startswith("psql:"):
            continue
        if s.startswith("(") and "row" in s:
            continue

        if current_q is not None:
            current_res = s

    missing = [i for i in range(expected_queries) if i not in lat_by_idx]
    if missing:
        raise RuntimeError(
            f"Failed to parse timing/result for {len(missing)} queries. Missing indices: {missing[:10]}"
        )

    ordered_results = [results_by_idx.get(i, []) for i in range(expected_queries)]
    ordered_lats = [lat_by_idx[i] for i in range(expected_queries)]
    return ordered_results, ordered_lats


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = pct * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def exact_topk_l2_numpy(
    query_vectors, candidate_ids, candidate_vectors, k: int
) -> List[List[int]]:
    out: List[List[int]] = []
    for q in query_vectors:
        if candidate_vectors.shape[0] == 0:
            out.append([])
            continue
        dists = np.sum((candidate_vectors - q) ** 2, axis=1)
        if k < dists.shape[0]:
            idx = np.argpartition(dists, kth=k - 1)[:k]
            idx = idx[np.argsort(dists[idx], kind="stable")]
        else:
            idx = np.argsort(dists, kind="stable")
        out.append(candidate_ids[idx][:k].astype(np.int64).tolist())
    return out


def exact_topk_l2_python(
    query_vectors: Sequence[Sequence[float]],
    candidate_ids: Sequence[int],
    candidate_vectors: Sequence[Sequence[float]],
    k: int,
) -> List[List[int]]:
    import heapq

    out: List[List[int]] = []
    for q in query_vectors:
        if not candidate_ids:
            out.append([])
            continue
        heap: List[Tuple[float, int]] = []
        for cid, cvec in zip(candidate_ids, candidate_vectors):
            dist = 0.0
            for qv, cv in zip(q, cvec):
                d = float(qv) - float(cv)
                dist += d * d
            item = (-dist, int(cid))
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        ranked = sorted([(-neg_d, cid) for (neg_d, cid) in heap], key=lambda x: x[0])
        out.append([cid for _, cid in ranked])
    return out


def compute_recall_metrics(
    ann_results: Sequence[Sequence[int]], gt_results: Sequence[Sequence[int]], k: int
) -> Tuple[List[dict], float]:
    rows = []
    recall_sum = 0.0
    for i, (ann, gt) in enumerate(zip(ann_results, gt_results)):
        ann_set = set(ann)
        gt_set = set(gt)
        overlap = len(ann_set.intersection(gt_set))
        denom = min(k, len(gt))
        if denom == 0:
            recall = 1.0 if not ann else 0.0
        else:
            recall = overlap / float(denom)
        recall_sum += recall
        rows.append(
            {
                "query_id": i,
                "ann_count": len(ann),
                "gt_count": len(gt),
                "overlap": overlap,
                "recall_at_k": recall,
                "ann_ids": ",".join(str(x) for x in ann),
                "gt_ids": ",".join(str(x) for x in gt),
            }
        )
    avg = recall_sum / len(rows) if rows else 0.0
    return rows, avg


def run_prepare(args: argparse.Namespace, out_dir: Path, sptag_threads: int) -> PreparedMetadata:
    ensure_file(args.base_fvecs, "base .fvecs")

    docker_prepare_environment(args.container_name)
    ensure_container_running(args.container_name)

    base_tsv = out_dir / "base_for_copy.tsv"
    prep = write_base_tsv(
        args.base_fvecs,
        base_tsv,
        nfilters=args.nfilters,
        base_limit=args.base_limit,
    )
    prep.db_name = args.db_name
    prep.table_name = DEFAULT_TABLE_NAME

    prepare_sql = out_dir / "prepare.sql"
    prepare_sql.write_text(
        generate_prepare_sql(
            db_name=args.db_name,
            table_name=DEFAULT_TABLE_NAME,
            copy_tsv_container_path=host_to_container_path(base_tsv),
            sptag_threads=sptag_threads,
        ),
        encoding="utf-8",
    )

    prepare_raw = run_psql_file_in_container(args.container_name, "vectordb", prepare_sql)
    (out_dir / "prepare_psql_output.txt").write_text(prepare_raw, encoding="utf-8")

    count_s = run_psql_scalar(
        args.container_name,
        args.db_name,
        f"SELECT count(*) FROM {DEFAULT_TABLE_NAME};",
    )
    loaded_rows = int(count_s)
    if loaded_rows != prep.loaded_base_count:
        raise RuntimeError(
            f"Loaded row count mismatch: expected {prep.loaded_base_count}, got {loaded_rows}"
        )

    meta = {
        "dim": prep.dim,
        "loaded_base_count": prep.loaded_base_count,
        "source_base_count": prep.source_base_count,
        "nfilters": prep.nfilters,
        "block_size": prep.block_size,
        "table_name": DEFAULT_TABLE_NAME,
        "db_name": args.db_name,
        "base_fvecs": str(args.base_fvecs.resolve()),
        "base_tsv": str(base_tsv.resolve()),
    }
    (out_dir / DEFAULT_META_JSON).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("Prepare phase completed successfully.")
    return prep


def run_query(args: argparse.Namespace, out_dir: Path, sptag_threads: int) -> None:
    del sptag_threads  # unused in query mode
    ensure_file(args.base_fvecs, "base .fvecs")
    ensure_file(args.query_fvecs, "query .fvecs")
    ensure_container_running(args.container_name)

    meta_path = out_dir / DEFAULT_META_JSON
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    table_name = str(meta.get("table_name", DEFAULT_TABLE_NAME))
    db_name = str(meta.get("db_name", args.db_name))
    nfilters = int(meta.get("nfilters", args.nfilters))
    validate_identifier(table_name, "table name")
    validate_identifier(db_name, "database name")

    row_count = int(
        run_psql_scalar(args.container_name, db_name, f"SELECT count(*) FROM {table_name};")
    )
    if row_count <= 0:
        raise RuntimeError(f"Table {table_name} has no rows; run prepare mode first.")

    base_info = inspect_fvec(args.base_fvecs)
    if row_count > base_info.total_vectors:
        raise RuntimeError(
            f"DB row count {row_count} exceeds vectors in base file {base_info.total_vectors}."
        )

    loaded_n = row_count
    dim = int(meta.get("dim", base_info.dim))
    if dim != base_info.dim:
        raise RuntimeError(f"Base dim mismatch: meta/db dim={dim}, base file dim={base_info.dim}")

    filter_bucket = args.filter_bucket if args.filter_bucket is not None else (nfilters - 1)
    if filter_bucket < 0 or filter_bucket >= nfilters:
        raise ValueError(f"--filter-bucket must be in [0, {nfilters - 1}]")

    query_vectors = read_first_queries(args.query_fvecs, dim, args.num_queries)
    query_count = len(query_vectors)

    query_sql = out_dir / "query.sql"
    query_sql.write_text(
        generate_query_sql(
            db_name=db_name,
            table_name=table_name,
            filter_bucket=filter_bucket,
            k=args.k,
            query_vectors=query_vectors,
        ),
        encoding="utf-8",
    )

    explain_sql = out_dir / "explain.sql"
    explain_sql.write_text(
        generate_explain_sql(
            db_name=db_name,
            table_name=table_name,
            filter_bucket=filter_bucket,
            k=args.k,
            first_query=query_vectors[0],
        ),
        encoding="utf-8",
    )

    explain_output = run_psql_file_in_container(args.container_name, "vectordb", explain_sql)
    explain_out_path = out_dir / "explain_output.txt"
    explain_out_path.write_text(explain_output, encoding="utf-8")
    index_used = DEFAULT_INDEX_NAME in explain_output

    raw_query_output = run_psql_file_in_container(args.container_name, "vectordb", query_sql)
    raw_out_path = out_dir / "query_psql_output.txt"
    raw_out_path.write_text(raw_query_output, encoding="utf-8")
    ann_results, latencies_ms = parse_query_output(raw_query_output, query_count)

    bucket_start, bucket_end = bucket_bounds(loaded_n, nfilters, filter_bucket)
    candidate_count = max(0, bucket_end - bucket_start)
    log(
        f"Computing exact filtered L2 recall on candidate bucket range "
        f"[{bucket_start}, {bucket_end}) with {candidate_count} vectors."
    )
    cand_ids, cand_vecs = load_base_vectors_in_range(args.base_fvecs, dim, bucket_start, bucket_end)

    if HAS_NUMPY:
        gt_results = exact_topk_l2_numpy(query_vectors, cand_ids, cand_vecs, args.k)
    else:
        gt_results = exact_topk_l2_python(query_vectors, cand_ids, cand_vecs, args.k)

    recall_rows, avg_recall = compute_recall_metrics(ann_results, gt_results, args.k)

    for i, row in enumerate(recall_rows):
        row["latency_ms"] = latencies_ms[i]

    per_query_csv = out_dir / "per_query.csv"
    with per_query_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "latency_ms",
                "ann_count",
                "gt_count",
                "overlap",
                "recall_at_k",
                "ann_ids",
                "gt_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(recall_rows)

    total_ms = sum(latencies_ms)
    qps = query_count / (total_ms / 1000.0) if total_ms > 0 else 0.0
    avg_latency = statistics.fmean(latencies_ms) if latencies_ms else float("nan")
    p50 = percentile(latencies_ms, 0.50)
    p95 = percentile(latencies_ms, 0.95)
    p99 = percentile(latencies_ms, 0.99)
    selectivity = candidate_count / loaded_n if loaded_n > 0 else 0.0

    summary_lines = [
        "sptag_fvec_filter_bench summary",
        f"mode: {args.mode}",
        f"base_fvecs: {args.base_fvecs}",
        f"query_fvecs: {args.query_fvecs}",
        f"db_name: {db_name}",
        f"table_name: {table_name}",
        f"container_name: {args.container_name}",
        f"dim: {dim}",
        f"loaded_base_count: {loaded_n}",
        f"nfilters: {nfilters}",
        f"filter_bucket: {filter_bucket}",
        f"bucket_range_start: {bucket_start}",
        f"bucket_range_end: {bucket_end}",
        f"filtered_candidate_count: {candidate_count}",
        f"selectivity_ratio: {selectivity:.6f}",
        f"k: {args.k}",
        f"num_queries_requested: {args.num_queries}",
        f"num_queries_executed: {query_count}",
        f"average_recall_at_k: {avg_recall:.6f}",
        f"latency_avg_ms: {avg_latency:.6f}",
        f"latency_p50_ms: {p50:.6f}",
        f"latency_p95_ms: {p95:.6f}",
        f"latency_p99_ms: {p99:.6f}",
        f"qps: {qps:.6f}",
        f"numpy_enabled_for_gt: {HAS_NUMPY}",
        f"explain_uses_expected_index_name: {index_used}",
        f"explain_output_path: {explain_out_path}",
        f"query_sql_path: {query_sql}",
        f"query_raw_output_path: {raw_out_path}",
        f"per_query_csv_path: {per_query_csv}",
    ]
    summary = "\n".join(summary_lines) + "\n"
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    log("Query phase completed successfully.")


def main() -> int:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.nfilters <= 0:
        raise ValueError("--nfilters must be > 0")
    if args.base_limit is not None and args.base_limit <= 0:
        raise ValueError("--base-limit must be > 0 when provided")

    validate_identifier(args.db_name, "database name")
    validate_identifier(DEFAULT_TABLE_NAME, "table name")
    validate_identifier(DEFAULT_INDEX_NAME, "index name")
    validate_identifier(DEFAULT_FILTER_INDEX_NAME, "filter index name")

    sptag_threads = args.sptag_threads if args.sptag_threads is not None else (os_cpu_count() or 1)
    if sptag_threads <= 0:
        raise ValueError("--sptag-threads must be > 0")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    if args.mode in ("all", "prepare"):
        run_prepare(args, out_dir, sptag_threads)
    if args.mode in ("all", "query"):
        run_query(args, out_dir, sptag_threads)

    return 0


def os_cpu_count() -> Optional[int]:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[sptag_bench] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
