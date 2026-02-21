#!/usr/bin/env python3
import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "faiss-cpu is required. Install with: pip install faiss-cpu"
    ) from exc


LAION_KEYS = ["NSFW", "similarity", "original_width", "original_height"]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    unique_count: int
    included: bool
    reason: str
    text_bytes: int


@dataclass
class DatasetProfile:
    name: str
    rows: int
    total_columns_considered: int
    included_columns: int
    included_column_names: List[str]
    excluded_column_names: List[str]
    embedding_raw: int
    embedding_binary: int
    embedding_compressed_raw: int
    embedding_compressed_binary: int
    graph_raw: int
    graph_binary: int
    graph_compressed_raw: int
    graph_compressed_binary: int
    metadata_raw: int
    metadata_binary: int
    metadata_compressed_raw: int
    metadata_compressed_binary: int
    column_profiles: List[ColumnProfile]
    notes: List[str]


def read_fvecs(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)
        if dim.size == 0:
            raise ValueError(f"Empty fvecs file: {path}")
        d = int(dim[0])
        f.seek(0)
        raw = np.fromfile(f, dtype=np.int32)

    if raw.size % (d + 1) != 0:
        raise ValueError(f"Invalid fvecs layout in {path}")

    raw = raw.reshape(-1, d + 1)
    if not np.all(raw[:, 0] == d):
        raise ValueError(f"Inconsistent dimensions in {path}")

    vecs = raw[:, 1:].astype(np.float32, copy=False)
    return vecs


def compress_bytes(data: bytes, level: int = 6) -> int:
    return len(zlib.compress(data, level))


def to_uint8_mapped_int(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    min_v = int(arr.min())
    max_v = int(arr.max())
    if min_v == max_v:
        return np.zeros(arr.shape, dtype=np.uint8)
    if max_v - min_v <= 255:
        return (arr - min_v).astype(np.uint8)
    scaled = ((arr - min_v) * 255.0 / (max_v - min_v)).clip(0, 255)
    return scaled.astype(np.uint8)


def parse_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_columns(rows: Sequence[dict], dataset_name: str) -> Dict[str, List[str]]:
    cols: Dict[str, List[str]] = {}
    if dataset_name == "laion":
        keys = LAION_KEYS
    else:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        keys = [k for k in sorted(all_keys) if k != "detail_desc"]

    for k in keys:
        vals: List[str] = []
        for r in rows:
            v = r.get(k, None)
            if v is None:
                vals.append("")
            else:
                vals.append(str(v))
        cols[k] = vals
    return cols


def profile_metadata(rows: Sequence[dict], dataset_name: str) -> Tuple[int, int, int, int, int, int, List[ColumnProfile], List[str], List[str]]:
    cols = collect_columns(rows, dataset_name)
    total_cols = len(cols)

    text_blob = bytearray()
    binary_blob = bytearray()
    col_profiles: List[ColumnProfile] = []
    included: List[str] = []
    excluded: List[str] = []

    for name, vals in cols.items():
        unique_vals = set(vals)
        unique_count = len(unique_vals)

        is_int = True
        int_values: List[int] = []
        for v in vals:
            try:
                iv = int(v)
                if str(iv) != v.strip():
                    is_int = False
                    break
                int_values.append(iv)
            except Exception:
                is_int = False
                break

        text_size = sum(len(v.encode("utf-8")) + 1 for v in vals)
        include = False
        reason = "excluded"

        if is_int:
            arr = np.array(int_values, dtype=np.int64)
            if unique_count <= 256:
                include = True
                reason = "int unique<=256 mapped"
                uniq_sorted = sorted(set(int_values))
                mapping = {v: i for i, v in enumerate(uniq_sorted)}
                mapped = bytes(mapping[v] for v in int_values)
                binary_blob.extend(mapped)
            else:
                include = True
                reason = "int unique>256 range->0..255"
                mapped = to_uint8_mapped_int(arr).tobytes()
                binary_blob.extend(mapped)
        else:
            if unique_count <= 256:
                include = True
                reason = "string unique<=256 mapped"
                uniq_sorted = sorted(unique_vals)
                mapping = {v: i for i, v in enumerate(uniq_sorted)}
                mapped = bytes(mapping[v] for v in vals)
                binary_blob.extend(mapped)
            else:
                include = False
                reason = "string unique>256"

        if include:
            included.append(name)
            for v in vals:
                text_blob.extend(v.encode("utf-8"))
                text_blob.append(0x0A)
        else:
            excluded.append(name)

        dtype = "int" if is_int else "string/other"
        col_profiles.append(
            ColumnProfile(
                name=name,
                dtype=dtype,
                unique_count=unique_count,
                included=include,
                reason=reason,
                text_bytes=text_size,
            )
        )

    metadata_raw = len(text_blob)
    metadata_binary = len(binary_blob)
    metadata_compressed_raw = compress_bytes(bytes(text_blob)) if text_blob else 0
    metadata_compressed_binary = compress_bytes(bytes(binary_blob)) if binary_blob else 0

    return (
        metadata_raw,
        metadata_binary,
        metadata_compressed_raw,
        metadata_compressed_binary,
        total_cols,
        len(included),
        col_profiles,
        included,
        excluded,
    )


def extract_hnsw_graph_bytes(index: "faiss.IndexHNSWFlat") -> Tuple[int, int, int, int, List[str]]:
    notes: List[str] = []
    hnsw = index.hnsw

    try:
        neighbors = faiss.vector_to_array(hnsw.neighbors)
        offsets = faiss.vector_to_array(hnsw.offsets)
        levels = faiss.vector_to_array(hnsw.levels)

        raw = neighbors.tobytes() + offsets.tobytes() + levels.tobytes()
        graph_raw = len(raw)

        # Keep graph unchanged across representations. Only metadata changes.
        graph_binary = graph_raw
        graph_compressed_raw = graph_raw
        graph_compressed_binary = graph_raw
        return graph_raw, graph_binary, graph_compressed_raw, graph_compressed_binary, notes
    except Exception as exc:
        notes.append(f"Graph arrays extraction fallback used: {exc}")

    # Fallback: estimate graph by subtracting embedding bytes from serialized index size.
    serialized = faiss.serialize_index(index)
    ser_bytes = int(serialized.nbytes)
    # This is an approximation; serialized index includes overhead.
    graph_raw = max(0, ser_bytes)
    graph_binary = graph_raw
    graph_compressed_raw = graph_raw
    graph_compressed_binary = graph_raw
    notes.append("Fallback graph size is serialized index bytes (approximation).")
    return graph_raw, graph_binary, graph_compressed_raw, graph_compressed_binary, notes


def profile_dataset(name: str, fvecs_path: Path, jsonl_path: Path, m: int, ef_construct: int) -> DatasetProfile:
    xb = read_fvecs(fvecs_path)

    index = faiss.IndexHNSWFlat(xb.shape[1], m)
    index.hnsw.efConstruction = ef_construct
    index.add(xb)

    graph_raw, graph_binary, graph_compressed_raw, graph_compressed_binary, notes = extract_hnsw_graph_bytes(index)

    # Keep embedding unchanged across representations. Only metadata changes.
    embedding_raw = int(xb.nbytes)
    embedding_binary = embedding_raw
    embedding_compressed_raw = embedding_raw
    embedding_compressed_binary = embedding_raw

    rows = parse_jsonl(jsonl_path)
    (
        metadata_raw,
        metadata_binary,
        metadata_compressed_raw,
        metadata_compressed_binary,
        total_cols,
        included_cols,
        col_profiles,
        included_names,
        excluded_names,
    ) = profile_metadata(rows, name)

    return DatasetProfile(
        name=name,
        rows=xb.shape[0],
        total_columns_considered=total_cols,
        included_columns=included_cols,
        included_column_names=included_names,
        excluded_column_names=excluded_names,
        embedding_raw=embedding_raw,
        embedding_binary=embedding_binary,
        embedding_compressed_raw=embedding_compressed_raw,
        embedding_compressed_binary=embedding_compressed_binary,
        graph_raw=graph_raw,
        graph_binary=graph_binary,
        graph_compressed_raw=graph_compressed_raw,
        graph_compressed_binary=graph_compressed_binary,
        metadata_raw=metadata_raw,
        metadata_binary=metadata_binary,
        metadata_compressed_raw=metadata_compressed_raw,
        metadata_compressed_binary=metadata_compressed_binary,
        column_profiles=col_profiles,
        notes=notes,
    )


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return part * 100.0 / total


def write_report(profiles: Sequence[DatasetProfile], out_txt: Path, out_json: Path, m: int, ef_construct: int) -> None:
    lines: List[str] = []
    lines.append("Memory Breakdown Profiling")
    lines.append("========================")
    lines.append(f"HNSW config: M={m}, efConstruction={ef_construct}")
    lines.append("")

    json_out: Dict[str, object] = {
        "hnsw": {"M": m, "efConstruction": ef_construct},
        "datasets": [],
    }

    for p in profiles:
        lines.append(f"Dataset: {p.name}")
        lines.append(f"Rows: {p.rows}")
        lines.append(
            f"Metadata included columns: {p.included_columns}/{p.total_columns_considered}"
        )
        lines.append(
            "Included column names: "
            + (", ".join(p.included_column_names) if p.included_column_names else "(none)")
        )
        lines.append(
            "Excluded column names: "
            + (", ".join(p.excluded_column_names) if p.excluded_column_names else "(none)")
        )

        raw_total = p.graph_raw + p.embedding_raw + p.metadata_raw
        bin_total = p.graph_binary + p.embedding_binary + p.metadata_binary
        comp_total = p.graph_compressed_raw + p.embedding_compressed_raw + p.metadata_compressed_raw
        comp_bin_total = (
            p.graph_compressed_binary + p.embedding_compressed_binary + p.metadata_compressed_binary
        )

        def add_section(title: str, g: int, e: int, md: int, total: int) -> None:
            lines.append(f"{title}: total={total} bytes")
            lines.append(
                f"  HNSW graph={g} bytes ({pct(g, total):.2f}%), "
                f"embedding={e} bytes ({pct(e, total):.2f}%), "
                f"metadata={md} bytes ({pct(md, total):.2f}%)"
            )

        add_section("Raw", p.graph_raw, p.embedding_raw, p.metadata_raw, raw_total)
        add_section("Binary", p.graph_binary, p.embedding_binary, p.metadata_binary, bin_total)
        add_section(
            "Compressed(raw)",
            p.graph_compressed_raw,
            p.embedding_compressed_raw,
            p.metadata_compressed_raw,
            comp_total,
        )
        add_section(
            "Compressed(after binary)",
            p.graph_compressed_binary,
            p.embedding_compressed_binary,
            p.metadata_compressed_binary,
            comp_bin_total,
        )

        if p.notes:
            lines.append("Notes:")
            for n in p.notes:
                lines.append(f"  - {n}")

        lines.append("Column details:")
        for c in p.column_profiles:
            lines.append(
                f"  {c.name}: type={c.dtype}, unique={c.unique_count}, "
                f"included={c.included}, reason={c.reason}, text_bytes={c.text_bytes}"
            )
        lines.append("")

        dataset_json = {
            "name": p.name,
            "rows": p.rows,
            "metadata_columns": {
                "included": p.included_columns,
                "total": p.total_columns_considered,
                "included_names": p.included_column_names,
                "excluded_names": p.excluded_column_names,
            },
            "sizes": {
                "raw": {
                    "graph": p.graph_raw,
                    "embedding": p.embedding_raw,
                    "metadata": p.metadata_raw,
                    "total": raw_total,
                },
                "binary": {
                    "graph": p.graph_binary,
                    "embedding": p.embedding_binary,
                    "metadata": p.metadata_binary,
                    "total": bin_total,
                },
                "compressed_raw": {
                    "graph": p.graph_compressed_raw,
                    "embedding": p.embedding_compressed_raw,
                    "metadata": p.metadata_compressed_raw,
                    "total": comp_total,
                },
                "compressed_after_binary": {
                    "graph": p.graph_compressed_binary,
                    "embedding": p.embedding_compressed_binary,
                    "metadata": p.metadata_compressed_binary,
                    "total": comp_bin_total,
                },
            },
            "notes": p.notes,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "unique_count": c.unique_count,
                    "included": c.included,
                    "reason": c.reason,
                    "text_bytes": c.text_bytes,
                }
                for c in p.column_profiles
            ],
        }
        cast_list = json_out["datasets"]
        assert isinstance(cast_list, list)
        cast_list.append(dataset_json)

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(json_out, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile memory breakdown for HNSW/embeddings/metadata.")
    parser.add_argument("--hnm-fvecs", type=Path, required=True)
    parser.add_argument("--hnm-jsonl", type=Path, required=True)
    parser.add_argument("--laion-fvecs", type=Path, required=True)
    parser.add_argument("--laion-jsonl", type=Path, required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=128)
    parser.add_argument("--out-txt", type=Path, default=Path("profiling.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("profiling.json"))
    args = parser.parse_args()

    profiles = [
        profile_dataset("hnm", args.hnm_fvecs, args.hnm_jsonl, args.m, args.ef_construct),
        profile_dataset("laion", args.laion_fvecs, args.laion_jsonl, args.m, args.ef_construct),
    ]

    write_report(profiles, args.out_txt, args.out_json, args.m, args.ef_construct)
    print(f"Wrote {args.out_txt} and {args.out_json}")


if __name__ == "__main__":
    main()
