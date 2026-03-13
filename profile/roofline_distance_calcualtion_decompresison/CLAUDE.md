# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Roofline model profiler comparing latency of three operations as a function of parallelism (N = 1,2,4,8,16,32):
1. **L2 distance calculation** — sparse random-access cold-cache reads from `.fvecs` base files (models HNSW traversal)
2. **IAA async decompression + scan_eq** — Intel QPL hardware decompression of real FID bitmap chunks
3. **LZ4 software decompression** — cold-cache random-access decompression of real FID bitmap chunks

Datasets profiled: SIFT1M (128-D), LAION (512-D), HNM (2048-D). FID file paths are hardcoded in `profile.cpp` under `FID_DATASETS`.

## Build & Run

**Prerequisites:** QPL must be built first (`./qpl_build.sh` from repo root).

```bash
# Build
make

# Run (pinned to core 8, NUMA node 0 by default)
./profile_roofline \
    --base-dir /storage/jykang5/compass_base_query \
    --out-dir out \
    --cpu-core 8 \
    --numa-node 0
```

Outputs three CSVs to `--out-dir`: `results_dist.csv`, `results_iaa.csv`, `results_lz4.csv`.

**Plot:**
```bash
python3 plot_roofline.py --chunk-size 64KB --out-dir out
# <size> accepts bytes (65536), KB (64KB), or MB (1MB)
```

Outputs `out/roofline_<chunk_size>.png` — 3-panel figure (one per dimension).

## Architecture

`profile.cpp` has three independent benchmark functions: `bench_distance`, `bench_iaa`, `bench_lz4`. Each writes its own CSV. All three use the same cold-cache methodology: cache lines are explicitly flushed with `_mm_clflush` before each timed measurement to guarantee DRAM-fetch cost.

IAA jobs use `qpl_submit_job` / `qpl_wait_job` in submit-all-then-wait-all order. The `QplJob` RAII wrapper handles `qpl_init_job` / `qpl_fini_job` lifetime. Jobs must be reinitialized via `reinit()` before each reuse.

`plot_roofline.py` reads all three CSVs, filters to the requested `chunk_size`, and for each dimension plots: IAA latency, distance latency, and distance+LZ4 sum (with shaded fill showing LZ4 overhead).
