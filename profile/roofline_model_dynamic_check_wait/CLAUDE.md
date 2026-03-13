# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Dynamic-wait roofline profiler that answers: "given the observed decompression fan-out (N concurrent IAA jobs) during HNSW traversal, does the IAA hardware keep up with the traversal rate?" Combines two data sources:

1. **Traversal logging** — three instrumented search binaries (`roofline_exact_iaa.run`, `roofline_range_iaa.run`, `roofline_multi_iaa.run`) that log per-step IAA fan-out (how many jobs were submitted/waited) during real queries. Sources in `src/`.
2. **Decompression benchmark** (`roofline_dynamic_wait_bench`) — standalone IAA+LZ4 latency sweep over chunk sizes and N values, at a given engine count.

The sweep is repeated for each IAA engine count (1, 2, 4, 8 by default), requiring a `sudo` IAA reconfiguration between runs.

## Build & Run

**Prerequisites:** QPL must be built first (`./qpl_build.sh` from repo root). IAA must be configured before running (`sudo scripts/iaa/configure_iaa_user_<N>.sh`).

```bash
# Build all four binaries
make

# Full sweep (reconfigures IAA, runs traversal + bench, plots)
./run_roofline_dynamic_wait.sh
```

Key variables at the top of `run_roofline_dynamic_wait.sh`:
- `ENGINE_COUNTS` — IAA engine counts to sweep (default: 1 2 4 8)
- `USE_CURRENT_CONFIG=1` — skip `sudo` reconfiguration, use current IAA setup
- `NUM_QUERIES`, `K` — query count and top-k
- `CHUNK_SIZES_CSV`, `N_LIST_CSV` — bench sweep parameters
- `CPU_CORE`, `NUMA_NODE` — affinity (applied via `numactl` if available)

**Plot only** (if CSVs already exist):
```bash
python3 plot_dynamic_wait_roofline.py \
    --expansion-glob "out/traversal/*.csv" \
    --decomp-csv out/decomp_latency.csv \
    --out-dir out
```

## Output Structure

```
out/
  traversal/         # per-scenario per-engine expansion metrics CSVs
  summaries/         # per-scenario per-engine text summary files
  decomp_latency.csv # IAA/LZ4 latency vs chunk_size x N x engine_count
  roofline_wait_check_*d.png  # final plots
```

## Architecture

**Traversal binaries** (`src/*.cpp`) are instrumented copies of the COMPASS IAA async search variants. They emit `--expansion-metrics-out` CSV (per-step N distribution) and `--summary-out` text. Scenario tags encode filter selectivity and engine count (e.g. `1pct_e4`).

**Bench binary** (`roofline_dynamic_wait_bench.cpp`) sweeps chunk sizes × N values × engine count. Uses the same cold-cache flush methodology (`_mm_clflush`) as the sibling profiler in `roofline_distance_calcualtion_decompresison/`. Appends to the CSV across engine-count iterations (`--append` flag).

**Plotter** (`plot_dynamic_wait_roofline.py`) joins the traversal fan-out distribution with the decompression latency surface to draw roofline zone plots — one per dimension (128D/512D/2048D).
