# Memory Breakdown Profiling

This folder now includes:
- `profile_memory_breakdown.py`: builds HNSW from `.fvecs` (M=32, efConstruction=128 by default), profiles memory for graph/embedding/metadata across:
  - raw
  - binary (1-byte mapped)
  - compressed(raw)
  - compressed(after binary)
- `plot_memory_breakdown.py`: plots stacked bars from `profiling.json`
- `metadata_column_profile.cpp`: standalone metadata-only profiling with the same inclusion rules.

## Metadata inclusion rules
For each candidate metadata column:
- string/char: include only if unique values <= 256
- integer:
  - unique <= 256: one-to-one code mapping to 0..255
  - unique > 256: range map to 0..255
- otherwise exclude

Dataset-specific columns:
- `laion`: only `NSFW`, `similarity`, `original_width`, `original_height`
- `hnm`: all columns except `detail_desc`

## Run full profiling (combined output)
```bash
python3 profile/metadata_compressability/profile_memory_breakdown.py \
  --hnm-fvecs /path/to/hnm_base.fvecs \
  --hnm-jsonl /path/to/hnm_payloads.jsonl \
  --laion-fvecs /path/to/laion_base.fvecs \
  --laion-jsonl /path/to/laion_payloads.jsonl \
  --m 32 \
  --ef-construct 128 \
  --out-txt profiling.txt \
  --out-json profiling.json
```

Outputs:
- `profiling.txt` (combined report for both datasets)
- `profiling.json` (machine-readable for plotting)

## Plot
```bash
python3 profile/metadata_compressability/plot_memory_breakdown.py \
  --input profiling.json \
  --output profiling_breakdown.png
```

## Optional: compile metadata-only C++ profiler
```bash
g++ -std=c++17 -O2 profile/metadata_compressability/metadata_column_profile.cpp -lz -o metadata_column_profile
```

Example:
```bash
./metadata_column_profile hnm /path/to/hnm_payloads.jsonl
./metadata_column_profile laion /path/to/laion_payloads.jsonl
```
