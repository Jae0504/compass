# FID/TB Problem Summary

## Symptoms observed
- `compass_search_w_lz4` showed low recall.
- Generated TB file had all zero bytes (no connector bits set).
- `build_FID_TB.run` crashed with `Segmentation fault (exit code 139)` on SIFT1M generation.
- `/home/jykang5/compass/debug/fid_tb` was initially empty because generation output was going to a different default path.

## Root causes
1. **`nfilters=256` overflow in TB group counting**
   - Code used:
     - `if (g < static_cast<uint8_t>(nfilters_local))`
   - With `nfilters_local=256`, `static_cast<uint8_t>(256)` becomes `0`, so the condition is always false.
   - Result: `group_counts` stayed zero, TB generation loop was skipped, TB became effectively empty/all-zero.

2. **Global/local state mismatch in `build_FID_TB` path**
   - `post_proc.h` helpers (for example `find_num_cluster`) read global variables `filter_ids` and `connector_bits`.
   - Refactored builder was using local `fid_values` and `local_connector_bits` without syncing required globals.
   - Result: helper code accessed inconsistent/empty global state and crashed (observed at `post_proc.h:785` under `gdb`).

3. **Output path expectation mismatch**
   - Build script default output is under:
     - `end_to_end/vectordb/dataset/fid_tb/<benchmark>`
   - Not under:
     - `/home/jykang5/compass/debug/fid_tb`
   - So checking only `/debug/fid_tb` can look like "nothing generated" unless `--out-dir` is explicitly set.

## Fixes applied
In:
- [build_FID_TB.cpp](/home/jykang5/compass/end_to_end/vectordb/cpp_examples/build_examples/build_FID_TB.cpp)

Changes:
1. Fixed overflow-prone check:
   - from `g < static_cast<uint8_t>(nfilters_local)`
   - to `static_cast<size_t>(g) < group_counts.size()`
2. Synced global state required by `post_proc.h`:
   - `filter_ids = fid_values;` at start of `build_tb_for_attribute(...)`
   - `connector_bits = local_connector_bits;` before `find_num_cluster(...)`

## Verification done
- Reproduced segfault before fix: `EXIT_CODE:139`.
- Captured backtrace:
  - `find_num_cluster(...)` in `post_proc.h:785`.
- Rebuilt `build_FID_TB.run` with fixes.
- Re-ran same command; it no longer crashed in the previous early-crash window.

## Practical takeaway
- Low recall was caused by broken TB generation (all-zero TB), not by the LZ4 filter search logic itself.
- Regenerate FID/TB with the fixed `build_FID_TB.run` and use the new manifest/files for search.
