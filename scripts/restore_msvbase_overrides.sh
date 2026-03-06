#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSVBASE_DIR="${1:-${ROOT_DIR}/MSVBASE}"
OVERRIDES_DIR="${ROOT_DIR}/scripts/msvbase_overrides"

if [[ ! -d "${MSVBASE_DIR}" ]]; then
    echo "[restore_msvbase_overrides] ERROR: missing MSVBASE dir: ${MSVBASE_DIR}" >&2
    exit 1
fi
if [[ ! -d "${MSVBASE_DIR}/scripts" ]]; then
    echo "[restore_msvbase_overrides] ERROR: missing scripts dir: ${MSVBASE_DIR}/scripts" >&2
    exit 1
fi

for file in sptag_fvec_filter_bench.py SPTAG_FVEC_FILTER_BENCH.md; do
    if [[ ! -f "${OVERRIDES_DIR}/${file}" ]]; then
        echo "[restore_msvbase_overrides] ERROR: missing override file: ${OVERRIDES_DIR}/${file}" >&2
        exit 1
    fi
done

install -m 755 "${OVERRIDES_DIR}/sptag_fvec_filter_bench.py" "${MSVBASE_DIR}/scripts/sptag_fvec_filter_bench.py"
install -m 644 "${OVERRIDES_DIR}/SPTAG_FVEC_FILTER_BENCH.md" "${MSVBASE_DIR}/scripts/SPTAG_FVEC_FILTER_BENCH.md"

echo "[restore_msvbase_overrides] Restored benchmark files into ${MSVBASE_DIR}/scripts"
echo "[restore_msvbase_overrides] - sptag_fvec_filter_bench.py"
echo "[restore_msvbase_overrides] - SPTAG_FVEC_FILTER_BENCH.md"
