#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/restore_msvbase_overrides.sh" "${ROOT_DIR}/MSVBASE"

python3 "${ROOT_DIR}/MSVBASE/scripts/sptag_fvec_filter_bench.py" "$@"
