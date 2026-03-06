#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSVBASE_PATH="MSVBASE"
MSVBASE_URL="${MSVBASE_URL:-https://github.com/microsoft/MSVBASE.git}"

if [[ ! -d "${ROOT_DIR}/.git" ]]; then
    echo "[setup_msvbase_submodule] ERROR: ${ROOT_DIR} is not a git repo root" >&2
    exit 1
fi

if git -C "${ROOT_DIR}" config -f .gitmodules --get "submodule.${MSVBASE_PATH}.path" >/dev/null 2>&1; then
    echo "[setup_msvbase_submodule] Submodule entry already exists for ${MSVBASE_PATH}"
else
    if [[ -e "${ROOT_DIR}/${MSVBASE_PATH}" ]]; then
        echo "[setup_msvbase_submodule] Registering existing ${MSVBASE_PATH} directory as submodule"
        git -C "${ROOT_DIR}" submodule add -f "${MSVBASE_URL}" "${MSVBASE_PATH}"
    else
        echo "[setup_msvbase_submodule] Cloning ${MSVBASE_URL} into ${MSVBASE_PATH} as submodule"
        git -C "${ROOT_DIR}" submodule add "${MSVBASE_URL}" "${MSVBASE_PATH}"
    fi
fi

echo "[setup_msvbase_submodule] Initializing/updating submodule"
git -C "${ROOT_DIR}" submodule update --init --recursive "${MSVBASE_PATH}"

"${ROOT_DIR}/scripts/restore_msvbase_overrides.sh" "${ROOT_DIR}/${MSVBASE_PATH}"

echo "[setup_msvbase_submodule] Done"
