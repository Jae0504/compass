#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QPL_DIR="$SCRIPT_DIR/qpl"
BUILD_DIR="$QPL_DIR/build"

if [[ ! -d "$QPL_DIR" || ! -f "$QPL_DIR/CMakeLists.txt" ]]; then
  echo "Error: qpl source not found at $QPL_DIR" >&2
  echo "Hint: run 'git submodule update --init qpl' first." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

if [[ -d "$QPL_DIR/.git" || -f "$QPL_DIR/.git" ]]; then
  echo "[0/3] Initialize qpl submodules"
  git -C "$QPL_DIR" submodule update --init --recursive
fi

cd "$BUILD_DIR"

echo "[1/3] cmake .."
cmake ..

echo "[2/3] make"
if command -v nproc >/dev/null 2>&1; then
  make -j"$(nproc)"
else
  make
fi

echo "[3/3] Skip install (local build only)"

QPL_STATIC_LIB=""
for candidate in \
  "$BUILD_DIR/sources/libqpl.a" \
  "$BUILD_DIR/lib/libqpl.a" \
  "$BUILD_DIR/libqpl.a"; do
  if [[ -f "$candidate" ]]; then
    QPL_STATIC_LIB="$candidate"
    break
  fi
done

QPL_INCLUDE_DIR="$QPL_DIR/include"
if [[ -z "$QPL_STATIC_LIB" ]]; then
  echo "Error: expected static library not found under $BUILD_DIR" >&2
  exit 1
fi

cat <<EOF
Build complete.
Link with:
  -I$QPL_INCLUDE_DIR
  $QPL_STATIC_LIB
EOF
