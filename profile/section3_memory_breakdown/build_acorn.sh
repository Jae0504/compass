#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_PATCH="$ROOT_DIR/acorn_cmakelist.patch"
DEFAULT_BUILD_DIR="$ROOT_DIR/ACORN/build"

ACORN_DIR=""
ACORN_BUILD_DIR="$DEFAULT_BUILD_DIR"
PATCH_FILE="$DEFAULT_PATCH"
JOBS=""

usage() {
  cat <<'EOF'
Build ACORN/FAISS static library for section3 profiling.

Usage:
  build_acorn.sh \
    [--acorn-dir /path/to/ACORN] \
    [--acorn-build-dir /path/to/build] \
    [--patch /path/to/acorn_cmakelist.patch] \
    [--jobs N]

Resolution order when --acorn-dir is not provided:
  1) /home/jykang5/compass/ACORN
  2) /home/jykang5/ACORN
  3) /home/jykang5/acorn_build/ACORN
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

resolve_source_dir() {
  local candidates=(
    "/home/jykang5/compass/ACORN"
    "/home/jykang5/ACORN"
    "/home/jykang5/acorn_build/ACORN"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c/CMakeLists.txt" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

abspath_or_raw() {
  local p="$1"
  if [[ -d "$p" ]]; then
    (cd "$p" && pwd)
  else
    echo "$p"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --acorn-dir)
      ACORN_DIR="$2"
      shift 2
      ;;
    --acorn-build-dir)
      ACORN_BUILD_DIR="$2"
      shift 2
      ;;
    --patch)
      PATCH_FILE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

if [[ -n "$JOBS" && ! "$JOBS" =~ ^[0-9]+$ ]]; then
  fail "--jobs must be a positive integer."
fi

if [[ -z "$ACORN_DIR" ]]; then
  if ! ACORN_DIR="$(resolve_source_dir)"; then
    fail "Could not find ACORN source with CMakeLists.txt. Initialize /home/jykang5/compass/ACORN submodule or pass --acorn-dir."
  fi
fi

if [[ ! -f "$ACORN_DIR/CMakeLists.txt" ]]; then
  fail "ACORN source does not contain CMakeLists.txt: $ACORN_DIR"
fi

ACORN_DIR="$(abspath_or_raw "$ACORN_DIR")"
ACORN_BUILD_DIR="$(abspath_or_raw "$ACORN_BUILD_DIR")"

if [[ -z "$JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  else
    JOBS="4"
  fi
fi

if [[ -f "$ACORN_BUILD_DIR/CMakeCache.txt" ]]; then
  cache_source="$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$ACORN_BUILD_DIR/CMakeCache.txt" | head -n1 | cut -d= -f2- || true)"
  if [[ -n "$cache_source" ]]; then
    cache_source_abs="$(abspath_or_raw "$cache_source")"
    if [[ "$cache_source_abs" != "$ACORN_DIR" ]]; then
      echo "Detected CMake cache source mismatch:"
      echo "  cache source: $cache_source_abs"
      echo "  requested:    $ACORN_DIR"
      echo "Removing build directory: $ACORN_BUILD_DIR"
      rm -rf "$ACORN_BUILD_DIR"
    fi
  fi
fi

if [[ -f "$PATCH_FILE" ]]; then
  if [[ ! -d "$ACORN_DIR/.git" ]]; then
    fail "ACORN source is not a git repository, cannot apply patch with git apply: $ACORN_DIR"
  fi
  if git -C "$ACORN_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "Applying patch: $PATCH_FILE"
    git -C "$ACORN_DIR" apply "$PATCH_FILE"
  elif git -C "$ACORN_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "Patch already applied: $PATCH_FILE"
  else
    fail "Patch does not match this ACORN tree: $PATCH_FILE"
  fi
else
  echo "Warning: patch file not found, skipping patch apply: $PATCH_FILE"
fi

echo "Configuring ACORN..."
cmake -S "$ACORN_DIR" -B "$ACORN_BUILD_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "Building faiss static library..."
cmake --build "$ACORN_BUILD_DIR" --target faiss -j "$JOBS"

FAISS_STATIC_LIB="$ACORN_BUILD_DIR/faiss/libfaiss.a"
if [[ ! -f "$FAISS_STATIC_LIB" ]]; then
  fail "Build completed but static library not found: $FAISS_STATIC_LIB"
fi

echo "ACORN_DIR=$ACORN_DIR"
echo "ACORN_BUILD_DIR=$ACORN_BUILD_DIR"
echo "FAISS_STATIC_LIB=$FAISS_STATIC_LIB"
