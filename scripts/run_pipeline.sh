#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CMAKE_BIN="${CMAKE_BIN:-${ROOT_DIR}/.venv/bin/cmake}"
if [[ ! -x "${CMAKE_BIN}" ]]; then
  CMAKE_BIN="cmake"
fi

"${CMAKE_BIN}" -S "${ROOT_DIR}" -B "${ROOT_DIR}/build"
"${CMAKE_BIN}" --build "${ROOT_DIR}/build" -j
"${ROOT_DIR}/build/cognitive_pipeline"
