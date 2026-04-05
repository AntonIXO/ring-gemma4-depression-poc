#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <vector_csv> <top_k>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DB_FILE="${ROOT_DIR}/models/state_vectors.db"
QUERY_VEC="$1"
TOP_K="$2"
CMAKE_BIN="${CMAKE_BIN:-${ROOT_DIR}/.venv/bin/cmake}"
if [[ ! -x "${CMAKE_BIN}" ]]; then
  CMAKE_BIN="cmake"
fi

"${CMAKE_BIN}" -S "${ROOT_DIR}" -B "${ROOT_DIR}/build"
"${CMAKE_BIN}" --build "${ROOT_DIR}/build" -j
"${ROOT_DIR}/build/cognitive_vector_db" search "${DB_FILE}" "${QUERY_VEC}" "${TOP_K}"
