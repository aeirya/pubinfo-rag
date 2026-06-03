#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DIRS=(
  pubinfo
  experiments
  tests
)

EXCLUDES=(
  '*/__pycache__/*'
  '*/__pycache__'
  '*.pyc'
  '*.pyo'
  '*/.git/*'
  '*/.venv/*'
  '*/venv/*'
  '*/env/*'
  '*/.mypy_cache/*'
  '*/.pytest_cache/*'
  '*/.ruff_cache/*'
  '*/dist/*'
  '*/build/*'
  '*.egg-info/*'
  '*/.DS_Store'
)

DIRS=("$@")
[[ ${#DIRS[@]} -eq 0 ]] && DIRS=("${DEFAULT_DIRS[@]}")

OUTPUT="$(
  [[ ${#DIRS[@]} -eq 1 ]] \
    && basename "${DIRS[0]}" \
    || echo project
).zip"

zip -r "$OUTPUT" "${DIRS[@]}" \
  "${EXCLUDES[@]/#/-x=}"