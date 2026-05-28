#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
QUESTIONS="${1:-data/questions/abstract_questions.csv}"
OUT="${OUT:-data/eval_results/qa_retrieval_modes.csv}"

"$PYTHON" scripts/eval_qa.py \
  --questions "$QUESTIONS" \
  --out "$OUT" \
  --prompt qa1 \
  --retriever hybrid \
  --columns default \
  --k 4 \
  --retrieval-mode dummy \
  --retrieval-mode normal \
  --retrieval-mode guided
