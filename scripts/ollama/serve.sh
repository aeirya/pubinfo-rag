#!/usr/bin/env bash
set -euo pipefail

# Directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env from the same directory as this script
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

: "${OLLAMA_HOST_IP:=127.0.0.1}"
: "${OLLAMA_PORT:=23114}"
: "${OLLAMA_MODELS_DIR:=$HOME/.ollama/models}"

export OLLAMA_HOST="${OLLAMA_HOST_IP}:${OLLAMA_PORT}"
export OLLAMA_MODELS="${OLLAMA_MODELS_DIR}"

mkdir -p "$OLLAMA_MODELS"


echo "Ollama config:"
echo -e "  base_url:\t${OLLAMA_HOST}"
echo -e "  models:\t${OLLAMA_MODELS}"
echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

if ss -ltn | grep -q ":${OLLAMA_PORT} "; then
    echo "Error: port ${OLLAMA_PORT} is already in use."
    echo "Change OLLAMA_PORT in $SCRIPT_DIR/.env"
    exit 1
fi

echo "Starting Ollama server..."
exec ollama serve
