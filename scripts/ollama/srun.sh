#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ollama_gpu() {
    srun -p GPU-ilara \
        --time=02:00:00 \
        --cpus-per-task=4 \
        --mem=32GB \
        --gres=gpu:1 \
        --pty "$script_dir/serve.sh"
}

ollama_gpu