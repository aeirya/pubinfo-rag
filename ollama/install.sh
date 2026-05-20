#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/.local"
TMP="$HOME/ollama-linux-amd64.tar.zst"

mkdir -p "$INSTALL_DIR"

echo "Downloading Ollama..."
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$TMP"

echo "Extracting..."
python - <<PY
import zstandard as zstd
from pathlib import Path

src = Path("$TMP")
dst = Path("$HOME/ollama-linux-amd64.tar")

with src.open("rb") as f_in, dst.open("wb") as f_out:
    zstd.ZstdDecompressor().copy_stream(f_in, f_out)

print(dst)
PY

tar -xf "$HOME/ollama-linux-amd64.tar" -C "$INSTALL_DIR"

echo "Adding Ollama to ~/.bashrc..."
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

grep -qxF 'export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"' ~/.bashrc || \
  echo 'export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"' >> ~/.bashrc

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"

echo "Installed:"
ollama --version