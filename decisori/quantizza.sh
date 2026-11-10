#!/bin/bash
# Quantizza il GGUF BF16 di un decisore (da `converti.sh <nome> bf16`) con il llama-quantize del motore.
#   ~/decisori/quantizza.sh <nome> Q6_K
# llama-quantize è compilato con CUDA e senza libcuda non parte: il container vede la 4090 (non la usa, lavora la CPU).
set -eu
NOME=$1; TIPO=$2
D=$HOME/models/decisori/$NOME/gguf
docker run --rm --network none --gpus device=GPU-91e731c9-4111-80e1-56b9-27e3652dcf89 --user "$(id -u):$(id -g)" -v "$D:/g" \
  "${IMMAGINE:-ubuntu-ai/decisori:cuda12.8}" /app/.build/bin/llama-quantize "/g/$NOME-BF16.gguf" "/g/$NOME-$TIPO.gguf" "$TIPO"
ls -la "$D"
