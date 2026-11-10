#!/bin/bash
# Converte un decisore da safetensors (bf16) a GGUF Q8_0 con il convert_hf_to_gguf.py del motore (llama.cpp 911f6cdc,
# lo stesso che lo esegue). Solo CPU: non tocca le schede. Uscita in ~/models/decisori/<nome>/gguf/<nome>-Q8_0.gguf.
#   ~/decisori/converti.sh decider-4b | h2o-lightning-4b   [tipo: q8_0 (predefinito) | bf16]
# Con bf16 l'uscita è <nome>-BF16.gguf, la base per llama-quantize (es. Q6_K) senza passare da un'altra quantizzazione.
#
# I decisori su Qwen3.5 dichiarano in config.json lo strato MTP della base (mtp_num_hidden_layers: 1) ma i pesi pubblicati
# non lo contengono: convertito così, il GGUF ha un blocco in più senza tensori e llama.cpp non lo carica
# («tensor 'blk.32.attn_norm.weight' not found»). Se nei safetensors non c'è nessun tensore MTP, si passa --no-mtp.
# TRANSFORMERS=5.17.0: il convertitore gira con quella versione di transformers (installata solo nel container usa e getta).
# Serve per i modelli su Gemma 4 (transformers 5): il loro tokenizer_config.json ha extra_special_tokens come lista e la
# 4.57.6 fissata da llama.cpp si ferma con «'list' object has no attribute 'keys'».
set -eu
NOME=$1
TIPO=${2:-q8_0}
D=$HOME/models/decisori/$NOME
mkdir -p "$D/gguf"
OPZ=()
SENZA_MTP=$(python3 - "$D" <<'PY'
import glob, json, os, struct, sys
d = sys.argv[1]
mtp = 0
for f in glob.glob(os.path.join(d, "*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        mtp += sum(1 for k in json.loads(fh.read(n)) if "mtp" in k)
c = json.load(open(os.path.join(d, "config.json")))
dichiarati = c.get("mtp_num_hidden_layers") or (c.get("text_config") or {}).get("mtp_num_hidden_layers")
print("si" if dichiarati and not mtp else "")
PY
)
[ -n "$SENZA_MTP" ] && { echo "config.json dichiara l'MTP ma i pesi non lo hanno: --no-mtp"; OPZ=(--no-mtp); }
RETE=(--network none); PRIMA=""
[ -n "${TRANSFORMERS:-}" ] && { RETE=(); PRIMA="pip3 install -q --user --break-system-packages transformers==$TRANSFORMERS && "; }
docker run --rm "${RETE[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$D:/modello" \
  "${IMMAGINE:-ubuntu-ai/decisori:cuda12.8}" bash -c "$PRIMA python3 /app/.runtime/llama.cpp/convert_hf_to_gguf.py \
  /modello --outtype $TIPO ${OPZ[*]:-} --outfile /modello/gguf/$NOME-${TIPO^^}.gguf"
ls -la "$D/gguf"
