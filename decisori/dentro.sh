#!/bin/bash
# Dentro il container: un decisore sulla sua scheda (il container ne vede una sola, --gpus device=UUID).
#   dentro.sh <nome> <porta>
# Winnow parla da solo (/v1/systemone nel winnow-server). Gli altri tre: winnow-server interno su <porta>+1000 con il GGUF,
# poi l'adattatore (il server dell'autore con il motore sostituito) su <porta>. Se uno dei due muore, il container finisce.
# Variabili: CTX (contesto totale di llama-server, condiviso fra le slot), PARALLELI (slot = richieste contemporanee).
set -u
NOME=$1; PORTA=$2; INTERNA=$((PORTA + 1000))
M=/modello
P=${PARALLELI:-4}
SERVER=/app/.build/bin/winnow-server

case "$NOME" in
  winnow-12b)
    exec python3 /app/scripts/serve.py --server "$SERVER" --model "$M/gguf/Winnow-12B-Q8_0.gguf" --text-only \
      --alias Winnow-12B --context "${CTX:-32768}" --decision-parallel "$P" --chat-parallel 1 --cache f16 --memory exclusive \
      --host 127.0.0.1 --port "$PORTA" ;;
  decider-4b)
    GGUF=$M/gguf/decider-4b-Q8_0.gguf; CTX=${CTX:-36864}
    ADATTATORE=(env DECIDER_MODEL="$M" python3 /opt/decisori/decider_llama.py --llama "http://127.0.0.1:$INTERNA" --port "$PORTA" --paralleli "$P") ;;
  h2o-lightning-4b)
    GGUF=$M/gguf/h2o-lightning-4b-Q8_0.gguf; CTX=${CTX:-40960}
    ADATTATORE=(python3 /opt/decisori/h2o_llama.py --modello "$M" --llama "http://127.0.0.1:$INTERNA" --port "$PORTA") ;;
  *) echo "decisore sconosciuto: $NOME"; exit 2 ;;
esac

"$SERVER" --model "$GGUF" --alias "$NOME" --host 127.0.0.1 --port "$INTERNA" --n-gpu-layers 999 \
  --ctx-size "$CTX" --parallel "$P" --kv-unified --flash-attn on --no-context-shift --threads 4 &
LS=$!
until curl -sf "http://127.0.0.1:$INTERNA/health" >/dev/null; do
  kill -0 $LS 2>/dev/null || { echo "llama-server di $NOME uscito durante il caricamento"; exit 1; }
  sleep 1
done
"${ADATTATORE[@]}" &
AD=$!
trap 'kill $LS $AD 2>/dev/null' TERM INT
wait -n $LS $AD
kill $LS $AD 2>/dev/null
wait
