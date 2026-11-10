#!/bin/bash
# Avvia un decisore in primo piano (per llama-swap, gruppo «decisori»): ~/decisori/avvia.sh <nome>
# Ogni decisore ha la sua scheda (per UUID, mai per indice: nvidia-smi e CUDA li numerano al contrario), la sua porta e
# quanta memoria libera gli serve sulla scheda; aspetta fino a 180 s che ci sia (Strata che si spegne, un altro decisore
# che si carica), poi resta attaccato al container finché llama-swap non lo ferma (docker stop decisore-<nome>).
set -u
V100=GPU-9547940b-edee-6807-bcc0-caf99c0747df
RTX4090=GPU-91e731c9-4111-80e1-56b9-27e3652dcf89
IMMAGINE=${IMMAGINE:-ubuntu-ai/decisori:cuda12.8}
NOME=$1

if [ "$NOME" = hub ]; then   # il punto unico: nessuna scheda, chiama gli altri attraverso llama-swap
  docker rm -f decisore-hub >/dev/null 2>&1 || true
  for i in $(seq 1 30); do docker ps -aq -f name="^decisore-hub$" | grep -q . || break; sleep 1; done
  exec docker run --rm --name decisore-hub --network host "$IMMAGINE" python3 /opt/decisori/hub.py --port 8130
fi

# La 4090 non regge Winnow più due 4B (misurato il 12/10: 5,1 + 5,1 + 13,5 GB, alla prima decisione bad_alloc): i tre
# sulla V100 (~25 GB), la 4090 resta libera.
#                         scheda    porta  MiB liberi che servono   contesto  slot
case "$NOME" in
  decider-4b)        set -- $V100    8131 5800  16384 2 ;;
  h2o-lightning-4b)  set -- $V100    8132 5800  16384 2 ;;
  winnow-12b)        set -- $V100    8133 14500 16384 2 ;;
  *) echo "decisore sconosciuto: $NOME"; exit 2 ;;
esac
export CTX=${CTX:-$4} PARALLELI=${PARALLELI:-$5}
SCHEDA=$1; PORTA=$2; SERVE=$3
# per le prove: SU=v100 | 4090 sposta il decisore sull'altra scheda
case "${SU:-}" in v100) SCHEDA=$V100 ;; 4090) SCHEDA=$RTX4090 ;; esac

libera() { nvidia-smi --id="$SCHEDA" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
T0=$(date +%s)
until [ "$(libera)" -ge "$SERVE" ]; do
  [ $(( $(date +%s) - T0 )) -gt 180 ] && { echo "$NOME: sulla scheda ${SCHEDA:4:8} restano $(libera) MiB, ne servono $SERVE"; exit 3; }
  sleep 2
done

docker rm -f "decisore-$NOME" >/dev/null 2>&1 || true
for i in $(seq 1 30); do docker ps -aq -f name="^decisore-$NOME$" | grep -q . || break; sleep 1; done
exec docker run --rm --name "decisore-$NOME" --gpus "device=$SCHEDA" --network host \
  -e CTX -e PARALLELI -v "$HOME/models/decisori/$NOME:/modello:ro" \
  "$IMMAGINE" /opt/decisori/dentro.sh "$NOME" "$PORTA"
