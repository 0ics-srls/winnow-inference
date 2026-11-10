#!/bin/bash
# Scarica i quattro decisori da Hugging Face in ~/models/decisori/<nome>/, a 30 MB/s in tutto (un file alla volta),
# con ripresa. Solo i file che servono: pesi, configurazione, tokenizer, codice del wrapper; niente asset e grafici.
# Uso: ./scarica.sh            (tutti, nell'ordine: i piccoli prima)
#      ./scarica.sh winnow     (uno solo)
set -u
T=$(cat ~/.cache/huggingface/token)
BASE=~/models/decisori
LIMITE=${LIMITE:-30M}

# nome | repo | revisione | regex dei file da prendere
MODELLI=(
  "decider-4b|Mapika/decider-4b|99d7fd5bfcc4d11ba37239dafe0ed653803819d3|.*"
  "h2o-lightning-4b|h2oai/h2o-lightning-4b|b43f759b6385d21182f87c3829931ec3d4dfa98a|^(model\.safetensors|[^/]*\.(json|jinja|txt|py|sh|md)|LICENSE|routing_reproducibility/.*)$"
  "winnow-12b|EldanRing/Winnow-12B|859a2cb24921d8b579badce6caa469f9290f3bc1|^(gguf/Winnow-12B-Q8_0\.gguf|gguf/mmproj-Winnow-12B\.gguf|[^/]*\.(json|jinja|md)|SHA256SUMS|LICENSE|NOTICE|docs/[^/]*\.(md|json))$"
)

for riga in "${MODELLI[@]}"; do
  IFS='|' read -r nome repo rev filtro <<<"$riga"
  [ $# -gt 0 ] && [ "$1" != "$nome" ] && continue
  D=$BASE/$nome
  mkdir -p "$D"
  echo "$(date +%T) === $nome ($repo @ ${rev:0:8})"
  curl -s -H "Authorization: Bearer $T" "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true" \
    | python3 -c '
import json, re, sys
d = json.load(sys.stdin); f = re.compile(sys.argv[1])
for s in d["siblings"]:
    if f.search(s["rfilename"]): print(s["rfilename"], s.get("size") or 0)
' "$filtro" > "$D/.elenco"
  while read -r f dim; do
    mkdir -p "$D/$(dirname "$f")"
    if [ -f "$D/$f" ] && [ "$(stat -c %s "$D/$f")" = "$dim" ]; then continue; fi
    echo "$(date +%T) inizio $f ($((dim / 1000000)) MB)"
    for tentativo in 1 2 3 4 5; do
      curl -sSL --fail --retry 5 --limit-rate "$LIMITE" -C - -H "Authorization: Bearer $T" \
        -o "$D/$f" "https://huggingface.co/$repo/resolve/$rev/$f" && break
      echo "$(date +%T) $f: tentativo $tentativo fallito, riprendo"; sleep 10
    done
    [ "$(stat -c %s "$D/$f")" = "$dim" ] || echo "$(date +%T) ATTENZIONE $f: $(stat -c %s "$D/$f") byte invece di $dim"
  done < "$D/.elenco"
  echo "$(date +%T) fine $nome"
done
echo "$(date +%T) SCARICAMENTO-FINITO"
