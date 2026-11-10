#!/bin/bash
# I 231 item pubblici di JevBench (github.com/fstandhartinger/jevbench, datasets/public: 48 facili, 72 «original», 111
# difficili) contro un decisore, con l'adattatore TypeSafe del banco (POST /v1/systemone). Poi l'accuratezza per livello.
#   ~/decisori/banco.sh <nome> <porta> [modello]      es.  banco.sh decider-4b 8131
# Risultati in ~/decisori/banco/<nome>/ (risposte grezze comprese). Il banco non ritenta e va in serie.
set -u
NOME=$1; PORTA=$2; MODELLO=${3:-$1}
JB=$HOME/src/jevbench
OUT=$HOME/decisori/banco/$NOME
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$JB" || exit 1
export TYPESAFE_API_KEY=locale
for livello in easy original hard; do
  python3 -m jevbench.cli run --tasks "datasets/public/$livello.jsonl" --adapter typesafe \
    --endpoint "http://127.0.0.1:$PORTA" --model "$MODELLO" --results "$OUT/$livello.jsonl" \
    --ledger "$OUT/ledger.jsonl" --raw-dir "$OUT/raw-$livello" --run-label "$NOME-$livello" > "$OUT/$livello.log" 2>&1
done
python3 - "$OUT" <<'EOF'
import json, sys, os
out = sys.argv[1]
tot_ok = tot = 0
for livello in ("easy", "original", "hard"):
    righe = [json.loads(l) for l in open(os.path.join(out, f"{livello}.jsonl")) if l.strip()]
    ok = sum(1 for r in righe if r.get("correct") is True)
    err = sum(1 for r in righe if not r.get("ok", True))
    lat = sorted(r.get("latency_s") or r.get("wall_s") or 0 for r in righe)
    print(f"{livello:9} {ok:3}/{len(righe):3} = {ok / max(1, len(righe)):.3f}   errori {err}   latenza mediana {lat[len(lat) // 2]:.3f} s")
    tot_ok += ok; tot += len(righe)
print(f"{'totale':9} {tot_ok:3}/{tot:3} = {tot_ok / max(1, tot):.3f}")
EOF
