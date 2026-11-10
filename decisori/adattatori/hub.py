"""Hub decisionale: una richiesta /v1/systemone, tutti i decisori del gruppo, una risposta combinata.

Ogni decisore riceve la richiesta così com'è (attraverso llama-swap, /upstream/<nome>/..., che lo carica se serve) e
risponde nel formato di TypeSafe. Il hub combina domanda per domanda:
  noul    media delle probabilità di «vero»
  choice  media delle distribuzioni sulle opzioni (una risposta senza distribuzione conta come certa), scelta = la più
          probabile della media
  score   media delle distribuzioni sui livelli, punteggio = valore atteso
più `accordo`: la frazione dei decisori la cui risposta coincide con quella combinata (per noul: dalla stessa parte di 0,5).
choice e score portano `confidence` (1 meno l'entropia normalizzata della media) e score la `legend` del primo decisore:
sono i campi che il formato System One chiede, senza i quali LiteLLM rifiuta la risposta.

    POST /v1/systemone  {"state": ..., "questions": {...}, "decisori": ["winnow-12b", ...] (facoltativo)}
      -> {"model": "decisori", "answers": {...combinate...}, "per_modello": {nome: answers}, "errori": {nome: testo},
          "secondi": {nome: s}}
    GET /health  /v1/models

    python3 hub.py --port 8130 --llama-swap http://127.0.0.1:8080      (in produzione, dentro il gruppo di llama-swap)
    python3 hub.py --port 8130 --diretto                              (prove: i decisori già accesi con avvia.sh)
"""

import argparse
import asyncio
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# nome in llama-swap -> porta del suo server (decisori/avvia.sh; --diretto le usa senza passare da llama-swap) e il
# nome del modello che il suo server si aspetta nel campo "model".
DECISORI = {"h2o-lightning-4b": 8132, "winnow-12b": 8133, "decider-4b": 8131}
MODELLO = {"h2o-lightning-4b": "h2oai/h2o-lightning-4b", "winnow-12b": "Winnow-12B", "decider-4b": "decider-4b-v2.1"}


def chiedi(base, nome, corpo, diretto=False, timeout=600):
    url = f"http://127.0.0.1:{DECISORI[nome]}/v1/systemone" if diretto else f"{base}/upstream/{nome}/v1/systemone"
    req = urllib.request.Request(url, data=json.dumps({**corpo, "model": MODELLO[nome]}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), None, time.time() - t0
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}", time.time() - t0
    except Exception as e:  # rete, timeout, risposta non JSON
        return None, f"{type(e).__name__}: {e}", time.time() - t0


def certezza(p):
    p = [x for x in p if x > 0]
    if len(p) <= 1:
        return 1.0
    return max(0.0, 1.0 - (-sum(x * math.log(x) for x in p)) / math.log(len(p)))


def distribuzione(risposta, chiavi):
    """La distribuzione di una risposta choice/score sulle chiavi date; None se non c'è modo di ricavarla."""
    prob = risposta.get("probabilities")
    if isinstance(prob, dict) and prob:
        v = [float(prob.get(k, 0.0)) for k in chiavi]
        s = sum(v)
        return [x / s for x in v] if s > 0 else None
    scelta = risposta.get("choice")
    if scelta is not None and str(scelta) in chiavi:
        return [1.0 if k == str(scelta) else 0.0 for k in chiavi]
    return None


def combina(domande, per_modello):
    out = {}
    for qid, q in domande.items():
        tipo = q.get("type", "choice")
        risposte = {n: a[qid] for n, a in per_modello.items() if isinstance(a, dict) and isinstance(a.get(qid), dict)}
        if not risposte:
            continue
        if tipo in ("noul", "bool"):
            pv = {n: float(r["noul"]) for n, r in risposte.items() if isinstance(r.get("noul"), (int, float))}
            if not pv:
                continue
            m = sum(pv.values()) / len(pv)
            out[qid] = {"type": "noul", "noul": round(m, 4),
                        "accordo": round(sum((p >= 0.5) == (m >= 0.5) for p in pv.values()) / len(pv), 4),
                        "decisori": len(pv)}
            continue
        crit = q.get("criteria", q.get("options"))
        if tipo == "score":
            chiavi = [str(i) for i in range(len(crit) if isinstance(crit, (list, tuple, dict)) else 0)]
        else:
            chiavi = [str(k) for k in (crit if isinstance(crit, dict) else (crit or []))]
        dist = {n: d for n, r in risposte.items() if (d := distribuzione(r, chiavi)) is not None}
        if not dist or not chiavi:
            continue
        media = [sum(d[i] for d in dist.values()) / len(dist) for i in range(len(chiavi))]
        j = max(range(len(chiavi)), key=media.__getitem__)
        a = {"type": tipo, "probabilities": {k: round(p, 4) for k, p in zip(chiavi, media)},
             "confidence": round(certezza(media), 4), "decisori": len(dist),
             "accordo": round(sum(max(range(len(chiavi)), key=d.__getitem__) == j for d in dist.values()) / len(dist), 4)}
        if tipo == "score":
            a["score"] = round(sum(i * p for i, p in enumerate(media)), 2)
            a["legend"] = next((r["legend"] for r in risposte.values() if isinstance(r.get("legend"), dict)), {})
        else:
            a["choice"] = chiavi[j]
        out[qid] = a
    return out


def main():
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--llama-swap", default="http://127.0.0.1:8080")
    ap.add_argument("--diretto", action="store_true", help="chiama le porte dei decisori senza passare da llama-swap (prove)")
    a = ap.parse_args()
    pool = ThreadPoolExecutor(max_workers=16)
    app = FastAPI(title="decisori", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        return {"status": "ok", "decisori": list(DECISORI)}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "decisori", "object": "model", "decisori": list(DECISORI)}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        corpo = await request.json()
        if not isinstance(corpo, dict) or not isinstance(corpo.get("questions"), dict) or "state" not in corpo:
            raise HTTPException(422, "serve un oggetto JSON con `state` e `questions`")
        nomi = corpo.pop("decisori", None) or list(DECISORI)
        sconosciuti = [n for n in nomi if n not in DECISORI]
        if sconosciuti:
            raise HTTPException(422, f"decisori sconosciuti: {sconosciuti}; ci sono {list(DECISORI)}")
        corpo.pop("model", None)
        risultati = await asyncio.gather(*(asyncio.wrap_future(pool.submit(chiedi, a.llama_swap, n, corpo, a.diretto)) for n in nomi))
        per_modello, errori, secondi = {}, {}, {}
        for n, (r, err, s) in zip(nomi, risultati):
            secondi[n] = round(s, 3)
            if err:
                errori[n] = err
            else:
                per_modello[n] = r.get("answers", {})
        if not per_modello:
            raise HTTPException(502, json.dumps(errori, ensure_ascii=False))
        return {"model": "decisori", "answers": combina(corpo["questions"], per_modello),
                "per_modello": per_modello, "errori": errori, "secondi": secondi}

    print(f"hub decisori su {a.host}:{a.port} -> {'porte dirette' if a.diretto else a.llama_swap} {list(DECISORI)}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
