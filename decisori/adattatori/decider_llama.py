"""decider-4b su llama.cpp: il server ufficiale (`decider/serve.py`, nella cartella del modello) invariato, con il motore
PyTorch (EngineV2) sostituito da `LlamaEngine`.

serve.py prepara le righe con il codice dell'autore (`decider.systemone`, `decider.prompt_fast`: una riga per domanda,
i token id dello stato e del blocco della domanda, la posizione della risposta "Answer: (") e chiede al motore le
probabilità delle lettere a quella posizione, divise per la temperatura del tipo di risposta. Qui il motore è
llama-server: per ogni posizione di risposta il prompt fino a quella posizione e le log-probabilità esatte delle lettere
(`logprob_token_ids`); la softmax su logprob/T sulle opzioni valide è quella di EngineV2 su logit/T.

Il serve.py di decider è asincrono e fa una "forward" per gruppo di righe: qui ogni gruppo va su llama-server in parallelo
(`--paralleli` richieste, quante sono le slot di llama-server), e la cache del prefisso di llama-server condivide lo stato.

    DECIDER_MODEL=/modello python3 decider_llama.py --llama http://127.0.0.1:9002 --port 8742
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tokenizers import Tokenizer

from llama_client import Llama


class HFTok:
    """Il minimo dell'AutoTokenizer che usano prompt.py e prompt_fast.py: encode senza token speciali."""

    def __init__(self, path, pad_token_id):
        self.t = Tokenizer.from_file(path)
        self.pad_token_id = pad_token_id
        self.chat_template = None

    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids


class LlamaEngine:
    def __init__(self, llama, tok, paralleli, max_options, label_ids, slot_temperatures):
        self.llama, self.tok = llama, tok
        self.pool = ThreadPoolExecutor(max_workers=paralleli, thread_name_prefix="llama")
        self.max_options, self.label_ids = max_options, label_ids
        self.slot_temperatures = slot_temperatures
        self.cfg = {"engine": "llama.cpp", "paralleli": paralleli}
        self.graphs, self.use_graphs, self.sealed = {}, False, False
        self.stats = {"rows": 0, "slots": 0}
        self.max_batch = 1 << 16

    # l'interfaccia di EngineV2 che serve.py usa
    def pad_len(self, n):
        return n

    def t_bucket(self, n):
        return n

    def max_rows(self, T):
        return self.max_batch

    def seal(self):
        self.sealed = True

    def _slot(self, ids, n, T):
        lp, _ = self.llama.logprobs(ids, self.label_ids[:n])
        v = np.asarray(lp, dtype=np.float64) / float(T)
        v -= v.max()
        p = np.exp(v)
        p /= p.sum()
        row = np.zeros(self.max_options, dtype=np.float32)
        row[:n] = p
        return row

    def score_items(self, items, temperature=1.0):
        temps = self.slot_temperatures(temperature, items)
        jobs, k = [], 0
        for it in items:
            for s, n in zip(it["slots"], it["nopts"]):
                T = temps if not isinstance(temps, (list, tuple)) else temps[k]
                k += 1
                jobs.append(self.pool.submit(self._slot, it["ids"][: s + 1], n, T))
        rows = [j.result() for j in jobs]
        out, k = [], 0
        for it in items:
            m = len(it["slots"])
            out.append(np.stack(rows[k:k + m]))
            k += m
        self.stats["rows"] += len(items)
        self.stats["slots"] += k
        return out

    score_shared = score_items        # la cache del prefisso di llama-server fa il lavoro di decider.shared_prefix


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llama", required=True, help="URL di llama-server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--paralleli", type=int, default=4, help="richieste contemporanee a llama-server (= sue slot)")
    a = ap.parse_args(argv)

    modello = os.environ["DECIDER_MODEL"]
    os.environ.setdefault("DECIDER_WARMUP", "0")
    sys.path.insert(0, modello)
    from decider import serve as SV
    from decider.prompt import MAX_OPTIONS, label_table
    from decider.temperature import slot_temperatures

    import json
    tcfg = json.load(open(os.path.join(modello, "tokenizer_config.json")))
    pad = tcfg.get("pad_token")
    tok = HFTok(os.path.join(modello, "tokenizer.json"), None)
    if isinstance(pad, str):
        tok.pad_token_id = tok.t.token_to_id(pad)
    label_ids = label_table(tok)[1]
    engine = LlamaEngine(Llama(a.llama), tok, a.paralleli, MAX_OPTIONS, label_ids, slot_temperatures)

    async def _start():
        SV.apply_config(SV.load_config(modello))
        if SV.LAYOUT != "plain":
            raise ValueError(f"layout {SV.LAYOUT}: l'adattatore conosce solo il layout plain")
        SV.eng = engine
        SV.CHAT = None
        SV.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        SV.eng.seal()
        print("[decider su llama.cpp] pronto", json.dumps(dict(model=SV.MODEL_NAME, temperature=SV.TEMP,
              temperature_by_type=SV.TT.effective(SV.TEMP, SV.TEMP_BY_TYPE), isolated_levels=SV.ISOLATED,
              llama=a.llama, paralleli=a.paralleli)), flush=True)
        SV.start_workers(asyncio.get_running_loop())

    SV._start = _start
    import uvicorn
    uvicorn.run(SV.app, host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
