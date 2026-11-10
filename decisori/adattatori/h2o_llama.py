"""H2O-Lightning-4B su llama.cpp: lo shim ufficiale (`h2o_lightning_shim.py`, nella cartella del modello) invariato, con la
sola classe `VLLM` sostituita da `LlamaBackend`.

Lo shim parla con vLLM per quattro cose, qui rifatte così:
  GET /v1/models               -> il nome del modello e il contesto di llama-server (/props)
  tokenize / detokenize        -> il tokenizer HF del modello (tokenizer.json, libreria `tokenizers`): gli stessi id di vLLM
  label_logprobs(prompt, ids)  -> prompt tokenizzato qui, `logprob_token_ids` su llama-server (patch del fork)
  la via delle immagini        -> non ancora: una richiesta con immagini riceve 422

    python3 h2o_llama.py --modello /modello --llama http://127.0.0.1:9001 --port 8741
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tokenizers import Tokenizer

from llama_client import Llama, LlamaError


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modello", required=True, help="cartella del modello HF (shim, serve_config.json, tokenizer.json)")
    ap.add_argument("--llama", required=True, help="URL di llama-server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, a.modello)
    import h2o_lightning_shim as S

    tok = Tokenizer.from_file(os.path.join(a.modello, "tokenizer.json"))
    llama = Llama(a.llama)

    class LlamaBackend(S.VLLM):
        def __init__(self, model):
            self.base = a.llama
            self.model = model

        def call(self, method, path, body=None):
            if method == "GET" and path == "/v1/models":
                try:
                    n_ctx = llama.n_ctx()
                except LlamaError as e:
                    raise S.UpstreamError(str(e), 503) from e
                return 200, {"data": [{"id": self.model, "max_model_len": n_ctx}]}
            raise S.UpstreamError(f"chiamata vLLM non prevista su llama.cpp: {method} {path}")

        def tokenize(self, text):
            return tok.encode(text, add_special_tokens=False).ids

        def detokenize(self, ids):
            return tok.decode(ids, skip_special_tokens=False)

        def label_logprobs(self, prompt, ids):
            try:
                vals, usage = llama.logprobs(self.tokenize(prompt), ids)
            except LlamaError as e:
                if "exceed" in str(e) or "context" in str(e):
                    raise S.Unprocessable(f"oltre il contesto di llama-server: {e}") from e
                raise S.UpstreamError(str(e)) from e
            return dict(zip(ids, vals)), {"prompt_tokens": usage["prompt_tokens"]}

        def tokenize_chat(self, messages):
            raise S.Unprocessable("immagini non ancora supportate su llama.cpp")

        def label_logprobs_chat(self, messages, ids, max_pixels=0):
            raise S.Unprocessable("immagini non ancora supportate su llama.cpp")

    cfg = json.load(open(os.path.join(a.modello, "serve_config.json")))
    cfg.pop("image", None)                 # la via delle immagini resta spenta finché non c'è il proiettore GGUF
    contract = S.Contract(cfg)
    shim = S.Shim(contract, LlamaBackend(contract.model))
    srv = S.Server((a.host, a.port), S.make_handler(shim))
    print(f"h2o su llama.cpp: {a.host}:{a.port} -> {a.llama} (T {contract.temperature_by_type}, "
          f"soglia sì/no {contract.noul_floor}, {len(contract.labels)} etichette)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
