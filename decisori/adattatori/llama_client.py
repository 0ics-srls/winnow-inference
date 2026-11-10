"""Client minimo per llama-server (il winnow-server del nostro fork, con la patch `logprob_token_ids`).

Una sola operazione serve a tutti e tre gli adattatori: dato un prompt in token id, le log-probabilità esatte (sull'intero
vocabolario) di alcuni token alla posizione dopo l'ultimo token del prompt. È quello che i wrapper degli autori chiedono a
vLLM con `logprob_token_ids` (H2O) o leggono dai logit di PyTorch (decider): la softmax sulle sole opzioni di
logprob/T è identica a quella di logit/T, perché le due differiscono di una costante.

Solo libreria standard: una connessione keep-alive per thread.
"""
from __future__ import annotations

import http.client
import json
import threading
import urllib.parse


class LlamaError(RuntimeError):
    pass


class Llama:
    def __init__(self, base: str, timeout: float = 600.0):
        u = urllib.parse.urlsplit(base if "://" in base else "http://" + base)
        self.host = u.netloc
        self.timeout = timeout
        self.local = threading.local()

    def _conn(self, fresh=False):
        c = getattr(self.local, "conn", None)
        if c is None or fresh:
            if c is not None:
                c.close()
            c = self.local.conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        return c

    def call(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data is not None else {}
        status, raw = 0, b""
        for attempt in (0, 1):
            try:
                c = self._conn(fresh=attempt == 1)
                c.request(method, path, body=data, headers=headers)
                r = c.getresponse()
                status, raw = r.status, r.read()
                break
            except (http.client.HTTPException, OSError) as e:
                if attempt == 1:
                    raise LlamaError(f"llama-server {self.host} irraggiungibile: {type(e).__name__}: {e}") from e
        try:
            out = json.loads(raw)
        except ValueError:
            out = raw.decode("utf-8", "replace")
        return status, out

    def props(self) -> dict:
        status, out = self.call("GET", "/props")
        if status != 200 or not isinstance(out, dict):
            raise LlamaError(f"/props HTTP {status}: {str(out)[:300]}")
        return out

    def n_ctx(self) -> int:
        p = self.props()
        return int(p.get("default_generation_settings", {}).get("n_ctx") or p.get("n_ctx") or 0)

    def logprobs(self, prompt_ids: list[int], token_ids: list[int]) -> tuple[list[float], dict]:
        """Log-probabilità esatte di `token_ids` dopo `prompt_ids`. -> (valori nell'ordine di token_ids, uso)."""
        status, out = self.call("POST", "/completion", {
            "prompt": list(prompt_ids), "n_predict": 1, "temperature": 0.0, "n_probs": 1,
            "logprob_token_ids": list(token_ids), "cache_prompt": True, "return_tokens": False,
        })
        if status != 200 or not isinstance(out, dict):
            raise LlamaError(f"/completion HTTP {status}: {json.dumps(out)[:500] if isinstance(out, dict) else str(out)[:500]}")
        try:
            got = out["completion_probabilities"][0]["id_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise LlamaError(f"llama-server non ha restituito id_logprobs ({e}): è il winnow-server con la patch?") from e
        if len(got) != len(token_ids):
            raise LlamaError(f"id_logprobs: {len(got)} valori per {len(token_ids)} token")
        t = out.get("timings") or {}
        usage = {"prompt_tokens": len(prompt_ids), "cached_tokens": int(t.get("cache_n", 0) or 0),
                 "evaluated_tokens": int(t.get("prompt_n", 0) or 0)}
        return [float(x) for x in got], usage
