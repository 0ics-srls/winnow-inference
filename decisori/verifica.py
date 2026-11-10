#!/usr/bin/env python3
"""Controlla i file scaricati da scarica.sh contro gli SHA-256 che Hugging Face pubblica (API, campo lfs.sha256).
I file piccoli (non LFS) si confrontano per dimensione. Uso: python3 verifica.py [nome ...]"""
import hashlib
import json
import os
import sys
import urllib.request

BASE = os.path.expanduser("~/models/decisori")
MODELLI = {  # nome: (repo, revisione), come in scarica.sh
    "decider-4b": ("Mapika/decider-4b", "99d7fd5bfcc4d11ba37239dafe0ed653803819d3"),
    "h2o-lightning-4b": ("h2oai/h2o-lightning-4b", "b43f759b6385d21182f87c3829931ec3d4dfa98a"),
    "winnow-12b": ("EldanRing/Winnow-12B", "859a2cb24921d8b579badce6caa469f9290f3bc1"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 24), b""):
            h.update(blocco)
    return h.hexdigest()


def main(nomi):
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    errori = 0
    for nome in nomi or MODELLI:
        repo, rev = MODELLI[nome]
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true",
                                     headers={"Authorization": f"Bearer {token}"})
        info = {s["rfilename"]: s for s in json.load(urllib.request.urlopen(req))["siblings"]}
        presi = [r.split(" ")[0] for r in open(os.path.join(BASE, nome, ".elenco")).read().splitlines() if r]
        for f in presi:
            p, s = os.path.join(BASE, nome, f), info[f]
            if not os.path.exists(p):
                print(f"MANCA   {nome}/{f}"); errori += 1; continue
            lfs = (s.get("lfs") or {}).get("sha256")
            if lfs:
                ok = sha256(p) == lfs
                print(f"{'OK     ' if ok else 'DIVERSO'} {nome}/{f} (sha256)", flush=True)
            else:
                ok = os.path.getsize(p) == s.get("size")
                if not ok:
                    print(f"DIVERSO {nome}/{f} (dimensione)")
            errori += not ok
    print("TUTTO OK" if not errori else f"{errori} FILE NON VANNO")
    return 1 if errori else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
