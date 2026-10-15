"""Demo adapter: pure-Python BM25 over chunked docs. LOCKED after `init`.

An adapter is the bridge between rag-autotune and YOUR retriever. It must define:
    build_index(config) -> any        # build/load the index for this config
    retrieve(index, question, config) -> list[str]   # ranked ids, best first
The ids it returns must match `relevant_ids` in the eval set (here: doc ids).
Every knob the agent may tune should be read from `config`.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
STOP = set("""a an and are as at be but by can do does for from has have how i if in
into is it its my no not of on or so that the their then there these they this to
was what when where which who why will with you your""".split())
SYNONYMS = {  # tiny hand-made expansion table, enabled by config["expand_synonyms"]
    "tracker": ["tag"], "device": ["tag"], "light": ["led"], "flashing": ["blinking"],
    "guarantee": ["warranty"], "money": ["refund"], "subscription": ["plan", "pro"],
    "notified": ["alert"], "zones": ["geofence"], "noise": ["sound"], "heavy": ["weighs", "grams"],
    "winter": ["cold"], "sea": ["saltwater", "seawater"], "companies": ["business"],
    "download": ["export"], "secretly": ["stalking"], "updates": ["firmware"],
}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str, cfg: dict) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if cfg.get("remove_stopwords"):
        toks = [t for t in toks if t not in STOP]
    if cfg.get("stem"):
        toks = [_stem(t) for t in toks]
    return toks


def _chunks(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for start in range(0, max(1, len(words)), step):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_index(config: dict):
    docs = [json.loads(l) for l in (HERE / "corpus.jsonl").read_text().splitlines() if l.strip()]
    chunks = []
    for d in docs:
        for c in _chunks(d["text"].split(), int(config["chunk_size"]), int(config["chunk_overlap"])):
            text = f"{d['title']}. {c}" if config.get("prepend_title") else c
            chunks.append((d["id"], tokenize(text, config)))
    df = Counter()
    for _, toks in chunks:
        df.update(set(toks))
    avgdl = sum(len(t) for _, t in chunks) / len(chunks)
    return {"chunks": chunks, "df": df, "n": len(chunks), "avgdl": avgdl,
            "tf": [Counter(t) for _, t in chunks]}


def retrieve(index, question: str, config: dict) -> list[str]:
    q = tokenize(question, config)
    if config.get("expand_synonyms"):
        raw = re.findall(r"[a-z0-9]+", question.lower())
        q += [tokenize(s, config)[0] for w in raw for s in SYNONYMS.get(w, []) if tokenize(s, config)]
    k1, b = float(config["bm25_k1"]), float(config["bm25_b"])
    n, avgdl, df = index["n"], index["avgdl"], index["df"]
    doc_scores = {}
    for (doc_id, toks), tf in zip(index["chunks"], index["tf"]):
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avgdl))
        if config.get("doc_aggregation", "max") == "sum":
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + s
        else:
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), s)
    ranked = sorted(doc_scores.items(), key=lambda x: -x[1])
    return [d for d, s in ranked[: int(config["top_k"])] if s > 0]
