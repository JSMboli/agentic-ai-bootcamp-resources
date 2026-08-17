"""A tiny transparent retrieval helper for teaching RAG without paid embedding APIs."""
from __future__ import annotations
import math, re
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def simple_tfidf_search(query: str, documents: List[Dict[str, str]], text_key: str = "content", top_k: int = 3) -> List[Dict[str, str]]:
    corpus = [doc.get(text_key, "") for doc in documents]
    doc_tokens = [tokenize(t) for t in corpus]
    vocab = sorted(set(t for toks in doc_tokens for t in toks))
    if not vocab:
        return []
    df = {term: sum(1 for toks in doc_tokens if term in toks) for term in vocab}
    def vec(tokens):
        counts = {t: tokens.count(t) for t in set(tokens)}
        n = len(doc_tokens)
        return {t: counts.get(t,0) * math.log((n+1)/(df[t]+1)) for t in vocab}
    qv = vec(tokenize(query))
    def cosine(a,b):
        dot = sum(a[t]*b[t] for t in vocab)
        na = math.sqrt(sum(a[t]*a[t] for t in vocab))
        nb = math.sqrt(sum(b[t]*b[t] for t in vocab))
        return 0.0 if na == 0 or nb == 0 else dot/(na*nb)
    scored = []
    for doc, toks in zip(documents, doc_tokens):
        scored.append((cosine(qv, vec(toks)), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        d = dict(doc)
        d['similarity'] = round(score, 3)
        results.append(d)
    return results
