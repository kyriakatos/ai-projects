# app/services/reranker.py
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def rerank(query: str, docs: list[dict], top_k: int = 8):
    pairs = [[query, d["text"]] for d in docs]
    scores = reranker.compute_score(pairs)

    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_k]