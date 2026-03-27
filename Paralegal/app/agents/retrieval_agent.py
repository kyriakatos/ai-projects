# app/agents/retrieval_agent.py
from app.services.embeddings import embed_query
from app.services.vector_store import search_dense
from app.services.bm25_store import search_keyword
from app.services.reranker import rerank

def build_legal_query(structured_case: dict) -> str:
    parts = []
    parts.extend(structured_case.get("legal_issues", []))
    parts.extend(structured_case.get("key_arguments", []))
    if structured_case.get("summary"):
        parts.append(structured_case["summary"])
    return " | ".join(parts)

def dedupe_docs(docs: list[dict]):
    seen = set()
    out = []
    for d in docs:
        if d["chunk_id"] not in seen:
            seen.add(d["chunk_id"])
            out.append(d)
    return out

def retrieve_legal_context(structured_case: dict):
    query = build_legal_query(structured_case)

    qvec = embed_query(query)
    dense = search_dense(qvec, top_k=20)
    keyword = search_keyword(query, top_k=20)

    merged = dedupe_docs(dense + keyword)
    top_docs = rerank(query, merged, top_k=8)

    return {
        "query": query,
        "documents": top_docs
    }