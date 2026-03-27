# scripts/ingest_corpus.py
from datasets import load_dataset
from app.services.chunking import chunk_legal_text
from app.services.embeddings import embed_passages
from app.services.vector_store import upsert_chunks

def load_glc():
    ds = load_dataset("AI-team-UoA/greek_legal_code", split="train")
    return ds

def normalize_glc_record(row):
    text = row.get("text", "") or row.get("document", "") or ""
    title = row.get("title", "") or row.get("name", "")
    topics = row.get("labels", []) or row.get("topic", [])
    return {
        "doc_id": str(row.get("id", title)),
        "source": "greek_legal_code",
        "title": title,
        "document_type": "law",
        "date": row.get("date", ""),
        "topic": topics,
        "text": text,
    }

def ingest_glc():
    ds = load_glc()
    batch = []

    for row in ds:
        rec = normalize_glc_record(row)
        chunks = chunk_legal_text(rec["text"], metadata=rec)
        batch.extend(chunks)

        if len(batch) >= 256:
            texts = [x["text"] for x in batch]
            vecs = embed_passages(texts)
            upsert_chunks(batch, vecs)
            batch = []

    if batch:
        texts = [x["text"] for x in batch]
        vecs = embed_passages(texts)
        upsert_chunks(batch, vecs)

if __name__ == "__main__":
    ingest_glc()