# app/services/vector_store.py
import psycopg
from pgvector.psycopg import register_vector

def get_conn():
    conn = psycopg.connect("dbname=legal user=postgres password=postgres host=localhost")
    register_vector(conn)
    return conn

def upsert_chunks(chunks, vectors):
    with get_conn() as conn:
        with conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors):
                cur.execute("""
                    INSERT INTO legal_chunks
                    (chunk_id, doc_id, source, title, document_type, date, topic, text, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding
                """, (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["source"],
                    chunk.get("title", ""),
                    chunk.get("document_type", ""),
                    chunk.get("date", ""),
                    chunk.get("topic", []),
                    chunk["text"],
                    vec
                ))

def search_dense(query_vec, top_k=20):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, doc_id, title, source, text, 1 - (embedding <=> %s::vector) AS score
                FROM legal_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, query_vec, top_k))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "doc_id": r[1],
            "title": r[2],
            "source": r[3],
            "text": r[4],
            "score": float(r[5]),
        }
        for r in rows
    ]