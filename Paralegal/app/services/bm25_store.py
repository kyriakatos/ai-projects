# app/services/bm25_store.py
import psycopg

def search_keyword(query: str, top_k: int = 20):
    with psycopg.connect("dbname=legal user=postgres password=postgres host=localhost") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, doc_id, title, source, text,
                       ts_rank_cd(tsv, websearch_to_tsquery('simple', %s)) AS score
                FROM legal_chunks
                WHERE tsv @@ websearch_to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, top_k))
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