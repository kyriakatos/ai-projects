CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE legal_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT,
  source TEXT,
  title TEXT,
  document_type TEXT,
  date TEXT,
  topic TEXT[],
  text TEXT,
  embedding vector(1024),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(text,''))) STORED
);

CREATE INDEX legal_chunks_embedding_idx
ON legal_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX legal_chunks_tsv_idx
ON legal_chunks USING GIN (tsv);