# app/services/embeddings.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
model = SentenceTransformer(MODEL_NAME)

QUERY_PREFIX = "Instruct: Retrieve Greek legal passages relevant to the case.\nQuery: "
PASSAGE_PREFIX = "passage: "

def embed_query(query: str):
    return model.encode(QUERY_PREFIX + query, normalize_embeddings=True)

def embed_passages(passages: list[str]):
    texts = [PASSAGE_PREFIX + p for p in passages]
    return model.encode(texts, normalize_embeddings=True)