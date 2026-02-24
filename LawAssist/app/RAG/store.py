from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from ..config import settings
import hashlib
import time

class RAGStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.embed_fn = DefaultEmbeddingFunction()  # CPU embeddings; replace with better model later
        self.col = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self.embed_fn
        )

    def _doc_id(self, filename: str, content: str) -> str:
        h = hashlib.sha256((filename + "\n" + content[:5000]).encode("utf-8", errors="ignore")).hexdigest()
        return h[:24]

    def add_document(self, filename: str, chunks: List[str], tags: Optional[List[str]] = None) -> Dict[str, Any]:
        doc_id = self._doc_id(filename, "\n".join(chunks))
        ids = []
        metadatas = []
        docs = []
        now = int(time.time())

        for idx, ch in enumerate(chunks):
            ids.append(f"{doc_id}-{idx}")
            metadatas.append({"doc_id": doc_id, "filename": filename, "chunk_index": idx, "tags": tags or [], "ts": now})
            docs.append(ch)

        if ids:
            self.col.upsert(ids=ids, documents=docs, metadatas=metadatas)

        return {"doc_id": doc_id, "chunks_added": len(ids)}

    def query(self, query_text: str, k: int = 4, tags: Optional[List[str]] = None) -> List[str]:
        if k <= 0:
            return []
        where = None
        if tags:
            where = {"tags": {"$contains": tags[0]}} if len(tags) == 1 else None  # simple filter example

        res = self.col.query(
            query_texts=[query_text],
            n_results=k,
            where=where
        )
        docs = res.get("documents", [[]])[0]
        return [d for d in docs if d and d.strip()]