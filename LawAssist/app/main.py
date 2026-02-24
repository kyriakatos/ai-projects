from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from .parsers import get_parser
from .schemas import SummarizeOptions, SummarizeResponse, IngestResponse
from .summarizer.llm import summarize_greek
from .rag.chunking import chunk_text
from .rag.store import RAGStore
import os
import time

app = FastAPI(title="Greek Summarization Agent", version="0.1.0")

rag = RAGStore()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    tags: str | None = Form(default=None),
):
    raw = await file.read()
    parser = get_parser(file.filename)
    text = parser.parse(raw)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    chunks = chunk_text(text)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or None
    res = rag.add_document(filename=file.filename, chunks=chunks, tags=tag_list)

    return IngestResponse(
        chunks_added=res["chunks_added"],
        doc_id=res["doc_id"],
        meta={"filename": file.filename, "chunks_total": len(chunks)}
    )

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    file: UploadFile = File(...),
    style: str = Form(default="executive"),
    bullets: bool = Form(default=True),
    include_actions: bool = Form(default=True),
    max_words: int | None = Form(default=None),
    k_retrieval: int = Form(default=4),
    tags: str | None = Form(default=None),
):
    raw = await file.read()
    parser = get_parser(file.filename)
    text = parser.parse(raw)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or None

    rag_ctx = None
    if k_retrieval and k_retrieval > 0:
        hits = rag.query(query_text=text[:2000], k=k_retrieval, tags=tag_list)
        if hits:
            rag_ctx = "\n\n".join([f"- {h}" for h in hits])

    result = summarize_greek(
        text=text,
        style=style,
        bullets=bullets,
        include_actions=include_actions,
        max_words=max_words,
        rag_context=rag_ctx,
    )

    return SummarizeResponse(
        language=result["language"],
        summary=result["summary"],
        key_points=result.get("key_points"),
        action_items=result.get("action_items"),
        meta={
            **result["meta"],
            "filename": file.filename,
            "text_chars": len(text),
            "used_rag": bool(rag_ctx),
        },
    )