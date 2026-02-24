from app.rag.chunking import chunk_text

def test_chunking_non_empty():
    text = "α" * 5000
    chunks = chunk_text(text, chunk_chars=2000, overlap=200)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)