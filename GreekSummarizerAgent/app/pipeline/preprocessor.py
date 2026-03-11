import re
from app.utils.config import settings

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str) -> list[str]:
    """Split long documents into overlapping chunks."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks