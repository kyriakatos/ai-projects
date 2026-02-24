from typing import List

def chunk_text(text: str, chunk_chars: int = 2400, overlap: int = 250) -> List[str]:
    """
    Simple char-based chunking (robust & language-agnostic).
    You can later replace with token-based chunking.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if i >= n:
            break
    return chunks