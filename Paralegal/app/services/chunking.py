# app/services/chunking.py
import re

def split_legal_sections(text: str):
    patterns = [
        r"(?=Άρθρο\s+\d+)",
        r"(?=ΚΕΦΑΛΑΙΟ\s+[Α-ΩA-Z]+)",
        r"(?=ΤΜΗΜΑ\s+[Α-ΩA-Z]+)",
    ]

    parts = [text]
    for pattern in patterns:
        new_parts = []
        for p in parts:
            split = re.split(pattern, p)
            split = [s.strip() for s in split if s.strip()]
            new_parts.extend(split if len(split) > 1 else [p])
        parts = new_parts

    return parts

def chunk_legal_text(text: str, metadata: dict, max_words: int = 700, overlap: int = 100):
    sections = split_legal_sections(text)
    chunks = []

    for sec_idx, sec in enumerate(sections):
        words = sec.split()
        step = max_words - overlap
        for i in range(0, len(words), step):
            part = " ".join(words[i:i+max_words]).strip()
            if not part:
                continue
            chunks.append({
                **metadata,
                "chunk_id": f'{metadata["doc_id"]}_s{sec_idx}_c{i}',
                "text": part
            })
    return chunks