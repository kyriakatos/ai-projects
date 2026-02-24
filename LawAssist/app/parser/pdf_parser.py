from pypdf import PdfReader
from io import BytesIO
from .base import DocumentParser

class PdfParser(DocumentParser):
    def parse(self, file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts).strip()