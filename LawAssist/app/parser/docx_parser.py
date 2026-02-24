from docx import Document
from io import BytesIO
from .base import DocumentParser

class DocxParser(DocumentParser):
    def parse(self, file_bytes: bytes) -> str:
        doc = Document(BytesIO(file_bytes))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras).strip()