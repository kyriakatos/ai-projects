from .pdf_parser import PdfParser
from .docx_parser import DocxParser
from .text_parser import TextParser

def get_parser(filename: str):
    name = filename.lower()
    if name.endswith(".pdf"):
        return PdfParser()
    if name.endswith(".docx"):
        return DocxParser()
    return TextParser()