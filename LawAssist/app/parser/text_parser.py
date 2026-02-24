from .base import DocumentParser

class TextParser(DocumentParser):
    def parse(self, file_bytes: bytes) -> str:
        # best-effort decoding
        for enc in ("utf-8", "utf-8-sig", "cp1253", "iso-8859-7"):
            try:
                return file_bytes.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace").strip()