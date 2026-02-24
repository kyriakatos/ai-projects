from app.parsers.text_parser import TextParser

def test_text_parser_utf8():
    p = TextParser()
    txt = p.parse("Καλημέρα κόσμε".encode("utf-8"))
    assert "Καλημέρα" in txt