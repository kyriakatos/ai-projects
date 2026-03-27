from typing import Any, Dict, List
from app.services.extractor import extract_text
from app.services.chunking import chunk_legal_text
from app.services.llm import call_llm

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "summary": "",
        "parties": [],
        "legal_issues": [],
        "dates": [],
        "key_arguments": [],
        "citations": []
    }

    for r in results:
        if r.get("summary"):
            merged["summary"] += r["summary"].strip() + "\n"
        merged["parties"].extend(r.get("parties", []))
        merged["legal_issues"].extend(r.get("legal_issues", []))
        merged["dates"].extend(r.get("dates", []))
        merged["key_arguments"].extend(r.get("key_arguments", []))
        merged["citations"].extend(r.get("citations", []))

    # de-duplicate while preserving order
    for key in ["parties", "legal_issues", "dates", "key_arguments", "citations"]:
        seen = set()
        deduped = []
        for item in merged[key]:
            norm = str(item).strip()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        merged[key] = deduped

    merged["summary"] = merged["summary"].strip()
    return merged

def document_understanding_agent(file_path: str) -> Dict[str, Any]:
    text = extract_text(file_path).strip()
    chunks = chunk_text(text, max_words=1200)
    partials = [call_llm(chunk) for chunk in chunks]
    return merge_results(partials)