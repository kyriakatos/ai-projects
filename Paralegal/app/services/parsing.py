import json
import re
from typing import Any, Dict

def fallback_output(raw_text: str) -> Dict[str, Any]:
    return {
        "summary": raw_text[:2000].strip(),
        "parties": [],
        "legal_issues": [],
        "dates": [],
        "key_arguments": [],
        "citations": []
    }

def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else "{}"

def parse_model_json(raw_text: str) -> Dict[str, Any]:
    json_text = extract_json_block(raw_text)
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return fallback_output(raw_text)