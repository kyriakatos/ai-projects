# app/agents/answer_agent.py
from app.services.llm import generate_with_krikri

def build_rag_prompt(case_data: dict, retrieved_docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {d['title']} ({d['source']})\n{d['text']}"
        for i, d in enumerate(retrieved_docs)
    )

    return f"""
Είσαι νομικός βοηθός. Σύνταξε σύντομο εκτελεστικό σημείωμα στα ελληνικά.

Χρησιμοποίησε μόνο τα παρακάτω αποσπάσματα.
Αν κάτι δεν υποστηρίζεται από τα αποσπάσματα, δήλωσέ το ρητά.

Υπόθεση:
{case_data}

Νομικό υλικό:
{context}

Επέστρεψε JSON:
{{
  "executive_summary": "...",
  "relevant_laws_and_cases": [
    {{"citation_id": 1, "why_it_matters": "..."}}
  ],
  "strategy_considerations": ["..."],
  "uncertainties": ["..."]
}}
""".strip()

def answer_with_rag(case_data: dict, retrieval_result: dict):
    prompt = build_rag_prompt(case_data, retrieval_result["documents"])
    return generate_with_krikri(prompt)