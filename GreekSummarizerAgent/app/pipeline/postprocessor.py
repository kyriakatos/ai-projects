from app.llm.inference import generate_summary

MERGE_PROMPT = """Έχεις τις παρακάτω μερικές περιλήψεις ενός μεγάλου κειμένου. 
Συνδύασέ τες σε μία ενιαία, συνεκτική περίληψη στα ελληνικά.

Μερικές περιλήψεις:
{summaries}

Τελική περίληψη:"""

def merge_summaries(summaries: list[str]) -> str:
    combined = "\n\n".join(f"- {s}" for s in summaries)
    prompt = MERGE_PROMPT.format(summaries=combined)
    return generate_summary(prompt)