from app.llm.inference import generate_summary
from app.pipeline.preprocessor import clean_text, chunk_text
from app.pipeline.postprocessor import merge_summaries
from app.utils.config import settings

SUMMARY_PROMPT = """Διάβασε το παρακάτω κείμενο και γράψε μια σύντομη περίληψη στα ελληνικά.

Κείμενο:
{text}

Περίληψη:"""

def summarize(text: str) -> str:
    text = clean_text(text)

    if len(text) <= settings.chunk_size:
        prompt = SUMMARY_PROMPT.format(text=text)
        return generate_summary(prompt)

    # Map-reduce for long documents
    chunks = chunk_text(text)
    chunk_summaries = []
    for chunk in chunks:
        prompt = SUMMARY_PROMPT.format(text=chunk)
        chunk_summaries.append(generate_summary(prompt))

    return merge_summaries(chunk_summaries)