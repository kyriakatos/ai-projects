from app.pipeline.summarizer import summarize
from app.utils.logger import get_logger

logger = get_logger(__name__)

class SummarizerAgent:
    def run(self, document: str) -> dict:
        logger.info("Agent received document, starting summarization...")
        summary = summarize(document)
        logger.info("Summarization complete.")
        return {"summary": summary, "input_length": len(document)}