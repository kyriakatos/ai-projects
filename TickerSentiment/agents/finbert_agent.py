from transformers import pipeline
 
class FinBERTAgent:
    def __init__(self):
        self.pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
 
    def score(self, text: str) -> dict:
        chunk   = text[:1500]   # FinBERT max input is 512 tokens
        results = self.pipe(chunk)[0]
        scores  = {r['label']: round(r['score'], 3) for r in results}
        label   = max(scores, key=scores.get)
        numeric = scores.get('positive', 0) - scores.get('negative', 0)
        return {
            "finbert_label":   label,
            "finbert_score":   round(numeric, 3),
            "finbert_details": scores,
        }
