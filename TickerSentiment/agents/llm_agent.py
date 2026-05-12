import ollama, json, re
 
class OllamaAnalystAgent:
    #def __init__(self, model: str = "llama3.1:70b"):
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
 
    def analyze(self, ticker: str, company: str, text: str) -> dict:
        prompt = f'''
You are a financial sentiment analyst.
Analyze the following data for {company} ({ticker}).
 
DATA:
{text[:3000]}
 
Return ONLY valid JSON:
{{
  "sentiment_score": <float -1.0 to 1.0>,
  "sentiment_label": "BULLISH" | "NEUTRAL" | "BEARISH",
  "confidence": <float 0.0 to 1.0>,
  "key_themes": [<3-5 strings>],
  "risks": [<1-3 strings>],
  "summary": "<2-3 sentence summary>"
}}
        '''
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1},
        )
        raw   = response['message']['content']
        clean = re.sub(r'```json|```', '', raw).strip()
        return json.loads(clean)
 
 
class OllamaValidatorAgent:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
 
    def validate(self, ticker: str, primary: dict, text: str) -> dict:
        prompt = f'''
Primary analysis of {ticker}:
  Score: {primary['sentiment_score']}
  Label: {primary['sentiment_label']}
 
Raw data:
{text[:2000]}
 
Do you agree? Return ONLY valid JSON:
{{
  "agree": <true|false>,
  "validator_score": <float -1.0 to 1.0>,
  "validator_label": "BULLISH" | "NEUTRAL" | "BEARISH",
  "disagreement_reason": "<string or null>"
}}
        '''
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1},
        )
        raw   = response['message']['content']
        clean = re.sub(r'```json|```', '', raw).strip()
        return json.loads(clean)
