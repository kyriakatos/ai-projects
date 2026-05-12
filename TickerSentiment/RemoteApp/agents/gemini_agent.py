import os, json
import google.generativeai as genai
from agents.grok_agent import SocialData
 
class GeminiAgent:
    def __init__(self):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-pro")
 
    async def validate(self, data: SocialData, claude_result: dict) -> dict:
        prompt = f'''
        A previous analysis of {data.company} ({data.ticker}) returned:
        sentiment_score={claude_result['sentiment_score']},
        label={claude_result['sentiment_label']}.
 
        Raw social data:
        {data.raw_grok_summary}
 
        Do you agree? Return JSON:
        {
          "agree": <bool>,
          "gemini_sentiment_score": <float>,
          "gemini_label": "BULLISH | NEUTRAL | BEARISH",
          "disagreement_reason": "<string or null>"
        }
        '''
        response = self.model.generate_content(prompt)
        return json.loads(response.text)
