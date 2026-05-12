import anthropic
from agents.grok_agent import SocialData
 
class ClaudeAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
 
    async def analyze(self, data: SocialData) -> dict:
        prompt = f'''
        You are a financial sentiment analyst. Based on the following
        social media data for {data.company} ({data.ticker}),
        provide a structured sentiment report.
 
        --- SOCIAL DATA ---
        {data.raw_grok_summary}
        --- END DATA ---
 
        Return JSON with:
        {
          "ticker": "...",
          "sentiment_score": <float -1.0 to 1.0>,
          "sentiment_label": "BULLISH | NEUTRAL | BEARISH",
          "confidence": <float 0.0 to 1.0>,
          "key_themes": [...],
          "risks": [...],
          "summary": "..."
        }
        Return only valid JSON, no markdown.
        '''
 
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        return json.loads(message.content[0].text)
