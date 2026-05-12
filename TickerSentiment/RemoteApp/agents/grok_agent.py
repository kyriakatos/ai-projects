import os, httpx
from dataclasses import dataclass
 
@dataclass
class SocialData:
    ticker: str
    company: str
    posts: list[str]
    raw_grok_summary: str
 
class GrokAgent:
    BASE_URL = "https://api.x.ai/v1"
 
    def __init__(self):
        self.api_key = os.environ["KK_XAI_key"]
        self.model = "grok-3"
 
    async def fetch_sentiment_data(self, ticker: str, company: str) -> SocialData:
        prompt = f'''
        Search for the latest social media posts and news about {company} ({ticker}).
        Focus on the last 24-48 hours. Return:
        1. 5-10 representative posts/headlines
        2. Your assessment of the overall sentiment tone
        '''
 
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
 
        return SocialData(
            ticker=ticker, company=company,
            posts=[], raw_grok_summary=content
        )
