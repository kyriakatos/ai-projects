import asyncio
import yfinance as yf
from agents.grok_agent import GrokAgent
from agents.claude_agent import ClaudeAgent
from agents.gemini_agent import GeminiAgent
 
def resolve_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", ticker)
    except Exception:
        return ticker
 
async def analyze_ticker(ticker: str) -> dict:
    company = resolve_company_name(ticker)
    grok, claude, gemini = GrokAgent(), ClaudeAgent(), GeminiAgent()
 
    social_data    = await grok.fetch_sentiment_data(ticker, company)
    claude_result  = await claude.analyze(social_data)
    gemini_result  = await gemini.validate(social_data, claude_result)
 
    final_score = (
        claude_result['sentiment_score']          * 0.5
        + gemini_result['gemini_sentiment_score'] * 0.3
        + (0.2 if claude_result['sentiment_score'] > 0 else -0.2)
    )
    return {
        'ticker':               ticker,
        'company':              company,
        'final_sentiment_score': round(final_score, 3),
        'claude_analysis':      claude_result,
        'gemini_validation':    gemini_result,
        'consensus':            gemini_result['agree'],
    }
 
async def main(tickers: list[str]):
    results = await asyncio.gather(*[analyze_ticker(t) for t in tickers])
    for r in results:
        print(f"{r['ticker']} | score={r['final_sentiment_score']:+.3f}"
              f" | {r['claude_analysis']['sentiment_label']}"
              f" | consensus={'Yes' if r['consensus'] else 'No'}")
 
if __name__ == "__main__":
    asyncio.run(main(["AAPL", "NVDA", "TSLA"]))
