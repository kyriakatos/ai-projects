import requests, feedparser
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from finviz_scraper import FinvizScraper
 
@dataclass
class LocalSocialData:
    ticker: str
    company: str
    stocktwits_posts: list[str] = field(default_factory=list)
    news_headlines:   list[str] = field(default_factory=list)
    combined_text:    str = ''
 
class DataAgent:
    """Stocktwits (free, no auth) + Google News RSS (free, no auth)."""
 
    """ def fetch_stocktwits(self, ticker: str) -> list[str]:
        url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
        try:
            resp     = requests.get(url, timeout=10)
            messages = resp.json().get('messages', [])
            return [
                f"[{m.get('likes', {}).get('total', 0)} likes] {m['body']}"
                for m in messages[:30]
            ]
        except Exception as e:
            print(f'Stocktwits error: {e}')
            return [] """
 
    def fetch_google_news(self, ticker: str, company: str) -> list[str]:
        #query = f'{ticker} stock'
        query = f'{ticker}'
        url   = (f'https://news.google.com/rss/search'
                 f'?q={query}&hl=en-US&gl=US&ceid=US:en')
        try:
            feed = feedparser.parse(url)
            return [entry.title for entry in feed.entries[:15]]
        except Exception as e:
            print(f'Google News error: {e}')
            return []
        
    def __init__(self):
        self.finviz = FinvizScraper()
 
    def fetch(self, ticker: str, company: str) -> LocalSocialData:
        #stocktwits = self.fetch_stocktwits(ticker)
        ##news       = self.fetch_google_news(ticker, company)
        #combined   = '\n'.join(stocktwits + news)
        news_finviz = [h.headline for h in self.finviz.get_headlines(ticker)]
        combined = news_finviz


        return LocalSocialData(
            ticker=ticker, company=company,
            #stocktwits_posts=stocktwits,
            news_headlines=news_finviz,
            combined_text=combined
            
        )
