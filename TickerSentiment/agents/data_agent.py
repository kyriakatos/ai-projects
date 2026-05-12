import requests, feedparser
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
 
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
        
    def fetch_finviz_news(self, ticker: str, company: str) -> list[str]:
        #query = f'{ticker} stock'
        query = f'{ticker}'
        url   = f'https://finviz.com/quote.ashx?t={query}'

        try:

        # 2. Fetch the page
           response = requests.get(url, headers=headers)
           soup = BeautifulSoup(response.text, 'html.parser')

        # 3. Find the news table
           news_table = soup.find(id='news-table')

        # 4. Loop through the rows to get the headlines
           for row in news_table.findAll('tr'):
                a_tag = row.find('a')
                if a_tag:
                    headline = a_tag.text
                    #link = a_tag['href']
        # The time is in the adjacent cell
                    #time_text = row.td.text.strip()
        #print(f"{time_text} | {headline} | {link}")


            #feed = feedparser.parse(url)
           return [entry.title for entry in headline.entries[:15]]
        except Exception as e:
            print(f'Finviz News error: {e}')
            return []
 
    def fetch(self, ticker: str, company: str) -> LocalSocialData:
        #stocktwits = self.fetch_stocktwits(ticker)
        ##news       = self.fetch_google_news(ticker, company)
        #combined   = '\n'.join(stocktwits + news)
        news_finviz = self.fetch_finviz_news(ticker, company)
        combined = news_finviz


        return LocalSocialData(
            ticker=ticker, company=company,
            #stocktwits_posts=stocktwits,
            news_headlines=news,
            combined_text=combined
            
        )
