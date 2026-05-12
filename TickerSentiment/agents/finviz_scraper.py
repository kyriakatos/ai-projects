import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class FinvizHeadline:
    time: str
    headline: str
    url: str
    source: str

class FinvizScraper:
    """
    Scrapes ticker-specific news headlines from Finviz's quote page.
    No API key needed. Finviz is freely scrapable with a proper User-Agent.
    """

    BASE_URL = "https://finviz.com/quote.ashx"
    HEADERS  = {
        # Finviz blocks the default requests user-agent — this mimics a browser
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def get_headlines(self, ticker: str) -> list[FinvizHeadline]:
        url      = f"{self.BASE_URL}?t={ticker.upper()}&p=d"
        response = requests.get(url, headers=self.HEADERS, timeout=15)
        response.raise_for_status()

        soup  = BeautifulSoup(response.text, "html.parser")
        # Finviz news is in a table with id="news-table"
        table = soup.find(id="news-table")

        if not table:
            print(f"[{ticker}] No news table found — ticker may be invalid.")
            return []

        headlines = []
        last_date = ""

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            # First cell contains date/time, second cell contains the link
            time_cell = cells[0].get_text(strip=True)
            link_cell = cells[1]
            anchor    = link_cell.find("a")

            if not anchor:
                continue

            headline_text = anchor.get_text(strip=True)
            headline_url  = anchor.get("href", "")

            # Finviz reuses the date only on the first row of each day
            # Subsequent rows show time only (e.g. "08:30AM")
            if len(time_cell) > 8:          # full "May-12-26 08:30AM"
                parts     = time_cell.split()
                last_date = parts[0]
                time_str  = parts[1] if len(parts) > 1 else ""
            else:
                time_str  = time_cell       # just "08:30AM"

            # Derive source from URL domain
            try:
                from urllib.parse import urlparse
                source = urlparse(headline_url).netloc.replace("www.", "")
            except Exception:
                source = "unknown"

            headlines.append(FinvizHeadline(
                time=f"{last_date} {time_str}".strip(),
                headline=headline_text,
                url=headline_url,
                source=source,
            ))

        return headlines


def print_headlines(ticker: str, headlines: list[FinvizHeadline]):
    print(f"\n{'='*65}")
    print(f"  Finviz Headlines for {ticker.upper()}  ({len(headlines)} found)")
    print(f"{'='*65}")
    for h in headlines:
        print(f"  [{h.time:<22}] {h.headline}")
        print(f"   └─ {h.source}")
        print()


if __name__ == "__main__":
    scraper   = FinvizScraper()
    tickers   = ["AAPL", "NVDA", "TSLA"]

    for ticker in tickers:
        headlines = scraper.get_headlines(ticker)
        print_headlines(ticker, headlines)