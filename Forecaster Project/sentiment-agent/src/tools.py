from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from agents import function_tool

from .providers.stocktwits import StockTwitsProvider
from .providers.x_stub import XProviderStub

log = logging.getLogger("sentiment_agent.tools")

COMPANIES: Dict[str, str] = {
    "Apple": "AAPL",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "Oracle": "ORCL",
}

def _date_years_ago(years: int) -> date:
    # Simple 365-day approximation; good enough for windowing daily series.
    return (date.today() - timedelta(days=years * 365))

@function_tool
def fetch_daily_company_sentiment(
    years: int = 5,
    source: str = "stocktwits",
    out_dir: str = "outputs",
    companies: Optional[Dict[str, str]] = None,
    stocktwits_api_base: Optional[str] = None,
    stocktwits_token: Optional[str] = None,
) -> Dict[str, str | int]:
    """
    Fetch average sentiment per day for the last N years for a set of companies.
    Exports to CSV and Parquet.

    source:
      - "stocktwits" (implemented)
      - "x" (stub; requires official X historical access)
    """
    if years < 1 or years > 10:
        raise ValueError("years must be between 1 and 10")

    companies = companies or COMPANIES
    end = date.today()
    start = _date_years_ago(years)

    if source.lower() == "stocktwits":
        provider = StockTwitsProvider(api_base=stocktwits_api_base, api_token=stocktwits_token)
    elif source.lower() == "x":
        provider = XProviderStub()
    else:
        raise ValueError("source must be one of: stocktwits, x")

    rows = []
    for name, symbol in companies.items():
        log.info("Fetching sentiment for %s (%s) from %s", name, symbol, provider.name)
        series = list(provider.fetch_daily_sentiment(symbol=symbol, start=start, end=end))
        for s in series:
            rows.append({
                "Name": name,
                "Symbol": symbol,
                "Date": s.dt.isoformat(),
                "Avg Sentiment": float(s.avg_sentiment),
                "Source": s.source,
            })

    if not rows:
        raise RuntimeError("No sentiment data fetched (check source credentials/plan).")

    df = pd.DataFrame(rows).sort_values(["Symbol", "Date"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"daily_sentiment_last_{years}_years_{source}.csv"
    pq_path = out_path / f"daily_sentiment_last_{years}_years_{source}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    log.info("Wrote CSV: %s", csv_path)
    log.info("Wrote Parquet: %s", pq_path)

    return {"rows": int(len(df)), "csv": str(csv_path), "parquet": str(pq_path)}