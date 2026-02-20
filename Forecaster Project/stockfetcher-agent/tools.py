from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from agents import function_tool  # Agents SDK decorator  [oai_citation:2‡openai.github.io](https://openai.github.io/openai-agents-python/tools/)

log = logging.getLogger("stock_agent.tools")

DEFAULT_COMPANIES: Dict[str, str] = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "Oracle": "ORCL",
}

@function_tool
def fetch_stock_prices(
    years: int = 5,
    out_dir: str = "outputs",
    companies: Optional[Dict[str, str]] = None,
) -> Dict[str, str | int]:
    """
    Fetch daily prices for the last N years and export CSV + Parquet.

    Args:
      years: Number of years of history to fetch (default 5).
      out_dir: Output directory for files.
      companies: Optional mapping of company name -> ticker.

    Returns:
      Metadata about outputs and row counts.
    """
    if years < 1 or years > 20:
        raise ValueError("years must be between 1 and 20")

    companies = companies or DEFAULT_COMPANIES

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=years * 365)

    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    log.info("Fetching daily prices from %s to %s (UTC)", start, end)

    rows: list[dict] = []

    for name, ticker in companies.items():
        log.info("Fetching %s (%s)...", name, ticker)
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)

        if hist.empty:
            log.warning("No data returned for %s (%s). Skipping.", name, ticker)
            continue

        for ts, r in hist.iterrows():
            rows.append(
                {
                    "Name": name,
                    "Date": ts.strftime("%Y-%m-%d"),
                    "Opening Price": float(round(r["Open"], 2)),
                    "Closing Price": float(round(r["Close"], 2)),
                }
            )

        log.info("Fetched %d rows for %s (%s).", len(hist), name, ticker)

    if not rows:
        raise RuntimeError("No data fetched for any ticker.")

    df = pd.DataFrame(rows).sort_values(["Name", "Date"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"nasdaq_stock_prices_last_{years}_years.csv"
    pq_path = out_path / f"nasdaq_stock_prices_last_{years}_years.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    log.info("Wrote CSV: %s", csv_path)
    log.info("Wrote Parquet: %s", pq_path)

    return {
        "rows": int(len(df)),
        "csv": str(csv_path),
        "parquet": str(pq_path),
        "start": start,
        "end": end,
    }