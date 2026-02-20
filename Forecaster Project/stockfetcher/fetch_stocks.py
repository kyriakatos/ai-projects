#!/usr/bin/env python3
"""
Fetch last 5 years of daily stock prices for selected NASDAQ companies and export
to BOTH CSV and Parquet with basic logging.

Columns: Name, Date, Opening Price, Closing Price
"""

import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLC history for ticker between [start, end).
    Returns a DataFrame with yfinance's standard columns, index as DatetimeIndex.
    """
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, interval="1d", auto_adjust=False)
    return hist


def main() -> None:
    setup_logging()

    companies = {
        "Apple": "AAPL",
        "Google": "GOOGL",  # Alphabet Class A
        "Meta": "META",
        "NVIDIA": "NVDA",
        "Oracle": "ORCL",
    }

    # Use UTC dates for consistency
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=5 * 365)

    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    logging.info("Fetching daily prices from %s to %s (UTC)", start, end)

    rows: list[dict] = []
    for name, ticker in companies.items():
        try:
            logging.info("Fetching %s (%s)...", name, ticker)
            hist = fetch_history(ticker, start=start, end=end)

            if hist.empty:
                logging.warning("No data returned for %s (%s). Skipping.", name, ticker)
                continue

            # Ensure we only use the columns we need
            # yfinance index is timestamps; convert to date string
            for ts, r in hist.iterrows():
                rows.append(
                    {
                        "Name": name,
                        "Date": ts.strftime("%Y-%m-%d"),
                        "Opening Price": float(round(r["Open"], 2)),
                        "Closing Price": float(round(r["Close"], 2)),
                    }
                )

            logging.info("Fetched %d rows for %s (%s).", len(hist), name, ticker)

        except Exception:
            logging.exception("Failed fetching %s (%s). Continuing.", name, ticker)

    if not rows:
        logging.error("No data fetched for any ticker. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Optional: sort for nicer outputs
    df.sort_values(["Name", "Date"], inplace=True)

    # Output filenames
    csv_file = "nasdaq_stock_prices_last_5_years.csv"
    parquet_file = "nasdaq_stock_prices_last_5_years.parquet"

    # Export CSV
    logging.info("Writing CSV: %s", csv_file)
    df.to_csv(csv_file, index=False)

    # Export Parquet (requires pyarrow or fastparquet)
    try:
        logging.info("Writing Parquet: %s", parquet_file)
        df.to_parquet(parquet_file, index=False)
    except Exception as e:
        logging.error(
            "Parquet export failed (%s). Install one of: pyarrow, fastparquet.\n"
            "Example: pip install pyarrow",
            e,
        )
        raise

    logging.info("Done. Rows total: %d", len(df))
    logging.info("Outputs: %s, %s", csv_file, parquet_file)


if __name__ == "__main__":
    main()