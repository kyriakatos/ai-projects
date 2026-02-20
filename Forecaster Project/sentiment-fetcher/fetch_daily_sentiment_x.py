#!/usr/bin/env python3
"""
Fetch daily average sentiment (0..100) for the last N years for specified companies
using X (Twitter) API v2 (full-archive search requires appropriate access).

Outputs:
  - daily_sentiment_last_{years}_years.csv
  - daily_sentiment_last_{years}_years.parquet

Columns:
  Company name, date, average sentiment (0..100)

Notes:
- X full-archive search is NOT available on all tiers.
- This script caps tweets per day per company to control cost/time.
- Sentiment scoring uses VADER (fast, local). 0..100 mapped from VADER compound [-1..1]:
    score_0_100 = (compound + 1) * 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------
# Logging
# ---------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


log = logging.getLogger("daily_sentiment_x")


# ---------------------------
# X API Client
# ---------------------------
@dataclass(frozen=True)
class XConfig:
    bearer_token: str
    base_url: str = "https://api.x.com/2"
    timeout_s: int = 30
    max_retries: int = 6
    backoff_base_s: float = 1.5


class XClient:
    """
    Minimal X API v2 client for full-archive search.
    Endpoint used:
      GET /2/tweets/search/all
    """

    def __init__(self, cfg: XConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {cfg.bearer_token}",
                "User-Agent": "daily-sentiment-script/1.0",
            }
        )

    def _request_with_retries(self, method: str, url: str, params: dict) -> dict:
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.session.request(
                    method, url, params=params, timeout=self.cfg.timeout_s
                )

                # Rate limit or transient errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self.cfg.backoff_base_s ** attempt
                    log.warning(
                        "HTTP %s from X API. attempt=%d/%d waiting=%.1fs",
                        resp.status_code,
                        attempt,
                        self.cfg.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                wait = self.cfg.backoff_base_s ** attempt
                log.warning(
                    "Request error: %s. attempt=%d/%d waiting=%.1fs",
                    e,
                    attempt,
                    self.cfg.max_retries,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError("X API request failed after retries.")

    def search_all(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        max_results: int = 100,
        next_token: Optional[str] = None,
    ) -> dict:
        """
        Full-archive search. Requires proper X API access.
        start_time/end_time must be timezone-aware UTC datetimes.
        """
        url = f"{self.cfg.base_url}/tweets/search/all"
        params = {
            "query": query,
            "start_time": start_time.isoformat().replace("+00:00", "Z"),
            "end_time": end_time.isoformat().replace("+00:00", "Z"),
            "max_results": max_results,  # 10..100
            "tweet.fields": "created_at,lang",
        }
        if next_token:
            params["next_token"] = next_token

        return self._request_with_retries("GET", url, params=params)


# ---------------------------
# Sentiment + aggregation
# ---------------------------
def vader_to_0_100(compound: float) -> float:
    # compound is [-1,1] -> [0,100]
    score = (compound + 1.0) * 50.0
    # clamp just in case
    return max(0.0, min(100.0, score))


def daterange(start: date, end: date) -> Iterable[date]:
    """Inclusive start, inclusive end."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def utc_day_window(d: date) -> Tuple[datetime, datetime]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def build_company_queries() -> Dict[str, str]:
    """
    Tune these queries as you like.
    Tips:
    - include cashtags ($AAPL) where relevant
    - use OR for alternate names
    - optionally filter: lang:en
    """
    return {
        "Apple": '(Apple OR $AAPL OR AAPL) lang:en -is:retweet',
        "Alphabet": '(Alphabet OR Google OR $GOOGL OR GOOGL) lang:en -is:retweet',
        "Meta": '(Meta OR Facebook OR Instagram OR $META OR META) lang:en -is:retweet',
        "NVIDIA": '(NVIDIA OR $NVDA OR NVDA) lang:en -is:retweet',
        "Oracle": '(Oracle OR $ORCL OR ORCL) lang:en -is:retweet',
    }


def fetch_texts_for_day(
    x: XClient,
    query: str,
    day: date,
    cap_per_day: int,
) -> List[str]:
    """
    Fetch up to cap_per_day tweet texts for a given day.
    Uses pagination with next_token.
    """
    start_dt, end_dt = utc_day_window(day)
    analyzer_limit = cap_per_day

    texts: List[str] = []
    next_token: Optional[str] = None

    while len(texts) < analyzer_limit:
        remaining = analyzer_limit - len(texts)
        max_results = min(100, max(10, remaining))  # X allows 10..100

        data = x.search_all(
            query=query,
            start_time=start_dt,
            end_time=end_dt,
            max_results=max_results,
            next_token=next_token,
        )

        tweets = data.get("data", [])
        for t in tweets:
            # Some payloads may omit text if not requested; search/all normally includes 'text'
            txt = t.get("text")
            if txt:
                texts.append(txt)
                if len(texts) >= analyzer_limit:
                    break

        meta = data.get("meta", {}) or {}
        next_token = meta.get("next_token")

        if not next_token or not tweets:
            break

    return texts


def compute_daily_avg_sentiment(texts: List[str], analyzer: SentimentIntensityAnalyzer) -> Optional[float]:
    if not texts:
        return None

    scores = []
    for txt in texts:
        compound = analyzer.polarity_scores(txt)["compound"]
        scores.append(vader_to_0_100(compound))

    return float(sum(scores) / len(scores))


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily average sentiment (0..100) from X for last N years.")
    parser.add_argument("--years", type=int, default=5, help="Number of years back (default: 5)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--cap-per-day", type=int, default=200, help="Max tweets per day per company (default: 200)")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level (default: INFO)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    bearer = os.getenv("X_BEARER_TOKEN")
    if not bearer:
        raise SystemExit("Missing env var X_BEARER_TOKEN (your X API v2 Bearer token).")

    cfg = XConfig(bearer_token=bearer)
    x = XClient(cfg)
    analyzer = SentimentIntensityAnalyzer()

    company_queries = build_company_queries()

    end_day = date.today()
    start_day = (datetime.now().date() - relativedelta(years=args.years))

    log.info("Computing daily sentiment from %s to %s (inclusive)", start_day, end_day)
    log.info("Cap per day per company: %d tweets", args.cap_per_day)

    rows = []
    total_days = (end_day - start_day).days + 1
    company_count = len(company_queries)

    for i, d in enumerate(daterange(start_day, end_day), start=1):
        if i % 30 == 1:
            log.info("Progress: day %d/%d", i, total_days)

        for company, query in company_queries.items():
            try:
                texts = fetch_texts_for_day(x, query, d, cap_per_day=args.cap_per_day)
                avg = compute_daily_avg_sentiment(texts, analyzer)

                if avg is None:
                    # No tweets returned that day; skip or write NaN. Here: write NaN row for completeness.
                    rows.append({"Company name": company, "date": d.isoformat(), "average sentiment (0..100)": float("nan")})
                else:
                    rows.append({"Company name": company, "date": d.isoformat(), "average sentiment (0..100)": round(avg, 2)})

            except Exception as e:
                log.warning("Failed for %s on %s: %s", company, d.isoformat(), e)
                rows.append({"Company name": company, "date": d.isoformat(), "average sentiment (0..100)": float("nan")})

    if not rows:
        raise SystemExit("No data collected.")

    df = pd.DataFrame(rows).sort_values(["Company name", "date"])

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"daily_sentiment_last_{args.years}_years.csv")
    pq_path = os.path.join(out_dir, f"daily_sentiment_last_{args.years}_years.parquet")

    log.info("Writing CSV: %s", csv_path)
    df.to_csv(csv_path, index=False)

    log.info("Writing Parquet: %s", pq_path)
    df.to_parquet(pq_path, index=False)

    log.info("Done. Rows: %d (days=%d * companies=%d = %d)", len(df), total_days, company_count, total_days * company_count)
    log.info("Outputs: %s | %s", csv_path, pq_path)


if __name__ == "__main__":
    main()