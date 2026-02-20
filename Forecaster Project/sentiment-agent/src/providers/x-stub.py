from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

from .base import DailySentiment, SentimentProvider

log = logging.getLogger("sentiment_agent.providers.x")

class XProviderStub(SentimentProvider):
    """
    Stub: Implement using official X API only.

    Reality check:
    - Pulling 5 years of posts typically requires specific historical access.
    - You must comply with X Developer Policy/ToS and your tier limits.
    """

    name = "x"

    def fetch_daily_sentiment(self, symbol: str, start: date, end: date) -> Iterable[DailySentiment]:
        raise NotImplementedError(
            "X provider not implemented. You need official X API credentials and historical access."
        )