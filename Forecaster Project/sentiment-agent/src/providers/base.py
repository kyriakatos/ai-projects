from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Iterable

@dataclass(frozen=True)
class DailySentiment:
    symbol: str
    dt: date
    avg_sentiment: float  # normalized score (provider-defined, document it)
    source: str

class SentimentProvider(Protocol):
    name: str

    def fetch_daily_sentiment(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> Iterable[DailySentiment]:
        ...