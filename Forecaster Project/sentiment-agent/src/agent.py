from agents import Agent
from .tools import fetch_daily_company_sentiment

def build_agent() -> Agent:
    return Agent(
        name="Daily Sentiment Agent",
        instructions=(
            "You fetch daily average sentiment for companies and export CSV + Parquet.\n"
            "Use fetch_daily_company_sentiment.\n"
            "Default to 5 years and stocktwits unless the user specifies otherwise.\n"
            "If the user requests X, explain it requires official historical access and then attempt if configured."
        ),
        tools=[fetch_daily_company_sentiment],
    )