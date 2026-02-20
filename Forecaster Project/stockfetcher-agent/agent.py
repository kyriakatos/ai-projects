from agents import Agent
from .tools import fetch_stock_prices

def build_agent() -> Agent:
    return Agent(
        name="Stock Export Agent",
        instructions=(
            "You help users fetch historical stock prices.\n"
            "When asked to fetch prices, call the fetch_stock_prices tool.\n"
            "Default to 5 years unless the user specifies otherwise.\n"
            "If the user names companies, use tickers if provided; otherwise keep defaults.\n"
            "Always confirm where files were written and row counts."
        ),
        tools=[fetch_stock_prices],
        # You can set a model explicitly if you want; otherwise SDK default config applies.
        # model="gpt-5.2"
    )