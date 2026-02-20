
import pandas as pd
import yfinance as yf

start = "2021-01-01"
end = "2025-12-31"

tickers = ["META", "GOOGL", "MSFT", "AAPL","NVDA", "AMZN", "ASML","AVGO"]


def compute_top20(ticker):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
        actions=False,
    )

    # If MultiIndex columns exist, flatten to first level (Open/High/Low/Close/...)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    # If Close is still a DataFrame (multiple columns), pick the first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = pd.DataFrame({"Close": close})
    out["Previous_Close"] = out["Close"].shift(1)

    # Percent change (algebraic, can be + or -)
    out["Percent_Change"] = (out["Close"] / out["Previous_Close"] - 1.0) * 100.0
    out = out.dropna()

    # Top 20 by algebraic move descending (largest positive moves)
    out = out.reset_index()[["Date", "Previous_Close", "Close", "Percent_Change"]]
    out = out.sort_values("Percent_Change", ascending=False).head(20)

    out.to_csv(f"{ticker}_top20_daily_moves_2021_2025.csv", index=False)

for t in tickers:
    compute_top20(t)

print("Done.")

