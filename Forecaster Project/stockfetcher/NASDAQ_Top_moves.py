import pandas as pd
import yfinance as yf

start = "2021-01-01"
end = "2025-12-31"

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

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    # Force Close into Series if it's still a DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = pd.DataFrame({"Close": close})
    out["Previous_Close"] = out["Close"].shift(1)
    out["Percent_Change"] = (out["Close"] / out["Previous_Close"] - 1.0) * 100.0
    out = out.dropna()
    out = out.reset_index()[["Date", "Previous_Close", "Close", "Percent_Change"]]

    # Top 20 Positive Moves
    top_pos = out.sort_values("Percent_Change", ascending=False).head(20)
    top_pos["Move_Type"] = "Positive"

    # Top 20 Negative Moves
    top_neg = out.sort_values("Percent_Change", ascending=True).head(20)
    top_neg["Move_Type"] = "Negative"

    # Combine
    final = pd.concat([top_pos, top_neg])

    final.to_csv(f"{ticker}_top20_up_down_daily_moves_2021_2025.csv", index=False)

for t in ["META", "GOOGL", "MSFT", "AAPL","AMZN", "ASML", "AVGO"]:
    compute_top20(t)

print("CSV files generated.")