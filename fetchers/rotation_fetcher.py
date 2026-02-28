"""
rotation_fetcher.py
Pulls Growth vs Value rotation data via XLK/IVE ratio,
plus moving averages for key sector ETFs.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

try:
    import streamlit as st
    def _cache(ttl=3600):
        return st.cache_data(ttl=ttl)
except Exception:
    def _cache(ttl=3600):
        return lambda f: f


ROTATION_TICKERS = ["XLK", "IVE", "XLI", "XLF", "XLE", "XLB"]


@_cache(ttl=3600)
def fetch_rotation_data() -> dict:
    """
    Returns a dict with keys:
      'prices'       -> DataFrame of daily closes (1 year)
      'ratio'        -> Series: XLK/IVE ratio
      'ratio_ma20'   -> 20-day MA of ratio
      'ratio_ma50'   -> 50-day MA of ratio
      'xlk_ma50'     -> 50-day MA of XLK price
      'xlk_ma200'    -> 200-day MA of XLK price
      'ratio_signal' -> str: "Growth Leading" or "Value Leading"
    """
    today = datetime.today()
    start = (today - timedelta(days=400)).strftime("%Y-%m-%d")

    raw = yf.download(
        ROTATION_TICKERS,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].dropna(how="all")

    # Trim to last 252 trading days (~1 year)
    prices = close.tail(252)

    ratio = prices["XLK"] / prices["IVE"]
    ratio.name = "GrowthValueRatio"

    ratio_ma20 = ratio.rolling(20).mean()
    ratio_ma50 = ratio.rolling(50).mean()

    xlk_ma50 = prices["XLK"].rolling(50).mean()
    xlk_ma200 = prices["XLK"].rolling(200).mean()

    # Signal: compare current ratio to its 20-day MA
    current_ratio = float(ratio.iloc[-1])
    ma20_current = float(ratio_ma20.iloc[-1])
    signal = "Growth Leading" if current_ratio > ma20_current else "Value Leading / Rotation"

    return {
        "prices": prices,
        "ratio": ratio,
        "ratio_ma20": ratio_ma20,
        "ratio_ma50": ratio_ma50,
        "xlk_ma50": xlk_ma50,
        "xlk_ma200": xlk_ma200,
        "ratio_signal": signal,
        "current_ratio": round(current_ratio, 4),
        "xlk_price": round(float(prices["XLK"].iloc[-1]), 2),
        "xlk_ma200_level": round(float(xlk_ma200.iloc[-1]), 2),
    }


if __name__ == "__main__":
    data = fetch_rotation_data()
    print("Signal:", data["ratio_signal"])
    print("Current Ratio (XLK/IVE):", data["current_ratio"])
    print("XLK Price:", data["xlk_price"])
    print("XLK 200MA:", data["xlk_ma200_level"])
    print("\nRatio tail:")
    print(data["ratio"].tail(10).to_string())
    print("\nPrices tail:")
    print(data["prices"].tail(5).to_string())
