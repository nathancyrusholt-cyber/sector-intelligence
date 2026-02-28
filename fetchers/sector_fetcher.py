"""
sector_fetcher.py
Pulls YTD, 1M, 3M performance, moving averages, RSI, and volume
for all 11 GICS sector ETFs plus SPY benchmark.
"""

import numpy as np
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


SECTORS = {
    "Materials":              "XLB",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Industrials":            "XLI",
    "Technology":             "XLK",
    "Consumer Staples":       "XLP",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Consumer Discretionary": "XLY",
    "Communications":         "XLC",
}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@_cache(ttl=3600)
def fetch_sector_data() -> pd.DataFrame:
    today = datetime.today()
    year_start = datetime(today.year, 1, 1)
    fetch_start = today - timedelta(days=420)  # enough for 200MA + buffer

    all_tickers = list(SECTORS.values()) + ["SPY"]

    raw = yf.download(
        all_tickers,
        start=fetch_start.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"]
    volume = raw["Volume"]

    rows = []
    entries = list(SECTORS.items()) + [("S&P 500", "SPY")]

    for sector_name, ticker in entries:
        try:
            prices = close[ticker].dropna()
            vols = volume[ticker].dropna()

            current_price = float(prices.iloc[-1])

            # YTD
            ytd = prices[prices.index >= pd.Timestamp(year_start)]
            ytd_pct = (ytd.iloc[-1] / ytd.iloc[0] - 1) * 100 if len(ytd) >= 2 else float("nan")

            # 1-month
            cut_1m = pd.Timestamp(today - timedelta(days=30))
            p1m = prices[prices.index >= cut_1m]
            pct_1m = (p1m.iloc[-1] / p1m.iloc[0] - 1) * 100 if len(p1m) >= 2 else float("nan")

            # 3-month
            cut_3m = pd.Timestamp(today - timedelta(days=91))
            p3m = prices[prices.index >= cut_3m]
            pct_3m = (p3m.iloc[-1] / p3m.iloc[0] - 1) * 100 if len(p3m) >= 2 else float("nan")

            ma50 = float(prices.rolling(50).mean().iloc[-1])
            ma200 = float(prices.rolling(200).mean().iloc[-1])
            vs_50ma = (current_price / ma50 - 1) * 100
            vs_200ma = (current_price / ma200 - 1) * 100

            rsi = float(_rsi(prices).iloc[-1])

            # Volume: last session vs trailing 30-session average
            current_vol = float(vols.iloc[-1])
            avg_vol = float(vols.iloc[-31:-1].mean())
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else float("nan")

            rows.append({
                "Sector":     sector_name,
                "Ticker":     ticker,
                "Price":      round(current_price, 2),
                "YTD %":      round(float(ytd_pct), 2),
                "1M %":       round(float(pct_1m), 2),
                "3M %":       round(float(pct_3m), 2),
                "50MA":       round(ma50, 2),
                "200MA":      round(ma200, 2),
                "vs 50MA %":  round(vs_50ma, 2),
                "vs 200MA %": round(vs_200ma, 2),
                "RSI (14)":   round(rsi, 1),
                "Vol Ratio":  round(vol_ratio, 2),
            })
        except Exception as e:
            rows.append({"Sector": sector_name, "Ticker": ticker, "Error": str(e)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.2f}".format)
    df = fetch_sector_data()
    print(df.to_string(index=False))
