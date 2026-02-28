"""
breadth_fetcher.py
Calculates market and sector breadth from S&P 500 constituents,
breadth thrust detection, and rolling sector correlations.

NOTE: The S&P 500 breadth fetch is network-intensive (~500 tickers).
      It is cached for 4 hours to avoid repeated slow fetches.
"""

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    def _cache(ttl):
        return st.cache_data(ttl=ttl)
except Exception:
    def _cache(ttl):
        return lambda f: f

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLV", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLY", "XLC"]

SECTOR_ETF_NAMES = {
    "XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
    "XLV": "Healthcare", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLY": "Consumer Discretionary", "XLC": "Communications",
}


def _get_sp500_constituents() -> pd.DataFrame:
    """Fetch S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0][["Symbol", "Security", "GICS Sector"]]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


@_cache(ttl=14400)  # 4-hour TTL — intensive fetch
def fetch_breadth_data() -> dict:
    """
    Returns dict with:
      'breadth_50'         -> float: % of S&P 500 above 50MA
      'breadth_200'        -> float: % of S&P 500 above 200MA
      'sector_breadth'     -> DataFrame: per-sector breadth scores
      'breadth_thrust'     -> DataFrame: std devs from 60-day mean per sector ETF
      'corr_matrix'        -> DataFrame: 60-day correlation matrix of 11 sector ETFs
      'high_corr_pairs'    -> list of (etf1, etf2, corr) for pairs > 0.85
    """
    # ── 1. S&P 500 constituents ──────────────────────────────────────────────
    constituents = _get_sp500_constituents()
    tickers = constituents["Symbol"].tolist()

    # Download 260 days to compute 200MA + 60-day breadth history
    raw = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

    # Drop tickers with insufficient data
    close = close.dropna(axis=1, thresh=200)

    # ── 2. Moving averages (vectorised) ──────────────────────────────────────
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    above_50 = (close > ma50).astype(int)
    above_200 = (close > ma200).astype(int)

    # Daily breadth series (all S&P 500)
    breadth_50_series = above_50.mean(axis=1) * 100
    breadth_200_series = above_200.mean(axis=1) * 100

    current_breadth_50 = float(breadth_50_series.iloc[-1])
    current_breadth_200 = float(breadth_200_series.iloc[-1])

    # ── 3. Sector breadth ────────────────────────────────────────────────────
    sector_rows = []
    for gics_sector in constituents["GICS Sector"].unique():
        sector_tickers = constituents.loc[
            constituents["GICS Sector"] == gics_sector, "Symbol"
        ].tolist()
        valid = [t for t in sector_tickers if t in close.columns]
        if not valid:
            continue
        sec_above_50 = above_50[valid].mean(axis=1) * 100
        sec_above_200 = above_200[valid].mean(axis=1) * 100
        sector_rows.append({
            "GICS Sector": gics_sector,
            "Stock Count": len(valid),
            "% Above 50MA": round(float(sec_above_50.iloc[-1]), 1),
            "% Above 200MA": round(float(sec_above_200.iloc[-1]), 1),
        })
    sector_breadth_df = pd.DataFrame(sector_rows).sort_values("% Above 200MA", ascending=False)

    # ── 4. Breadth thrust detection (sector ETFs) ────────────────────────────
    # For each sector ETF: compute daily % of its "peer" stocks above 200MA,
    # then check how far current reading is from 60-day mean.
    # We approximate using the all-market breadth_200_series rolling stats.
    last_60 = breadth_200_series.iloc[-60:]
    mean_60 = float(last_60.mean())
    std_60 = float(last_60.std())

    thrust_rows = []
    for gics_sector in constituents["GICS Sector"].unique():
        sector_tickers = constituents.loc[
            constituents["GICS Sector"] == gics_sector, "Symbol"
        ].tolist()
        valid = [t for t in sector_tickers if t in close.columns]
        if not valid:
            continue
        sec_breadth = (above_200[valid].mean(axis=1) * 100).iloc[-60:]
        s_mean = float(sec_breadth.mean())
        s_std = float(sec_breadth.std()) if float(sec_breadth.std()) > 0 else 1.0
        current = float(sec_breadth.iloc[-1])
        z_score = (current - s_mean) / s_std
        thrust_rows.append({
            "GICS Sector": gics_sector,
            "Current Breadth %": round(current, 1),
            "60D Mean %": round(s_mean, 1),
            "60D Std": round(s_std, 1),
            "Z-Score": round(z_score, 2),
            "Signal": (
                "⚠️ Overbought" if z_score >= 2
                else "🟢 Oversold Opportunity" if z_score <= -2
                else "Normal"
            ),
        })
    breadth_thrust_df = pd.DataFrame(thrust_rows).sort_values("Z-Score", ascending=False)

    # ── 5. 60-day correlation matrix (sector ETFs) ───────────────────────────
    etf_raw = yf.download(
        SECTOR_ETFS,
        period="6mo",
        auto_adjust=True,
        progress=False,
    )
    etf_close = etf_raw["Close"].dropna(how="all")
    etf_returns = etf_close.pct_change().dropna()
    corr_60d = etf_returns.tail(60).corr().round(3)

    # Find highly-correlated pairs
    high_corr = []
    etfs = corr_60d.columns.tolist()
    for i in range(len(etfs)):
        for j in range(i + 1, len(etfs)):
            c = corr_60d.iloc[i, j]
            if c > 0.85:
                high_corr.append((etfs[i], etfs[j], round(c, 3)))

    return {
        "breadth_50": round(current_breadth_50, 1),
        "breadth_200": round(current_breadth_200, 1),
        "sector_breadth": sector_breadth_df,
        "breadth_thrust": breadth_thrust_df,
        "corr_matrix": corr_60d,
        "high_corr_pairs": high_corr,
        "breadth_50_series": breadth_50_series,
        "breadth_200_series": breadth_200_series,
    }


if __name__ == "__main__":
    print("Fetching S&P 500 breadth data (this may take ~30-60 seconds)...")
    data = fetch_breadth_data()
    print(f"\nMarket Breadth:")
    print(f"  % Above 50MA:  {data['breadth_50']:.1f}%")
    print(f"  % Above 200MA: {data['breadth_200']:.1f}%")
    print(f"\nSector Breadth:")
    print(data["sector_breadth"].to_string(index=False))
    print(f"\nBreadth Thrust Detection:")
    print(data["breadth_thrust"].to_string(index=False))
    print(f"\n60-Day Correlation Matrix:")
    print(data["corr_matrix"].to_string())
    if data["high_corr_pairs"]:
        print(f"\n⚠️ High Correlation Pairs (>0.85):")
        for a, b, c in data["high_corr_pairs"]:
            print(f"  {a} / {b}: {c:.3f}")
