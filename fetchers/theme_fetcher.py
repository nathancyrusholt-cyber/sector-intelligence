"""
theme_fetcher.py
Fetches performance, valuation, and positioning data for
four structural macro themes.
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


THEMES = {
    "AI Enablers": {
        "description": "Semiconductors & AI hardware infrastructure",
        "tickers": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "MU"],
    },
    "AI Adopters": {
        "description": "Software companies with AI-driven margin expansion",
        "tickers": ["MSFT", "CRM", "NOW", "PANW", "ADBE", "ORCL", "SAP"],
    },
    "Nearshoring": {
        "description": "Mexico beneficiaries & supply-chain reshoring plays",
        "tickers": ["FUNO11.MX", "VESTA.MX", "EWW", "IYT", "GXC"],
    },
    "Capex Revival": {
        "description": "Industrials & electrification capex cycle",
        "tickers": ["CAT", "DE", "ETN", "PWR", "HUBB", "GEV", "VRT"],
    },
}

# Company display names
COMPANY_NAMES = {
    "NVDA": "NVIDIA", "AMD": "AMD", "AVGO": "Broadcom", "TSM": "TSMC",
    "ASML": "ASML", "AMAT": "Applied Materials", "LRCX": "Lam Research", "MU": "Micron",
    "MSFT": "Microsoft", "CRM": "Salesforce", "NOW": "ServiceNow", "PANW": "Palo Alto",
    "ADBE": "Adobe", "ORCL": "Oracle", "SAP": "SAP",
    "FUNO11.MX": "Fibra Uno", "VESTA.MX": "Vesta", "EWW": "iShares Mexico ETF",
    "IYT": "iShares Transport", "GXC": "SPDR China ETF",
    "CAT": "Caterpillar", "DE": "Deere & Co", "ETN": "Eaton", "PWR": "Quanta Services",
    "HUBB": "Hubbell", "GEV": "GE Vernova", "VRT": "Vertiv",
}


def _pct(a, b):
    try:
        return round((float(a) / float(b) - 1) * 100, 2)
    except Exception:
        return float("nan")


@_cache(ttl=3600)
def fetch_theme_data() -> dict:
    """
    Returns a dict keyed by theme name, each value is a DataFrame with:
      Ticker, Company, YTD%, 3M%, vs 52W High%, vs 52W Low%, Market Cap, Fwd P/E
    Also returns:
      'price_history' -> dict of theme_name -> normalized price DataFrame
    """
    today = datetime.today()
    year_start = datetime(today.year, 1, 1)
    start_fetch = (today - timedelta(days=400)).strftime("%Y-%m-%d")

    all_tickers = list({t for theme in THEMES.values() for t in theme["tickers"]})

    raw = yf.download(
        all_tickers,
        start=start_fetch,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"]

    # Get fundamental data per ticker via yf.Ticker
    fundamentals = {}
    for ticker in all_tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                "market_cap": info.get("marketCap"),
                "fwd_pe": info.get("forwardPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            fundamentals[ticker] = {}

    theme_dfs = {}
    price_histories = {}

    for theme_name, theme_info in THEMES.items():
        tickers = theme_info["tickers"]
        rows = []

        for ticker in tickers:
            try:
                prices = close[ticker].dropna() if ticker in close.columns else pd.Series(dtype=float)
                if prices.empty:
                    continue

                current = float(prices.iloc[-1])

                ytd = prices[prices.index >= pd.Timestamp(year_start)]
                ytd_pct = _pct(ytd.iloc[-1], ytd.iloc[0]) if len(ytd) >= 2 else float("nan")

                cut_3m = pd.Timestamp(today - timedelta(days=91))
                p3m = prices[prices.index >= cut_3m]
                pct_3m = _pct(p3m.iloc[-1], p3m.iloc[0]) if len(p3m) >= 2 else float("nan")

                fund = fundamentals.get(ticker, {})
                w52h = fund.get("week52_high") or float(prices.tail(252).max())
                w52l = fund.get("week52_low") or float(prices.tail(252).min())

                vs_52h = _pct(current, w52h)
                vs_52l = _pct(current, w52l)

                mkt_cap = fund.get("market_cap")
                if mkt_cap:
                    mkt_cap_str = f"${mkt_cap / 1e9:.1f}B" if mkt_cap < 1e12 else f"${mkt_cap / 1e12:.2f}T"
                else:
                    mkt_cap_str = "N/A"

                fwd_pe = fund.get("fwd_pe")
                fwd_pe_str = f"{fwd_pe:.1f}x" if fwd_pe else "N/A"

                rows.append({
                    "Ticker":       ticker,
                    "Company":      COMPANY_NAMES.get(ticker, ticker),
                    "YTD %":        ytd_pct,
                    "3M %":         pct_3m,
                    "vs 52W High":  vs_52h,
                    "vs 52W Low":   vs_52l,
                    "Market Cap":   mkt_cap_str,
                    "Fwd P/E":      fwd_pe_str,
                })
            except Exception:
                continue

        theme_dfs[theme_name] = pd.DataFrame(rows)

        # Normalized price history (rebased to 100 at year start)
        valid_tickers = [t for t in tickers if t in close.columns]
        if valid_tickers:
            hist = close[valid_tickers].copy()
            ytd_hist = hist[hist.index >= pd.Timestamp(year_start)].dropna(how="all")
            if not ytd_hist.empty:
                base = ytd_hist.iloc[0]
                price_histories[theme_name] = (ytd_hist / base * 100).round(2)

    return {
        "themes": theme_dfs,
        "price_histories": price_histories,
    }


@_cache(ttl=3600)
def fetch_spy_normalized() -> pd.Series:
    """SPY price history normalized to 100 at year start, for benchmark overlay."""
    today = datetime.today()
    year_start = datetime(today.year, 1, 1)
    spy = yf.download("SPY", start=year_start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)["Close"]["SPY"]
    base = float(spy.iloc[0])
    return (spy / base * 100).round(2)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    data = fetch_theme_data()
    for theme_name, df in data["themes"].items():
        print(f"\n{'='*60}")
        print(f"  {theme_name}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
