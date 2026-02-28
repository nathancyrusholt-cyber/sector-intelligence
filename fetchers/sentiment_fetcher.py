"""
sentiment_fetcher.py
Aggregates market sentiment from NewsAPI and StockTwits.

StockTwits API requires no authentication — completely free.
NewsAPI key is read from the NEWS_API_KEY environment variable.
"""

import os
from datetime import datetime, timedelta, timezone

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import streamlit as st
    def _cache(ttl=3600):
        return st.cache_data(ttl=ttl)
except Exception:
    def _cache(ttl=3600):
        return lambda f: f

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_API_KEY            = os.environ.get("NEWS_API_KEY", "")
STOCKTWITS_ACCESS_TOKEN = os.environ.get("STOCKTWITS_ACCESS_TOKEN", "")

CROWDED_THRESHOLD = 0.6   # composite score above this = crowded trade warning
NEWS_QUERIES      = ["stock market", "S&P 500", "sector rotation", "Fed", "earnings"]

STOCKTWITS_TICKERS = ["SPY", "QQQ", "XLK", "XLE", "XLB", "NVDA", "BTC.X", "ETH.X"]
STOCKTWITS_BASE    = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

# StockTwits note:
# Their API now requires a free access token even for public read-only streams.
# Register at https://api.stocktwits.com/developers/apps/new (free, instant).
# Set STOCKTWITS_ACCESS_TOKEN in your .env file.
STOCKTWITS_AUTH_NOTE = (
    "StockTwits API now requires a free access token. "
    "Register at https://api.stocktwits.com/developers/apps/new "
    "and set STOCKTWITS_ACCESS_TOKEN in your .env file."
)

_vader = SentimentIntensityAnalyzer()


# ── News Sentiment ─────────────────────────────────────────────────────────────

@_cache(ttl=3600)
def fetch_news_sentiment() -> dict:
    """
    Pulls last-24h headlines from NewsAPI and scores with VADER.
    Returns avg score, headline count, best/worst headline, per-query breakdown.
    Falls back to empty result if NEWS_API_KEY is missing.
    """
    if not NEWS_API_KEY:
        return _empty_news_result(reason="NEWS_API_KEY not set")

    results_by_query = {}
    all_scores = []
    all_articles = []

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for query in NEWS_QUERIES:
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={requests.utils.quote(query)}"
            f"&from={since}"
            "&language=en"
            "&sortBy=publishedAt"
            "&pageSize=20"
            f"&apiKey={NEWS_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            results_by_query[query] = {"error": str(e), "avg_score": 0.0, "count": 0}
            continue

        scores = []
        for art in articles:
            text = f"{art.get('title', '')} {art.get('description', '') or ''}"
            score = _vader.polarity_scores(text)["compound"]
            scores.append(score)
            all_scores.append(score)
            all_articles.append({
                "query":     query,
                "headline":  art.get("title", ""),
                "score":     score,
                "url":       art.get("url", ""),
                "published": art.get("publishedAt", ""),
            })

        avg = sum(scores) / len(scores) if scores else 0.0
        results_by_query[query] = {"avg_score": round(avg, 3), "count": len(scores)}

    composite = sum(all_scores) / len(all_scores) if all_scores else 0.0
    best  = max(all_articles, key=lambda x: x["score"], default=None)
    worst = min(all_articles, key=lambda x: x["score"], default=None)

    return {
        "composite_score":      round(composite, 3),
        "total_articles":       len(all_articles),
        "crowded_trade_warning": composite > CROWDED_THRESHOLD,
        "best_headline":        best,
        "worst_headline":       worst,
        "by_query":             results_by_query,
        "all_articles":         all_articles,
        "error":                None,
    }


def _empty_news_result(reason: str) -> dict:
    return {
        "composite_score":      0.0,
        "total_articles":       0,
        "crowded_trade_warning": False,
        "best_headline":        None,
        "worst_headline":       None,
        "by_query":             {},
        "all_articles":         [],
        "error":                reason,
    }


# ── StockTwits Sentiment ───────────────────────────────────────────────────────

@_cache(ttl=3600)
def fetch_stocktwits_sentiment() -> dict:
    """
    Pulls the last 30 messages per ticker from StockTwits.
    Requires a free access token (STOCKTWITS_ACCESS_TOKEN env var).
    Register at https://api.stocktwits.com/developers/apps/new

    Runs VADER on each message body and uses StockTwits' own Bullish/Bearish
    sentiment field where available.

    Returns:
      'ticker_rows'       -> list of per-ticker dicts
      'composite_score'   -> weighted avg sentiment across all tickers
      'total_messages'    -> total messages processed
      'error'             -> None on success, str on total failure
    """
    import pandas as pd

    if not STOCKTWITS_ACCESS_TOKEN:
        return {
            "ticker_rows": [_empty_ticker_row(t, reason="No token") for t in STOCKTWITS_TICKERS],
            "ticker_df": pd.DataFrame([
                {"Ticker": t, "Messages": 0, "Avg Sentiment": 0.0,
                 "Bullish Count": 0, "Bearish Count": 0,
                 "Bullish %": 0.0, "Bearish %": 0.0, "Signal": "N/A"}
                for t in STOCKTWITS_TICKERS
            ]),
            "composite_score": 0.0,
            "total_messages": 0,
            "error": STOCKTWITS_AUTH_NOTE,
        }

    ticker_rows = []
    all_scores  = []

    for ticker in STOCKTWITS_TICKERS:
        url = STOCKTWITS_BASE.format(ticker=ticker)
        params = {"access_token": STOCKTWITS_ACCESS_TOKEN}
        try:
            resp = requests.get(url, params=params, timeout=10,
                                headers={"User-Agent": "sector-intelligence/1.0"})
            if resp.status_code == 401:
                return {
                    "ticker_rows": [], "ticker_df": pd.DataFrame(),
                    "composite_score": 0.0, "total_messages": 0,
                    "error": "StockTwits: Invalid access token (401). Check STOCKTWITS_ACCESS_TOKEN.",
                }
            if resp.status_code == 429:
                ticker_rows.append(_empty_ticker_row(ticker, reason="Rate limited"))
                continue
            resp.raise_for_status()
            messages = resp.json().get("messages", [])
        except Exception as e:
            ticker_rows.append(_empty_ticker_row(ticker, reason=str(e)))
            continue

        scores        = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        recent_bullish = None
        recent_bearish = None

        for msg in messages:
            body  = msg.get("body", "")
            score = _vader.polarity_scores(body)["compound"]
            scores.append(score)
            all_scores.append(score)

            # StockTwits native sentiment field (most reliable signal)
            native = (
                (msg.get("entities") or {})
                .get("sentiment", {}) or {}
            ).get("basic", "")

            if native == "Bullish":
                bullish_count += 1
                if recent_bullish is None:
                    recent_bullish = {"text": body[:120], "score": score}
            elif native == "Bearish":
                bearish_count += 1
                if recent_bearish is None:
                    recent_bearish = {"text": body[:120], "score": score}
            else:
                neutral_count += 1
                # Fall back to VADER classification for un-tagged messages
                if score > 0.05 and recent_bullish is None:
                    recent_bullish = {"text": body[:120], "score": score}
                elif score < -0.05 and recent_bearish is None:
                    recent_bearish = {"text": body[:120], "score": score}

        avg_score    = sum(scores) / len(scores) if scores else 0.0
        total        = bullish_count + bearish_count + neutral_count
        bull_pct     = round(bullish_count / total * 100, 1) if total else 0.0
        bear_pct     = round(bearish_count / total * 100, 1) if total else 0.0

        ticker_rows.append({
            "Ticker":          ticker,
            "Messages":        len(messages),
            "Avg Sentiment":   round(avg_score, 3),
            "Bullish Count":   bullish_count,
            "Bearish Count":   bearish_count,
            "Bullish %":       bull_pct,
            "Bearish %":       bear_pct,
            "Signal":          (
                "Bullish" if avg_score > 0.05 and bullish_count >= bearish_count
                else "Bearish" if avg_score < -0.05 or bearish_count > bullish_count
                else "Neutral"
            ),
            "Recent Bullish":  recent_bullish,
            "Recent Bearish":  recent_bearish,
            "error":           None,
        })

    composite = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "ticker_rows":     ticker_rows,
        "ticker_df":       pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("Recent Bullish", "Recent Bearish", "error")}
            for r in ticker_rows
        ]),
        "composite_score": round(composite, 3),
        "total_messages":  len(all_scores),
        "error":           None if ticker_rows else "No data returned",
    }


def _empty_ticker_row(ticker: str, reason: str) -> dict:
    return {
        "Ticker": ticker, "Messages": 0, "Avg Sentiment": 0.0,
        "Bullish Count": 0, "Bearish Count": 0,
        "Bullish %": 0.0, "Bearish %": 0.0, "Signal": "N/A",
        "Recent Bullish": None, "Recent Bearish": None, "error": reason,
    }


# ── Composite Score ────────────────────────────────────────────────────────────

def compute_composite_sentiment(news_result: dict, stocktwits_result: dict) -> float:
    """Blend news (60%) and StockTwits (40%) into a single −1 to +1 score."""
    n = news_result.get("composite_score", 0.0)
    s = stocktwits_result.get("composite_score", 0.0)
    if news_result.get("error") and not stocktwits_result.get("error"):
        return s
    if stocktwits_result.get("error") and not news_result.get("error"):
        return n
    return round(0.6 * n + 0.4 * s, 3)


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("=== News Sentiment ===")
    news = fetch_news_sentiment()
    print(f"Composite: {news['composite_score']}")
    print(f"Articles:  {news['total_articles']}")
    if news["best_headline"]:
        print(f"Best:  {news['best_headline']['headline'][:80]}")
    if news["worst_headline"]:
        print(f"Worst: {news['worst_headline']['headline'][:80]}")
    if news["error"]:
        print(f"Error: {news['error']}")

    print("\n=== StockTwits Sentiment ===")
    st_data = fetch_stocktwits_sentiment()
    print(f"Composite: {st_data['composite_score']}")
    print(f"Messages:  {st_data['total_messages']}")
    print(st_data["ticker_df"].to_string(index=False))
