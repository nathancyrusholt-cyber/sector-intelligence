"""
sentiment_fetcher.py
Aggregates market sentiment from NewsAPI and Reddit (PRAW).
All API keys are read from environment variables.

X/Twitter note: The X API v2 now requires a paid tier ($100+/month).
  Use Reddit (free) or StockTwits (free) as alternatives.
  A mock function is provided for the X panel.
"""

import os
import re
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

# ── Config (from environment variables) ──────────────────────────────────────
NEWS_API_KEY         = os.environ.get("NEWS_API_KEY", "")
REDDIT_CLIENT_ID     = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = "sector-intelligence/1.0"

_vader = SentimentIntensityAnalyzer()

CROWDED_THRESHOLD = 0.6   # composite score above this = crowded trade warning
NEWS_QUERIES = ["stock market", "S&P 500", "sector rotation", "Fed", "earnings"]


# ── News Sentiment ────────────────────────────────────────────────────────────

@_cache(ttl=3600)
def fetch_news_sentiment() -> dict:
    """
    Pulls last-24h headlines from NewsAPI and scores with VADER.
    Returns avg score, headline count, best/worst headline, per-query breakdown.
    Falls back to empty result if API key is missing.
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
                "query": query,
                "headline": art.get("title", ""),
                "score": score,
                "url": art.get("url", ""),
                "published": art.get("publishedAt", ""),
            })

        avg = sum(scores) / len(scores) if scores else 0.0
        results_by_query[query] = {"avg_score": round(avg, 3), "count": len(scores)}

    composite = sum(all_scores) / len(all_scores) if all_scores else 0.0
    best = max(all_articles, key=lambda x: x["score"], default=None)
    worst = min(all_articles, key=lambda x: x["score"], default=None)

    return {
        "composite_score": round(composite, 3),
        "total_articles": len(all_articles),
        "crowded_trade_warning": composite > CROWDED_THRESHOLD,
        "best_headline": best,
        "worst_headline": worst,
        "by_query": results_by_query,
        "all_articles": all_articles,
        "error": None,
    }


def _empty_news_result(reason: str) -> dict:
    return {
        "composite_score": 0.0,
        "total_articles": 0,
        "crowded_trade_warning": False,
        "best_headline": None,
        "worst_headline": None,
        "by_query": {},
        "all_articles": [],
        "error": reason,
    }


# ── Reddit Sentiment ──────────────────────────────────────────────────────────

TICKER_PATTERN = re.compile(r"\b\$?([A-Z]{2,5})\b")
SUBREDDITS = ["wallstreetbets", "investing", "stocks"]
# Common English words to exclude from ticker detection
_STOP_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "ITS", "LET", "PUT",
    "SAY", "SHE", "TOO", "USE", "USA", "USD", "NYSE", "SEC", "IPO", "ETF",
    "YTD", "CEO", "CFO", "CTO", "IMO", "IMHO", "TLDR", "WSB", "DD", "FUD",
    "ATH", "ATL", "YOLO", "HODL", "BUY", "SELL", "HOLD", "LONG", "SHORT",
}


@_cache(ttl=3600)
def fetch_reddit_sentiment() -> dict:
    """
    Pulls top-50 posts from r/wallstreetbets, r/investing, r/stocks.
    Scores each post with VADER, extracts ticker mentions.
    Falls back to empty result if PRAW credentials are missing.
    """
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        return _empty_reddit_result(reason="REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")

    try:
        import praw
    except ImportError:
        return _empty_reddit_result(reason="praw not installed")

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    ticker_data: dict[str, dict] = {}
    post_scores = []

    for sub_name in SUBREDDITS:
        try:
            sub = reddit.subreddit(sub_name)
            posts = list(sub.top(time_filter="day", limit=50))
        except Exception:
            continue

        for post in posts:
            text = f"{post.title} {post.selftext[:500] if post.selftext else ''}"
            compound = _vader.polarity_scores(text)["compound"]
            post_scores.append(compound)

            mentions = set(TICKER_PATTERN.findall(text.upper())) - _STOP_WORDS
            for ticker in mentions:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "ticker": ticker,
                        "mention_count": 0,
                        "scores": [],
                        "upvotes": 0,
                    }
                ticker_data[ticker]["mention_count"] += 1
                ticker_data[ticker]["scores"].append(compound)
                ticker_data[ticker]["upvotes"] += post.score

    # Aggregate per ticker
    ticker_rows = []
    for td in ticker_data.values():
        avg_score = sum(td["scores"]) / len(td["scores"]) if td["scores"] else 0.0
        ticker_rows.append({
            "Ticker": td["ticker"],
            "Mentions": td["mention_count"],
            "Avg Sentiment": round(avg_score, 3),
            "Total Upvotes": td["upvotes"],
            "Signal": "Bullish" if avg_score > 0.05 else "Bearish" if avg_score < -0.05 else "Neutral",
        })

    import pandas as pd
    ticker_df = pd.DataFrame(ticker_rows).sort_values("Mentions", ascending=False).head(20)

    composite = sum(post_scores) / len(post_scores) if post_scores else 0.0

    return {
        "composite_score": round(composite, 3),
        "total_posts": len(post_scores),
        "ticker_df": ticker_df,
        "error": None,
    }


def _empty_reddit_result(reason: str) -> dict:
    import pandas as pd
    return {
        "composite_score": 0.0,
        "total_posts": 0,
        "ticker_df": pd.DataFrame(),
        "error": reason,
    }


# ── X / Twitter Placeholder ───────────────────────────────────────────────────
# X API v2 Basic tier now costs $100/month (as of 2023).
# StockTwits offers a free API alternative: https://api.stocktwits.com/api/2/
# The function below returns mock data as an example of what would be shown.

def fetch_twitter_mock() -> dict:
    """
    PLACEHOLDER — X (Twitter) API requires a paid tier ($100+/month).
    Returns mock data to demonstrate the UI layout.

    FREE ALTERNATIVE: Use the StockTwits API instead:
      GET https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
      No authentication required for public streams.
    """
    return {
        "is_mock": True,
        "note": (
            "X/Twitter API now requires a paid Basic tier ($100/month). "
            "Consider StockTwits (free) as an alternative — see fetch_stocktwits()."
        ),
        "mock_data": [
            {"ticker": "NVDA", "sentiment": 0.72, "mentions": 1420, "signal": "Bullish"},
            {"ticker": "SPY",  "sentiment": 0.31, "mentions": 980,  "signal": "Bullish"},
            {"ticker": "TSLA", "sentiment": -0.18,"mentions": 870,  "signal": "Bearish"},
            {"ticker": "AAPL", "sentiment": 0.45, "mentions": 650,  "signal": "Bullish"},
            {"ticker": "META", "sentiment": 0.29, "mentions": 510,  "signal": "Bullish"},
        ],
    }


def fetch_stocktwits_placeholder(symbol: str = "SPY") -> dict:
    """
    Placeholder for StockTwits free API integration.
    Endpoint: GET https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
    No auth required. Returns bullish/bearish ratio from watchers.
    """
    return {
        "is_placeholder": True,
        "note": f"StockTwits data for {symbol} — implement with requests.get(url).json()",
        "endpoint": f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
    }


# ── Composite Score ───────────────────────────────────────────────────────────

def compute_composite_sentiment(news_result: dict, reddit_result: dict) -> float:
    """Blend news (60%) and Reddit (40%) into a single -1 to +1 score."""
    n = news_result.get("composite_score", 0.0)
    r = reddit_result.get("composite_score", 0.0)
    # If one source is unavailable, use the other fully
    if news_result.get("error") and not reddit_result.get("error"):
        return r
    if reddit_result.get("error") and not news_result.get("error"):
        return n
    return round(0.6 * n + 0.4 * r, 3)


if __name__ == "__main__":
    news = fetch_news_sentiment()
    print("=== News Sentiment ===")
    print(f"Composite: {news['composite_score']}")
    print(f"Articles:  {news['total_articles']}")
    if news["best_headline"]:
        print(f"Best:  {news['best_headline']['headline'][:80]}")
    if news["worst_headline"]:
        print(f"Worst: {news['worst_headline']['headline'][:80]}")
    if news["error"]:
        print(f"Error: {news['error']}")

    reddit = fetch_reddit_sentiment()
    print("\n=== Reddit Sentiment ===")
    print(f"Composite: {reddit['composite_score']}")
    print(f"Posts:     {reddit['total_posts']}")
    if not reddit["ticker_df"].empty:
        print(reddit["ticker_df"].head(10).to_string(index=False))
    if reddit["error"]:
        print(f"Error: {reddit['error']}")

    print("\n=== Twitter (Mock) ===")
    tw = fetch_twitter_mock()
    print(tw["note"])
