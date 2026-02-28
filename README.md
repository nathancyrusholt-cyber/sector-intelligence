# 📊 Market Sector Intelligence

A real-time Streamlit dashboard for sector rotation, market breadth, sentiment analysis, and thematic macro research — powered by live market data and Claude AI.

---

## Features

| Tab | What it shows |
|-----|---------------|
| 🗺️ Sector Overview | Heatmap of all 11 GICS sectors + sortable metrics table |
| 🔄 Rotation & Breadth | Growth/Value ratio, sector radar, 60-day correlation matrix, S&P 500 breadth |
| 💬 Sentiment | VADER-scored news + Reddit bubble map + X/Twitter placeholder |
| 🎯 Theme Drill-Downs | AI Enablers, AI Adopters, Nearshoring, Capex Revival |
| 🤖 AI Briefing | Claude-powered market regime analysis (sidebar) |

---

## Quick Start

```bash
cd sector-intelligence
pip install -r requirements.txt
cp .env.example .env          # fill in your API keys
streamlit run app.py
```

---

## API Keys Setup

All keys are **optional** — the app degrades gracefully when they are missing. `yfinance` requires no key.

### 1. Anthropic API Key (AI Briefing)

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account → API Keys → Create Key
3. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

**Cost:** Pay-per-use. A briefing uses ~1,000 tokens ≈ $0.003.

---

### 2. NewsAPI (News Sentiment)

1. Go to [newsapi.org](https://newsapi.org) → Get API Key (free)
2. Free tier: **100 requests/day**, English headlines, last 30 days
3. Add to `.env`: `NEWS_API_KEY=your_key`

**Free tier limit:** 100 req/day is sufficient for ~20 dashboard loads.

---

### 3. Reddit / PRAW (Reddit Sentiment)

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"Create another app"**
3. Choose type: **script**
4. Name: `sector-intelligence` (anything)
5. Redirect URI: `http://localhost:8080` (required but unused)
6. Copy the **client ID** (under the app name) and **client secret**
7. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   ```

**Free tier:** Reddit PRAW is free for read-only access. Rate limit: 100 req/min.

---

### 4. X / Twitter (Not Recommended)

X API now requires a **paid Basic tier at $100/month**. The dashboard shows mock data instead.

**Free alternative:** [StockTwits API](https://api.stocktwits.com/api/2/streams/symbol/SPY.json) — no auth needed for public streams. A placeholder function `fetch_stocktwits_placeholder()` is included in `fetchers/sentiment_fetcher.py`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Load `.env` before running:

```bash
# Option A: use python-dotenv (add to top of app.py if needed)
pip install python-dotenv

# Option B: export manually (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run app.py

# Option C: export manually (bash)
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push repo to GitHub (ensure `.env` and `secrets.toml` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → Deploy
3. In the app settings → **Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real keys

---

## Project Structure

```
sector-intelligence/
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── .streamlit/
│   ├── config.toml               # Dark theme + teal primary colour
│   └── secrets.toml.example      # Streamlit Cloud secrets template
└── fetchers/
    ├── sector_fetcher.py         # 11 GICS sector ETF metrics
    ├── breadth_fetcher.py        # S&P 500 breadth + correlations
    ├── rotation_fetcher.py       # Growth vs Value rotation
    ├── sentiment_fetcher.py      # NewsAPI + Reddit VADER sentiment
    └── theme_fetcher.py          # AI / Nearshoring / Capex themes
```

---

## Performance Notes

- **S&P 500 breadth** (`breadth_fetcher.py`) downloads ~500 tickers and is cached for **4 hours**. First load takes 30–60 seconds — subsequent loads are instant.
- All other fetchers cache for **1 hour**.
- Use the **🔄 Refresh All Data** button in the sidebar to force a reload.

---

## Data Sources

| Data | Source | Refresh |
|------|--------|---------|
| Price / volume / fundamentals | yfinance (Yahoo Finance) | 1h cache |
| S&P 500 constituent list | Wikipedia | 4h cache |
| News headlines | NewsAPI | 1h cache |
| Reddit posts | PRAW (Reddit API) | 1h cache |
| AI Briefing | Claude claude-sonnet-4-6 | 2h cache |
