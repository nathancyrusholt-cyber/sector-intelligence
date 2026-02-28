"""
Market Sector Intelligence — app.py
Full Streamlit dashboard: sector heatmap, rotation, breadth,
sentiment, theme drill-downs, and AI briefing via Claude.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sector Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (lazy, with error feedback) ──────────────────────────────────────
from fetchers.sector_fetcher import fetch_sector_data, SECTORS
from fetchers.rotation_fetcher import fetch_rotation_data
from fetchers.breadth_fetcher import fetch_breadth_data, SECTOR_ETF_NAMES
from fetchers.sentiment_fetcher import (
    fetch_news_sentiment, fetch_reddit_sentiment,
    fetch_twitter_mock, compute_composite_sentiment,
)
from fetchers.theme_fetcher import fetch_theme_data, fetch_spy_normalized, THEMES

# ── Session state defaults ────────────────────────────────────────────────────
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = "All"
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "AI Enablers"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "ai_briefing" not in st.session_state:
    st.session_state.ai_briefing = None
if "ai_briefing_time" not in st.session_state:
    st.session_state.ai_briefing_time = None

# ── Color helpers ─────────────────────────────────────────────────────────────
TEAL   = "#00D4AA"
RED    = "#FF4B4B"
GREEN  = "#00C853"
YELLOW = "#FFD600"
GRAY   = "#9E9E9E"
BG     = "#0E1117"
CARD   = "#1E2530"


def pct_color(v):
    if pd.isna(v):
        return GRAY
    return GREEN if v >= 0 else RED


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### 📊 Market Intelligence")
    st.markdown("---")

    # Data source status
    st.markdown("**Data Sources**")
    news_key_ok = bool(os.environ.get("NEWS_API_KEY"))
    reddit_ok   = bool(os.environ.get("REDDIT_CLIENT_ID"))
    anthropic_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def status_dot(ok, label):
        dot = "🟢" if ok else "🔴"
        note = "" if ok else " *(key missing)*"
        st.markdown(f"{dot} {label}{note}")

    status_dot(True,         "yfinance (free)")
    status_dot(news_key_ok,  "NewsAPI")
    status_dot(reddit_ok,    "Reddit PRAW")
    status_dot(anthropic_ok, "Claude AI")
    st.markdown("---")

    # Refresh controls
    st.markdown("**Refresh**")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

    auto_refresh = st.toggle("Auto-refresh (15 min)", value=False)
    if auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        remaining = max(0, 900 - int(elapsed))
        st.caption(f"Next refresh in {remaining // 60}m {remaining % 60}s")
        if elapsed >= 900:
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

    st.markdown("---")

    # CSV export
    st.markdown("**Export**")
    if st.button("📥 Export Sector Data as CSV", use_container_width=True):
        try:
            df = fetch_sector_data()
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sector_data_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.markdown("---")

    # AI Briefing panel
    st.markdown("**🤖 AI Briefing**")
    briefing_disabled = not anthropic_ok
    if briefing_disabled:
        st.caption("Set ANTHROPIC_API_KEY to enable")

    if st.button("✨ Get AI Briefing", use_container_width=True, disabled=briefing_disabled):
        st.session_state.ai_briefing = None
        st.session_state.ai_briefing_time = None
        st.session_state.ai_briefing_requested = True

    if st.session_state.ai_briefing:
        age_mins = int((time.time() - st.session_state.ai_briefing_time) / 60)
        st.caption(f"Generated {age_mins}m ago · cached 2h")

    st.markdown("---")
    st.caption(f"Last data pull: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')}")


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING WITH PROGRESS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_all_fast_data():
    """Load all non-intensive data sources."""
    sector_df   = fetch_sector_data()
    rotation    = fetch_rotation_data()
    news        = fetch_news_sentiment()
    reddit      = fetch_reddit_sentiment()
    themes      = fetch_theme_data()
    return sector_df, rotation, news, reddit, themes


def load_data_with_progress():
    if "data_loaded" not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("### Loading market data...")
            prog = st.progress(0)
            status = st.empty()

            status.text("📈 Fetching sector ETF data...")
            prog.progress(15)
            sector_df = fetch_sector_data()

            status.text("🔄 Fetching rotation data...")
            prog.progress(30)
            rotation = fetch_rotation_data()

            status.text("💬 Fetching news sentiment...")
            prog.progress(45)
            news = fetch_news_sentiment()

            status.text("📱 Fetching Reddit sentiment...")
            prog.progress(60)
            reddit = fetch_reddit_sentiment()

            status.text("🎯 Fetching theme data...")
            prog.progress(80)
            themes = fetch_theme_data()

            prog.progress(100)
            status.text("✅ Data loaded!")
            time.sleep(0.5)

        placeholder.empty()
        st.session_state.data_loaded = True
        return sector_df, rotation, news, reddit, themes
    else:
        return load_all_fast_data()


sector_df, rotation_data, news_data, reddit_data, theme_data = load_data_with_progress()

spy_row = sector_df[sector_df["Ticker"] == "SPY"].iloc[0] if "SPY" in sector_df["Ticker"].values else {}
sectors_only = sector_df[sector_df["Ticker"] != "SPY"].copy()
composite_sentiment = compute_composite_sentiment(news_data, reddit_data)


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <h1 style='color:{TEAL}; margin-bottom:0'>📊 Market Sector Intelligence</h1>
    <p style='color:{GRAY}; margin-top:4px'>
        Real-time sector rotation, breadth, sentiment & thematic analysis
        &nbsp;·&nbsp; Updated {datetime.today().strftime('%b %d, %Y %H:%M')}
    </p>
    """,
    unsafe_allow_html=True,
)

# Banner stats
try:
    import yfinance as yf
    vix_data = yf.download("^VIX", period="2d", progress=False)
    vix_level = round(float(vix_data["Close"].iloc[-1]), 2)
except Exception:
    vix_level = None

spy_ytd = float(spy_row.get("YTD %", 0)) if spy_row is not None and len(spy_row) else 0.0
gv_ratio = rotation_data.get("current_ratio", "N/A")
gv_signal = rotation_data.get("ratio_signal", "N/A")

b1, b2, b3 = st.columns(3)
b1.metric("SPY YTD", f"{spy_ytd:+.2f}%", delta_color="normal")
b2.metric("VIX", f"{vix_level:.2f}" if vix_level else "N/A",
          delta="Fear Elevated" if vix_level and vix_level > 20 else "Calm",
          delta_color="inverse" if vix_level and vix_level > 20 else "normal")
b3.metric("Growth/Value Ratio (XLK/IVE)", f"{gv_ratio}", delta=gv_signal,
          delta_color="normal" if "Growth" in str(gv_signal) else "inverse")

st.markdown("---")

# AI Briefing panel (if triggered)
if st.session_state.ai_briefing is None and anthropic_ok:
    pass  # Will be generated on button press (handled below)

if st.session_state.get("_run_ai_briefing"):
    st.session_state._run_ai_briefing = False
    _generate_briefing = True
else:
    _generate_briefing = False


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Sector Overview",
    "🔄 Rotation & Breadth",
    "💬 Sentiment",
    "🎯 Theme Drill-Downs",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SECTOR OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Sector Heatmap")

    # Build heatmap grid: 4 cols × 3 rows
    GRID = [
        ["XLK", "XLF", "XLV", "XLY"],
        ["XLI", "XLE", "XLB", "XLC"],
        ["XLP", "XLRE", "XLU", ""],
    ]

    sector_lookup = sectors_only.set_index("Ticker")

    def _sector_fig():
        z_vals, text_vals, custom_vals = [], [], []
        for row in GRID:
            z_row, t_row, c_row = [], [], []
            for ticker in row:
                if ticker == "" or ticker not in sector_lookup.index:
                    z_row.append(None)
                    t_row.append("")
                    c_row.append(["", "", ""])
                else:
                    r = sector_lookup.loc[ticker]
                    ytd = r["YTD %"] if not pd.isna(r["YTD %"]) else 0
                    z_row.append(ytd)
                    t_row.append(
                        f"<b>{r['Sector']}</b><br>{ticker}<br>"
                        f"YTD: {ytd:+.1f}%<br>1M: {r['1M %']:+.1f}%"
                    )
                    c_row.append([ticker, r["Sector"], ytd])
            z_vals.append(z_row)
            text_vals.append(t_row)
            custom_vals.append(c_row)

        fig = go.Figure(go.Heatmap(
            z=z_vals,
            text=text_vals,
            texttemplate="%{text}",
            customdata=custom_vals,
            colorscale=[[0, "#C62828"], [0.5, "#37474F"], [1, "#1B5E20"]],
            zmid=0,
            showscale=True,
            colorbar=dict(title="YTD %", tickformat="+.1f"),
            hovertemplate="<b>%{customdata[1]}</b> (%{customdata[0]})<br>YTD: %{z:+.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="white", size=11),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        return fig

    heatmap_fig = _sector_fig()
    heatmap_event = st.plotly_chart(
        heatmap_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="sector_heatmap",
    )

    # Handle click-to-filter
    if heatmap_event and heatmap_event.get("selection", {}).get("points"):
        pt = heatmap_event["selection"]["points"][0]
        custom = pt.get("customdata", ["", ""])
        if custom[0]:
            st.session_state.selected_sector = custom[1]  # sector name

    # Sector filter
    col_filter, col_reset = st.columns([3, 1])
    with col_filter:
        all_sector_names = ["All"] + sorted(sectors_only["Sector"].tolist())
        selected = st.selectbox(
            "Filter by sector:",
            all_sector_names,
            index=all_sector_names.index(st.session_state.selected_sector)
            if st.session_state.selected_sector in all_sector_names else 0,
            key="sector_select",
        )
        st.session_state.selected_sector = selected
    with col_reset:
        if st.button("Show All", key="reset_sector"):
            st.session_state.selected_sector = "All"

    # Metrics table
    display_df = (
        sectors_only if st.session_state.selected_sector == "All"
        else sectors_only[sectors_only["Sector"] == st.session_state.selected_sector]
    )

    table_cols = ["Sector", "Ticker", "Price", "YTD %", "1M %", "3M %",
                  "vs 200MA %", "RSI (14)", "Vol Ratio"]
    st.dataframe(
        display_df[table_cols].style
        .format({
            "Price": "${:,.2f}", "YTD %": "{:+.2f}%", "1M %": "{:+.2f}%",
            "3M %": "{:+.2f}%", "vs 200MA %": "{:+.2f}%",
            "RSI (14)": "{:.1f}", "Vol Ratio": "{:.2f}x",
        })
        .applymap(lambda v: f"color: {GREEN}" if isinstance(v, (int, float)) and v > 0
                  else f"color: {RED}" if isinstance(v, (int, float)) and v < 0 else "",
                  subset=["YTD %", "1M %", "3M %", "vs 200MA %"]),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ROTATION & BREADTH
# ─────────────────────────────────────────────────────────────────────────────

with tab2:

    # ── Growth vs Value Rotation Chart ───────────────────────────────────────
    st.subheader("Growth vs Value Rotation (XLK / IVE)")

    ratio        = rotation_data["ratio"]
    ratio_ma20   = rotation_data["ratio_ma20"]
    prices       = rotation_data["prices"]
    xlk_ma50     = rotation_data["xlk_ma50"]
    xlk_ma200    = rotation_data["xlk_ma200"]
    xlk_ma200_lvl = rotation_data["xlk_ma200_level"]

    # Background shading: green when ratio rising, red when falling
    ratio_diff = ratio.diff()
    fig_rot = go.Figure()

    # Shade regions
    prev_i = 0
    dates = ratio.index.tolist()
    for i in range(1, len(ratio_diff)):
        sign_now  = 1 if float(ratio_diff.iloc[i]) >= 0 else -1
        sign_prev = 1 if float(ratio_diff.iloc[i - 1]) >= 0 else -1
        if sign_now != sign_prev or i == len(ratio_diff) - 1:
            color = "rgba(0,200,83,0.07)" if sign_prev == 1 else "rgba(255,75,75,0.07)"
            fig_rot.add_vrect(
                x0=dates[prev_i], x1=dates[i],
                fillcolor=color, layer="below", line_width=0,
            )
            prev_i = i

    fig_rot.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values,
        name="XLK/IVE Ratio", line=dict(color=TEAL, width=2),
        yaxis="y1",
    ))
    fig_rot.add_trace(go.Scatter(
        x=ratio_ma20.index, y=ratio_ma20.values,
        name="20MA Ratio", line=dict(color=YELLOW, width=1, dash="dot"),
        yaxis="y1",
    ))
    fig_rot.add_trace(go.Scatter(
        x=prices.index, y=prices["XLK"].values,
        name="XLK Price", line=dict(color="white", width=1.5),
        yaxis="y2",
    ))
    fig_rot.add_trace(go.Scatter(
        x=xlk_ma50.index, y=xlk_ma50.values,
        name="XLK 50MA", line=dict(color="#29B6F6", width=1, dash="dot"),
        yaxis="y2",
    ))
    fig_rot.add_trace(go.Scatter(
        x=xlk_ma200.index, y=xlk_ma200.values,
        name="XLK 200MA", line=dict(color="#EF5350", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    # 200MA annotation
    fig_rot.add_annotation(
        x=xlk_ma200.index[-1], y=xlk_ma200_lvl,
        text=f"200MA: ${xlk_ma200_lvl:.2f}",
        showarrow=True, arrowhead=2, arrowcolor=RED,
        font=dict(color=RED, size=11), yref="y2",
    )
    fig_rot.update_layout(
        height=420,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.05),
        yaxis=dict(title="XLK/IVE Ratio", showgrid=False),
        yaxis2=dict(title="XLK Price ($)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_rot, use_container_width=True)

    # ── Sector Rotation Wheel ─────────────────────────────────────────────────
    st.subheader("Sector Momentum Radar (3-Month)")

    radar_sectors = ["XLK", "XLF", "XLE", "XLI", "XLV", "XLC"]
    radar_labels  = [SECTOR_ETF_NAMES.get(t, t) for t in radar_sectors]
    radar_vals    = []
    for t in radar_sectors:
        row = sectors_only[sectors_only["Ticker"] == t]
        radar_vals.append(float(row["3M %"].iloc[0]) if not row.empty else 0.0)

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor=f"rgba(0,212,170,0.2)",
        line=dict(color=TEAL, width=2),
        name="3M Momentum %",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, color=GRAY),
            angularaxis=dict(color="white"),
        ),
        paper_bgcolor=BG,
        font=dict(color="white"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Rolling Correlation Heatmap ───────────────────────────────────────────
    st.subheader("60-Day Sector Correlation Matrix")

    with st.spinner("Computing correlations..."):
        breadth = fetch_breadth_data()

    corr = breadth["corr_matrix"]
    high_pairs = breadth["high_corr_pairs"]

    if high_pairs:
        warnings = ", ".join([f"{a}/{b}: {c:.2f}" for a, b, c in high_pairs])
        st.warning(f"⚠️ High correlation pairs (>0.85) — diversification benefit is low: {warnings}")

    # Rename columns to sector names
    corr_display = corr.rename(columns=SECTOR_ETF_NAMES, index=SECTOR_ETF_NAMES)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_display.values,
        x=list(corr_display.columns),
        y=list(corr_display.index),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=corr_display.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig_corr.update_layout(
        height=500,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="white", size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Market Breadth ────────────────────────────────────────────────────────
    st.subheader("Market Breadth — S&P 500")

    b50 = breadth["breadth_50"]
    b200 = breadth["breadth_200"]
    FEB_2026_BENCHMARK = 65.0

    m1, m2, m3 = st.columns(3)
    m1.metric("% Above 50MA", f"{b50:.1f}%",
              delta=f"{b50 - 50:+.1f}% vs neutral",
              delta_color="normal" if b50 > 50 else "inverse")
    m2.metric("% Above 200MA", f"{b200:.1f}%",
              delta=f"{b200 - FEB_2026_BENCHMARK:+.1f}% vs Feb '26 benchmark ({FEB_2026_BENCHMARK}%)",
              delta_color="normal" if b200 >= FEB_2026_BENCHMARK else "inverse")
    m3.metric("Rotation Signal", rotation_data["ratio_signal"])

    # Sector breadth bar chart
    sec_breadth = breadth["sector_breadth"]
    thrust_df   = breadth["breadth_thrust"]
    thrust_map  = dict(zip(thrust_df["GICS Sector"], thrust_df["Z-Score"]))

    bar_colors = []
    for sector in sec_breadth["GICS Sector"]:
        z = thrust_map.get(sector, 0)
        if z >= 2:
            bar_colors.append(RED)     # overbought
        elif z <= -2:
            bar_colors.append(GREEN)   # oversold opportunity
        else:
            bar_colors.append(TEAL)

    fig_breadth = go.Figure(go.Bar(
        x=sec_breadth["% Above 200MA"],
        y=sec_breadth["GICS Sector"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in sec_breadth["% Above 200MA"]],
        textposition="outside",
    ))
    fig_breadth.add_vline(
        x=FEB_2026_BENCHMARK, line_dash="dash", line_color=YELLOW,
        annotation_text=f"Feb '26: {FEB_2026_BENCHMARK}%",
        annotation_font_color=YELLOW,
    )
    fig_breadth.update_layout(
        height=400,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="white"),
        xaxis=dict(title="% Stocks Above 200MA", range=[0, 110], showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=60, t=10, b=30),
    )
    st.plotly_chart(fig_breadth, use_container_width=True)

    st.caption("🔴 Red = Overbought (Z > +2σ from 60-day mean)  |  🟢 Green = Oversold Opportunity (Z < −2σ)")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Market Sentiment")

    # ── Overall Gauge ─────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=composite_sentiment,
        number={"suffix": "", "font": {"color": "white", "size": 36}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "white"},
            "bar": {"color": TEAL},
            "steps": [
                {"range": [-1, -0.3], "color": "#B71C1C"},
                {"range": [-0.3, 0.3], "color": "#F57F17"},
                {"range": [0.3, 0.6],  "color": "#1B5E20"},
                {"range": [0.6, 1.0],  "color": "#B71C1C"},  # crowded/overbought
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": composite_sentiment},
        },
        title={"text": "Composite Sentiment (−1 Fear → +1 Greed)", "font": {"color": "white"}},
        domain={"x": [0.1, 0.9], "y": [0, 1]},
    ))
    fig_gauge.update_layout(
        height=300, paper_bgcolor=BG, font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if composite_sentiment > CROWDED_THRESHOLD if hasattr(st, "session_state") else False:
        st.error("⚠️ **CROWDED TRADE WARNING** — Sentiment is in extreme greed territory. Risk of reversal elevated.")

    try:
        from fetchers.sentiment_fetcher import CROWDED_THRESHOLD as _CT
        if composite_sentiment > _CT:
            st.error("⚠️ **CROWDED TRADE WARNING** — Sentiment in extreme greed. Reversal risk elevated.")
    except Exception:
        pass

    # ── News Sentiment ────────────────────────────────────────────────────────
    st.subheader("News Sentiment — Last 24h")

    if news_data.get("error"):
        st.warning(f"NewsAPI unavailable: {news_data['error']}")
    else:
        nc1, nc2 = st.columns(2)
        with nc1:
            best = news_data.get("best_headline")
            if best:
                st.markdown(
                    f"""<div style='background:{CARD};border-left:4px solid {GREEN};
                    padding:12px;border-radius:4px'>
                    <b style='color:{GREEN}'>Most Positive</b><br>
                    <span style='font-size:13px'>{best['headline']}</span><br>
                    <small style='color:{GRAY}'>Score: {best['score']:.3f}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )
        with nc2:
            worst = news_data.get("worst_headline")
            if worst:
                st.markdown(
                    f"""<div style='background:{CARD};border-left:4px solid {RED};
                    padding:12px;border-radius:4px'>
                    <b style='color:{RED}'>Most Negative</b><br>
                    <span style='font-size:13px'>{worst['headline']}</span><br>
                    <small style='color:{GRAY}'>Score: {worst['score']:.3f}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        # Sentiment by topic bar chart
        topics = list(news_data["by_query"].keys())
        scores = [news_data["by_query"][t]["avg_score"] for t in topics]
        fig_news = go.Figure(go.Bar(
            x=topics, y=scores,
            marker_color=[GREEN if s >= 0 else RED for s in scores],
            text=[f"{s:+.3f}" for s in scores], textposition="outside",
        ))
        fig_news.update_layout(
            height=300, title="Avg Sentiment by Topic",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color="white"),
            yaxis=dict(title="VADER Score", range=[-1, 1], showgrid=False),
            xaxis=dict(showgrid=False),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_news, use_container_width=True)

    # ── Reddit Sentiment ───────────────────────────────────────────────────────
    st.subheader("Reddit Sentiment (WSB / Investing / Stocks)")

    if reddit_data.get("error"):
        st.warning(f"Reddit unavailable: {reddit_data['error']}")
    elif not reddit_data["ticker_df"].empty:
        rdf = reddit_data["ticker_df"].head(20)

        # Bubble chart
        fig_bubble = go.Figure(go.Scatter(
            x=rdf["Mentions"],
            y=rdf["Avg Sentiment"],
            mode="markers+text",
            text=rdf["Ticker"],
            textposition="top center",
            marker=dict(
                size=rdf["Total Upvotes"].clip(lower=50) / rdf["Total Upvotes"].max() * 50 + 10,
                color=[GREEN if s > 0.05 else RED if s < -0.05 else YELLOW
                       for s in rdf["Avg Sentiment"]],
                opacity=0.8,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="<b>%{text}</b><br>Mentions: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        ))
        fig_bubble.add_hline(y=0, line_dash="dash", line_color=GRAY)
        fig_bubble.update_layout(
            height=400, title="Reddit Ticker Sentiment Bubble Map",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color="white"),
            xaxis=dict(title="Mention Count", showgrid=False),
            yaxis=dict(title="Avg VADER Score", showgrid=False),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.dataframe(
            rdf[["Ticker", "Mentions", "Avg Sentiment", "Total Upvotes", "Signal"]].style
            .applymap(lambda v: f"color:{GREEN}" if v == "Bullish"
                      else f"color:{RED}" if v == "Bearish" else f"color:{YELLOW}",
                      subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No Reddit data available.")

    # ── X/Twitter Placeholder ──────────────────────────────────────────────────
    st.subheader("X / Twitter")
    tw = fetch_twitter_mock()
    st.markdown(
        f"""<div style='background:{CARD};border:1px solid #333;
        padding:16px;border-radius:8px;margin-bottom:12px'>
        <b style='color:{YELLOW}'>⚠️ X API Requires Paid Access</b><br>
        <span style='color:{GRAY};font-size:13px'>{tw['note']}</span><br><br>
        <b style='color:{TEAL}'>Free Alternative:</b>
        <span style='color:{GRAY};font-size:13px'> StockTwits API —
        <code>https://api.stocktwits.com/api/2/streams/symbol/SPY.json</code>
        — no auth required for public streams.</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Mock X data (illustrative only):")
    mock_df = pd.DataFrame(tw["mock_data"])
    st.dataframe(
        mock_df.style.applymap(
            lambda v: f"color:{GREEN}" if v == "Bullish" else f"color:{RED}" if v == "Bearish" else "",
            subset=["signal"],
        ),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — THEME DRILL-DOWNS
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Structural Macro Themes")

    # ── Theme selector cards ──────────────────────────────────────────────────
    theme_names = list(THEMES.keys())
    theme_icons = {"AI Enablers": "🤖", "AI Adopters": "💡", "Nearshoring": "🌎", "Capex Revival": "🏗️"}

    tcols = st.columns(len(theme_names))
    for i, tname in enumerate(theme_names):
        with tcols[i]:
            is_active = st.session_state.selected_theme == tname
            border_color = TEAL if is_active else "#333"
            avg_ytd = theme_data["themes"].get(tname, pd.DataFrame())
            avg = avg_ytd["YTD %"].mean() if not avg_ytd.empty and "YTD %" in avg_ytd.columns else 0
            avg_color = GREEN if avg >= 0 else RED
            st.markdown(
                f"""<div style='background:{CARD};border:2px solid {border_color};
                padding:12px;border-radius:8px;text-align:center;cursor:pointer'>
                <div style='font-size:24px'>{theme_icons.get(tname, '📈')}</div>
                <b>{tname}</b><br>
                <span style='color:{avg_color};font-size:14px'>Avg YTD: {avg:+.1f}%</span>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Select", key=f"theme_{i}", use_container_width=True):
                st.session_state.selected_theme = tname
                st.rerun()

    st.markdown("---")
    active_theme = st.session_state.selected_theme
    theme_desc = THEMES[active_theme]["description"]
    st.markdown(f"### {theme_icons.get(active_theme, '📈')} {active_theme}")
    st.caption(theme_desc)

    # ── Performance chart (normalized to 100) ─────────────────────────────────
    hist = theme_data.get("price_histories", {}).get(active_theme)
    if hist is not None and not hist.empty:
        try:
            spy_norm = fetch_spy_normalized()
            fig_perf = go.Figure()

            for col in hist.columns:
                fig_perf.add_trace(go.Scatter(
                    x=hist.index, y=hist[col].values,
                    name=col, mode="lines", line=dict(width=1.5),
                ))
            # SPY benchmark in grey
            aligned_spy = spy_norm.reindex(hist.index, method="ffill")
            fig_perf.add_trace(go.Scatter(
                x=aligned_spy.index, y=aligned_spy.values,
                name="SPY", mode="lines",
                line=dict(color=GRAY, width=1.5, dash="dot"),
            ))
            fig_perf.add_hline(y=100, line_dash="dash", line_color=GRAY, line_width=0.5)
            fig_perf.update_layout(
                height=420, title=f"{active_theme} — YTD Performance (Normalized to 100)",
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color="white"),
                legend=dict(orientation="h", y=-0.15),
                yaxis=dict(title="Normalized Price (Jan 1 = 100)", showgrid=False),
                xaxis=dict(showgrid=False),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render performance chart: {e}")
    else:
        st.info("Price history not available for this theme.")

    # ── Watchlist table ───────────────────────────────────────────────────────
    st.subheader("Watchlist")
    active_df = theme_data["themes"].get(active_theme, pd.DataFrame())
    if not active_df.empty:
        def color_ytd(v):
            if isinstance(v, float) and not pd.isna(v):
                return f"color:{GREEN}" if v >= 0 else f"color:{RED}"
            return ""

        fmt_cols = {
            c: "{:+.2f}%" for c in ["YTD %", "3M %", "vs 52W High", "vs 52W Low"]
            if c in active_df.columns
        }
        st.dataframe(
            active_df.style
            .format(fmt_cols)
            .applymap(color_ytd, subset=[c for c in ["YTD %", "3M %"] if c in active_df.columns]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No data available.")

    # ── Theme vs Theme comparison ─────────────────────────────────────────────
    st.subheader("Theme vs Theme — Average YTD Performance")
    comparison_rows = []
    for tname, tdf in theme_data["themes"].items():
        avg = tdf["YTD %"].mean() if not tdf.empty and "YTD %" in tdf.columns else float("nan")
        comparison_rows.append({"Theme": tname, "Avg YTD %": round(avg, 2)})
    cdf = pd.DataFrame(comparison_rows).sort_values("Avg YTD %", ascending=False)

    fig_comp = go.Figure(go.Bar(
        x=cdf["Theme"],
        y=cdf["Avg YTD %"],
        marker_color=[GREEN if v >= 0 else RED for v in cdf["Avg YTD %"]],
        text=[f"{v:+.2f}%" for v in cdf["Avg YTD %"]],
        textposition="outside",
    ))
    fig_comp.update_layout(
        height=320,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="white"),
        yaxis=dict(title="Avg YTD %", showgrid=False),
        xaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# AI BRIEFING (sidebar trigger → inline panel)
# ═════════════════════════════════════════════════════════════════════════════

def _should_generate_briefing():
    """Return True if we need to regenerate (never generated, or >2h old)."""
    if st.session_state.ai_briefing is None:
        return True
    age = time.time() - (st.session_state.ai_briefing_time or 0)
    return age > 7200


# Detect button press via a flag written by the sidebar button
if not st.session_state.ai_briefing and anthropic_ok:
    # Auto-generate on first visit if key is set
    pass

# Check if briefing button was pressed (sidebar sets briefing to None)
if st.session_state.ai_briefing is not None:
    st.markdown("---")
    st.markdown("### 🤖 AI Market Briefing")
    age_mins = int((time.time() - st.session_state.ai_briefing_time) / 60) if st.session_state.ai_briefing_time else 0
    st.caption(f"Generated {age_mins}m ago · cached for 2 hours")
    st.markdown(
        f"""<div style='background:{CARD};border-left:4px solid {TEAL};
        padding:20px;border-radius:8px;line-height:1.7'>
        {st.session_state.ai_briefing.replace(chr(10), '<br>')}
        </div>""",
        unsafe_allow_html=True,
    )
elif anthropic_ok and _should_generate_briefing() and st.sidebar.button is not None:
    pass  # briefing generated on explicit button click below


def generate_ai_briefing():
    """Call Claude to produce a structured market briefing."""
    import anthropic as _anthropic

    sectors_sorted = sectors_only.sort_values("YTD %", ascending=False)
    top3 = sectors_sorted.head(3)[["Sector", "YTD %"]].to_dict("records")
    bot3 = sectors_sorted.tail(3)[["Sector", "YTD %"]].to_dict("records")

    best_theme = max(
        theme_data["themes"].items(),
        key=lambda x: x[1]["YTD %"].mean() if not x[1].empty and "YTD %" in x[1].columns else -999,
        default=("N/A", pd.DataFrame()),
    )
    best_theme_name = best_theme[0]
    best_ticker_row = best_theme[1].sort_values("YTD %", ascending=False).iloc[0] if not best_theme[1].empty else {}
    best_ticker = best_ticker_row.get("Ticker", "N/A") if isinstance(best_ticker_row, dict) else best_ticker_row["Ticker"]

    has_crowded = news_data.get("crowded_trade_warning", False)
    has_alerts = any(abs(z) >= 2 for z in breadth["breadth_thrust"]["Z-Score"]) if "breadth_thrust" in breadth else False

    prompt = f"""You are a professional macro market analyst. Based on the real-time data below, provide a structured market briefing.

MARKET DATA ({datetime.today().strftime('%B %d, %Y')}):
- Top 3 Sectors YTD: {top3}
- Bottom 3 Sectors YTD: {bot3}
- Growth vs Value Signal: {rotation_data.get('ratio_signal', 'N/A')}
- XLK/IVE Ratio: {rotation_data.get('current_ratio', 'N/A')}
- Market Breadth: {breadth.get('breadth_50', 'N/A')}% above 50MA, {breadth.get('breadth_200', 'N/A')}% above 200MA
- Feb 2026 Breadth Benchmark: 65% — Current: {breadth.get('breadth_200', 'N/A')}%
- Composite Sentiment Score: {composite_sentiment:.2f} (−1=Fear, +1=Greed){'  ⚠️ CROWDED TRADE WARNING' if has_crowded else ''}
- Best Theme: {best_theme_name} (Top ticker: {best_ticker})
{'- ⚠️ BREADTH THRUST ALERTS DETECTED' if has_alerts else ''}

Return a briefing with EXACTLY these four sections:

**1. Market Regime**
(2 sentences — what phase of the cycle are we in based on breadth + rotation)

**2. Top Opportunity**
(Specific sector or theme with data-driven reasoning)

**3. Risk to Watch**
(What could derail the current trend)

**4. Contrarian Take**
(What the market may be mispricing or ignoring)

Be specific, concise, and data-driven."""

    client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# Trigger generation when sidebar button sets briefing to None
if anthropic_ok and st.session_state.ai_briefing is None:
    # Only auto-generate if explicitly requested (button was clicked)
    if "ai_briefing_requested" in st.session_state and st.session_state.ai_briefing_requested:
        st.session_state.ai_briefing_requested = False
        with st.spinner("🤖 Claude is analyzing the markets..."):
            try:
                result = generate_ai_briefing()
                st.session_state.ai_briefing = result
                st.session_state.ai_briefing_time = time.time()
                st.rerun()
            except Exception as e:
                st.error(f"AI Briefing failed: {e}")
