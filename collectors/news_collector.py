"""
collectors/news_collector.py
─────────────────────────────
Fetches company news headlines via Google News RSS feeds and scores
sentiment using VADER. No API key required.

Google News RSS gives the ~10 most recent headlines per ticker query,
updated in near real-time. We fetch once per day after market close.

For each ticker:
  - Pulls latest headlines from Google News RSS
  - Scores each headline with VADER (compound, pos, neg, neu)
  - Aggregates to a single daily sentiment row

Outputs:
    data/news/news_sentiment.csv — daily sentiment per ticker (appended)

Dependencies:
    pip install vaderSentiment feedparser

Run directly:
    python collectors/news_collector.py
"""

import time
from pathlib import Path
from datetime import date
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NEWS_DIR, MAX_WORKERS
from utils.helpers import get_logger, retry, save_csv, load_or_empty

log         = get_logger("news_collector")
NEWS_OUT    = NEWS_DIR / "news_sentiment.csv"

# Conservative workers — Google News will throttle aggressive crawlers
NEWS_WORKERS = min(MAX_WORKERS, 3)

# Small delay between requests per worker to be polite
REQUEST_DELAY = 0.5  # seconds


# ─── Lazy imports ─────────────────────────────────────────────────────────────
_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            log.error("vaderSentiment not installed. Run: pip install vaderSentiment")
            raise
    return _vader


def _get_feedparser():
    try:
        import feedparser
        return feedparser
    except ImportError:
        log.error("feedparser not installed. Run: pip install feedparser")
        raise


# ─── RSS fetch + score ────────────────────────────────────────────────────────
@retry(max_attempts=3, wait_seconds=3.0)
def fetch_ticker_sentiment(ticker: str) -> dict | None:
    """
    Fetch Google News RSS headlines for a ticker and return
    a sentiment row dict, or None if no articles found.
    """
    feedparser = _get_feedparser()
    vader      = _get_vader()

    # Google News RSS — searches for ticker symbol + company name
    url = (
        f"https://news.google.com/rss/search"
        f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    )

    time.sleep(REQUEST_DELAY)
    feed = feedparser.parse(url)

    if not feed.entries:
        return None

    scores = []
    for entry in feed.entries:
        text = entry.get("title", "") + " " + entry.get("summary", "")
        if text.strip():
            s = vader.polarity_scores(text)
            scores.append(s)

    if not scores:
        return None

    df_scores = pd.DataFrame(scores)
    return {
        "ticker":              ticker,
        "date":                date.today().isoformat(),
        "sentiment_compound":  round(df_scores["compound"].mean(), 4),
        "sentiment_pos":       round(df_scores["pos"].mean(), 4),
        "sentiment_neg":       round(df_scores["neg"].mean(), 4),
        "sentiment_neu":       round(df_scores["neu"].mean(), 4),
        "news_count":          len(scores),
    }


# ─── Main runner ──────────────────────────────────────────────────────────────
def run_news_collection(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch and score news sentiment for all tickers via Google News RSS.
    Appends today's row to news_sentiment.csv (one row per ticker per day).
    """
    if tickers is None:
        from collectors.ohlcv_collector import get_sp500_tickers
        tickers = get_sp500_tickers()

    today = date.today().isoformat()

    # Skip tickers already collected today
    df_existing = load_or_empty(NEWS_OUT, parse_dates=["date"])
    if not df_existing.empty and "date" in df_existing.columns:
        already_done = set(
            df_existing[df_existing["date"].astype(str).str[:10] == today]["ticker"]
        )
        tickers = [t for t in tickers if t not in already_done]
        if not tickers:
            log.info("News sentiment already collected for today.")
            return df_existing

    log.info(f"Fetching news sentiment for {len(tickers)} tickers "
             f"via Google News RSS (workers={NEWS_WORKERS}) ...")

    rows = []
    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker_sentiment, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception as e:
                ticker = futures[future]
                log.warning(f"[{ticker}] news fetch failed: {e}")

            if i % 50 == 0 or i == len(tickers):
                log.info(f"  Progress: {i}/{len(tickers)} tickers | "
                         f"{len(rows)} with data")

    if not rows:
        log.warning("No sentiment data collected today.")
        return df_existing

    df_new = pd.DataFrame(rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(
        subset=["ticker", "date"]
    ).sort_values(["ticker", "date"])

    save_csv(df_combined, NEWS_OUT, index=False)
    log.info(f"News sentiment saved: {len(df_new)} new rows → {NEWS_OUT}")
    return df_combined


if __name__ == "__main__":
    df = run_news_collection()
    if not df.empty:
        print(f"\nTotal rows: {len(df)}")
        print(df.tail(10))
