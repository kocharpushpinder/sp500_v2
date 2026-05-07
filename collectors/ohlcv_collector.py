"""
collectors/ohlcv_collector.py
─────────────────────────────
Downloads OHLCV price data for the full S&P 500 universe via yfinance.

- First run  : pulls full history from HISTORICAL_START to today.
- Daily runs : incremental — only fetches data since the last stored date.
- Output     : data/raw/<TICKER>.csv  (one file per ticker)
               data/raw/combined_ohlcv.csv  (all tickers merged, long format)

Run directly:
    python collectors/ohlcv_collector.py
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DIR, HISTORICAL_START, OHLCV_INTERVAL,
    MAX_WORKERS, TICKER_OVERRIDE
)
from utils.helpers import get_logger, retry, load_or_empty, save_csv, date_range_since

log = get_logger("ohlcv_collector")


# ─── Ticker universe ──────────────────────────────────────────────────────────
def _cached_tickers() -> list[str]:
    """Best-effort local fallback when live universe fetch is unavailable."""
    raw_files = sorted(p.stem for p in RAW_DIR.glob("*.csv") if p.stem != "combined_ohlcv")
    if raw_files:
        log.warning(f"Using {len(raw_files)} tickers from local raw CSV cache")
        return sorted(raw_files)
    return []


def get_sp500_tickers() -> list[str]:
    """
    Fetch the current S&P 500 constituent list from Wikipedia.
    Uses requests with a browser User-Agent to avoid 403 blocks,
    then parses the HTML with pd.read_html via StringIO.
    """
    if TICKER_OVERRIDE:
        log.info(f"Using ticker override: {TICKER_OVERRIDE}")
        return sorted(TICKER_OVERRIDE)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        cached = _cached_tickers()
        if cached:
            return cached
        raise RuntimeError(f"Failed to download S&P 500 Wikipedia page: {e}") from e

    from io import StringIO
    try:
        tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return sorted(tickers)
    except Exception as e:
        cached = _cached_tickers()
        if cached:
            return cached
        raise RuntimeError(f"Failed to parse S&P 500 Wikipedia table: {e}") from e


# ─── Column normalizer ───────────────────────────────────────────────────────
def _normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Bulletproof column handler for yfinance output.

    Newer yfinance versions always return a MultiIndex (Price, Ticker) even for
    single-ticker downloads. Sometimes it duplicates columns (multiples of 5).
    This handles all observed cases:
      - Normal MultiIndex: (Open/AAPL, High/AAPL, ...)
      - Doubled MultiIndex: (Open/AAPL x2, High/AAPL x2, ...)
      - Flat columns: Open, High, Low, Close, Volume (older yfinance)
    """
    # Step 1: if MultiIndex, select this ticker's slice via level 1
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            # ticker name may differ in case or have suffix — try fuzzy match
            lvl1_vals = df.columns.get_level_values(1).unique()
            match = next((v for v in lvl1_vals if ticker.upper() in v.upper()), None)
            if match:
                df = df.xs(match, axis=1, level=1)
            else:
                # Last resort: flatten and deduplicate
                df.columns = df.columns.get_level_values(0)

    # Step 2: remove duplicate columns (keep first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Step 3: select the 5 OHLCV columns we need (case-insensitive lookup)
    col_map = {c.lower(): c for c in df.columns}
    needed  = ["open", "high", "low", "close", "volume"]
    missing = [n for n in needed if n not in col_map]
    if missing:
        raise ValueError(
            f"[{ticker}] Missing columns after normalization: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df = df[[col_map[n] for n in needed]].copy()
    df.columns = needed
    return df


# ─── Single ticker fetch ──────────────────────────────────────────────────────
@retry(max_attempts=3, wait_seconds=3.0)
def fetch_ticker_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for a single ticker between start and end dates."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=OHLCV_INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()

    df = _normalize_ohlcv_columns(df, ticker)

    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["ticker"] = ticker
    return df.reset_index()


# ─── Incremental update for one ticker ───────────────────────────────────────
def update_ticker(ticker: str) -> dict:
    """Fetch new data since the last stored date and append to the ticker CSV."""
    out_path = RAW_DIR / f"{ticker}.csv"
    start, end = date_range_since(out_path, HISTORICAL_START, date_col="date")

    if start >= end:
        return {"ticker": ticker, "rows": 0, "status": "up_to_date"}

    try:
        df_new = fetch_ticker_ohlcv(ticker, start, end)
        if df_new.empty:
            return {"ticker": ticker, "rows": 0, "status": "no_data"}

        df_existing = load_or_empty(out_path, parse_dates=["date"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["date"]).sort_values("date")
        save_csv(df_combined, out_path, index=False)
        return {"ticker": ticker, "rows": len(df_new), "status": "ok"}
    except Exception as e:
        log.error(f"[{ticker}] OHLCV fetch failed: {e}")
        return {"ticker": ticker, "rows": 0, "status": f"error: {e}"}


# ─── Bulk collection ──────────────────────────────────────────────────────────
def run_ohlcv_collection(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Run incremental OHLCV collection for all tickers in parallel.
    Returns a summary DataFrame of results.
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    log.info(f"Starting OHLCV collection for {len(tickers)} tickers "
             f"(max_workers={MAX_WORKERS})")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(update_ticker, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if i % 50 == 0 or i == len(tickers):
                log.info(f"  Progress: {i}/{len(tickers)} tickers processed")

    summary = pd.DataFrame(results)
    ok      = (summary["status"] == "ok").sum()
    uptodate= (summary["status"] == "up_to_date").sum()
    errors  = summary["status"].str.startswith("error").sum()
    total_rows = summary["rows"].sum()

    log.info(f"OHLCV collection complete: "
             f"{ok} updated | {uptodate} up-to-date | {errors} errors | "
             f"{total_rows:,} new rows")

    # Build combined long-format CSV
    _build_combined(tickers)

    return summary


def _build_combined(tickers: list[str]):
    """Merge all per-ticker CSVs into one combined_ohlcv.csv."""
    log.info("Building combined_ohlcv.csv ...")
    frames = []
    for ticker in tickers:
        path = RAW_DIR / f"{ticker}.csv"
        if path.exists():
            df = load_or_empty(path, parse_dates=["date"])
            if not df.empty:
                frames.append(df)

    if not frames:
        log.warning("No ticker files found — combined CSV not written.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    out_path = RAW_DIR / "combined_ohlcv.csv"
    save_csv(combined, out_path, index=False)
    log.info(f"Combined OHLCV saved: {len(combined):,} rows → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_ohlcv_collection()
    errors = summary[summary["status"].str.startswith("error")]
    if not errors.empty:
        log.warning(f"Tickers with errors:\n{errors.to_string(index=False)}")
