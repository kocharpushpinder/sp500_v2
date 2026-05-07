"""
collectors/macro_collector.py
──────────────────────────────
Downloads macro / market-wide features via yfinance.
These are shared features broadcast to every ticker during feature engineering.

Outputs:
    data/macro/macro_daily.csv   — one row per date, one column per macro ticker

Run directly:
    python collectors/macro_collector.py
"""

from pathlib import Path
import sys
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MACRO_DIR, MACRO_TICKERS, HISTORICAL_START
from utils.helpers import get_logger, retry, load_or_empty, save_csv, date_range_since

log = get_logger("macro_collector")

MACRO_OUT = MACRO_DIR / "macro_daily.csv"


@retry(max_attempts=3, wait_seconds=5.0)
def _fetch_macro_series(yf_symbol: str, col_name: str,
                        start: str, end: str) -> pd.Series:
    """Download a single macro time series and return as a named Series indexed by date."""
    df = yf.download(yf_symbol, start=start, end=end,
                     auto_adjust=True, progress=False, threads=False)
    if df.empty:
        log.debug(f"No data for macro ticker {yf_symbol} (may be weekend/holiday)")
        return pd.Series(dtype=float, name=col_name)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    series = df["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.index.name = "date"
    series.name = col_name
    return series


def run_macro_collection() -> pd.DataFrame:
    """
    Fetch all macro tickers and merge into a single wide CSV.
    Incremental: only fetches data since the last stored date.
    """
    start, end = date_range_since(MACRO_OUT, HISTORICAL_START, date_col="date")

    if start >= end:
        log.info("Macro data is up to date.")
        df_existing = load_or_empty(MACRO_OUT, parse_dates=["date"])
        return df_existing

    log.info(f"Fetching macro data from {start} to {end} ...")

    # Check if the date range contains any weekdays before fetching.
    # pd.bdate_range snaps weekends to Monday, so we must check weekday() directly.
    import datetime as _dt
    start_dt = _dt.date.fromisoformat(start)
    end_dt   = _dt.date.fromisoformat(end)
    has_weekday = any(
        (start_dt + _dt.timedelta(days=i)).weekday() < 5
        for i in range((end_dt - start_dt).days + 1)
    )
    if not has_weekday:
        log.info(f"No trading days in range {start} → {end} (weekend/holiday) — nothing to fetch.")
        return load_or_empty(MACRO_OUT, parse_dates=["date"])

    series_list = []
    for yf_symbol, col_name in MACRO_TICKERS.items():
        s = _fetch_macro_series(yf_symbol, col_name, start, end)
        if not s.empty:
            series_list.append(s)
            log.info(f"  {col_name:12s} ({yf_symbol}): {len(s)} rows")
        else:
            log.debug(f"  {col_name:12s} ({yf_symbol}): no data returned (holiday/weekend/delisted)")

    if not series_list:
        log.info("No new macro data for this date range (likely all non-trading days).")
        return load_or_empty(MACRO_OUT, parse_dates=["date"])

    df_new = pd.concat(series_list, axis=1).reset_index()
    df_new.columns.name = None

    # Derive useful signals
    df_new = _add_derived_features(df_new)

    # Append to existing
    df_existing = load_or_empty(MACRO_OUT, parse_dates=["date"])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["date"]).sort_values("date")

    save_csv(df_combined, MACRO_OUT, index=False)
    log.info(f"Macro data saved: {len(df_combined):,} rows → {MACRO_OUT}")
    return df_combined


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived macro signals on top of raw close prices."""
    df = df.sort_values("date").copy()

    if "vix" in df.columns:
        df["vix_7d_change"]  = df["vix"].pct_change(7)
        df["vix_high_regime"] = (df["vix"] > 25).astype(int)

    if "sp500" in df.columns:
        df["sp500_ret_1d"]  = df["sp500"].pct_change(1)
        df["sp500_ret_5d"]  = df["sp500"].pct_change(5)
        df["sp500_ret_20d"] = df["sp500"].pct_change(20)
        # Simple trend regime: 1 = above 50-day SMA, 0 = below
        df["sp500_sma50"]   = df["sp500"].rolling(50).mean()
        df["bull_regime"]   = (df["sp500"] > df["sp500_sma50"]).astype(int)

    if "yield_10y" in df.columns and "sp500" in df.columns:
        df["yield_10y_5d_change"] = df["yield_10y"].pct_change(5)

    if "dxy" in df.columns:
        df["dxy_5d_change"] = df["dxy"].pct_change(5)

    return df


if __name__ == "__main__":
    df = run_macro_collection()
    if not df.empty:
        print(df.tail())
