"""
utils/helpers.py — Shared utilities used across all collectors.
"""

import logging
import time
import functools
from pathlib import Path
from datetime import datetime, date

import pandas as pd
from pandas.errors import ParserError

from config import LOG_FILE, LOG_LEVEL


# ─── Logger ──────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── Rate limiter ────────────────────────────────────────────────────────────
class RateLimiter:
    """Token-bucket rate limiter.  Usage: limiter.wait() before each API call."""

    def __init__(self, calls_per_minute: int):
        self.min_interval = 60.0 / calls_per_minute
        self._last_call   = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        sleep   = self.min_interval - elapsed
        if sleep > 0:
            time.sleep(sleep)
        self._last_call = time.monotonic()


# ─── Retry decorator ─────────────────────────────────────────────────────────
def retry(max_attempts: int = 3, wait_seconds: float = 5.0, exceptions=(Exception,)):
    """Retry a function up to max_attempts times on specified exceptions."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log = get_logger("retry")
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log.error(f"{fn.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    log.warning(f"{fn.__name__} attempt {attempt} failed: {e}. "
                                f"Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
        return wrapper
    return decorator


# ─── CSV helpers ─────────────────────────────────────────────────────────────
def load_or_empty(path: Path, parse_dates: list = None) -> pd.DataFrame:
    """Load a CSV if it exists, otherwise return an empty DataFrame."""
    if path.exists():
        try:
            return pd.read_csv(path, parse_dates=parse_dates)
        except ParserError as e:
            # Recover from occasional malformed rows in incremental CSVs.
            log = get_logger("helpers")
            log.warning(f"CSV parse issue in {path.name}; skipping bad lines ({e})")
            return pd.read_csv(
                path,
                parse_dates=parse_dates,
                on_bad_lines="skip",
                engine="python",
            )
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path, index: bool = True):
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def market_date_today() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().strftime("%Y-%m-%d")


def date_range_since(csv_path: Path, start_fallback: str,
                     date_col: str = "date") -> tuple[str, str]:
    """
    Return (start, end) for an incremental fetch:
    - start = last date in existing CSV + 1 day (or start_fallback)
    - end   = today
    """
    df = load_or_empty(csv_path, parse_dates=[date_col])
    if not df.empty and date_col in df.columns:
        # Be defensive: mixed CSV dtypes can yield non-datetimelike values.
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            last = dates.max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = start_fallback
    else:
        start = start_fallback
    end = market_date_today()
    return start, end
