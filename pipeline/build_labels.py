"""
pipeline/build_labels.py
─────────────────────────
Builds the prediction labels SEPARATELY from features.

DESIGN PRINCIPLE:
  Labels are NEVER stored alongside features in features.csv.
  They are computed at training time, joined to features by (date, ticker),
  and used only in memory — never written to disk as part of the feature matrix.

  This makes leakage structurally impossible: features.csv contains only
  data known at market close on day T. Labels are computed from T+1 to T+N.

TARGET: Cross-sectional alpha vs S&P 500
  label = +1 if stock return over next LABEL_HORIZON days
              exceeds S&P 500 return by more than LABEL_THRESHOLD
  label =  0 if within ±LABEL_THRESHOLD of S&P 500 (neutral)
  label = -1 if underperforms S&P 500 by more than LABEL_THRESHOLD

  This is regime-invariant: ~35% long, 30% flat, 35% short in all markets.
  A stock that rises 1% when the market rises 3% = UNDERPERFORM (short).
  A stock that falls 1% when the market falls 5% = OUTPERFORM (long).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, MACRO_DIR, FEATURES_DIR,
    LABEL_HORIZON, LABEL_THRESHOLD
)
from utils.helpers import get_logger

log = get_logger("build_labels")


def build_labels(horizon: int   = LABEL_HORIZON,
                 threshold: float = LABEL_THRESHOLD) -> pd.DataFrame:
    """
    Build cross-sectional alpha labels for all tickers.

    Returns a DataFrame with columns:
      date, ticker, label (−1/0/+1), forward_alpha (raw value)

    This is joined to features at training time. Never stored in features.csv.
    """
    # Load OHLCV for forward stock returns
    ohlcv_path = RAW_DIR / "combined_ohlcv.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError("combined_ohlcv.csv not found. Run Phase 1.")

    ohlcv = pd.read_csv(ohlcv_path, parse_dates=["date"])
    # Ensure a consistent merge key type even if CSV parsing yields mixed/object dtypes.
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce").dt.normalize()
    ohlcv = ohlcv.dropna(subset=["date"])
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Load S&P 500 index for market return benchmark
    macro_path = MACRO_DIR / "macro_daily.csv"
    if not macro_path.exists():
        raise FileNotFoundError("macro_daily.csv not found. Run Phase 1.")

    macro = pd.read_csv(macro_path, parse_dates=["date"])
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
    macro = macro.dropna(subset=["date"])
    sp500 = macro[["date", "sp500"]].dropna().sort_values("date").reset_index(drop=True)

    # ── Forward return for each stock ─────────────────────────────────────────
    # forward_ret[T] = log(close[T+horizon] / close[T])
    # This uses future prices — that's correct and intentional for labels
    ohlcv["forward_ret"] = ohlcv.groupby("ticker")["close"].transform(
        lambda x: np.log(x.shift(-horizon) / x)
    )

    # ── Forward S&P 500 return ────────────────────────────────────────────────
    # market_ret[T] = log(sp500[T+horizon] / sp500[T])
    sp500["market_forward_ret"] = np.log(sp500["sp500"].shift(-horizon) / sp500["sp500"])

    # ── Merge market return onto OHLCV ────────────────────────────────────────
    ohlcv = ohlcv.merge(
        sp500[["date", "market_forward_ret"]],
        on="date", how="left"
    )

    # ── Cross-sectional alpha ─────────────────────────────────────────────────
    # alpha = stock outperformance vs market
    ohlcv["forward_alpha"] = ohlcv["forward_ret"] - ohlcv["market_forward_ret"]

    # ── 3-class label ─────────────────────────────────────────────────────────
    ohlcv["label"] = 0
    ohlcv.loc[ohlcv["forward_alpha"] >  threshold, "label"] =  1
    ohlcv.loc[ohlcv["forward_alpha"] < -threshold, "label"] = -1

    # ── Drop rows without valid labels ────────────────────────────────────────
    labels = ohlcv[["date", "ticker", "label", "forward_alpha"]].dropna()
    labels = labels.reset_index(drop=True)

    # ── Distribution check ────────────────────────────────────────────────────
    dist = labels["label"].value_counts(normalize=True)
    log.info(f"Label distribution: "
             f"long={dist.get(1,0):.1%} | "
             f"flat={dist.get(0,0):.1%} | "
             f"short={dist.get(-1,0):.1%} | "
             f"total={len(labels):,}")

    # Warn if collapsed (regime bias still present)
    if dist.get(1, 0) < 0.15 or dist.get(-1, 0) < 0.15:
        log.warning("Label distribution is skewed — check macro data quality")

    return labels


def load_training_dataset(min_ticker_rows: int = 252) -> pd.DataFrame:
    """
    Join features + labels into a training-ready DataFrame.

    This is the ONLY place where features and labels come together.
    Called by the trainer — never writes to disk.

    Returns DataFrame with all feature columns + 'label' column.
    No future-derived columns are present.
    """
    feat_path = FEATURES_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.csv not found. Run: python run_pipeline.py --features")

    log.info("Loading features ...")
    features = pd.read_csv(feat_path, parse_dates=["date"])

    # ASSERT: no future columns in features.
    # Only block columns that are explicitly computed from future prices (shift(-N)).
    # "pe_forward" is fine — analyst estimate known today, not computed from future prices.
    LEAKY_EXACT = {
        "future_ret", "future_market_ret", "future_alpha",
        "forward_ret", "forward_alpha", "forward_market_ret",
        "market_forward_ret",
    }
    bad = [c for c in features.columns if c in LEAKY_EXACT]
    if bad:
        raise RuntimeError(
            f"Features file contains future-derived columns: {bad}. "
            "Rebuild features with: python run_pipeline.py --features"
        )

    log.info("Building labels ...")
    labels = build_labels()

    # Join on (date, ticker)
    df = features.merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="inner")

    # Drop tickers with insufficient history
    counts = df.groupby("ticker").size()
    valid  = counts[counts >= min_ticker_rows].index
    dropped = df["ticker"].nunique() - len(valid)
    if dropped > 0:
        log.info(f"Dropping {dropped} thin tickers (< {min_ticker_rows} rows)")
    df = df[df["ticker"].isin(valid)].reset_index(drop=True)

    log.info(f"Training dataset: {df.shape} | "
             f"{df['ticker'].nunique()} tickers | "
             f"{df['date'].min().date()} → {df['date'].max().date()}")

    return df
