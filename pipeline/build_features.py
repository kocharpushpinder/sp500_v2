"""
pipeline/build_features.py
───────────────────────────
Builds the feature matrix from Phase 1 raw data.

LEAKAGE PREVENTION — ARCHITECTURAL:
  Every feature is computed using only data available at market close on day T.
  Indicators use .shift(1) where needed so same-day close is excluded.
  The sp500 and other macro columns are LAG-1 values (yesterday's close).
  Cross-sectional ranks use LAG-1 returns — never forward returns.

  The label is built in a SEPARATE function (build_labels.py) and joined
  to features only at training time, never stored together.

Output: data/features/features.csv
  Columns: date, ticker, [feature columns]
  NO target, NO future_ret, NO future_alpha — labels live elsewhere.
"""

import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DIR, MACRO_DIR, FUNDAMENTALS_DIR, NEWS_DIR, FEATURES_DIR,
    RSI_WINDOW, ATR_WINDOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_WINDOW, BB_STD, MAX_WORKERS
)
from utils.helpers import get_logger, save_csv

log = get_logger("build_features")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(s: pd.Series, w: int) -> pd.Series:
    d  = s.diff()
    up = d.clip(lower=0).ewm(com=w-1, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(com=w-1, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def _atr(h, l, c, w=14) -> pd.Series:
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PER-TICKER FEATURE COMPUTATION
# All features use only data available at end of day T (no shift(-N))
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ticker_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sorted OHLCV DataFrame for ONE ticker, return feature rows.
    Input columns: date, open, high, low, close, volume
    """
    df = ohlcv.sort_values("date").copy().reset_index(drop=True)
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    feat = pd.DataFrame({
        "date":   df["date"],
        "close":  c,                  # needed by mean-reversion/regime helpers
        "high":   h,
        "low":    l,
    })

    # ── Past returns (all backward-looking) ──────────────────────────────────
    for n in [1, 2, 3, 5, 10, 20, 60]:
        feat[f"ret_{n}d"] = np.log(c / c.shift(n))

    # ── Intraday ─────────────────────────────────────────────────────────────
    feat["gap_pct"]        = (o - c.shift(1)) / c.shift(1)
    feat["intraday_range"] = (h - l) / c
    feat["close_loc"]      = (c - l) / (h - l + 1e-9)  # 0=bottom, 1=top

    # ── RSI ───────────────────────────────────────────────────────────────────
    feat[f"rsi_{RSI_WINDOW}"] = _rsi(c, RSI_WINDOW)
    feat["rsi_5"]              = _rsi(c, 5)
    feat["rsi_28"]             = _rsi(c, 28)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema_f  = c.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_s  = c.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    feat["macd_hist"]     = macd - sig
    feat["macd_hist_chg"] = (macd - sig) - (macd - sig).shift(1)

    # ── ATR / volatility ──────────────────────────────────────────────────────
    atr14 = _atr(h, l, c, ATR_WINDOW)
    feat["atr_pct"]     = atr14 / c           # normalised ATR
    feat["atr_chg"]     = atr14 / (atr14.shift(5) + 1e-9) - 1
    for w in [10, 20, 60]:
        feat[f"hv_{w}d"] = np.log(c / c.shift(1)).rolling(w).std() * np.sqrt(252)
    feat["vol_ratio"]   = feat["hv_10d"] / (feat["hv_60d"] + 1e-9)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    sma  = c.rolling(BB_WINDOW).mean()
    std  = c.rolling(BB_WINDOW).std()
    feat["bb_pct"]   = (c - sma) / (BB_STD * std + 1e-9)  # position in bands
    feat["bb_width"] = (BB_STD * 2 * std) / (sma + 1e-9)

    # ── Trend ─────────────────────────────────────────────────────────────────
    for w in [20, 50, 100, 200]:
        sma_w = c.rolling(w).mean()
        feat[f"dist_sma{w}"]    = (c - sma_w) / (sma_w + 1e-9)
        feat[f"above_sma{w}"]   = (c > sma_w).astype(np.int8)
    feat["sma20_50_slope"]  = (c.rolling(20).mean() / c.rolling(50).mean()) - 1
    feat["sma50_200_slope"] = (c.rolling(50).mean() / c.rolling(200).mean()) - 1

    # ── Price vs rolling range ────────────────────────────────────────────────
    for w in [20, 52]:
        days = 252 if w == 52 else w
        feat[f"pct_hi_{w}w"] = (c - h.rolling(days).max()) / (h.rolling(days).max() + 1e-9)
        feat[f"pct_lo_{w}w"] = (c - l.rolling(days).min()) / (l.rolling(days).min() + 1e-9)

    # ── Volume ────────────────────────────────────────────────────────────────
    for w in [5, 20]:
        v_ma = v.rolling(w).mean()
        feat[f"vol_ratio_{w}d"] = v / (v_ma + 1e-9)
        feat[f"vol_zs_{w}d"]    = (v - v_ma) / (v.rolling(w).std() + 1e-9)

    # ── Stochastic ────────────────────────────────────────────────────────────
    for w in [14, 28]:
        feat[f"stoch_{w}"] = (c - l.rolling(w).min()) / (
            h.rolling(w).max() - l.rolling(w).min() + 1e-9)

    # ── Rate of change ────────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        feat[f"roc_{w}"] = c.pct_change(w)

    # ── Price patterns ────────────────────────────────────────────────────────
    feat["new_52w_high"]  = (c >= h.rolling(252).max()).astype(np.int8)
    feat["new_52w_low"]   = (c <= l.rolling(252).min()).astype(np.int8)
    feat["new_20d_high"]  = (c >= h.rolling(20).max().shift(1)).astype(np.int8)
    feat["inside_bar"]    = ((h < h.shift(1)) & (l > l.shift(1))).astype(np.int8)

    # Consecutive up/down days
    d1 = np.sign(c.diff())
    feat["consec_up"] = d1.groupby((d1 != 1).cumsum()).cumcount().clip(upper=10)
    feat["consec_dn"] = (-d1).groupby((d1 != -1).cumsum()).cumcount().clip(upper=10)

    # ── Calendar ──────────────────────────────────────────────────────────────
    dt = pd.to_datetime(df["date"])
    feat["dow"]        = dt.dt.dayofweek.astype(np.int8)    # 0=Mon
    feat["month"]      = dt.dt.month.astype(np.int8)
    feat["quarter"]    = dt.dt.quarter.astype(np.int8)
    feat["month_end"]  = (dt.dt.day >= 25).astype(np.int8)
    feat["month_start"]= (dt.dt.day <= 5).astype(np.int8)

    # Mean-reversion features (balances momentum dominance)
    feat["ticker"] = "tmp"  # needed by helper functions
    feat = _add_mean_reversion_features(feat)
    feat = _add_regime_features(feat)
    feat = feat.drop(columns=["ticker"])

    # Drop price columns used only by helper functions (close/high/low
    # are already in the main ohlcv data and would be duplicates)
    feat = feat.drop(columns=["close", "high", "low"], errors="ignore")

    # Drop rows with insufficient history (first 200 rows of each ticker)
    feat = feat.iloc[200:].reset_index(drop=True)
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CROSS-SECTIONAL FEATURES
# Computed across all tickers at each date — uses LAGGED returns only
# ══════════════════════════════════════════════════════════════════════════════

def _add_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentile rank features comparing each stock to the full universe.
    Uses LAGGED (past) returns — no forward-looking information.
    """
    log.info("Computing cross-sectional rank features ...")

    # Compute lagged returns for ranking (these are already in the feature set
    # but we recompute cleanly here to avoid any ordering issues)
    ohlcv = pd.read_csv(RAW_DIR / "combined_ohlcv.csv", parse_dates=["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce").dt.normalize()
    ohlcv = ohlcv.dropna(subset=["date"])
    ohlcv = ohlcv.sort_values(["ticker", "date"])

    # Past 5d and 20d returns (backward-looking — safe)
    ohlcv["_r5"]  = ohlcv.groupby("ticker")["close"].transform(lambda x: x.pct_change(5))
    ohlcv["_r20"] = ohlcv.groupby("ticker")["close"].transform(lambda x: x.pct_change(20))
    ohlcv["_v5"]  = ohlcv.groupby("ticker")["volume"].transform(lambda x: x.rolling(5).mean())

    # Percentile rank within each date (0=worst, 1=best)
    for col, feat_name in [("_r5", "cs_rank_ret5"), ("_r20", "cs_rank_ret20"),
                           ("_v5", "cs_rank_vol5")]:
        ohlcv[feat_name] = ohlcv.groupby("date")[col].rank(pct=True)

    rank_cols = ["date", "ticker", "cs_rank_ret5", "cs_rank_ret20", "cs_rank_vol5"]
    ranks = ohlcv[rank_cols].dropna()
    ranks["date"] = pd.to_datetime(ranks["date"], errors="coerce").dt.normalize()
    ranks = ranks.dropna(subset=["date"])

    # Momentum quartile flags
    ranks["top_momentum"] = (ranks["cs_rank_ret5"] >= 0.75).astype(np.int8)
    ranks["bot_momentum"] = (ranks["cs_rank_ret5"] <= 0.25).astype(np.int8)

    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.normalize()
    df_all = df_all.dropna(subset=["date"])

    df_all = df_all.merge(
        ranks[["date", "ticker", "cs_rank_ret5", "cs_rank_ret20",
               "cs_rank_vol5", "top_momentum", "bot_momentum"]],
        on=["date", "ticker"], how="left"
    )
    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MACRO FEATURES
# These are LAGGED by 1 day: model sees yesterday's VIX, not today's
# ══════════════════════════════════════════════════════════════════════════════

def _load_macro_features() -> pd.DataFrame | None:
    path = MACRO_DIR / "macro_daily.csv"
    if not path.exists():
        log.warning("macro_daily.csv not found — skipping macro features")
        return None

    macro = pd.read_csv(path, parse_dates=["date"])
    macro = macro.sort_values("date")

    # Use LAGGED macro (shift by 1 day): model knows yesterday's values
    # This prevents any same-day circular dependency
    macro_cols = [c for c in macro.columns if c != "date"]
    macro_lagged = macro[["date"]].copy()
    for col in macro_cols:
        macro_lagged[f"macro_{col}"] = macro[col].shift(1)

    # Derived signals from lagged data
    if "sp500" in macro.columns:
        s = macro["sp500"].shift(1)
        macro_lagged["macro_sp500_ret5"]  = s.pct_change(5)
        macro_lagged["macro_sp500_ret20"] = s.pct_change(20)
        macro_lagged["macro_bull"]        = (s > s.rolling(50).mean()).astype(np.int8)
    if "vix" in macro.columns:
        v = macro["vix"].shift(1)
        macro_lagged["macro_vix_ret5"]    = v.pct_change(5)
        macro_lagged["macro_high_vix"]    = (v > 25).astype(np.int8)

    return macro_lagged.dropna(subset=["macro_sp500_ret5"] if "macro_sp500_ret5" in macro_lagged.columns else [])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FUNDAMENTAL FEATURES
# Static snapshot — refreshed daily via run_collection.py
# ══════════════════════════════════════════════════════════════════════════════

def _load_fundamental_features() -> pd.DataFrame | None:
    path = FUNDAMENTALS_DIR / "fundamentals.csv"
    if not path.exists():
        return None
    fund = pd.read_csv(path)
    keep = ["ticker", "pe_ttm", "pe_forward", "pb", "ps", "eps_ttm",
            "revenue_growth_yoy", "net_margin", "roe", "debt_equity",
            "beta", "short_ratio", "dividend_yield"]
    cols = [c for c in keep if c in fund.columns]
    return fund[cols].copy()


def _load_earnings_features() -> pd.DataFrame | None:
    path = FUNDAMENTALS_DIR / "earnings.csv"
    if not path.exists():
        return None
    # Phase 1 collector stores "next_earnings_date" (single upcoming date per ticker)
    earn = pd.read_csv(path)
    # Normalise to a common column name for _add_earnings_proximity
    if "next_earnings_date" in earn.columns:
        earn = earn.rename(columns={"next_earnings_date": "earnings_date"})
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"], errors="coerce")
    return earn.dropna(subset=["earnings_date"])


def _add_earnings_proximity(df: pd.DataFrame,
                             df_earn: pd.DataFrame) -> pd.DataFrame:
    if df_earn is None or df_earn.empty:
        return df

    earn_map = df_earn.dropna(subset=["earnings_date"]).copy()
    # Handle both formats: single next date per ticker, or multiple historical dates
    earn_map = earn_map.groupby("ticker")["earnings_date"].apply(list).to_dict()

    def days_to_next(ticker, d):
        dates = earn_map.get(ticker, [])
        future = [e for e in dates if e >= d]
        return (min(future) - d).days if future else 999

    df["days_to_earn"]  = [days_to_next(t, d)
                            for t, d in zip(df["ticker"], pd.to_datetime(df["date"]))]
    df["earn_week"]     = (df["days_to_earn"] <= 5).astype(np.int8)
    df["earn_month"]    = (df["days_to_earn"] <= 21).astype(np.int8)
    return df



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6a — MEAN-REVERSION FEATURES
# Balances the momentum dominance. Overbought/oversold signals that detect
# when a stock has moved too far and is likely to revert.
# All computed from OHLCV — always available, no extra data needed.
# ══════════════════════════════════════════════════════════════════════════════

def _add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mean-reversion features to the per-ticker feature DataFrame.
    These complement momentum features — the model can learn WHEN to
    apply momentum vs when to fade it.
    """
    c = df["close"] if "close" in df.columns else None
    if c is None:
        return df

    # Distance from historical mean (z-score of price relative to own history)
    for w in [20, 60, 126]:
        roll_mean = c.rolling(w).mean()
        roll_std  = c.rolling(w).std()
        df[f"zscore_{w}d"] = (c - roll_mean) / (roll_std + 1e-9)

    # RSI extremes — how overbought/oversold is this stock
    rsi14 = _rsi(c, 14)
    df["rsi_overbought"]  = (rsi14 > 70).astype(np.int8)
    df["rsi_oversold"]    = (rsi14 < 30).astype(np.int8)
    df["rsi_extreme"]     = ((rsi14 > 75) | (rsi14 < 25)).astype(np.int8)
    # Distance from RSI midpoint (50) — larger = more extreme
    df["rsi_dist_mid"]    = (rsi14 - 50) / 50.0

    # Consecutive move magnitude — stocks moving far fast tend to revert
    ret1 = np.log(c / c.shift(1))
    df["ret5_zscore"]  = ret1.rolling(5).sum() / (ret1.rolling(60).std() * np.sqrt(5) + 1e-9)
    df["ret10_zscore"] = ret1.rolling(10).sum() / (ret1.rolling(60).std() * np.sqrt(10) + 1e-9)

    # Distance from 52-week high — stocks near highs have momentum,
    # stocks far below highs have mean-reversion potential
    high_52w = df["high"].rolling(252).max() if "high" in df.columns else c.rolling(252).max()
    low_52w  = df["low"].rolling(252).min()  if "low" in df.columns  else c.rolling(252).min()
    range_52w = high_52w - low_52w + 1e-9
    df["dist_from_52w_high"] = (high_52w - c) / range_52w  # 0=at high, 1=at low
    df["dist_from_52w_low"]  = (c - low_52w) / range_52w   # 0=at low, 1=at high

    # Bollinger band mean-reversion signal
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_pos = (c - sma20) / (2 * std20 + 1e-9)  # -1=lower band, +1=upper band
    df["bb_reversion_signal"] = -bb_pos   # negative of BB position = reversion pressure
    df["outside_bb_upper"]    = (bb_pos > 1.0).astype(np.int8)
    df["outside_bb_lower"]    = (bb_pos < -1.0).astype(np.int8)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b — REGIME FEATURES
# Detects momentum vs mean-reversion market conditions so the model can
# weight signals differently across regimes.
# Computed at the individual stock level (not just market-wide).
# ══════════════════════════════════════════════════════════════════════════════

def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime-detection features. These tell the model WHAT KIND of
    market we're in, so it can apply the right signal type.
    """
    c = df["close"] if "close" in df.columns else None
    if c is None:
        return df

    ret1 = np.log(c / c.shift(1))

    # Trend strength: how consistent is the direction of recent moves?
    # High = trending, Low = choppy/mean-reverting
    for w in [10, 20]:
        sign_sum = np.sign(ret1).rolling(w).sum()
        df[f"trend_consistency_{w}d"] = sign_sum / w  # -1=down trend, +1=up trend

    # Autocorrelation of returns: positive = momentum, negative = mean-reversion
    # Computed as rolling 1-day autocorrelation of returns
    ret_lag1 = ret1.shift(1)
    for w in [20, 60]:
        # Rolling covariance / rolling variance = rolling autocorrelation
        cov  = ret1.rolling(w).cov(ret_lag1)
        var  = ret1.rolling(w).var()
        df[f"ret_autocorr_{w}d"] = cov / (var + 1e-9)
        # Positive = momentum regime, negative = mean-reversion regime

    # Hurst exponent proxy (simplified): compares short vs long vol
    # H > 0.5 = trending (momentum), H < 0.5 = mean-reverting
    vol5  = ret1.rolling(5).std()
    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    # If vol scales with sqrt(time), H=0.5 (random walk)
    # vol5/vol20 > sqrt(5/20) = momentum; < sqrt(5/20) = mean-reversion
    df["hurst_proxy_5_20"]  = vol5  / (vol20 * np.sqrt(5/20)  + 1e-9)
    df["hurst_proxy_20_60"] = vol20 / (vol60 * np.sqrt(20/60) + 1e-9)

    # Market microstructure: bid-ask spread proxy (high-low range / close)
    # Higher = more uncertainty, lower = market confident
    if "high" in df.columns and "low" in df.columns:
        spread = (df["high"] - df["low"]) / (c + 1e-9)
        df["spread_proxy"]     = spread
        df["spread_vs_avg"]    = spread / (spread.rolling(20).mean() + 1e-9)
        df["high_uncertainty"] = (df["spread_vs_avg"] > 1.5).astype(np.int8)

    # Volatility regime: current vol relative to its own history
    vol20_current = ret1.rolling(20).std()
    vol20_hist    = vol20_current.rolling(252).mean()
    df["vol_regime"] = vol20_current / (vol20_hist + 1e-9)
    df["high_vol_regime"] = (df["vol_regime"] > 1.3).astype(np.int8)
    df["low_vol_regime"]  = (df["vol_regime"] < 0.7).astype(np.int8)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6c — EARNINGS REVISION FEATURES
# The #1 most academically validated equity alpha signal.
# When analysts raise EPS estimates, stocks outperform. When they cut, they
# underperform. Computable from yfinance fundamentals data.
# ══════════════════════════════════════════════════════════════════════════════

def _add_earnings_revision_features(df_all: pd.DataFrame,
                                     df_fund: pd.DataFrame | None) -> pd.DataFrame:
    """
    Add earnings revision features from fundamentals data.
    These are static per ticker (snapshot from last collection run)
    but provide strong cross-sectional signal.
    """
    if df_fund is None or df_fund.empty:
        return df_all

    rev_features = []

    for _, row in df_fund.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue

        feats = {"ticker": ticker}

        # EPS revision: forward EPS vs trailing EPS
        # Rising ratio = analysts upgrading expectations
        eps_fwd = row.get("eps_forward")
        eps_ttm = row.get("eps_ttm")
        if eps_fwd and eps_ttm and eps_ttm != 0:
            feats["eps_revision_ratio"] = float(eps_fwd) / (abs(float(eps_ttm)) + 1e-9)
            feats["eps_accelerating"]   = int(float(eps_fwd) > float(eps_ttm))
        else:
            feats["eps_revision_ratio"] = np.nan
            feats["eps_accelerating"]   = np.nan

        # P/E compression signal: forward P/E < trailing P/E = earnings growing
        # faster than price = positive revision signal
        pe_fwd = row.get("pe_forward")
        pe_ttm = row.get("pe_ttm")
        if pe_fwd and pe_ttm and pe_ttm != 0 and pe_fwd > 0:
            feats["pe_compression"] = float(pe_ttm) / (float(pe_fwd) + 1e-9)
            feats["earnings_growing_faster"] = int(float(pe_fwd) < float(pe_ttm))
        else:
            feats["pe_compression"]          = np.nan
            feats["earnings_growing_faster"] = np.nan

        # Revenue growth quality: gross margin stable while revenue grows = quality
        rev_growth = row.get("revenue_growth_yoy")
        gross_margin = row.get("gross_margin") if hasattr(row, "get") else None
        if rev_growth is not None and not (isinstance(rev_growth, float) and np.isnan(rev_growth)):
            feats["strong_revenue_growth"] = int(float(rev_growth) > 0.10)  # >10% YoY
            feats["revenue_growth_yoy_val"] = float(rev_growth)
        else:
            feats["strong_revenue_growth"]  = np.nan
            feats["revenue_growth_yoy_val"] = np.nan

        # Short interest as contrarian signal
        # High short interest + rising price = short squeeze potential
        short_ratio = row.get("short_ratio")
        if short_ratio is not None and not (isinstance(short_ratio, float) and np.isnan(short_ratio)):
            feats["short_ratio_val"]       = float(short_ratio)
            feats["high_short_interest"]   = int(float(short_ratio) > 5)
            feats["extreme_short_interest"]= int(float(short_ratio) > 10)
        else:
            feats["short_ratio_val"]        = np.nan
            feats["high_short_interest"]    = np.nan
            feats["extreme_short_interest"] = np.nan

        # Quality score: profitable + growing + low debt
        net_margin  = row.get("net_margin")
        debt_equity = row.get("debt_equity")
        roe         = row.get("roe")
        quality_score = 0
        if net_margin  and not np.isnan(float(net_margin  if net_margin  else np.nan)): quality_score += int(float(net_margin)  > 0.10)
        if roe         and not np.isnan(float(roe         if roe         else np.nan)): quality_score += int(float(roe)         > 0.15)
        if debt_equity and not np.isnan(float(debt_equity if debt_equity else np.nan)): quality_score += int(float(debt_equity) < 1.0)
        feats["quality_score"] = quality_score  # 0-3

        rev_features.append(feats)

    if not rev_features:
        return df_all

    df_rev = pd.DataFrame(rev_features)
    df_all = df_all.merge(df_rev, on="ticker", how="left")
    log.info(f"Earnings revision features merged: "
             f"{len([c for c in df_rev.columns if c != 'ticker'])} cols")
    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    ticker, group = args
    try:
        feat = _compute_ticker_features(group)
        feat["ticker"] = ticker
        return ticker, feat, "ok"
    except Exception as e:
        return ticker, None, f"error: {e}"


def build_features(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Build the full feature matrix for all tickers.
    Writes to data/features/features.csv.
    Returns the DataFrame.
    """
    from config import TICKER_OVERRIDE, MIN_TICKER_ROWS

    combined_path = RAW_DIR / "combined_ohlcv.csv"
    if not combined_path.exists():
        raise FileNotFoundError("combined_ohlcv.csv not found. Run Phase 1 first.")

    ohlcv = pd.read_csv(combined_path, parse_dates=["date"])
    log.info(f"Loaded combined_ohlcv: {ohlcv.shape}")

    # Filter universe
    # Drop NaN tickers before sorting (blank rows from yfinance can creep in)
    valid_tickers = [t for t in ohlcv["ticker"].unique()
                     if isinstance(t, str) and t.strip()]
    use_tickers = tickers or TICKER_OVERRIDE or sorted(valid_tickers)
    ticker_counts = ohlcv.groupby("ticker").size()
    valid = ticker_counts[ticker_counts >= MIN_TICKER_ROWS].index
    use_tickers = [t for t in use_tickers if t in valid]
    log.info(f"Building features for {len(use_tickers)} tickers ...")

    # Per-ticker technical features (parallel)
    args = [(t, ohlcv[ohlcv["ticker"] == t]) for t in use_tickers]
    frames, ok, errs = [], 0, 0
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(use_tickers))) as ex:
        futs = {ex.submit(_worker, a): a[0] for a in args}
        for i, fut in enumerate(as_completed(futs), 1):
            ticker, df, status = fut.result()
            if status == "ok" and df is not None:
                frames.append(df)
                ok += 1
            else:
                errs += 1
            if i % 100 == 0 or i == len(use_tickers):
                log.info(f"  {i}/{len(use_tickers)} | ok={ok} err={errs}")

    if not frames:
        raise RuntimeError("No features computed. Check OHLCV data.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], format="mixed")
    df_all = df_all.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info(f"Combined: {df_all.shape}")

    # Cross-sectional features
    try:
        df_all = _add_cross_sectional_features(df_all)
    except Exception as e:
        log.warning(f"Cross-sectional features failed: {e}")

    # Macro features (lagged)
    macro = _load_macro_features()
    if macro is not None:
        macro["date"] = pd.to_datetime(macro["date"])
        df_all = df_all.merge(macro, on="date", how="left")
        log.info(f"Macro features merged: {len([c for c in df_all.columns if c.startswith('macro_')])} cols")

    # Fundamental features (static per ticker)
    fund = _load_fundamental_features()
    if fund is not None:
        fund_cols = [c for c in fund.columns if c != "ticker"]
        df_all = df_all.merge(fund, on="ticker", how="left")
        log.info(f"Fundamental features merged: {len(fund_cols)} cols")

    # Earnings proximity
    earn = _load_earnings_features()
    df_all = _add_earnings_proximity(df_all, earn)

    # Earnings revision features (eps revision, pe compression, quality score)
    df_all = _add_earnings_revision_features(df_all, fund)

    # NaN handling: ffill within ticker, then median fill
    log.info("Handling NaNs ...")
    feat_cols = [c for c in df_all.columns if c not in ("date", "ticker")]

    def _fill(g):
        g[feat_cols] = g[feat_cols].ffill(limit=5).bfill()
        return g

    ticker_backup = df_all["ticker"].copy()
    df_all = df_all.groupby("ticker", group_keys=False).apply(_fill)
    if "ticker" not in df_all.columns:
        df_all["ticker"] = ticker_backup.values

    # Fill remaining NaNs with column medians
    medians = df_all[feat_cols].median()
    df_all[feat_cols] = df_all[feat_cols].fillna(medians)

    # Ensure correct types
    df_all["date"] = pd.to_datetime(df_all["date"], format="mixed")

    # Save — NO labels, NO future columns
    out = FEATURES_DIR / "features.csv"
    save_csv(df_all, out, index=False)
    log.info(f"Features saved: {df_all.shape} → {out}")
    log.info(f"Columns: {len(df_all.columns)} total "
             f"({len(feat_cols)} features + date + ticker)")

    # Sanity check — assert no future-derived columns snuck in
    # Use exact names only (substring matching causes false positives on
    # legitimate features like "zscore", "autocorr", etc.)
    LEAKY_EXACT = {
        "future_ret", "future_market_ret", "future_alpha",
        "forward_ret", "forward_alpha", "forward_market_ret",
        "market_forward_ret", "label", "target",
    }
    bad = [c for c in df_all.columns if c in LEAKY_EXACT]
    if bad:
        raise RuntimeError(f"LEAKAGE DETECTED in features.csv: {bad}")
    log.info("Leakage check passed — no future columns in feature matrix")

    return df_all
