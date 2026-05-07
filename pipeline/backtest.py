"""
pipeline/backtest.py
─────────────────────
Event-driven backtester. Long-only (Wealthsimple), daily bars.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURES_DIR, BACKTEST_DIR, MODELS_DIR, RAW_DIR, MACRO_DIR,
    INITIAL_CAPITAL, MAX_POSITION_PCT, MAX_SECTOR_PCT, MAX_POSITIONS,
    KELLY_FRACTION, STOP_LOSS_PCT, TAKE_PROFIT_PCT, HOLD_DAYS,
    SLIPPAGE_PCT, MIN_CONFIDENCE,
)
from utils.helpers import get_logger

log = get_logger("backtest")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:      str
    entry_date:  pd.Timestamp
    entry_price: float          # always a valid float > 0
    shares:      float          # always a valid float > 0
    cost_basis:  float
    stop_loss:   float
    take_profit: float
    expire_date: pd.Timestamp
    confidence:  float
    sector:      str = "Unknown"

@dataclass
class Trade:
    ticker:      str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      float
    pnl:         float
    pnl_pct:     float
    exit_reason: str
    confidence:  float
    sector:      str  = "Unknown"
    regime:      int  = 1


# ── Price lookup ──────────────────────────────────────────────────────────────

def _build_price_lookup(ohlcv: pd.DataFrame) -> dict:
    """
    Build a {ticker: {date_str: price}} dict for O(1) lookups.
    All dates normalised to 'YYYY-MM-DD' strings.
    """
    log.info("Building price lookup table ...")
    # Robust to CSV/object dtypes: enforce datetimelike before .dt usage.
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce").dt.normalize()
    ohlcv = ohlcv.dropna(subset=["date"])
    lookup = {}
    for ticker, grp in ohlcv.groupby("ticker"):
        grp = grp.sort_values("date")
        dates = grp["date"].dt.strftime("%Y-%m-%d").values
        prices = grp["close"].values
        lookup[ticker] = dict(zip(dates, prices))
    log.info(f"Price lookup: {len(lookup)} tickers")
    return lookup


def _get_price(lookup: dict, ticker: str, date: pd.Timestamp) -> float | None:
    """
    Get closing price for ticker on date.
    Falls back to the most recent prior date if exact date missing.
    Returns None only if ticker has NO data at all before this date.
    Never returns NaN — always returns a valid float or None.
    """
    d = date.strftime("%Y-%m-%d")
    ticker_data = lookup.get(ticker)
    if not ticker_data:
        return None

    # Exact match
    val = ticker_data.get(d)
    if val is not None and np.isfinite(float(val)) and float(val) > 0:
        return float(val)

    # Most recent prior date
    keys = sorted(k for k in ticker_data if k <= d)
    if not keys:
        return None
    val = ticker_data[keys[-1]]
    if val is not None and np.isfinite(float(val)) and float(val) > 0:
        return float(val)
    return None


# ── Position sizing ───────────────────────────────────────────────────────────

def _size_position(confidence: float, portfolio_val: float,
                   price: float, n_open: int) -> float:
    """
    Quarter Kelly position sizing. Returns shares (always >= 0).
    Never returns NaN — all inputs validated before calling.
    """
    p       = max(float(confidence), 0.501)
    odds    = 1.5
    kelly   = max((p - (1 - p) / odds), 0.01)
    target  = min(kelly * KELLY_FRACTION, MAX_POSITION_PCT)

    # Scale down near position limit
    if n_open >= int(MAX_POSITIONS * 0.8):
        target *= 0.5

    position_value = portfolio_val * target
    shares = position_value / (price * (1 + SLIPPAGE_PCT))
    return max(float(shares), 0.0)


# ── Regime classifier ─────────────────────────────────────────────────────────

def _build_regime_series(macro: pd.DataFrame) -> dict:
    """Returns {date_str: regime_int} mapping."""
    if "sp500" not in macro.columns:
        return {}
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
    macro = macro.dropna(subset=["date"])
    macro = macro.sort_values("date").copy()
    sp    = macro["sp500"]
    macro["sma200"]  = sp.rolling(200).mean()
    macro["ret60"]   = sp.pct_change(60)
    macro["regime"]  = 1
    macro.loc[(sp > macro["sma200"]) & (macro["ret60"] >  0.05), "regime"] = 2
    macro.loc[(sp < macro["sma200"]) & (macro["ret60"] < -0.05), "regime"] = 0
    return dict(zip(macro["date"].dt.strftime("%Y-%m-%d"), macro["regime"]))


# ── Signal generation ─────────────────────────────────────────────────────────

def _signals_for_date(model, feature_cols: list,
                      features_today: pd.DataFrame,
                      min_conf: float) -> list[dict]:
    """Generate long-only signals for one trading day."""
    if features_today.empty:
        return []

    avail = [c for c in feature_cols if c in features_today.columns]
    if len(avail) < len(feature_cols) * 0.7:
        return []

    X = features_today[avail].copy().reindex(columns=feature_cols, fill_value=0)
    X = X.astype(np.float32).fillna(0)

    probs      = model.predict_proba(X)      # (n, 3): [short, flat, long]
    pred_class = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)

    out = []
    for i, (_, row) in enumerate(features_today.iterrows()):
        if pred_class[i] == 2 and confidence[i] >= min_conf:  # long only
            out.append({
                "ticker":     str(row["ticker"]),
                "confidence": float(confidence[i]),
            })
    return out


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(start_date: str = "2019-01-01",
                 end_date:   str = None,
                 capital:    float = INITIAL_CAPITAL,
                 min_conf:   float = MIN_CONFIDENCE) -> dict:

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading data ...")

    feat_path = FEATURES_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Run: python run_pipeline.py --features")

    features = pd.read_csv(feat_path, parse_dates=["date"])
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    features = features.dropna(subset=["date"])
    features = features.sort_values(["date", "ticker"])

    ohlcv = pd.read_csv(RAW_DIR / "combined_ohlcv.csv", parse_dates=["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce").dt.normalize()
    ohlcv = ohlcv.dropna(subset=["date"])
    price_lookup = _build_price_lookup(ohlcv)

    macro  = pd.read_csv(MACRO_DIR / "macro_daily.csv", parse_dates=["date"])
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
    macro = macro.dropna(subset=["date"])
    regime_map = _build_regime_series(macro)

    sector_map: dict[str, str] = {}
    sect_path = FEATURES_DIR.parent / "fundamentals" / "sectors.csv"
    if sect_path.exists():
        s = pd.read_csv(sect_path)
        sector_map = dict(zip(s["ticker"], s["sector"]))

    import pickle
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        art = pickle.load(f)
    model = art["model"]
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    log.info(f"Model loaded: {len(feature_cols)} features")

    # ── Validate price lookup ─────────────────────────────────────────────────
    test_tickers = list(ohlcv["ticker"].unique())[:5]
    test_date    = pd.Timestamp(start_date)
    hits = sum(1 for t in test_tickers if _get_price(price_lookup, t, test_date))
    log.info(f"Price lookup check: {hits}/{len(test_tickers)} tickers found on {start_date}")
    if hits == 0:
        log.warning("No prices found — check OHLCV coverage")

    # ── Setup ──────────────────────────────────────────────────────────────────
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date) if end_date else features["date"].max()

    trading_dates = sorted(
        features[(features["date"] >= start_dt) & (features["date"] <= end_dt)]
        ["date"].unique()
    )
    log.info(f"Backtest: {start_dt.date()} → {end_dt.date()} "
             f"({len(trading_dates)} trading days)")

    cash:      float = capital
    positions: dict[str, Position] = {}
    trades:    list[Trade] = []
    equity:    list[dict]  = []
    rets:      list[float] = []
    prev_val:  float = capital

    def port_val() -> float:
        """Portfolio value — fully NaN-safe."""
        val = cash
        for pos in positions.values():
            p = _get_price(price_lookup, pos.ticker, date)
            # Fallback to entry price if live price unavailable
            p = p if (p and np.isfinite(p)) else pos.entry_price
            val += pos.shares * p
        return float(val) if np.isfinite(val) else prev_val

    # ── Main loop ──────────────────────────────────────────────────────────────
    for i, date in enumerate(trading_dates):
        date_str = date.strftime("%Y-%m-%d")
        regime   = int(regime_map.get(date_str, 1))

        # ── Exits ─────────────────────────────────────────────────────────────
        to_close: list[tuple] = []
        for ticker, pos in positions.items():
            p = _get_price(price_lookup, ticker, date)
            if p is None:
                continue
            reason = None
            if   p <= pos.stop_loss:   reason = "stop_loss"
            elif p >= pos.take_profit: reason = "take_profit"
            elif date >= pos.expire_date: reason = "hold_expiry"
            if reason:
                to_close.append((ticker, pos, p, reason))

        for ticker, pos, raw_price, reason in to_close:
            ep   = float(raw_price) * (1 - SLIPPAGE_PCT)
            proc = pos.shares * ep
            pnl  = proc - pos.cost_basis
            pnl_pct = pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0
            cash += proc
            del positions[ticker]
            trades.append(Trade(
                ticker=ticker, entry_date=pos.entry_date,
                exit_date=date, entry_price=pos.entry_price,
                exit_price=ep, shares=pos.shares,
                pnl=float(pnl), pnl_pct=float(pnl_pct),
                exit_reason=reason, confidence=pos.confidence,
                sector=pos.sector, regime=regime,
            ))

        # ── New signals ────────────────────────────────────────────────────────
        if len(positions) < MAX_POSITIONS:
            today_feat = features[features["date"] == date]
            sigs = _signals_for_date(model, feature_cols, today_feat, min_conf)
            sigs = sorted(sigs, key=lambda x: -x["confidence"])

            pv_now = port_val()

            # Sector exposure for concentration check
            sect_exp: dict[str, float] = {}
            for pos in positions.values():
                p = _get_price(price_lookup, pos.ticker, date) or pos.entry_price
                sect_exp[pos.sector] = sect_exp.get(pos.sector, 0) + pos.shares * p

            for sig in sigs:
                if len(positions) >= MAX_POSITIONS:
                    break

                ticker = sig["ticker"]
                if ticker in positions:
                    continue

                # CRITICAL: validate price before ANY computation
                price = _get_price(price_lookup, ticker, date)
                if price is None:
                    continue     # no data for this ticker on this date
                if not (np.isfinite(price) and price > 0.01):
                    continue

                sector = sector_map.get(ticker, "Unknown")

                # Sector cap
                sv = sect_exp.get(sector, 0)
                if pv_now > 0 and sv / pv_now > MAX_SECTOR_PCT:
                    continue

                # Size: use validated portfolio value
                if not (np.isfinite(pv_now) and pv_now > 0):
                    break   # portfolio value corrupted — stop trading

                shares = _size_position(sig["confidence"], pv_now, price,
                                        len(positions))
                if shares < 0.001:
                    continue

                cost = shares * price * (1 + SLIPPAGE_PCT)
                if cost > cash * 0.98:
                    shares = (cash * 0.98) / (price * (1 + SLIPPAGE_PCT))
                    cost   = shares * price * (1 + SLIPPAGE_PCT)
                if shares < 0.001 or cost > cash or cost <= 0:
                    continue

                # Final NaN guard before committing
                if not (np.isfinite(shares) and np.isfinite(cost)):
                    continue

                cash -= cost
                sect_exp[sector] = sect_exp.get(sector, 0) + shares * price

                positions[ticker] = Position(
                    ticker      = ticker,
                    entry_date  = date,
                    entry_price = price,
                    shares      = shares,
                    cost_basis  = cost,
                    stop_loss   = price * (1 - STOP_LOSS_PCT),
                    take_profit = price * (1 + TAKE_PROFIT_PCT),
                    expire_date = date + pd.offsets.BDay(HOLD_DAYS),
                    confidence  = sig["confidence"],
                    sector      = sector,
                )

        # ── Record ─────────────────────────────────────────────────────────────
        pv = port_val()
        if np.isfinite(pv) and prev_val > 0:
            ret = (pv - prev_val) / prev_val
        else:
            ret = 0.0
        rets.append(ret)
        prev_val = pv if np.isfinite(pv) else prev_val

        equity.append({"date": date, "portfolio": pv, "cash": cash,
                       "n_positions": len(positions), "regime": regime})

        if i % 60 == 0:
            log.info(f"  {date.date()} | ${pv:,.0f} | "
                     f"pos={len(positions)} | trades={len(trades)}")

    # ── Close remaining ────────────────────────────────────────────────────────
    final_date = trading_dates[-1] if trading_dates else end_dt
    for ticker, pos in list(positions.items()):
        p = _get_price(price_lookup, ticker, final_date) or pos.entry_price
        ep   = float(p) * (1 - SLIPPAGE_PCT)
        proc = pos.shares * ep
        pnl  = proc - pos.cost_basis
        cash += proc
        trades.append(Trade(
            ticker=ticker, entry_date=pos.entry_date,
            exit_date=final_date, entry_price=pos.entry_price,
            exit_price=ep, shares=pos.shares,
            pnl=float(pnl), pnl_pct=float(pnl/pos.cost_basis) if pos.cost_basis>0 else 0,
            exit_reason="end_of_backtest", confidence=pos.confidence,
            sector=pos.sector, regime=1,
        ))

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = _metrics(trades, equity, rets, capital, start_dt, end_dt)
    _save(metrics, equity, trades)
    _print(metrics)
    return metrics


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(trades, equity, rets, capital, start_dt, end_dt) -> dict:
    eq   = pd.DataFrame(equity)
    fv   = float(eq["portfolio"].iloc[-1]) if not eq.empty else capital
    fv   = fv if np.isfinite(fv) else capital

    n_yrs = max((end_dt - start_dt).days / 365.25, 0.01)
    tr    = (fv - capital) / capital
    cagr  = (1 + tr) ** (1 / n_yrs) - 1

    r = np.array([x for x in rets if np.isfinite(x)])
    sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(252)) if len(r) > 1 else 0.0

    pvs  = eq["portfolio"].values if not eq.empty else np.array([capital])
    pvs  = np.where(np.isfinite(pvs), pvs, capital)
    peak = np.maximum.accumulate(pvs)
    dd   = (pvs - peak) / (peak + 1e-9)
    mdd  = float(dd.min())
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    n    = len(trades)
    if n == 0:
        return {"error": "no trades", "n_trades": 0}

    wins  = [t for t in trades if np.isfinite(t.pnl) and t.pnl > 0]
    loss  = [t for t in trades if np.isfinite(t.pnl) and t.pnl <= 0]
    wr    = len(wins) / n
    aw    = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    al    = float(np.mean([t.pnl_pct for t in loss])) if loss else 0.0
    gross_win  = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in loss))
    pf    = gross_win / max(gross_loss, 1e-9)

    hold  = float(np.mean([(t.exit_date - t.entry_date).days for t in trades]))

    exits: dict = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    rp: dict = {}
    for rid, rname in [(0,"bear"),(1,"sideways"),(2,"bull")]:
        rt = [t for t in trades if t.regime == rid]
        if rt:
            rw = [t for t in rt if np.isfinite(t.pnl) and t.pnl > 0]
            rp[rname] = {
                "n_trades": len(rt),
                "win_rate": len(rw)/len(rt),
                "avg_pnl_pct": float(np.nanmean([t.pnl_pct for t in rt])),
                "total_pnl": float(sum(t.pnl for t in rt if np.isfinite(t.pnl))),
            }

    # Monthly
    bm = wm = 0.0
    pm = nm = 0
    if not eq.empty:
        eq["m"] = pd.to_datetime(eq["date"]).dt.to_period("M")
        mo = eq.groupby("m")["portfolio"].last().pct_change().dropna()
        mo = mo[np.isfinite(mo)]
        if len(mo):
            bm = float(mo.max()); wm = float(mo.min())
            pm = int((mo>0).sum()); nm = int((mo<=0).sum())

    return dict(
        initial_capital=capital, final_value=fv,
        total_return=tr, cagr=cagr, n_years=n_yrs,
        sharpe=sharpe, calmar=calmar, max_drawdown=mdd,
        volatility=float(r.std()*np.sqrt(252)) if len(r)>1 else 0,
        n_trades=n, win_rate=wr, avg_winner_pct=aw, avg_loser_pct=al,
        profit_factor=pf, avg_hold_days=hold, exit_reasons=exits,
        best_month=bm, worst_month=wm, positive_months=pm, negative_months=nm,
        regime_performance=rp,
        start_date=str(start_dt.date()), end_date=str(end_dt.date()),
        config=dict(min_confidence=MIN_CONFIDENCE, stop_loss=STOP_LOSS_PCT,
                    take_profit=TAKE_PROFIT_PCT, hold_days=HOLD_DAYS,
                    kelly=KELLY_FRACTION, max_pos=MAX_POSITIONS),
    )


def _save(metrics, equity, trades):
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(BACKTEST_DIR/"results.json","w") as f:
        json.dump(metrics, f, indent=2, default=str)
    pd.DataFrame(equity).to_csv(BACKTEST_DIR/"equity_curve.csv", index=False)
    if trades:
        pd.DataFrame([dict(
            ticker=t.ticker,
            entry_date=str(t.entry_date.date()),
            exit_date=str(t.exit_date.date()),
            entry_price=round(t.entry_price,4),
            exit_price=round(t.exit_price,4),
            shares=round(t.shares,4),
            pnl=round(t.pnl,2),
            pnl_pct=round(t.pnl_pct*100,2),
            exit_reason=t.exit_reason,
            confidence=round(t.confidence,3),
            sector=t.sector,
            regime={0:"bear",1:"sideways",2:"bull"}.get(t.regime,"?"),
        ) for t in trades]).to_csv(BACKTEST_DIR/"trades.csv", index=False)
    log.info(f"Saved → {BACKTEST_DIR}")


def _print(m):
    log.info("\n" + "="*56)
    log.info("BACKTEST RESULTS")
    log.info("="*56)
    log.info(f"Period:       {m['start_date']} → {m['end_date']} ({m['n_years']:.1f}y)")
    log.info(f"Capital:      ${m['initial_capital']:,.0f} → ${m['final_value']:,.0f}")
    log.info(f"Total return: {m['total_return']:.1%}  |  CAGR: {m['cagr']:.1%}")
    log.info(f"Sharpe: {m['sharpe']:.2f}  Calmar: {m['calmar']:.2f}  "
             f"Max DD: {m['max_drawdown']:.1%}  Vol: {m['volatility']:.1%}")
    log.info(f"Trades: {m['n_trades']}  Win: {m['win_rate']:.1%}  "
             f"Avg W: +{m['avg_winner_pct']:.1%}  Avg L: {m['avg_loser_pct']:.1%}  "
             f"PF: {m['profit_factor']:.2f}")
    log.info(f"Avg hold: {m['avg_hold_days']:.1f}d  "
             f"+months: {m['positive_months']}  -months: {m['negative_months']}")
    log.info(f"Exits: {m['exit_reasons']}")
    log.info("Regime:")
    for r, p in m.get("regime_performance",{}).items():
        log.info(f"  {r:9s}: {p['n_trades']:4d} trades | "
                 f"win={p['win_rate']:.1%} | avg={p['avg_pnl_pct']:.1%} | "
                 f"total=${p['total_pnl']:,.0f}")
    log.info("="*56)
    targets = [
        ("Sharpe > 1.5",       m["sharpe"]       > 1.5),
        ("Calmar > 1.0",       m["calmar"]        > 1.0),
        ("Max DD > -20%",      m["max_drawdown"]  > -0.20),
        ("Win rate > 52%",     m["win_rate"]      > 0.52),
        ("Profit factor >1.3", m["profit_factor"] > 1.3),
        ("CAGR > 10%",         m["cagr"]          > 0.10),
    ]
    all_pass = all(p for _, p in targets)
    for name, passed in targets:
        log.info(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log.info("→ READY FOR PAPER TRADING" if all_pass else "→ Review regime breakdown")
