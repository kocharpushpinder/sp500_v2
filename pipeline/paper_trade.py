"""
pipeline/paper_trade.py
────────────────────────
Daily paper trading loop.
"""

import json
import pickle
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURES_DIR, MODELS_DIR, PAPER_DIR,
    PAPER_POSITIONS_FILE, PAPER_TRADES_FILE,
    PAPER_EQUITY_FILE, PAPER_CAPITAL,
    MIN_CONFIDENCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    HOLD_DAYS, SLIPPAGE_PCT, MAX_POSITIONS,
    MAX_POSITION_PCT, MAX_SECTOR_PCT, KELLY_FRACTION,
)
from utils.helpers import get_logger

log = get_logger("paper_trade")

# ── State ─────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if PAPER_POSITIONS_FILE.exists():
        with open(PAPER_POSITIONS_FILE) as f:
            state = json.load(f)
        log.info(f"Loaded state: {len(state['positions'])} open positions, "
                 f"cash=${state['cash']:,.2f}")
        return state
    log.info(f"No existing state — starting fresh with ${PAPER_CAPITAL:,}")
    return {"cash": float(PAPER_CAPITAL), "positions": {},
            "started_at": date.today().isoformat(), "last_run": None}

def _save_state(state: dict):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPER_POSITIONS_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def _append_trade(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(PAPER_TRADES_FILE, mode="a",
              header=not PAPER_TRADES_FILE.exists(), index=False)

def _append_equity(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(PAPER_EQUITY_FILE, mode="a",
              header=not PAPER_EQUITY_FILE.exists(), index=False)

# ── Price fetch — batched + retry to avoid rate limiting ─────────────────────

def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch closing prices via yfinance.
    Batches of 50 with 3s delay between batches.
    Retries once per batch after 12s on rate-limit errors.
    """
    if not tickers:
        return {}

    BATCH_SIZE  = 50
    BATCH_DELAY = 3    # seconds between batches
    RETRY_WAIT  = 12   # seconds before retrying a failed batch

    prices  = {}
    batches = [tickers[i:i+BATCH_SIZE]
               for i in range(0, len(tickers), BATCH_SIZE)]
    log.info(f"Fetching {len(tickers)} prices in {len(batches)} batches ...")

    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            _time.sleep(BATCH_DELAY)

        for attempt in range(2):
            try:
                raw   = yf.download(batch, period="3d", interval="1d",
                                    auto_adjust=True, progress=False,
                                    threads=False)
                close = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=batch[0])
                for t in batch:
                    if t in close.columns:
                        s = close[t].dropna()
                        if not s.empty:
                            v = float(s.iloc[-1])
                            if np.isfinite(v) and v > 0:
                                prices[t] = v
                break
            except Exception as e:
                if attempt == 0:
                    log.warning(f"Batch {b_idx+1} rate-limited — "
                                f"retrying in {RETRY_WAIT}s ...")
                    _time.sleep(RETRY_WAIT)
                else:
                    log.error(f"Batch {b_idx+1} failed: {e}")

    log.info(f"Got {len(prices)}/{len(tickers)} prices")
    return prices

# ── Model + signals ───────────────────────────────────────────────────────────

def _load_model():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        art = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return art["model"], feature_cols

def _todays_features() -> pd.DataFrame | None:
    path = FEATURES_DIR / "features.csv"
    if not path.exists():
        log.error("features.csv missing.")
        return None
    feat   = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    latest = feat["date"].max()
    today_feat = feat[feat["date"] == latest]
    log.info(f"Features date: {latest.date()} | tickers: {len(today_feat)}")
    return today_feat

def _generate_signals(model, feature_cols, feat) -> list[dict]:
    avail = [c for c in feature_cols if c in feat.columns]
    if len(avail) < len(feature_cols) * 0.7:
        log.warning(f"Only {len(avail)}/{len(feature_cols)} features available")
        return []
    X = (feat[avail].reindex(columns=feature_cols, fill_value=0)
         .astype(np.float32).fillna(0))
    probs      = model.predict_proba(X)
    pred_class = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    sigs = [{"ticker": str(row["ticker"]), "confidence": float(confidence[i])}
            for i, (_, row) in enumerate(feat.iterrows())
            if pred_class[i] == 2 and confidence[i] >= MIN_CONFIDENCE]
    sigs.sort(key=lambda x: -x["confidence"])
    log.info(f"Signals: {len(sigs)} long above {MIN_CONFIDENCE:.0%} confidence")
    return sigs

# ── Position sizing ───────────────────────────────────────────────────────────

def _size(conf, pv, price, n_open) -> float:
    p      = max(conf, 0.501)
    kelly  = max((p - (1 - p) / 1.5), 0.01)
    target = min(kelly * KELLY_FRACTION, MAX_POSITION_PCT)
    if n_open >= int(MAX_POSITIONS * 0.8):
        target *= 0.5
    return max((pv * target) / (price * (1 + SLIPPAGE_PCT)), 0.0)

# ── Main update ───────────────────────────────────────────────────────────────

def run_paper_update(reset: bool = False) -> dict:
    today     = date.today()
    today_str = today.isoformat()
    log.info(f"\n{'='*54}\nPaper trading update — {today}\n{'='*54}")

    if reset:
        for f in [PAPER_POSITIONS_FILE, PAPER_TRADES_FILE, PAPER_EQUITY_FILE]:
            if f.exists(): f.unlink()
        log.info("State reset.")

    state     = _load_state()
    cash      = float(state["cash"])
    positions = state["positions"]

    # Signals
    try:
        model, feature_cols = _load_model()
        feat    = _todays_features()
        signals = _generate_signals(model, feature_cols, feat) if feat is not None else []
    except Exception as e:
        log.error(f"Signal generation failed: {e}")
        signals = []

    # Prices — wait 5s after feature load to reduce rate-limit risk
    _time.sleep(5)
    all_tickers = list(set(list(positions.keys()) +
                           [s["ticker"] for s in signals]))
    prices = _fetch_prices(all_tickers)

    def pv() -> float:
        return float(cash + sum(
            pos["shares"] * prices.get(tk, pos["entry_price"])
            for tk, pos in positions.items()))

    # ── Exits ─────────────────────────────────────────────────────────────────
    closed_today, to_close = [], []
    for ticker, pos in positions.items():
        price = prices.get(ticker)
        if price is None: continue
        reason = None
        if   price <= pos["stop_loss"]:       reason = "stop_loss"
        elif price >= pos["take_profit"]:     reason = "take_profit"
        elif today_str >= pos["expire_date"]: reason = "hold_expiry"
        if reason:
            to_close.append((ticker, pos, price, reason))

    for ticker, pos, raw_price, reason in to_close:
        ep       = raw_price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * ep
        pnl      = proceeds - pos["cost_basis"]
        pnl_pct  = pnl / pos["cost_basis"] if pos["cost_basis"] > 0 else 0.0
        cash    += proceeds
        del positions[ticker]
        trade = dict(date=today_str, ticker=ticker, action="closed",
                     entry_date=pos["entry_date"],
                     entry_price=round(pos["entry_price"], 4),
                     exit_price=round(ep, 4),
                     shares=round(pos["shares"], 4),
                     pnl=round(pnl, 2),
                     pnl_pct=round(pnl_pct * 100, 2),
                     exit_reason=reason,
                     confidence=pos["confidence"])
        _append_trade(trade)
        closed_today.append(trade)
        log.info(f"  CLOSED {ticker:6s} {reason:12s} ${pnl:+,.2f} ({pnl_pct:+.1%})")

    # ── New positions ──────────────────────────────────────────────────────────
    opened_today = []
    if len(positions) < MAX_POSITIONS and signals:
        pv_now   = pv()
        sect_exp = {}
        for pos in positions.values():
            p = prices.get(pos["ticker"], pos["entry_price"])
            s = pos.get("sector", "Unknown")
            sect_exp[s] = sect_exp.get(s, 0) + pos["shares"] * p

        for sig in signals:
            if len(positions) >= MAX_POSITIONS: break
            ticker = sig["ticker"]
            if ticker in positions: continue
            price = prices.get(ticker)
            if not price or not (np.isfinite(price) and price > 0.01): continue
            sector = sig.get("sector", "Unknown")
            if pv_now > 0 and sect_exp.get(sector, 0) / pv_now > MAX_SECTOR_PCT:
                continue
            shares = _size(sig["confidence"], pv_now, price, len(positions))
            if shares < 0.001: continue
            cost = shares * price * (1 + SLIPPAGE_PCT)
            if cost > cash * 0.98:
                shares = (cash * 0.98) / (price * (1 + SLIPPAGE_PCT))
                cost   = shares * price * (1 + SLIPPAGE_PCT)
            if shares < 0.001 or cost > cash or not np.isfinite(cost): continue
            expire = (pd.Timestamp(today) +
                      pd.offsets.BDay(HOLD_DAYS)).strftime("%Y-%m-%d")
            cash -= cost
            sect_exp[sector] = sect_exp.get(sector, 0) + shares * price
            positions[ticker] = dict(
                ticker=ticker, entry_date=today_str, entry_price=price,
                shares=shares, cost_basis=cost,
                stop_loss=price * (1 - STOP_LOSS_PCT),
                take_profit=price * (1 + TAKE_PROFIT_PCT),
                expire_date=expire, confidence=sig["confidence"],
                sector=sector)
            _append_trade(dict(date=today_str, ticker=ticker, action="buy",
                               entry_date=today_str, entry_price=round(price, 4),
                               exit_price=None, shares=round(shares, 4),
                               pnl=None, pnl_pct=None, exit_reason=None,
                               confidence=sig["confidence"]))
            opened_today.append(ticker)
            log.info(f"  OPENED {ticker:6s} conf={sig['confidence']:.0%} "
                     f"${cost:,.0f} ({shares:.2f}sh @ ${price:.2f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    pv_close = pv()
    state.update({"cash": cash, "positions": positions, "last_run": today_str})
    _save_state(state)
    day_pnl = sum(t["pnl"] for t in closed_today if t["pnl"] is not None)
    _append_equity(dict(date=today_str, portfolio=round(pv_close, 2),
                        cash=round(cash, 2), n_positions=len(positions),
                        n_opened=len(opened_today), n_closed=len(closed_today),
                        day_pnl=round(day_pnl, 2)))

    total_return = (pv_close - PAPER_CAPITAL) / PAPER_CAPITAL
    log.info(f"\n{'─'*54}")
    log.info(f"Portfolio:  ${pv_close:>12,.2f}  ({total_return:+.1%} total)")
    log.info(f"Cash:       ${cash:>12,.2f}")
    log.info(f"Positions:  {len(positions)} open")
    log.info(f"Today:      opened={opened_today}  "
             f"closed={[t['ticker'] for t in closed_today]}")
    if closed_today:
        log.info(f"Day P&L:    ${day_pnl:+,.2f}")
    log.info("Open positions:")
    for tk, pos in positions.items():
        p      = prices.get(tk, pos["entry_price"])
        unreal = (p - pos["entry_price"]) / pos["entry_price"]
        log.info(f"  {tk:6s} entry=${pos['entry_price']:.2f} "
                 f"now=${p:.2f} {unreal:+.1%}  exp={pos['expire_date']}")
    return {"portfolio": pv_close, "day_pnl": day_pnl}

# ── Status ────────────────────────────────────────────────────────────────────

def show_status():
    state     = _load_state()
    positions = state["positions"]
    cash      = float(state["cash"])
    prices    = _fetch_prices(list(positions.keys())) if positions else {}
    pv        = cash + sum(
        pos["shares"] * prices.get(tk, pos["entry_price"])
        for tk, pos in positions.items())
    total_return = (pv - PAPER_CAPITAL) / PAPER_CAPITAL

    print(f"\n{'='*54}")
    print(f"PAPER TRADING STATUS — {date.today()}")
    print(f"{'='*54}")
    print(f"Started:       {state.get('started_at','?')}")
    print(f"Last run:      {state.get('last_run','never')}")
    print(f"Portfolio:     ${pv:,.2f}")
    print(f"Cash:          ${cash:,.2f}")
    print(f"Total return:  {total_return:+.1%}")
    print(f"\nOpen positions ({len(positions)}):")
    for tk, pos in positions.items():
        p      = prices.get(tk, pos["entry_price"])
        unreal = (p - pos["entry_price"]) / pos["entry_price"]
        days   = (date.today() - date.fromisoformat(pos["entry_date"])).days
        print(f"  {tk:6s}  entry=${pos['entry_price']:.2f}  "
              f"now=${p:.2f}  P&L={unreal:+.1%}  "
              f"held={days}d  exp={pos['expire_date']}")
    if PAPER_TRADES_FILE.exists():
        t      = pd.read_csv(PAPER_TRADES_FILE)
        closed = t[(t["action"] == "closed") & t["pnl"].notna()]
        if not closed.empty:
            print(f"\nClosed trades:  {len(closed)}")
            print(f"  Win rate:     {(closed['pnl'] > 0).mean():.1%}")
            print(f"  Avg P&L:      {closed['pnl_pct'].mean():+.1f}%")
            print(f"  Total P&L:    ${closed['pnl'].sum():+,.2f}")
    print(f"{'='*54}\n")
