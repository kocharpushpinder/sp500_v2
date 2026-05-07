"""
run_scheduler.py
─────────────────
Intraday scheduler. Runs every 15 minutes during market hours.

Each 15-minute tick (9:30am–4:00pm ET):
  - Fetch latest prices for open positions
  - Check exits: stop-loss, take-profit, hold expiry

Once at 4:05pm ET:
  - Collect data → rebuild features → generate signals → open positions

Usage:
    nohup .venv/bin/python run_scheduler.py >> logs/scheduler.log 2>&1 &
Stop:
    kill $(cat logs/scheduler.pid)
"""

import os
import sys
import time
import signal
import subprocess
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("scheduler")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | scheduler | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    log.addHandler(_sh)

ET           = ZoneInfo("America/New_York")
MARKET_OPEN  = (9,  30)
MARKET_CLOSE = (16,  0)
EOD_COLLECT  = (16,  5)
INTERVAL_MIN = 15
PID_FILE     = LOG_DIR / "scheduler.pid"
PYTHON       = sys.executable

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    log.info(f"Signal {sig} — shutting down")
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

def _now() -> datetime:
    return datetime.now(ET)

def _is_trading_day(d: date = None) -> bool:
    d = d or _now().date()
    if d.weekday() >= 5:
        return False
    year = d.year
    fixed = {date(year,1,1), date(year,7,4), date(year,12,25)}
    adjusted = set()
    for h in fixed:
        if   h.weekday() == 5: adjusted.add(h - timedelta(days=1))
        elif h.weekday() == 6: adjusted.add(h + timedelta(days=1))
        else:                  adjusted.add(h)
    thursdays = [date(year,11,day) for day in range(1,31)
                 if date(year,11,day).weekday() == 3]
    if len(thursdays) >= 4:
        adjusted.add(thursdays[3])
    return d not in adjusted

def _market_is_open(now=None) -> bool:
    now = now or _now()
    return _is_trading_day(now.date()) and MARKET_OPEN <= (now.hour, now.minute) < MARKET_CLOSE

def _is_eod_window(now=None) -> bool:
    now = now or _now()
    t = (now.hour, now.minute)
    return _is_trading_day(now.date()) and EOD_COLLECT <= t < (EOD_COLLECT[0], EOD_COLLECT[1] + INTERVAL_MIN)

def _sleep(seconds: float):
    end = time.monotonic() + seconds
    while not _shutdown and time.monotonic() < end:
        time.sleep(min(5, end - time.monotonic()))

def _sleep_until_market_open():
    while not _shutdown:
        now   = _now()
        today = now.date()
        if _is_trading_day(today):
            open_today = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1],
                                     second=0, microsecond=0)
            if now < open_today:
                wait = (open_today - now).total_seconds()
                log.info(f"Market opens in {int(wait//3600)}h {int((wait%3600)//60)}m")
                _sleep(min(wait, 3600))
                continue
            return
        next_day = today + timedelta(days=1)
        while not _is_trading_day(next_day):
            next_day += timedelta(days=1)
        next_open = datetime(next_day.year, next_day.month, next_day.day,
                             MARKET_OPEN[0], MARKET_OPEN[1], tzinfo=ET)
        wait = (next_open - now).total_seconds()
        log.info(f"Next trading day: {next_day} — sleeping {int(wait//3600)}h")
        _sleep(min(wait, 3600))

def _run(args: list, label: str) -> bool:
    log.info(f"  [{label}] starting ...")
    t0 = time.time()
    try:
        # Explicitly open stdout/stderr so child processes don't inherit
        # nohup's redirected (and potentially broken) file descriptors.
        log_path = LOG_DIR / "scheduler.log"
        with open(log_path, "a") as log_fd:
            r = subprocess.run(
                [PYTHON] + args,
                cwd=str(Path(__file__).parent),
                stdout=log_fd,
                stderr=log_fd,
                timeout=3600,
            )
        elapsed = time.time() - t0
        if r.returncode == 0:
            log.info(f"  [{label}] done in {elapsed:.0f}s")
            return True
        log.error(f"  [{label}] FAILED (exit {r.returncode})")
        return False
    except Exception as e:
        log.error(f"  [{label}] ERROR: {e}")
        return False

def _run_intraday_exits():
    """Fetch intraday prices and check exits for open positions."""
    import json
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import time as _t

    from config import (
        PAPER_POSITIONS_FILE, PAPER_TRADES_FILE,
        PAPER_EQUITY_FILE, PAPER_DIR,
        STOP_LOSS_PCT, TAKE_PROFIT_PCT, SLIPPAGE_PCT,
    )

    if not PAPER_POSITIONS_FILE.exists():
        return

    with open(PAPER_POSITIONS_FILE) as f:
        state = json.load(f)

    positions = state["positions"]
    if not positions:
        return

    # Fetch intraday prices — small batch, just open positions
    tickers = list(positions.keys())
    prices  = {}
    RETRY_WAIT = 12

    for attempt in range(2):
        try:
            raw   = yf.download(tickers, period="1d", interval="1m",
                                auto_adjust=True, progress=False, threads=False)
            close = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0])
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        v = float(s.iloc[-1])
                        if np.isfinite(v) and v > 0:
                            prices[t] = v
            break
        except Exception as e:
            if attempt == 0:
                log.warning(f"Intraday price fetch failed ({e}) — retrying in {RETRY_WAIT}s")
                _t.sleep(RETRY_WAIT)
            else:
                log.error(f"Intraday price fetch failed twice: {e}")
                return

    if not prices:
        log.warning("No intraday prices fetched — skipping exit check")
        return

    today_str = date.today().isoformat()
    cash      = float(state["cash"])
    closed    = []

    for ticker, pos in list(positions.items()):
        price = prices.get(ticker)
        if not price: continue
        reason = None
        if   price <= pos["stop_loss"]:       reason = "stop_loss"
        elif price >= pos["take_profit"]:     reason = "take_profit"
        elif today_str >= pos["expire_date"]: reason = "hold_expiry"
        if not reason: continue

        ep       = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * ep
        pnl      = proceeds - pos["cost_basis"]
        pnl_pct  = pnl / pos["cost_basis"] if pos["cost_basis"] > 0 else 0.0
        cash    += proceeds
        del positions[ticker]

        row = dict(date=today_str, ticker=ticker, action="closed",
                   entry_date=pos["entry_date"],
                   entry_price=round(pos["entry_price"], 4),
                   exit_price=round(ep, 4),
                   shares=round(pos["shares"], 4),
                   pnl=round(pnl, 2), pnl_pct=round(pnl_pct*100, 2),
                   exit_reason=reason, confidence=pos["confidence"])

        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        df.to_csv(PAPER_TRADES_FILE, mode="a",
                  header=not PAPER_TRADES_FILE.exists(), index=False)
        closed.append(row)
        log.info(f"  INTRADAY EXIT {ticker:6s} {reason:12s} "
                 f"${pnl:+,.2f} ({pnl_pct:+.1%})")

    if closed:
        state["cash"]      = cash
        state["positions"] = positions
        state["last_run"]  = today_str
        with open(PAPER_POSITIONS_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        pv = cash + sum(
            pos["shares"] * prices.get(tk, pos["entry_price"])
            for tk, pos in positions.items())
        now_str = f"{today_str} {_now().strftime('%H:%M')}"
        snap = pd.DataFrame([dict(date=now_str, portfolio=round(pv,2),
                                  cash=round(cash,2),
                                  n_positions=len(positions),
                                  n_opened=0, n_closed=len(closed),
                                  day_pnl=round(sum(r["pnl"] for r in closed),2))])
        snap.to_csv(PAPER_EQUITY_FILE, mode="a",
                    header=not PAPER_EQUITY_FILE.exists(), index=False)

def _run_eod():
    log.info("=" * 54)
    log.info(f"EOD PIPELINE — {_now().strftime('%Y-%m-%d %H:%M %Z')}")
    log.info("=" * 54)
    ok  = _run(["run_collection.py"],              "Data collection")
    ok &= _run(["run_pipeline.py", "--features"],  "Feature rebuild")
    ok &= _run(["run_pipeline.py", "--paper"],     "Paper trading")
    log.info(f"EOD pipeline {'OK' if ok else 'PARTIAL FAILURE'}")

def main():
    PID_FILE.write_text(str(os.getpid()))
    log.info(f"Scheduler started (PID {os.getpid()})")
    log.info(f"Intraday exits every {INTERVAL_MIN} min | EOD pipeline at "
             f"{EOD_COLLECT[0]}:{EOD_COLLECT[1]:02d}pm ET")
    log.info(f"Python: {PYTHON}")
    log.info(f"Stop:   kill $(cat logs/scheduler.pid)")

    _eod_done_today: date | None = None

    while not _shutdown:
        now   = _now()
        today = now.date()

        if not _is_trading_day(today) or (now.hour, now.minute) < MARKET_OPEN:
            _sleep_until_market_open()
            continue

        if (now.hour, now.minute) >= (EOD_COLLECT[0], EOD_COLLECT[1] + INTERVAL_MIN):
            if _eod_done_today != today:
                log.warning("EOD window missed — running now")
                _run_eod()
                _eod_done_today = today
            _sleep_until_market_open()
            continue

        if _is_eod_window(now) and _eod_done_today != today:
            _run_eod()
            _eod_done_today = today
            _sleep(INTERVAL_MIN * 60)
            continue

        log.info(f"Intraday check — {now.strftime('%H:%M %Z')} | "
                 f"{len(__import__('json').load(open(__import__('pathlib').Path(__file__).parent/'paper_trading'/'positions.json')) ['positions']) if (__import__('pathlib').Path(__file__).parent/'paper_trading'/'positions.json').exists() else 0} open positions")
        try:
            _run_intraday_exits()
        except Exception as e:
            log.error(f"Intraday exits error: {e}")

        _sleep(INTERVAL_MIN * 60)

    log.info("Scheduler stopped")
    if PID_FILE.exists():
        PID_FILE.unlink()

if __name__ == "__main__":
    main()
