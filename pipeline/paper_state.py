"""
pipeline/paper_state.py
────────────────────────
Manages persistent paper trading state across daily runs.

State is stored in paper_trading/state.json and updated after each run.
This allows the paper trader to survive restarts, weekends, and holidays.

State includes:
  - Open positions (entry price, shares, stops, etc.)
  - Cash balance
  - Trade history
  - Daily P&L log
  - Performance metrics
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAPER_DIR, PAPER_CAPITAL
from utils.helpers import get_logger

log = get_logger("paper_state")

STATE_FILE  = PAPER_DIR / "state.json"
TRADES_FILE = PAPER_DIR / "trades.csv"
DAILY_FILE  = PAPER_DIR / "daily_pnl.csv"


# ── State schema ──────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "version":        2,
        "started":        datetime.now().isoformat(),
        "last_run":       None,
        "cash":           float(PAPER_CAPITAL),
        "peak_value":     float(PAPER_CAPITAL),
        "positions":      {},      # ticker → position dict
        "trades":         [],      # list of closed trade dicts
        "daily_log":      [],      # list of {date, portfolio, cash, n_pos, daily_ret}
        "n_trading_days": 0,
        "halted":         False,
        "halt_reason":    None,
    }


# ── Load / save ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load state from disk, or create fresh state if none exists."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            log.info(f"State loaded: cash=${state['cash']:,.0f} | "
                     f"{len(state['positions'])} open positions | "
                     f"{len(state['trades'])} closed trades")
            return state
        except Exception as e:
            log.warning(f"Could not load state: {e} — starting fresh")

    log.info(f"No existing state — initialising with ${PAPER_CAPITAL:,} capital")
    return _empty_state()


def save_state(state: dict):
    """Persist state to disk."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def reset_state():
    """Wipe all state and start fresh. Use with caution."""
    if STATE_FILE.exists():
        backup = STATE_FILE.with_suffix(".json.bak")
        STATE_FILE.rename(backup)
        log.info(f"Previous state backed up → {backup}")
    save_state(_empty_state())
    log.info("State reset to fresh start")


# ── Position helpers ──────────────────────────────────────────────────────────

def add_position(state: dict, ticker: str, entry_date: str,
                 entry_price: float, shares: float, cost_basis: float,
                 stop_loss: float, take_profit: float, expire_date: str,
                 confidence: float, sector: str = "Unknown"):
    """Add a new open position to state."""
    state["positions"][ticker] = {
        "ticker":       ticker,
        "entry_date":   entry_date,
        "entry_price":  entry_price,
        "shares":       shares,
        "cost_basis":   cost_basis,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "expire_date":  expire_date,
        "confidence":   confidence,
        "sector":       sector,
    }
    state["cash"] -= cost_basis


def close_position(state: dict, ticker: str, exit_date: str,
                   exit_price: float, reason: str, regime: int = 1):
    """Close a position and record the trade."""
    pos = state["positions"].pop(ticker, None)
    if pos is None:
        return

    from config import SLIPPAGE_PCT
    actual_exit = exit_price * (1 - SLIPPAGE_PCT)
    proceeds    = pos["shares"] * actual_exit
    pnl         = proceeds - pos["cost_basis"]
    pnl_pct     = pnl / pos["cost_basis"] if pos["cost_basis"] > 0 else 0.0

    state["cash"] += proceeds

    trade = {
        "ticker":       ticker,
        "entry_date":   pos["entry_date"],
        "exit_date":    exit_date,
        "entry_price":  round(pos["entry_price"], 4),
        "exit_price":   round(actual_exit, 4),
        "shares":       round(pos["shares"], 4),
        "pnl":          round(pnl, 2),
        "pnl_pct":      round(pnl_pct * 100, 2),
        "exit_reason":  reason,
        "confidence":   pos["confidence"],
        "sector":       pos["sector"],
        "regime":       {0:"bear", 1:"sideways", 2:"bull"}.get(regime, "?"),
    }
    state["trades"].append(trade)

    # Append to trades CSV
    _append_csv(TRADES_FILE, trade)
    return trade


def portfolio_value(state: dict, price_lookup: dict,
                    today: str) -> float:
    """Compute current portfolio value using live prices."""
    val = state["cash"]
    for ticker, pos in state["positions"].items():
        prices = price_lookup.get(ticker, {})
        # Use today's price or most recent available
        price = prices.get(today)
        if price is None:
            # Fallback to latest price available
            available = {k: v for k, v in prices.items() if k <= today}
            price = available[max(available)] if available else pos["entry_price"]
        val += pos["shares"] * float(price)
    return float(val)


# ── Daily log ─────────────────────────────────────────────────────────────────

def record_daily(state: dict, today: str, port_val: float, n_positions: int):
    """Record today's portfolio snapshot."""
    prev_val = state["daily_log"][-1]["portfolio"] if state["daily_log"] else PAPER_CAPITAL
    daily_ret = (port_val - prev_val) / prev_val if prev_val > 0 else 0.0

    entry = {
        "date":       today,
        "portfolio":  round(port_val, 2),
        "cash":       round(state["cash"], 2),
        "n_positions":n_positions,
        "daily_ret":  round(daily_ret * 100, 3),
    }
    state["daily_log"].append(entry)
    state["n_trading_days"] += 1

    # Update peak
    if port_val > state["peak_value"]:
        state["peak_value"] = port_val

    _append_csv(DAILY_FILE, entry)
    return daily_ret


# ── Performance summary ───────────────────────────────────────────────────────

def performance_summary(state: dict, current_value: float) -> dict:
    """Compute current performance metrics."""
    initial    = PAPER_CAPITAL
    total_ret  = (current_value - initial) / initial
    n_days     = state["n_trading_days"]
    n_years    = max(n_days / 252, 0.001)
    cagr       = (1 + total_ret) ** (1 / n_years) - 1

    # Drawdown from peak
    drawdown = (current_value - state["peak_value"]) / state["peak_value"]

    trades = state["trades"]
    n      = len(trades)
    if n > 0:
        wins     = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / n
        avg_win  = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        losers   = [t for t in trades if t["pnl"] <= 0]
        avg_loss = np.mean([t["pnl_pct"] for t in losers]) if losers else 0
        total_pnl = sum(t["pnl"] for t in trades)
    else:
        win_rate = avg_win = avg_loss = total_pnl = 0

    # Sharpe from daily log
    if len(state["daily_log"]) > 5:
        rets   = np.array([d["daily_ret"] / 100 for d in state["daily_log"]])
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    else:
        sharpe = 0.0

    return {
        "current_value":  current_value,
        "total_return":   total_ret,
        "cagr":           cagr,
        "drawdown":       drawdown,
        "sharpe":         sharpe,
        "n_trades":       n,
        "win_rate":       win_rate,
        "avg_win_pct":    avg_win,
        "avg_loss_pct":   avg_loss,
        "total_pnl":      total_pnl,
        "n_open":         len(state["positions"]),
        "n_trading_days": n_days,
    }


# ── Halt logic ────────────────────────────────────────────────────────────────

def check_halts(state: dict, port_val: float, daily_ret: float) -> bool:
    """
    Check if trading should be halted due to risk limits.
    Returns True if halted.
    """
    from config import DAILY_LOSS_LIMIT, MAX_DRAWDOWN_HALT

    if state.get("halted"):
        return True

    drawdown = (port_val - state["peak_value"]) / state["peak_value"]

    if daily_ret < -DAILY_LOSS_LIMIT:
        state["halted"] = True
        state["halt_reason"] = f"Daily loss {daily_ret:.1%} exceeded limit {-DAILY_LOSS_LIMIT:.1%}"
        log.warning(f"TRADING HALTED: {state['halt_reason']}")
        return True

    if drawdown < -MAX_DRAWDOWN_HALT:
        state["halted"] = True
        state["halt_reason"] = f"Drawdown {drawdown:.1%} exceeded limit {-MAX_DRAWDOWN_HALT:.1%}"
        log.warning(f"TRADING HALTED: {state['halt_reason']}")
        return True

    return False


# ── CSV helper ────────────────────────────────────────────────────────────────

def _append_csv(path: Path, row: dict):
    """Append one row to a CSV, writing header if file is new."""
    import csv
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
