"""
reconcile_paper_trading.py
───────────────────────────
Fixes the paper trading state after the sell→closed transition:
  1. Deduplicates trades.csv (keeps "closed" over "sell" for same ticker+entry_date)
  2. Removes any tickers from positions.json that already have a closed exit in trades.csv
  3. Prints a clean summary so you can verify before anything is saved

Run from project root: python reconcile_paper_trading.py
"""
import json
import shutil
import pandas as pd
from datetime import date
from pathlib import Path

BASE    = Path("/Users/sinder/PycharmProjects/sp500_agent_V2")
PAPER   = BASE / "paper_trading"
POS_F   = PAPER / "positions.json"
TRADES_F = PAPER / "trades.csv"

# ── Load current state ────────────────────────────────────────────────────────
with open(POS_F) as f:
    state = json.load(f)

trades = pd.read_csv(TRADES_F)
print(f"Loaded {len(trades)} trade rows, {len(state['positions'])} open positions\n")

# ── Step 1: Deduplicate exits ─────────────────────────────────────────────────
# For each (ticker, entry_date) pair, keep only ONE exit row.
# Priority: "closed" > "sell" (closed is the correct label)
# If both exist, keep "closed" and drop "sell".
# If only "sell" exists, rename it to "closed".

buys  = trades[trades["action"] == "buy"].copy()
exits = trades[trades["action"].isin(["closed", "sell"])].copy()

print(f"Buy rows: {len(buys)}")
print(f"Exit rows before dedup: {len(exits)}")

# For each (ticker, entry_date), prefer "closed" over "sell"
exits["priority"] = exits["action"].map({"closed": 0, "sell": 1})
exits = (exits
         .sort_values("priority")
         .drop_duplicates(subset=["ticker", "entry_date"], keep="first")
         .drop(columns=["priority"])
         .copy())

# Normalise all remaining exit actions to "closed"
exits["action"] = "closed"
print(f"Exit rows after dedup: {len(exits)}")

# Rebuild clean trades df
clean_trades = pd.concat([buys, exits], ignore_index=True)
clean_trades = clean_trades.sort_values(["date", "action"], ascending=[True, False])

# ── Step 2: Identify which positions are already closed ───────────────────────
closed_keys = set(zip(exits["ticker"], exits["entry_date"].astype(str)))
print(f"\nClosed (ticker, entry_date) pairs:")
for k in sorted(closed_keys):
    print(f"  {k[0]:6s}  entered {k[1]}")

still_open = {}
removed    = []
for ticker, pos in state["positions"].items():
    key = (ticker, pos["entry_date"])
    if key in closed_keys:
        removed.append(ticker)
    else:
        still_open[ticker] = pos

print(f"\nRemoving from positions.json (already closed): {removed}")
print(f"Keeping as open: {list(still_open.keys())}")

# ── Step 3: Preview and save ──────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("CHANGES TO BE SAVED:")
print(f"  trades.csv:      {len(trades)} rows → {len(clean_trades)} rows (deduped)")
print(f"  positions.json:  {len(state['positions'])} → {len(still_open)} open positions")
print(f"{'─'*50}")

confirm = input("\nApply these changes? [y/N]: ").strip().lower()
if confirm != "y":
    print("Aborted — no changes made.")
    exit(0)

# Backup originals
shutil.copy(TRADES_F,  TRADES_F.with_suffix(".csv.bak"))
shutil.copy(POS_F,     POS_F.with_suffix(".json.bak"))
print("Backups saved (.bak files)")

# Save clean trades
clean_trades.to_csv(TRADES_F, index=False)
print(f"trades.csv saved: {len(clean_trades)} rows")

# Save clean positions
state["positions"] = still_open
state["last_run"]  = date.today().isoformat()
with open(POS_F, "w") as f:
    json.dump(state, f, indent=2, default=str)
print(f"positions.json saved: {len(still_open)} open positions")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("RECONCILIATION COMPLETE")
print(f"{'='*50}")
closed_exits = clean_trades[clean_trades["action"] == "closed"]
print(f"Closed trades:  {len(closed_exits)}")
print(f"Win rate:       {(closed_exits['pnl'] > 0).mean():.0%}")
print(f"Avg P&L:        {closed_exits['pnl_pct'].mean():+.1f}%")
print(f"Total P&L:      ${closed_exits['pnl'].sum():+.2f}")
print(f"\nOpen positions: {list(still_open.keys())}")
