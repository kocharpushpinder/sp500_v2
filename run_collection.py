"""
run_collection.py
──────────────────
Master entry point for Phase 1 data collection.
Run this every day after 4pm ET (market close).

Usage:
    python run_collection.py              # full run (all collectors)
    python run_collection.py --ohlcv      # OHLCV only
    python run_collection.py --macro      # macro only
    python run_collection.py --fundamentals  # fundamentals + earnings + sectors
    python run_collection.py --news       # news sentiment only
    python run_collection.py --fast       # OHLCV + macro only (fastest, ~5 min)

Schedule (cron example — runs Mon-Fri at 4:30pm ET):
    30 16 * * 1-5 cd /path/to/sp500_agent && python run_collection.py >> logs/cron.log 2>&1
"""

import argparse
import sys
import time
from datetime import datetime

from utils.helpers import get_logger

log = get_logger("run_collection")


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Agent — Data Collection")
    parser.add_argument("--ohlcv",          action="store_true", help="Run OHLCV collector")
    parser.add_argument("--macro",          action="store_true", help="Run macro collector")
    parser.add_argument("--fundamentals",   action="store_true", help="Run fundamentals + sectors")
    parser.add_argument("--news",           action="store_true", help="Run news sentiment")
    parser.add_argument("--fast",           action="store_true", help="OHLCV + macro only")
    args = parser.parse_args()

    # Default: run everything
    run_all = not any([args.ohlcv, args.macro, args.fundamentals, args.news, args.fast])
    run_ohlcv         = run_all or args.ohlcv or args.fast
    run_macro         = run_all or args.macro or args.fast
    run_fundamentals  = (run_all or args.fundamentals) and not args.fast
    run_news          = (run_all or args.news) and not args.fast

    log.info("=" * 60)
    log.info(f"S&P 500 Agent — Data Collection started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    # ── 1. OHLCV ──────────────────────────────────────────────────────────────
    if run_ohlcv:
        _run_stage("OHLCV", lambda: _import_and_run_ohlcv())

    # ── 2. Macro ──────────────────────────────────────────────────────────────
    if run_macro:
        _run_stage("Macro", lambda: _import_and_run_macro())

    # ── 3. Sectors ────────────────────────────────────────────────────────────
    if run_fundamentals:
        _run_stage("Sectors", lambda: _import_and_run_sectors())

    # ── 4. Fundamentals + Earnings ────────────────────────────────────────────
    if run_fundamentals:
        _run_stage("Fundamentals + Earnings", lambda: _import_and_run_fundamentals())

    # ── 5. News sentiment ─────────────────────────────────────────────────────
    if run_news:
        _run_stage("News Sentiment", lambda: _import_and_run_news())

    log.info("=" * 60)
    log.info(f"Collection complete at {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)


def _run_stage(name: str, fn):
    log.info(f"\n{'─'*40}")
    log.info(f"Stage: {name}")
    log.info(f"{'─'*40}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        log.info(f"✓ {name} completed in {elapsed:.1f}s")
    except Exception as e:
        log.error(f"✗ {name} failed: {e}", exc_info=True)


def _import_and_run_ohlcv():
    from collectors.ohlcv_collector import run_ohlcv_collection
    run_ohlcv_collection()

def _import_and_run_macro():
    from collectors.macro_collector import run_macro_collection
    run_macro_collection()

def _import_and_run_sectors():
    from collectors.sector_collector import run_sector_collection
    run_sector_collection()

def _import_and_run_fundamentals():
    from collectors.fundamentals_collector import run_fundamentals_collection
    run_fundamentals_collection()

def _import_and_run_news():
    from collectors.news_collector import run_news_collection
    run_news_collection()


if __name__ == "__main__":
    main()
