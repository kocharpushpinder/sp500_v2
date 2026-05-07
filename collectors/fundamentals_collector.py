"""
collectors/fundamentals_collector.py
──────────────────────────────────────
Fetches company fundamentals and earnings calendars via yfinance.
No API key required.

Per ticker:
  - Basic financials: P/E, P/B, EPS, revenue, margins, ROE, debt/equity, beta
  - Earnings calendar: next earnings date
  - Quarterly income statement: revenue growth, EPS trend

Outputs:
    data/fundamentals/fundamentals.csv   — one row per ticker (latest snapshot)
    data/fundamentals/earnings.csv       — next earnings date per ticker

Rate consideration: yfinance is unofficial/scraping-based.
We use a small thread pool (4 workers max) to avoid getting blocked.

Run directly:
    python collectors/fundamentals_collector.py
"""

import time
from pathlib import Path
from datetime import date
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FUNDAMENTALS_DIR, MAX_WORKERS
from utils.helpers import get_logger, retry, save_csv, load_or_empty

log          = get_logger("fundamentals_collector")
FUND_OUT     = FUNDAMENTALS_DIR / "fundamentals.csv"
EARNINGS_OUT = FUNDAMENTALS_DIR / "earnings.csv"

# Use fewer workers for yfinance .info calls to avoid rate limiting
YF_WORKERS = min(MAX_WORKERS, 4)


# ─── Per-ticker fetch ─────────────────────────────────────────────────────────
@retry(max_attempts=3, wait_seconds=4.0)
def fetch_ticker_fundamentals(ticker: str) -> tuple[dict, dict]:
    """
    Fetch fundamentals and next earnings date for one ticker via yfinance.
    Returns (fundamentals_row, earnings_row).
    """
    t = yf.Ticker(ticker)

    # ── Basic info ────────────────────────────────────────────────────────────
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    fund_row = {
        "ticker":               ticker,
        "market_cap":           info.get("marketCap"),
        "pe_ttm":               info.get("trailingPE"),
        "pe_forward":           info.get("forwardPE"),
        "pb":                   info.get("priceToBook"),
        "ps":                   info.get("priceToSalesTrailing12Months"),
        "peg":                  info.get("pegRatio"),
        "eps_ttm":              info.get("trailingEps"),
        "eps_forward":          info.get("forwardEps"),
        "revenue_ttm":          info.get("totalRevenue"),
        "revenue_growth_yoy":   info.get("revenueGrowth"),
        "earnings_growth_yoy":  info.get("earningsGrowth"),
        "gross_margin":         info.get("grossMargins"),
        "operating_margin":     info.get("operatingMargins"),
        "net_margin":           info.get("profitMargins"),
        "roe":                  info.get("returnOnEquity"),
        "roa":                  info.get("returnOnAssets"),
        "debt_equity":          info.get("debtToEquity"),
        "current_ratio":        info.get("currentRatio"),
        "quick_ratio":          info.get("quickRatio"),
        "beta":                 info.get("beta"),
        "shares_outstanding":   info.get("sharesOutstanding"),
        "float_shares":         info.get("floatShares"),
        "short_ratio":          info.get("shortRatio"),
        "dividend_yield":       info.get("dividendYield"),
        "52w_high":             info.get("fiftyTwoWeekHigh"),
        "52w_low":              info.get("fiftyTwoWeekLow"),
        "sector":               info.get("sector"),
        "industry":             info.get("industry"),
        "fetched_date":         date.today().isoformat(),
    }

    # ── Next earnings date ────────────────────────────────────────────────────
    earn_row = {"ticker": ticker, "next_earnings_date": None}
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            if "Earnings Date" in cal.index:
                earn_dates = cal.loc["Earnings Date"]
                if hasattr(earn_dates, "iloc"):
                    earn_row["next_earnings_date"] = str(earn_dates.iloc[0])[:10]
                else:
                    earn_row["next_earnings_date"] = str(earn_dates)[:10]
    except Exception:
        pass

    # ── Quarterly revenue growth ───────────────────────────────────────────────
    try:
        q_income = t.quarterly_income_stmt
        if q_income is not None and not q_income.empty and "Total Revenue" in q_income.index:
            rev = q_income.loc["Total Revenue"].dropna()
            if len(rev) >= 2:
                fund_row["revenue_growth_qoq"] = float(
                    (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])
                ) if rev.iloc[1] != 0 else None
    except Exception:
        pass

    return fund_row, earn_row


# ─── Main runner ──────────────────────────────────────────────────────────────
def run_fundamentals_collection(tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch fundamentals and earnings for all tickers via yfinance.
    Returns (fundamentals_df, earnings_df).
    """
    if tickers is None:
        from collectors.ohlcv_collector import get_sp500_tickers
        tickers = get_sp500_tickers()

    log.info(f"Fetching fundamentals for {len(tickers)} tickers via yfinance "
             f"(workers={YF_WORKERS}) ...")
    log.info(f"Estimated time: ~{len(tickers) * 1.5 / YF_WORKERS / 60:.1f} minutes")

    fund_rows = []
    earn_rows = []

    with ThreadPoolExecutor(max_workers=YF_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker_fundamentals, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                fund_row, earn_row = future.result()
                fund_rows.append(fund_row)
                earn_rows.append(earn_row)
            except Exception as e:
                ticker = futures[future]
                log.warning(f"[{ticker}] fundamentals failed: {e}")
                fund_rows.append({"ticker": ticker, "fetched_date": date.today().isoformat()})

            if i % 50 == 0 or i == len(tickers):
                log.info(f"  Progress: {i}/{len(tickers)} tickers")

    # ── Fundamentals CSV ──────────────────────────────────────────────────────
    df_fund_new = pd.DataFrame(fund_rows)
    df_fund_existing = load_or_empty(FUND_OUT)
    if not df_fund_existing.empty and "ticker" in df_fund_existing.columns:
        df_fund_existing = df_fund_existing[
            ~df_fund_existing["ticker"].isin(df_fund_new["ticker"])
        ]
    df_fund = pd.concat([df_fund_existing, df_fund_new], ignore_index=True)
    save_csv(df_fund, FUND_OUT, index=False)
    log.info(f"Fundamentals saved: {len(df_fund)} tickers → {FUND_OUT}")

    # ── Earnings CSV ──────────────────────────────────────────────────────────
    df_earn_new = pd.DataFrame(earn_rows).dropna(subset=["next_earnings_date"])
    df_earn_existing = load_or_empty(EARNINGS_OUT)
    df_earn = pd.concat([df_earn_existing, df_earn_new], ignore_index=True)
    df_earn = df_earn.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    save_csv(df_earn, EARNINGS_OUT, index=False)
    log.info(f"Earnings saved: {len(df_earn)} upcoming dates → {EARNINGS_OUT}")

    return df_fund, df_earn


if __name__ == "__main__":
    df_f, df_e = run_fundamentals_collection()
    if not df_f.empty:
        print("\nFundamentals sample:")
        print(df_f[["ticker", "pe_ttm", "beta", "net_margin", "sector"]].head(10))
    if not df_e.empty:
        print("\nNext earnings dates:")
        print(df_e.head(10))
