"""
collectors/sector_collector.py
───────────────────────────────
Fetches GICS sector and industry classification for each S&P 500 ticker.
Sector data is used as a categorical feature in the model and for
computing sector-relative strength signals in feature engineering.

Source: Wikipedia S&P 500 table (free, no API key needed).
Output: data/fundamentals/sectors.csv

Run directly:
    python collectors/sector_collector.py
"""

from pathlib import Path
import sys
from io import StringIO

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FUNDAMENTALS_DIR
from utils.helpers import get_logger, save_csv

log      = get_logger("sector_collector")
SECT_OUT = FUNDAMENTALS_DIR / "sectors.csv"


def run_sector_collection() -> pd.DataFrame:
    """
    Pull sector/industry from Wikipedia S&P 500 table.
    This is fast and requires no API key.
    """
    log.info("Fetching S&P 500 sector classifications from Wikipedia ...")
    try:
        # Wikipedia can reject default bot-like clients; use an explicit UA.
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-agent/1.0)"},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        df = tables[0][["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
        df.columns = ["ticker", "sector", "sub_industry"]
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

        # Encode sector as integer for model use
        sectors = sorted(df["sector"].dropna().unique())
        sector_map = {s: i for i, s in enumerate(sectors)}
        df["sector_id"] = df["sector"].map(sector_map)

        save_csv(df, SECT_OUT, index=False)
        log.info(f"Sector data saved: {len(df)} tickers, "
                 f"{df['sector'].nunique()} sectors → {SECT_OUT}")
        return df
    except Exception as e:
        if SECT_OUT.exists():
            log.warning(f"Sector collection failed ({e}); using cached sectors file.")
            return pd.read_csv(SECT_OUT)
        log.warning(f"Sector collection skipped (no cache available): {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    df = run_sector_collection()
    if not df.empty:
        print(df.groupby("sector")["ticker"].count())
