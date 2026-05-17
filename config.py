"""
config.py — Single source of truth for the S&P 500 trading agent.
"""
import os
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.resolve()
DATA_DIR         = BASE_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
MACRO_DIR        = DATA_DIR / "macro"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
NEWS_DIR         = DATA_DIR / "news"
FEATURES_DIR     = DATA_DIR / "features"
MODELS_DIR       = BASE_DIR / "models"
LOG_DIR          = BASE_DIR / "logs"

for d in [RAW_DIR, MACRO_DIR, FUNDAMENTALS_DIR, NEWS_DIR,
          FEATURES_DIR, MODELS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "agent.log"

# ── Universe ──────────────────────────────────────────────────────────────────
TICKER_OVERRIDE = []          # e.g. ["AAPL","MSFT"] for dev runs

# ── Data collection ───────────────────────────────────────────────────────────
HISTORICAL_START   = "2015-01-01"
OHLCV_INTERVAL     = "1d"
NEWS_LOOKBACK_DAYS = 30
MAX_WORKERS        = 8

MACRO_TICKERS = {
    "^GSPC":    "sp500",
    "^IXIC":    "nasdaq",
    "^VIX":     "vix",
    "^TNX":     "yield_10y",
    "DX-Y.NYB": "dxy",
    "GLD":      "gold",
    "USO":      "oil",
    "TLT":      "bonds",
}

# ── Feature engineering ───────────────────────────────────────────────────────
# Prediction target
LABEL_HORIZON    = 10      # trading days forward (10d = 2 weeks, smoother signal)
LABEL_THRESHOLD  = 0.01    # 1% alpha vs market to be labelled long/short

# Technical windows
RSI_WINDOW    = 14
ATR_WINDOW    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
BB_WINDOW     = 20
BB_STD        = 2.0

# ── Training ──────────────────────────────────────────────────────────────────
# Walk-forward folds: each entry = (train_end_year, test_year)
WF_FOLDS = [
    (2018, 2019),
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]
WF_GAP_DAYS = 5   # trading-day gap between train end and test start

CONFIDENCE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_TRAIN_ROWS  = 50_000
MIN_TEST_ROWS   = 10_000
MIN_TICKER_ROWS = 252   # drop tickers with < 1yr history

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ── Optional API keys (Phase 6+) ──────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_DIR        = BASE_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL     = 1000   # CAD starting capital
MAX_POSITION_PCT    = 0.05      # max 5% of portfolio per position
MAX_SECTOR_PCT      = 0.30      # max 30% in one sector
MAX_POSITIONS       = 20        # max concurrent open positions
KELLY_FRACTION      = 0.25      # quarter Kelly sizing

# Exit rules
STOP_LOSS_PCT       = 0.05      # 5% hard stop loss
TAKE_PROFIT_PCT     = 0.15      # 15% take profit
HOLD_DAYS           = 30        # max hold = label horizon (then re-evaluate)

# Trading costs (Wealthsimple)
COMMISSION          = 0.0       # $0 commission
SLIPPAGE_PCT        = 0.001     # 0.1% slippage per trade (bid-ask + impact)

# Signal filter
MIN_CONFIDENCE      = 0.45      # minimum model confidence to trade

# ── Paper trading ─────────────────────────────────────────────────────────────
PAPER_DIR           = BASE_DIR / "paper_trading"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
PAPER_INITIAL_CAPITAL = 1000   # simulated starting capital

# ── Paper trading ─────────────────────────────────────────────────────────────
PAPER_DIR           = BASE_DIR / "paper_trading"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
PAPER_CAPITAL       = 1000   # starting simulated capital
PAPER_LOG_FILE      = PAPER_DIR / "paper_trading.log"
DAILY_LOSS_LIMIT    = 0.03      # halt trading if down >3% in one day
MAX_DRAWDOWN_HALT   = 0.15      # halt if drawdown exceeds 15% from peak

# ── Paper trading ─────────────────────────────────────────────────────────────
PAPER_DIR           = BASE_DIR / "paper_trading"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
PAPER_POSITIONS_FILE = PAPER_DIR / "positions.json"
PAPER_TRADES_FILE    = PAPER_DIR / "trades.csv"
PAPER_EQUITY_FILE    = PAPER_DIR / "equity.csv"
PAPER_LOG_FILE       = PAPER_DIR / "paper.log"
PAPER_CAPITAL        = 1000  # simulated starting capital
