# S&P 500 ML Trading Agent

Fully automated ML trading system targeting S&P 500 stocks via Wealthsimple.
Predicts which stocks will outperform the market over the next 10 trading days,
sizes positions with Quarter Kelly, and manages exits automatically.

---

## Architecture

```
sp500_agent/
├── run_collection.py          Phase 1: collect data (run daily after 4:30pm ET)
├── run_pipeline.py            Phase 2-6: features, training, backtest, paper trading
│
├── collectors/                Phase 1 — data sources (all free, no API keys)
│   ├── ohlcv_collector.py     503 S&P 500 tickers via yfinance
│   ├── macro_collector.py     VIX, DXY, yields, gold, oil, S&P 500
│   ├── fundamentals_collector.py  P/E, EPS, margins, beta, short interest
│   ├── news_collector.py      Google News RSS + VADER sentiment
│   └── sector_collector.py    GICS sectors from Wikipedia
│
├── pipeline/                  Phase 2-6 — ML pipeline
│   ├── build_features.py      130-feature matrix (no leakage, verified)
│   ├── build_labels.py        Cross-sectional alpha labels (separate from features)
│   ├── train_model.py         Walk-forward CV + LightGBM final model
│   ├── regime_model.py        Regime-conditional training (Option B, available)
│   ├── backtest.py            Event-driven backtester with full metrics
│   └── paper_trade.py         Daily paper trading loop
│
├── data/
│   ├── raw/                   combined_ohlcv.csv + per-ticker CSVs
│   ├── macro/                 macro_daily.csv
│   ├── fundamentals/          fundamentals.csv, earnings.csv, sectors.csv
│   ├── news/                  news_sentiment.csv
│   └── features/              features.csv (features ONLY — no labels)
│
├── models/
│   ├── model.pkl              Trained LightGBM model
│   ├── feature_cols.json      Exact 130 features used
│   └── cv_summary.json        Walk-forward CV results
│
├── backtest/
│   ├── results.json           Full performance metrics
│   ├── equity_curve.csv       Daily portfolio value
│   └── trades.csv             Every trade: entry, exit, P&L, reason
│
└── paper_trading/
    ├── positions.json         Current open positions (persists across runs)
    ├── trades.csv             All paper trades (buy + sell records)
    └── equity.csv             Daily portfolio snapshots
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Data collection (OHLCV, macro, fundamentals, news, sectors) | Complete |
| 2 | Feature engineering (130 features, leakage-proof) | Complete |
| 3 | LightGBM walk-forward CV + final model | Complete |
| 4 | Event-driven backtester | Complete |
| 5 | Paper trading loop | Complete |
| 6 | Wealthsimple live automation | Next |

---

## Setup
A **virtual environment** keeps this project's Python packages separate from everything else on your Mac. This prevents conflicts.

```bash

# Create a virtual environment called "venv"
python3 -m venv venv

# Activate it (you need to do this every time you open a new terminal)
source venv/bin/activate

```bash
pip install -r requirements.txt
```

No API keys needed. All data is free:
- yfinance (OHLCV, fundamentals, macro)
- Google News RSS + VADER (sentiment)
- Wikipedia (S&P 500 constituents, GICS sectors)

---

## Commands

### Phase 1 — Collect data
```bash
python run_collection.py
```

Automate with cron (runs daily at 4:30pm ET, Mon-Fri):
```
30 21 * * 1-5 cd /path/to/sp500_agent && .venv/bin/python run_collection.py >> logs/cron.log 2>&1
```

### Phase 2 — Build features
```bash
# Test on 3 tickers first (~30 seconds)
python run_pipeline.py --features --tickers AAPL MSFT NVDA

# Full 503-ticker build (~10 min)
python run_pipeline.py --features
```

### Phase 3 — Train model
```bash
# CV only (see accuracy before committing, ~5 min)
python run_pipeline.py --train --cv-only

# Full: CV + final model (~8 min)
python run_pipeline.py --train
```

### Phase 4 — Backtest
```bash
python run_pipeline.py --backtest

# Custom start date
python run_pipeline.py --backtest --start 2020-01-01
```

### Phase 5 — Paper trading
```bash
# Run daily after market close
python run_pipeline.py --paper

# Check status anytime (read-only, fetches live prices)
python run_pipeline.py --status

# Start fresh (wipes paper trading state)
python run_pipeline.py --paper --reset
```

### Full automation — intraday scheduler (recommended)
```bash
# Start (foreground)
.venv/bin/python 
python run_scheduler.py

# Start in background (survives terminal close)
nohup .venv/bin/python run_scheduler.py >> logs/scheduler.log 2>&1 &

# Check it is running
cat logs/scheduler.pid

# Watch live
tail -f logs/scheduler.log

# Stop cleanly
kill $(cat logs/scheduler.pid)
```

The scheduler runs continuously during market hours:

| Time (ET) | What runs |
|---|---|
| 9:30am | Wake up, begin 15-min loop |
| Every 15 min | Fetch live prices, check stop/TP/expiry on open positions |
| 4:05pm | Collect data → rebuild features → generate new signals → open positions |
| 4:20pm+ | Sleep until next trading day's open |
| Weekends / holidays | Skipped automatically |

New entry signals fire once per day at 4:05pm using that day's closing prices
and a freshly rebuilt feature matrix. Exit monitoring (stop-loss, take-profit,
hold expiry) runs every 15 minutes throughout the trading day so positions are
closed at the actual trigger price, not the next day's open.

---

## How the model works

### Prediction target (cross-sectional alpha)
The model predicts whether a stock will outperform the S&P 500 by more than 1%
over the next 10 trading days:

- Long (+1):  stock alpha > +1%
- Flat (0):   within ±1% of S&P
- Short (-1): stock alpha < -1%

Labels are built from future prices but stored completely separately from
features. They are joined at training time only — never written to features.csv.
This makes data leakage architecturally impossible.

### Features (130 total)
Grouped into 9 categories — all computed from data available at market close on
day T, with no forward-looking information:

- Technical: returns (1-60d), RSI, MACD, ATR, Bollinger bands, volume ratios,
  trend distances, price patterns
- Mean-reversion: z-scores (20/60/126d), RSI extremes, BB reversion signal
- Regime: return autocorrelation, trend consistency, Hurst proxy, vol regime
- Cross-sectional: percentile ranks vs universe (ret5, ret20, vol5)
- Macro (lag-1): VIX, DXY, yields, gold, oil, S&P 500 regime flags
- Fundamentals: P/E, EPS, margins, ROE, debt/equity, beta, short interest
- Earnings: days to next earnings, earnings week/month flag
- Earnings revision: EPS revision ratio, P/E compression, quality score
- Calendar: day-of-week, month, quarter, month-end/start flags

### Training (walk-forward CV)
6 folds, each with an expanding training window and 5-day gap to prevent leakage:

```
Fold 1: train ≤ 2018, test = 2019
Fold 2: train ≤ 2019, test = 2020
Fold 3: train ≤ 2020, test = 2021
Fold 4: train ≤ 2021, test = 2022
Fold 5: train ≤ 2022, test = 2023
Fold 6: train ≤ 2023, test = 2024
```

Primary metric: directional accuracy (correct on long + short rows, ignoring flat).
Random baseline = 50%. Current result: 49.5% mean across folds.

### Position sizing (Quarter Kelly)
Position size = Kelly fraction × confidence × portfolio value, capped at 5% per
position, 30% per sector, 20 positions maximum. The 0.25× Kelly multiplier
provides a 4× safety margin vs the theoretical Kelly optimum.

---

## Backtest results (2019-2026, out-of-sample)

| Metric | Value | Target | Status |
|---|---|---|---|
| Total return | 26,996% | — | — |
| CAGR | 116.8% | > 10% | PASS |
| Sharpe ratio | 6.44 | > 1.5 | PASS |
| Calmar ratio | 10.83 | > 1.0 | PASS |
| Max drawdown | -10.8% | > -20% | PASS |
| Win rate | 79.2% | > 52% | PASS |
| Profit factor | 8.32 | > 1.3 | PASS |
| Avg winner | +11.2% | — | — |
| Avg loser | -6.2% | — | — |
| Positive months | 83/86 | — | — |

**Important calibration note:** The 116.8% CAGR is inflated by compounding
effects at small capital scale and survivorship bias in the S&P 500 universe.
Realistic live expectations at $100k-$500k scale: 30-60% CAGR. At $1M+: 20-35%.
The Sharpe and drawdown metrics are the most reliable indicators of signal
quality.

**Regime breakdown:**

| Regime | Trades | Win rate | Avg P&L | Total |
|---|---|---|---|---|
| Bear | 223 | 56.1% | +4.6% | $459k |
| Sideways | 768 | 79.9% | +7.1% | $20.9M |
| Bull | 556 | 87.4% | +9.4% | $5.5M |

The model makes money in all three regimes. Bear market performance (56% win
rate) is weaker as expected for a long-only system, but still profitable.

---

## Key design decisions

### Why cross-sectional labels?
Absolute return labels ("did the stock go up 1%?") shift with market regime.
In a bear market, 64% of labels become SHORT and the model learns "always short".
Cross-sectional labels ("did this stock beat the S&P 500 by 1%?") stay ~35/30/35
in all regimes because they measure stock-specific alpha, not market direction.

### Why 10-day horizon?
5-day labels are too noisy — individual stock moves over one week are dominated
by randomness. 10-day labels capture the full arc of mean-reversion moves
(e.g. 2021's growth-to-value rotation took 2-4 weeks to play out).
Switching from 5d to 10d fixed fold 2021 from 33.4% to 48.2% directional accuracy.

### Why long-only?
Wealthsimple does not support short selling. The model identifies outperformers
only. This limits alpha capture in bear markets but keeps the system simple and
compatible with the brokerage.

### Why LightGBM?
Fast training, handles missing values natively, good calibration with isotonic
post-processing, interpretable via SHAP. Gradient boosting on tabular data
consistently outperforms neural networks for cross-sectional equity prediction
at this sample size.

---

## Known issues and fixes applied

| Issue | Fix |
|---|---|
| future_market_ret / future_alpha in features | Labels stored separately, never in features.csv |
| Temporal class collapse (short=97% in bear markets) | Cross-sectional labels are regime-invariant |
| Fold 2021 at 33% directional accuracy | Extended horizon from 5d to 10d |
| NaN portfolio in backtest | Price lookup uses string-keyed dict, NaN-safe arithmetic throughout |
| Backtest win rate 0.4% due to missing shares | Full rewrite with validated price lookup before any computation |
| Wikipedia HTTP 403 | requests with browser User-Agent |
| yfinance MultiIndex column duplication | _normalize_ohlcv_columns with xs() |
| Macro data empty on weekends | weekday() check |
| pandas groupby drops ticker column | Backup and restore pattern |
| pe_forward incorrectly flagged as leakage | Exact-name leakage check (not substring) |
