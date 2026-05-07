"""
pipeline/regime_model.py
─────────────────────────
Regime-conditional training: the structural fix for fold 2021.

Problem: A single LightGBM model trained on mixed regimes learns
the dominant regime (momentum/bull) and fails when a different
regime (mean-reversion/sideways) arrives in the test period.

Solution: Train THREE separate models, one per market regime.
At prediction time, classify the current regime and route to
the appropriate model. Each model only sees data from its own
regime type, so it learns the right signal for that context.

Regimes are defined by S&P 500 behavior:
  Bull:      S&P 500 above 200-day SMA, positive 3-month return
  Bear:      S&P 500 below 200-day SMA, negative 3-month return
  Sideways:  Everything else (2015, 2018, 2021 type markets)

Usage:
  from pipeline.regime_model import train_regime_models, predict_with_regime
  models = train_regime_models(df)
  predictions = predict_with_regime(models, X_live, macro_date)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LGBM_PARAMS, MODELS_DIR, MACRO_DIR
from utils.helpers import get_logger

warnings.filterwarnings("ignore")
log = get_logger("regime_model")

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REGIME CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each date as bull / bear / sideways based on S&P 500 macro data.

    Bull:     macro_sp500 > its 200d SMA AND macro_sp500_ret20 > +3%
    Bear:     macro_sp500 < its 200d SMA AND macro_sp500_ret20 < -3%
    Sideways: everything else

    Returns df with 'regime' column added (0=bear, 1=sideways, 2=bull).
    """
    macro_path = MACRO_DIR / "macro_daily.csv"
    if not macro_path.exists():
        log.warning("macro_daily.csv not found — defaulting all rows to sideways regime")
        df["regime"] = 1
        return df

    macro = pd.read_csv(macro_path, parse_dates=["date"])
    macro = macro.sort_values("date")

    if "sp500" not in macro.columns:
        log.warning("sp500 column not in macro — defaulting to sideways")
        df["regime"] = 1
        return df

    sp = macro["sp500"]
    macro["sma200"]     = sp.rolling(200).mean()
    macro["ret_60d"]    = sp.pct_change(60)   # ~3 month return
    macro["above_sma"]  = (sp > macro["sma200"]).astype(int)

    macro["regime"] = 1  # sideways default
    macro.loc[(macro["above_sma"] == 1) & (macro["ret_60d"] > 0.05), "regime"] = 2   # bull
    macro.loc[(macro["above_sma"] == 0) & (macro["ret_60d"] < -0.05), "regime"] = 0  # bear

    regime_map = macro.set_index("date")["regime"].to_dict()

    df["date"] = pd.to_datetime(df["date"])
    df["regime"] = df["date"].map(regime_map).fillna(1).astype(int)

    dist = df["regime"].value_counts(normalize=True)
    log.info(f"Regime distribution: "
             f"bear={dist.get(0,0):.1%} "
             f"sideways={dist.get(1,0):.1%} "
             f"bull={dist.get(2,0):.1%}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PER-REGIME TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _train_one(X_tr, y_tr, X_val, y_val, regime_name: str):
    """Train a single regime model with calibration."""
    counts = y_tr.value_counts()
    total  = len(y_tr)
    cw     = {int(c): total / (len(counts) * cnt) for c, cnt in counts.items()}
    params = {**LGBM_PARAMS, "class_weight": cw}

    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    try:
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        log.info(f"  [{regime_name}] trained on {len(X_tr):,} rows, calibrated on {len(X_val):,}")
        return cal
    except Exception:
        log.info(f"  [{regime_name}] trained on {len(X_tr):,} rows (no calibration)")
        return model


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return usable feature columns — same logic as train_model.py."""
    NEVER = {
        "date", "ticker", "label", "regime",
        "future_ret", "future_market_ret", "future_alpha",
        "forward_ret", "forward_alpha", "forward_market_ret",
        "market_forward_ret",
    }
    return [c for c in df.columns
            if c not in NEVER
            and not df[c].isna().all()
            and df[c].isna().mean() <= 0.8]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WALK-FORWARD WITH REGIME ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def run_regime_walk_forward(df: pd.DataFrame) -> dict:
    """
    Walk-forward CV with regime-conditional training.

    For each fold:
      1. Label training data by regime
      2. Train bull / bear / sideways models separately
      3. On test data, route each row to its regime's model
      4. Evaluate combined predictions
    """
    from config import WF_FOLDS, WF_GAP_DAYS, MIN_TRAIN_ROWS, MIN_TEST_ROWS

    df = _build_regime_labels(df.copy())
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = _get_feature_cols(df)
    log.info(f"Regime walk-forward: {len(feature_cols)} features, {len(WF_FOLDS)} folds")

    REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}
    fold_results = []

    for fold_idx, (train_end_yr, test_yr) in enumerate(WF_FOLDS, 1):
        log.info(f"\n{'─'*55}")
        log.info(f"Fold {fold_idx} | Train ≤ {train_end_yr} | Test = {test_yr}")

        import pandas as _pd
        train_end  = _pd.Timestamp(f"{train_end_yr}-12-31")
        test_start = _pd.Timestamp(f"{test_yr}-01-01") + _pd.offsets.BDay(WF_GAP_DAYS)
        test_end   = _pd.Timestamp(f"{test_yr}-12-31")
        if fold_idx == len(WF_FOLDS):
            test_end = df["date"].max()

        df_train = df[df["date"] <= train_end]
        df_test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(df_train) < MIN_TRAIN_ROWS or len(df_test) < MIN_TEST_ROWS:
            log.warning(f"  Skipping — insufficient data")
            continue

        # Train one model per regime
        models = {}
        for regime_id, regime_name in REGIME_NAMES.items():
            regime_train = df_train[df_train["regime"] == regime_id]
            if len(regime_train) < 5000:
                log.info(f"  [{regime_name}] only {len(regime_train)} rows — skipping")
                continue

            X_r = regime_train[feature_cols].astype(np.float32)
            y_r = regime_train["label"].map(LABEL_MAP)

            # Val split: last 20% chronologically
            n_val = int(len(X_r) * 0.2)
            X_tr, X_val = X_r.iloc[:-n_val], X_r.iloc[-n_val:]
            y_tr, y_val = y_r.iloc[:-n_val], y_r.iloc[-n_val:]

            models[regime_id] = _train_one(X_tr, y_tr, X_val, y_val, regime_name)

        if not models:
            log.warning("  No regime models trained — skipping fold")
            continue

        # Predict on test set with regime routing
        X_test = df_test[feature_cols].astype(np.float32)
        y_test = df_test["label"].map(LABEL_MAP)
        regimes_test = df_test["regime"].values

        all_probs = np.zeros((len(X_test), 3))
        all_probs[:, 1] = 1.0  # default to "flat" if no model for regime

        for regime_id, model in models.items():
            mask = regimes_test == regime_id
            if mask.sum() > 0:
                all_probs[mask] = model.predict_proba(X_test.iloc[mask])

        preds = np.argmax(all_probs, axis=1)
        y     = y_test.values

        # Directional accuracy
        dir_mask = y != 1
        dir_acc  = (preds[dir_mask] == y[dir_mask]).mean() if dir_mask.sum() > 0 else 0.0

        acc_3class = (preds == y).mean()

        class_acc = {}
        for cls, name in [(0,"short"), (1,"flat"), (2,"long")]:
            mask = y == cls
            if mask.sum() > 0:
                class_acc[name] = float((preds[mask] == cls).mean())

        log.info(f"  Dir acc: {dir_acc:.3f} | 3-class: {acc_3class:.3f}")
        log.info(f"  Class: short={class_acc.get('short',0):.3f} "
                 f"flat={class_acc.get('flat',0):.3f} "
                 f"long={class_acc.get('long',0):.3f}")

        fold_results.append({
            "fold": fold_idx, "test_year": test_yr,
            "dir_acc": dir_acc, "acc_3class": acc_3class,
            "class_acc": class_acc,
            "n_regime_models": len(models),
        })

    # Summary
    if fold_results:
        dir_accs = [r["dir_acc"] for r in fold_results]
        log.info(f"\n{'='*55}")
        log.info(f"REGIME CV SUMMARY")
        log.info(f"Mean dir acc: {np.mean(dir_accs):.3f} ± {np.std(dir_accs):.3f}")
        for r in fold_results:
            log.info(f"  {r['test_year']}: dir={r['dir_acc']:.3f} "
                     f"[short={r['class_acc'].get('short',0):.2f} "
                     f"long={r['class_acc'].get('long',0):.2f}]")

    return {
        "fold_results": fold_results,
        "feature_cols": feature_cols,
        "mean_dir_acc": float(np.mean([r["dir_acc"] for r in fold_results])) if fold_results else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN AND SAVE FINAL REGIME MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save_regime_models(df: pd.DataFrame) -> None:
    """Train final regime models on full data and save to models/."""
    df = _build_regime_labels(df.copy())
    feature_cols = _get_feature_cols(df)

    REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}
    models = {}

    for regime_id, regime_name in REGIME_NAMES.items():
        regime_df = df[df["regime"] == regime_id]
        if len(regime_df) < 5000:
            log.info(f"[{regime_name}] insufficient data ({len(regime_df)} rows) — skipping")
            continue

        X = regime_df[feature_cols].astype(np.float32)
        y = regime_df["label"].map(LABEL_MAP)

        cw     = {int(c): len(y) / (3 * cnt) for c, cnt in y.value_counts().items()}
        params = {**LGBM_PARAMS, "class_weight": cw}
        model  = LGBMClassifier(**params)
        model.fit(X, y)
        models[regime_id] = {"model": model, "name": regime_name}
        log.info(f"[{regime_name}] trained on {len(X):,} rows")

    artifacts = {
        "models":       models,
        "feature_cols": feature_cols,
        "regime_names": REGIME_NAMES,
    }
    path = MODELS_DIR / "regime_models.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    log.info(f"Regime models saved → {path}")


def load_regime_models() -> dict:
    """Load trained regime models."""
    path = MODELS_DIR / "regime_models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No regime models at {path}. Run training first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_regime(artifacts: dict, X: pd.DataFrame,
                        current_regime: int = 1) -> np.ndarray:
    """
    Generate predictions using the appropriate regime model.
    current_regime: 0=bear, 1=sideways, 2=bull
    """
    models = artifacts["models"]
    feature_cols = artifacts["feature_cols"]

    X_feat = X[feature_cols].astype(np.float32)
    probs  = np.zeros((len(X_feat), 3))
    probs[:, 1] = 1.0  # default flat

    if current_regime in models:
        model = models[current_regime]["model"]
        probs = model.predict_proba(X_feat)
    elif 1 in models:
        # Fallback to sideways model
        probs = models[1]["model"].predict_proba(X_feat)

    return probs
