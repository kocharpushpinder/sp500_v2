"""
pipeline/train_model.py
────────────────────────
Walk-forward cross-validation and final model training.

LEAKAGE PREVENTION:
  - Train set: all rows where date <= train_end
  - Gap:       WF_GAP_DAYS trading days excluded
  - Test set:  rows where date >= test_start AND date <= test_end
  - Features come from features.csv (no future columns)
  - Labels come from build_labels.py (computed separately)
  - The join happens at training time in memory only

METRICS:
  Primary:   directional_acc = accuracy on long/short rows only (excludes flat)
  Secondary: 3-class accuracy, log-loss
  Sweep:     threshold sweep optimises directional accuracy on confident predictions

WHAT GOOD LOOKS LIKE:
  Directional accuracy 52-58% is excellent for equities.
  The baseline is 50% (random on long/short).
  Each 1% above 50% = real, exploitable edge.
"""

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    WF_FOLDS, WF_GAP_DAYS, LGBM_PARAMS,
    CONFIDENCE_THRESHOLDS, MODELS_DIR,
    MIN_TRAIN_ROWS, MIN_TEST_ROWS
)
from utils.helpers import get_logger

warnings.filterwarnings("ignore")
log = get_logger("train_model")

# LightGBM multiclass maps internally: label {-1,0,1} → need 0-indexed {0,1,2}
LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


@dataclass
class FoldResult:
    fold:             int
    train_end:        int
    test_year:        int
    train_rows:       int
    test_rows:        int
    n_features:       int
    dir_acc:          float   # PRIMARY: accuracy on long+short rows
    acc_3class:       float
    log_loss_val:     float
    best_threshold:   float
    thresh_dir_acc:   float
    thresh_trade_pct: float
    class_acc:        dict = field(default_factory=dict)
    label_dist:       dict = field(default_factory=dict)
    sweep:            dict = field(default_factory=dict)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return usable feature columns. Explicitly forbids any future-derived column.
    """
    # Exact set of columns that must never be model features
    # (computed from future prices — genuine leakage)
    NEVER_FEATURES = {
        "date", "ticker", "label",
        "future_ret", "future_market_ret", "future_alpha",
        "forward_ret", "forward_alpha", "forward_market_ret",
        "market_forward_ret",
    }

    cols = []
    for c in df.columns:
        if c in NEVER_FEATURES:
            continue
        if df[c].isna().all():
            continue
        if df[c].isna().mean() > 0.8:
            continue
        cols.append(c)

    return cols


def _compute_class_weights(y: pd.Series) -> dict:
    """Inverse-frequency class weights. Handles temporal imbalance."""
    counts = y.value_counts()
    total  = len(y)
    return {int(cls): total / (len(counts) * cnt) for cls, cnt in counts.items()}


def _train_fold(X_tr, y_tr, X_val, y_val) -> object:
    """Train LightGBM with early stopping + isotonic calibration."""
    cw = _compute_class_weights(y_tr)
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

    # Calibrate probabilities on validation set
    try:
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        return cal
    except Exception:
        return model


def _evaluate(probs: np.ndarray, y_true: pd.Series) -> dict:
    """
    Compute metrics. Primary = directional_acc.
    directional_acc: accuracy ONLY on rows where true label is long(2) or short(0).
    This is what matters for trading — we don't care about flat prediction quality.
    """
    preds = np.argmax(probs, axis=1)
    y     = y_true.values

    acc_3class = (preds == y).mean()
    ll         = log_loss(y, probs, labels=[0, 1, 2])

    # Directional accuracy: on true long/short rows, are predictions correct?
    dir_mask = y != 1   # exclude true flat rows
    dir_acc  = (preds[dir_mask] == y[dir_mask]).mean() if dir_mask.sum() > 0 else 0.0

    class_acc = {}
    for cls, name in [(0, "short"), (1, "flat"), (2, "long")]:
        mask = y == cls
        if mask.sum() > 0:
            class_acc[name] = float((preds[mask] == cls).mean())

    return {
        "dir_acc":   dir_acc,
        "acc_3class": acc_3class,
        "log_loss":  ll,
        "class_acc": class_acc,
        "label_dist": {name: int((y == cls).sum())
                       for cls, name in [(0,"short"),(1,"flat"),(2,"long")]},
    }


def _threshold_sweep(probs: np.ndarray, y_true: pd.Series,
                     thresholds: list[float]) -> tuple[float, float, float, dict]:
    """
    Sweep confidence thresholds.
    At each threshold: only consider rows where max(prob) >= threshold
    AND predicted class is not flat (class 1).
    Optimise for directional accuracy on those rows.
    Returns: best_threshold, best_dir_acc, trade_pct, full_sweep_dict
    """
    y = y_true.values
    preds_all = np.argmax(probs, axis=1)
    best_t, best_acc, best_pct = thresholds[0], 0.0, 1.0
    sweep = {}

    for t in thresholds:
        confident   = probs.max(axis=1) >= t
        directional = preds_all != 1
        mask = confident & directional

        trade_pct = mask.mean()
        if mask.sum() < 100 or trade_pct < 0.10:
            continue

        acc = (preds_all[mask] == y[mask]).mean()
        sweep[t] = {"dir_acc": round(float(acc), 4),
                    "trade_pct": round(float(trade_pct), 3)}

        if acc > best_acc:
            best_acc = acc
            best_t   = t
            best_pct = trade_pct

    return best_t, best_acc, best_pct, sweep


def run_walk_forward(df: pd.DataFrame) -> list[FoldResult]:
    """
    Run 6-fold walk-forward CV on the training dataset.
    df must have: date, ticker, label, [feature columns]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = _get_feature_cols(df)
    log.info(f"Using {len(feature_cols)} features")

    results = []

    for fold_idx, (train_end_yr, test_yr) in enumerate(WF_FOLDS, 1):
        log.info(f"\n{'─'*55}")
        log.info(f"Fold {fold_idx}/{len(WF_FOLDS)} | Train ≤ {train_end_yr} | Test = {test_yr}")

        train_end   = pd.Timestamp(f"{train_end_yr}-12-31")
        test_start  = pd.Timestamp(f"{test_yr}-01-01") + pd.offsets.BDay(WF_GAP_DAYS)
        test_end    = pd.Timestamp(f"{test_yr}-12-31")
        if fold_idx == len(WF_FOLDS):
            test_end = df["date"].max()

        df_train = df[df["date"] <= train_end]
        df_test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(df_train) < MIN_TRAIN_ROWS or len(df_test) < MIN_TEST_ROWS:
            log.warning(f"  Skipping fold — insufficient data "
                        f"(train={len(df_train):,}, test={len(df_test):,})")
            continue

        # Prepare X, y
        X_train = df_train[feature_cols].astype(np.float32)
        y_train = df_train["label"].map(LABEL_MAP)
        X_test  = df_test[feature_cols].astype(np.float32)
        y_test  = df_test["label"].map(LABEL_MAP)

        # Val split: last 20% of training data (chronological)
        n_val   = int(len(X_train) * 0.2)
        X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
        y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

        log.info(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Verify label distribution in train
        dist = y_tr.value_counts(normalize=True)
        log.info(f"  Train labels: "
                 f"short={dist.get(0,0):.1%} flat={dist.get(1,0):.1%} long={dist.get(2,0):.1%}")

        # Train
        model  = _train_fold(X_tr, y_tr, X_val, y_val)
        probs  = model.predict_proba(X_test)
        m      = _evaluate(probs, y_test)
        best_t, thresh_acc, trade_pct, sweep = _threshold_sweep(
            probs, y_test, CONFIDENCE_THRESHOLDS
        )

        res = FoldResult(
            fold=fold_idx, train_end=train_end_yr, test_year=test_yr,
            train_rows=len(X_train), test_rows=len(X_test), n_features=len(feature_cols),
            dir_acc=m["dir_acc"], acc_3class=m["acc_3class"], log_loss_val=m["log_loss"],
            best_threshold=best_t, thresh_dir_acc=thresh_acc, thresh_trade_pct=trade_pct,
            class_acc=m["class_acc"], label_dist=m["label_dist"], sweep=sweep,
        )
        results.append(res)

        log.info(f"  Dir acc: {m['dir_acc']:.3f} | 3-class: {m['acc_3class']:.3f} | LL: {m['log_loss']:.4f}")
        log.info(f"  Class: short={m['class_acc'].get('short',0):.3f} "
                 f"flat={m['class_acc'].get('flat',0):.3f} "
                 f"long={m['class_acc'].get('long',0):.3f}")
        log.info(f"  Best threshold: {best_t:.2f} → dir_acc={thresh_acc:.3f} "
                 f"trade_pct={trade_pct:.1%}")
        if sweep:
            log.info("  Sweep: " + " | ".join(
                f"{t:.2f}→{v['dir_acc']:.3f}({v['trade_pct']:.0%})"
                for t, v in sweep.items()
            ))

        # Save fold model
        with open(MODELS_DIR / f"fold_{fold_idx}.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    return results


def train_final_model(df: pd.DataFrame) -> None:
    """
    Train final model on ALL available data and save all artifacts.
    """
    log.info("\nTraining final model on full dataset ...")
    feature_cols = _get_feature_cols(df)

    X = df[feature_cols].astype(np.float32)
    y = df["label"].map(LABEL_MAP)

    cw     = _compute_class_weights(y)
    params = {**LGBM_PARAMS, "class_weight": cw}
    model  = LGBMClassifier(**params)
    model.fit(X, y)

    # SHAP feature importance
    importance = _compute_importance(model, X, feature_cols)

    # Save artifacts
    artifacts = {
        "model":        model,
        "feature_cols": feature_cols,
        "trained_at":   datetime.now().isoformat(),
        "n_rows":       len(X),
        "n_features":   len(feature_cols),
    }
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    log.info(f"Model saved → {MODELS_DIR}/model.pkl")
    log.info(f"Top 10 features:\n{importance.head(10).to_string(index=False)}")


def save_cv_summary(results: list[FoldResult], best_threshold: float) -> None:
    """Save CV summary to JSON for the backtester to reference."""
    summary = {
        "n_folds":          len(results),
        "mean_dir_acc":     float(np.mean([r.dir_acc for r in results])),
        "std_dir_acc":      float(np.std([r.dir_acc for r in results])),
        "mean_thresh_acc":  float(np.mean([r.thresh_dir_acc for r in results])),
        "best_threshold":   best_threshold,
        "folds": [
            {"test_year": r.test_year, "dir_acc": r.dir_acc,
             "thresh_acc": r.thresh_dir_acc, "best_threshold": r.best_threshold}
            for r in results
        ],
    }
    with open(MODELS_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"CV summary saved → {MODELS_DIR}/cv_summary.json")


def _compute_importance(model, X: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Compute SHAP or fallback importance."""
    try:
        import shap
        sample = X.sample(min(3000, len(X)), random_state=42)
        ex     = shap.TreeExplainer(model)
        sv     = ex.shap_values(sample)
        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv).mean(0)
        return pd.DataFrame({"feature": feature_cols, "importance": mean_abs}
                             ).sort_values("importance", ascending=False)
    except Exception:
        imp = getattr(model, "feature_importances_",
                      getattr(getattr(model, "calibrated_classifiers_", [{}])[0],
                              "estimator", model).feature_importances_
                      if hasattr(model, "calibrated_classifiers_") else np.ones(len(feature_cols)))
        return pd.DataFrame({"feature": feature_cols, "importance": imp}
                             ).sort_values("importance", ascending=False)


def load_model() -> tuple:
    """Load trained model and feature columns."""
    path = MODELS_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run training first.")
    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    thresh_path = MODELS_DIR / "cv_summary.json"
    threshold = 0.55
    if thresh_path.exists():
        with open(thresh_path) as f:
            threshold = json.load(f).get("best_threshold", 0.55)

    return artifacts["model"], artifacts["feature_cols"], threshold
