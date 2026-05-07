"""
run_pipeline.py
────────────────
Single entry point for all pipeline stages.

Usage:
    python run_pipeline.py --features              # Build feature matrix
    python run_pipeline.py --train                 # CV + final model
    python run_pipeline.py --train --cv-only       # CV only
    python run_pipeline.py --features --train      # Full rebuild
    python run_pipeline.py --tickers AAPL MSFT NVDA --features  # Dev run

Daily automation (run after market close):
    python run_pipeline.py --collect               # Collect new data
    python run_pipeline.py --collect --features --train  # Full daily update
"""

import argparse
import time
from datetime import datetime

from utils.helpers import get_logger

log = get_logger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Agent Pipeline")
    parser.add_argument("--collect",  action="store_true", help="Run data collection")
    parser.add_argument("--features", action="store_true", help="Build feature matrix")
    parser.add_argument("--train",    action="store_true", help="Train model (CV + final)")
    parser.add_argument("--cv-only",  action="store_true", help="CV only, skip final model")
    parser.add_argument("--regime",   action="store_true", help="Use regime-conditional training")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on trained model")
    parser.add_argument("--start",    default="2019-01-01",help="Backtest start date")
    parser.add_argument("--paper",    action="store_true", help="Run daily paper trading update")
    parser.add_argument("--status",   action="store_true", help="Show paper trading status (read-only)")
    parser.add_argument("--reset",    action="store_true", help="Reset paper trading state")
    parser.add_argument("--tickers",  nargs="+",           help="Subset of tickers for dev")
    args = parser.parse_args()

    if not any([args.collect, args.features, args.train, args.backtest, args.paper, args.status]):
        parser.print_help()
        return

    t0 = time.time()
    log.info("=" * 60)
    log.info(f"Pipeline started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    # ── Collection ────────────────────────────────────────────────────────────
    if args.collect:
        log.info("\n[1/3] Data collection")
        from run_collection import main as collect
        collect()

    # ── Features ──────────────────────────────────────────────────────────────
    if args.features:
        log.info("\n[2/3] Feature engineering")
        from pipeline.build_features import build_features
        tickers = args.tickers or None
        df = build_features(tickers=tickers)
        log.info(f"Features: {df.shape}")

    # ── Training ──────────────────────────────────────────────────────────────
    if args.train:
        log.info("\n[3/3] Model training")
        from pipeline.build_labels import load_training_dataset
        from pipeline.train_model import (
            run_walk_forward, train_final_model, save_cv_summary
        )
        import numpy as np

        df = load_training_dataset()

        # Walk-forward CV — standard or regime-conditional
        if args.regime:
            log.info("Using regime-conditional training (Option B)")
            from pipeline.regime_model import run_regime_walk_forward
            regime_output = run_regime_walk_forward(df)
            log.info(f"\nRegime CV mean dir acc: {regime_output['mean_dir_acc']:.3f}")
            if not args.cv_only:
                from pipeline.regime_model import train_and_save_regime_models
                train_and_save_regime_models(df)
            return
        results = run_walk_forward(df)

        if not results:
            log.error("No CV folds completed.")
            return

        # Summary
        dir_accs     = [r.dir_acc for r in results]
        thresh_accs  = [r.thresh_dir_acc for r in results]
        thresholds   = [r.best_threshold for r in results]
        best_threshold = float(np.median(thresholds))

        log.info("\n" + "=" * 60)
        log.info("CV RESULTS")
        log.info("=" * 60)
        log.info(f"Folds:            {len(results)}")
        log.info(f"Mean dir acc:     {np.mean(dir_accs):.3f} ± {np.std(dir_accs):.3f}")
        log.info(f"Mean thresh acc:  {np.mean(thresh_accs):.3f}")
        log.info(f"Best threshold:   {best_threshold:.2f}")
        log.info("\nPer-fold:")
        for r in results:
            log.info(f"  {r.test_year}: dir={r.dir_acc:.3f} "
                     f"thresh={r.thresh_dir_acc:.3f} "
                     f"@ {r.best_threshold:.2f} "
                     f"[short={r.class_acc.get('short',0):.2f} "
                     f"long={r.class_acc.get('long',0):.2f}]")

        mean_dir = np.mean(dir_accs)
        if mean_dir >= 0.52:
            log.info(f"\nTarget 52%+ directional accuracy ACHIEVED ({mean_dir:.1%})")
        else:
            log.info(f"\nDirectional accuracy: {mean_dir:.1%} "
                     f"(baseline=50%, target=52%+)")
            log.info("Note: each 1% above 50% = real exploitable edge.")
            log.info("Check per-threshold sweep for best operating point.")

        save_cv_summary(results, best_threshold)

        # Final model
        if not args.cv_only:
            train_final_model(df)

    # ── Backtest ──────────────────────────────────────────────────────────────
    if args.backtest:
        log.info("\n[4/4] Running backtest ...")
        from pipeline.backtest import run_backtest
        run_backtest(start_date=args.start)

    # ── Paper trading ─────────────────────────────────────────────────────────
    if args.status:
        from pipeline.paper_trade import show_status
        show_status()

    if args.paper:
        log.info("\n[Paper trading] Running daily update ...")
        from pipeline.paper_trade import run_paper_update
        run_paper_update(reset=args.reset)

    elapsed = time.time() - t0
    log.info(f"\nPipeline complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
