"""
End-to-end training and inference pipeline for grocery sales forecasting.

Entry point for CLI execution:
    python src/pipeline.py --data-dir data/ --output-dir outputs/
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_all_tables, build_master_frame
from feature_engineering import build_features, FEATURE_COLS
from validation import (
    walk_forward_folds, split_fold, evaluate_predictions, rmsle
)
from models import train_lgbm, train_xgb, SeasonalNaiveBaseline, WeightedEnsemble
from error_analysis import full_error_report, feature_importance_report


def run_pipeline(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load and merge tables ─────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    tables = load_all_tables(data_dir)
    train_raw, test_raw = build_master_frame(tables)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/6] Engineering features...")
    # Concatenate train + test so lags computed on combined timeline
    train_raw["is_test"] = 0
    test_raw["is_test"] = 1
    test_raw["sales"] = np.nan
    combined = pd.concat([train_raw, test_raw], ignore_index=True).sort_values(
        ["store_nbr", "family", "date"]
    )
    combined = build_features(combined, is_train=True)

    train_df = combined[combined["is_test"] == 0].copy()
    test_df = combined[combined["is_test"] == 1].copy()

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    target_col = "sales"

    print(f"  Feature columns: {len(feature_cols)}")

    # ── 3. Walk-forward validation ───────────────────────────────────────────
    print("\n[3/6] Walk-forward validation...")
    folds = walk_forward_folds(train_df, n_folds=3, horizon_days=16)

    lgbm_models = []
    xgb_models = []
    fold_metrics = []

    for fold in folds:
        tr, val = split_fold(train_df, fold)
        tr = tr.dropna(subset=feature_cols + [target_col])
        val = val.dropna(subset=feature_cols + [target_col])

        print(f"\n  Fold {fold.fold_id}: train {fold.train_start.date()} to {fold.train_end.date()} | "
              f"val {fold.val_start.date()} to {fold.val_end.date()} "
              f"({len(tr):,} / {len(val):,} rows)")

        lgbm_model = train_lgbm(
            tr, tr[target_col], val, val[target_col], feature_cols
        )
        lgbm_models.append(lgbm_model)

        xgb_model = train_xgb(
            tr, tr[target_col], val, val[target_col], feature_cols
        )
        xgb_models.append(xgb_model)

        # Evaluate ensemble on this fold
        import lightgbm as lgb
        import xgboost as xgb
        lgbm_pred = np.expm1(lgbm_model.predict(val[feature_cols]))
        xgb_pred = np.expm1(xgb_model.predict(xgb.DMatrix(val[feature_cols])))
        ensemble_pred = 0.6 * lgbm_pred + 0.4 * xgb_pred
        ensemble_pred = np.clip(ensemble_pred, 0, None)

        metrics = evaluate_predictions(
            val[target_col].values, ensemble_pred,
            label=f"Fold {fold.fold_id} Ensemble"
        )
        fold_metrics.append(metrics)

        # Store predictions for error analysis on last fold
        if fold.fold_id == 0:
            val_with_preds = val.copy()
            val_with_preds["predicted"] = ensemble_pred

    avg_rmsle = np.mean([m["rmsle"] for m in fold_metrics])
    print(f"\n  Average cross-val RMSLE: {avg_rmsle:.4f}")

    # ── 4. Train final models on all training data ───────────────────────────
    print("\n[4/6] Training final models on full training data...")
    train_clean = train_df.dropna(subset=feature_cols + [target_col])

    val_end = train_df["date"].max()
    val_start = val_end - pd.Timedelta(days=15)
    final_train = train_clean[train_clean["date"] < val_start]
    final_val = train_clean[train_clean["date"] >= val_start]

    final_lgbm = train_lgbm(
        final_train, final_train[target_col],
        final_val, final_val[target_col],
        feature_cols,
    )
    final_xgb = train_xgb(
        final_train, final_train[target_col],
        final_val, final_val[target_col],
        feature_cols,
    )

    # ── 5. Error analysis ────────────────────────────────────────────────────
    print("\n[5/6] Running error analysis...")
    if "val_with_preds" in dir():
        full_error_report(val_with_preds)
        feature_importance_report(final_lgbm, feature_cols)

    # ── 6. Generate submission ───────────────────────────────────────────────
    print("\n[6/6] Generating submission file...")
    test_clean = test_df.dropna(subset=feature_cols)

    import xgboost as xgboost_lib
    lgbm_test = np.expm1(final_lgbm.predict(test_clean[feature_cols]))
    xgb_test = np.expm1(final_xgb.predict(xgboost_lib.DMatrix(test_clean[feature_cols])))
    ensemble_test = np.clip(0.6 * lgbm_test + 0.4 * xgb_test, 0, None)

    submission = pd.DataFrame({
        "id": test_clean["id"],
        "sales": ensemble_test,
    })

    submission_path = Path(output_dir) / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n  Submission saved to {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Sales range: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/", help="Path to Kaggle data directory")
    parser.add_argument("--output-dir", default="outputs/", help="Path for outputs")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir)
