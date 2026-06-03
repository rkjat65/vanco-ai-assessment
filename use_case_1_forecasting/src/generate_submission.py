"""
Fast submission generator — skips cross-validation, trains final models directly.
Run this after pipeline.py to regenerate submission.csv quickly.
"""
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import load_all_tables, build_master_frame
from feature_engineering import build_features, FEATURE_COLS
from models import train_lgbm, train_xgb
from validation import evaluate_predictions

DATA_DIR   = "../data/"
OUTPUT_DIR = "../outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1/4] Loading data...")
tables = load_all_tables(DATA_DIR)
train_raw, test_raw = build_master_frame(tables)

print("[2/4] Engineering features...")
train_raw["is_test"] = 0
test_raw["is_test"]  = 1
test_raw["sales"]    = np.nan
combined = pd.concat([train_raw, test_raw], ignore_index=True).sort_values(
    ["store_nbr", "family", "date"]
)
combined = build_features(combined, is_train=True)
train_df = combined[combined["is_test"] == 0].copy()
test_df  = combined[combined["is_test"] == 1].copy()

feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
print(f"  Features: {len(feature_cols)}")

print("[3/4] Training final LightGBM + XGBoost...")
train_clean = train_df.dropna(subset=feature_cols + ["sales"])
cutoff      = train_clean["date"].max() - pd.Timedelta(days=15)
final_train = train_clean[train_clean["date"] <= cutoff]
final_val   = train_clean[train_clean["date"] > cutoff]

print(f"  Train: {len(final_train):,}  |  Val: {len(final_val):,}")

final_lgbm = train_lgbm(final_train, final_train["sales"], final_val, final_val["sales"], feature_cols)
final_xgb  = train_xgb (final_train, final_train["sales"], final_val, final_val["sales"], feature_cols)

# Quick eval on val
lgbm_val = np.expm1(final_lgbm.predict(final_val[feature_cols]))
xgb_val  = np.expm1(final_xgb.predict(xgb.DMatrix(final_val[feature_cols])))
ens_val  = np.clip(0.6 * lgbm_val + 0.4 * xgb_val, 0, None)
evaluate_predictions(final_val["sales"].values, ens_val, label="Final ensemble (val)")

print("[4/4] Generating submission...")
# Fill NaN features in test set (transactions/lags unavailable for future dates)
# Use median imputation — conservative, avoids zeroing out important features
test_pred = test_df.copy()
for col in feature_cols:
    if col in test_pred.columns and test_pred[col].isna().any():
        fill_val = train_clean[col].median() if col in train_clean.columns else 0
        test_pred[col] = test_pred[col].fillna(fill_val)

print(f"  Test rows: {len(test_pred):,}")
lgbm_test    = np.expm1(final_lgbm.predict(test_pred[feature_cols]))
xgb_test     = np.expm1(final_xgb.predict(xgb.DMatrix(test_pred[feature_cols])))
ensemble_test = np.clip(0.6 * lgbm_test + 0.4 * xgb_test, 0, None)

# Align with sample_submission IDs
sample_sub = pd.read_csv("../data/sample_submission.csv")
pred_map   = dict(zip(test_pred["id"].astype(int), ensemble_test))
sample_sub["sales"] = sample_sub["id"].map(pred_map).fillna(0)

out_path = f"{OUTPUT_DIR}/submission.csv"
sample_sub.to_csv(out_path, index=False)
print(f"\nSubmission saved: {out_path}")
print(f"Shape: {sample_sub.shape}")
print(f"Sales range: [{sample_sub['sales'].min():.2f}, {sample_sub['sales'].max():.2f}]")
print(f"Zero predictions: {(sample_sub['sales'] == 0).sum()}")
