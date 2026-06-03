"""
Model definitions and training for grocery sales forecasting.

Models: Naive baseline, LightGBM, XGBoost, CatBoost, and ensemble.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional


# ─── Baseline ────────────────────────────────────────────────────────────────

class SeasonalNaiveBaseline:
    """
    Naive baseline: use same day from the previous year (lag-365).

    This establishes the floor. Any ML model must beat this.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = "sales_lag_365"
        if lag_col not in X.columns:
            raise ValueError(f"Need '{lag_col}' in features for seasonal naive baseline")
        return X[lag_col].fillna(0).clip(0).values


# ─── LightGBM ────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "regression_l1",        # MAE loss — robust to outliers
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 511,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 1500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Why LightGBM as the primary model:
# - Handles 54 stores × 33 product families × 4+ years efficiently
# - Native support for categorical features (no one-hot explosion)
# - Leaf-wise growth captures irregular holiday/promo spikes
# - Fast enough for hyperparameter tuning
# Trade-off vs. neural forecasting (TFT, N-BEATS):
# - Loses global temporal structure modeling
# - Cannot extrapolate trends beyond training range as well
# - But wins on speed, interpretability, and tabular feature utilization


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    cat_cols: Optional[list[str]] = None,
) -> lgb.Booster:
    cat_cols = cat_cols or ["store_nbr", "family", "type", "cluster",
                            "day_of_week", "month", "holiday_type_enc"]
    cat_cols = [c for c in cat_cols if c in feature_cols]

    dtrain = lgb.Dataset(
        X_train[feature_cols], label=np.log1p(y_train.clip(0)),
        categorical_feature=cat_cols,
    )
    dval = lgb.Dataset(
        X_val[feature_cols], label=np.log1p(y_val.clip(0)),
        categorical_feature=cat_cols,
        reference=dtrain,
    )

    params = {k: v for k, v in LGBM_PARAMS.items()
              if k not in ("n_estimators", "early_stopping_rounds")}

    callbacks = [
        lgb.early_stopping(LGBM_PARAMS["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    return model


# ─── XGBoost ─────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "alpha": 0.1,
    "lambda": 0.1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "seed": 42,
}


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train[feature_cols], label=np.log1p(y_train.clip(0)))
    dval = xgb.DMatrix(X_val[feature_cols], label=np.log1p(y_val.clip(0)))

    params = {k: v for k, v in XGB_PARAMS.items()
              if k not in ("n_estimators", "early_stopping_rounds")}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_PARAMS["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
        verbose_eval=100,
    )

    return model


# ─── Ensemble ────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """
    Weighted average of multiple forecasters.

    Weights are determined by validation RMSLE (lower RMSLE → higher weight).
    """

    def __init__(self, models: list, weights: Optional[list[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        preds = []
        for model, weight in zip(self.models, self.weights):
            if isinstance(model, lgb.Booster):
                p = np.expm1(model.predict(X[feature_cols]))
            elif isinstance(model, xgb.Booster):
                p = np.expm1(model.predict(xgb.DMatrix(X[feature_cols])))
            elif hasattr(model, "predict"):
                p = model.predict(X)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            preds.append(np.clip(p, 0, None) * weight)
        return np.sum(preds, axis=0)
