"""
Feature engineering for grocery sales forecasting.

Generates temporal, lag, rolling, event, and external features.
All functions are designed to prevent future data leakage.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─── Temporal Features ──────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")        # 0=Mon
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int8")
    df["quarter"] = df["date"].dt.quarter.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
    df["day_of_year"] = df["date"].dt.dayofyear.astype("int16")
    # Cyclical encoding for day_of_week and month (avoids ordinal jump from 6→0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ─── Lag & Rolling Features ──────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: list[str] = ["store_nbr", "family"],
    lags: list[int] = [7, 14, 28, 35, 364, 365],
) -> pd.DataFrame:
    """
    Compute lag features at the (store, family) level.
    Uses direct groupby().shift() — 10x faster than transform(lambda).
    """
    df = df.copy().sort_values(["store_nbr", "family", "date"])
    grouped = df.groupby(group_cols)[target_col]

    for lag in lags:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: list[str] = ["store_nbr", "family"],
    windows: list[int] = [7, 14, 28],
    lags: list[int] = [7, 14, 28],
) -> pd.DataFrame:
    """
    Rolling mean and std over recent windows, shifted by lag to prevent leakage.
    Uses vectorized groupby().shift() then groupby().rolling() — no slow lambda.
    """
    df = df.copy().sort_values(["store_nbr", "family", "date"])
    grp_keys = [df[c] for c in group_cols]

    for lag in lags:
        shifted = df.groupby(group_cols)[target_col].shift(lag)
        for window in windows:
            if window <= lag:
                continue
            rolled = (
                shifted.groupby(grp_keys)
                .rolling(window, min_periods=1)
                .agg(["mean", "std"])
                .reset_index(level=list(range(len(group_cols))), drop=True)
            )
            df[f"sales_roll_mean_{window}_lag{lag}"] = rolled["mean"]
            df[f"sales_roll_std_{window}_lag{lag}"]  = rolled["std"].fillna(0)

    return df


# ─── Promotion Features ─────────────────────────────────────────────────────

def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["store_nbr", "family", "date"])

    df["onpromotion"] = df["onpromotion"].astype("int8")

    # Rolling promotion count over last 7 and 14 days
    shifted_promo = df.groupby(["store_nbr", "family"])["onpromotion"].shift(1)
    grp_keys = [df["store_nbr"], df["family"]]
    for window in [7, 14]:
        df[f"promo_count_{window}d"] = (
            shifted_promo.groupby(grp_keys)
            .rolling(window, min_periods=1).sum()
            .reset_index(level=[0, 1], drop=True)
        )

    return df


# ─── Oil Price Features ──────────────────────────────────────────────────────

def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")

    df["oil_7d_ma"] = df["oil_price"].shift(1).rolling(7, min_periods=1).mean()
    df["oil_28d_ma"] = df["oil_price"].shift(1).rolling(28, min_periods=1).mean()
    df["oil_pct_change_7d"] = df["oil_price"].pct_change(7)

    return df


# ─── Holiday & Event Features ────────────────────────────────────────────────

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    holiday_type_map = {
        "None": 0, "Holiday": 1, "Transfer": 2, "Bridge": 3,
        "Additional": 4, "Event": 5, "Work Day": 6,
    }
    df["holiday_type_enc"] = df["holiday_type"].map(holiday_type_map).fillna(0).astype("int8")
    df["is_national_holiday"] = df["is_national"].astype("int8")
    df["is_transferred"] = df["transferred"].astype("int8")

    # Days until/since nearest holiday — O(n log m) with searchsorted
    holiday_dates = df[df["is_holiday"] == 1]["date"].sort_values().unique()
    if len(holiday_dates) > 0:
        dates_d     = df["date"].values.astype("datetime64[D]")
        holidays_d  = holiday_dates.astype("datetime64[D]")
        # searchsorted gives index of first holiday >= each date
        idx_next = np.searchsorted(holidays_d, dates_d, side="left")
        idx_prev = idx_next - 1

        days_to = np.where(
            idx_next < len(holidays_d),
            (holidays_d[np.minimum(idx_next, len(holidays_d)-1)] - dates_d).astype("int64"),
            99
        )
        days_since = np.where(
            idx_prev >= 0,
            (dates_d - holidays_d[np.maximum(idx_prev, 0)]).astype("int64"),
            99
        )
        df["days_to_holiday"]    = np.clip(days_to,    0, 30).astype("int8")
        df["days_since_holiday"] = np.clip(days_since, 0, 30).astype("int8")

    return df


# ─── Store/Family Encodings ──────────────────────────────────────────────────

def add_store_family_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal encode categorical columns (LightGBM handles these natively)
    for col in ["family", "city", "state", "type", "holiday_type", "holiday_locale"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype("int16")

    df["cluster"] = df["cluster"].astype("int8")

    return df


# ─── Transaction Features ────────────────────────────────────────────────────

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["store_nbr", "date"])

    grp = df.groupby("store_nbr")["transactions"]
    df["transactions_lag7"] = grp.shift(7)
    shifted_tx = grp.shift(1)
    df["transactions_roll7"] = (
        shifted_tx.groupby(df["store_nbr"])
        .rolling(7, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    return df


# ─── Master Feature Builder ──────────────────────────────────────────────────

def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.
    """
    df = add_temporal_features(df)
    df = add_holiday_features(df)
    df = add_oil_features(df)
    df = add_promotion_features(df)
    df = add_transaction_features(df)

    if is_train:
        # Lag and rolling features require the target column to exist
        df = add_lag_features(df)
        df = add_rolling_features(df)

    df = add_store_family_encodings(df)

    # Drop columns not needed for modeling
    drop_cols = [
        "date", "id", "holiday_type", "holiday_locale",
        "city", "state",  # already encoded above
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


FEATURE_COLS = [
    "store_nbr", "family", "onpromotion",
    "day_of_week", "day_of_month", "month", "year", "week_of_year",
    "quarter", "is_weekend", "is_month_start", "is_month_end", "day_of_year",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "cluster", "type",
    "oil_price", "oil_7d_ma", "oil_28d_ma", "oil_pct_change_7d",
    "is_holiday", "holiday_type_enc", "is_national_holiday",
    "is_transferred", "days_to_holiday", "days_since_holiday",
    "promo_count_7d", "promo_count_14d",
    "transactions", "transactions_lag7", "transactions_roll7",
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_35",
    "sales_lag_364", "sales_lag_365",
    "sales_roll_mean_14_lag7", "sales_roll_mean_28_lag7",
    "sales_roll_std_14_lag7",
    "sales_roll_mean_28_lag14",
]
