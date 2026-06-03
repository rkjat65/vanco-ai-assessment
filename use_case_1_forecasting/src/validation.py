"""
Time-aware validation strategies for time series forecasting.

Critical design principle: NEVER allow future data to appear in training.
All splits are based on cutoff dates, not random sampling.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Generator


@dataclass
class TimeSeriesFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def walk_forward_folds(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 3,
    horizon_days: int = 16,
    min_train_days: int = 180,
) -> list[TimeSeriesFold]:
    """
    Generate walk-forward validation folds.

    Each fold has an expanding training window and a fixed-size validation window
    that looks exactly like the test period.

    Why walk-forward instead of random k-fold:
    - Simulates how the model will be used in production
    - Respects temporal ordering — no future data leaks into training
    - Multiple folds give more reliable error estimates than a single hold-out
    """
    dates = sorted(df[date_col].unique())
    max_date = max(dates)
    min_date = min(dates)

    folds = []
    for fold_id in range(n_folds):
        # Each fold's validation window ends progressively earlier
        val_end = max_date - pd.Timedelta(days=fold_id * horizon_days)
        val_start = val_end - pd.Timedelta(days=horizon_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        train_start = min_date

        # Skip fold if training window is too short
        train_days = (train_end - train_start).days
        if train_days < min_train_days:
            continue

        folds.append(TimeSeriesFold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

    return folds


def split_fold(
    df: pd.DataFrame,
    fold: TimeSeriesFold,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) for a given fold."""
    train = df[(df[date_col] >= fold.train_start) & (df[date_col] <= fold.train_end)]
    val = df[(df[date_col] >= fold.val_start) & (df[date_col] <= fold.val_end)]
    return train, val


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error — official Kaggle metric.

    Clips predictions to 0 to avoid log of negative numbers.
    """
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — weighted by true sales volume."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1e-8)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean forecast bias (positive = over-forecast, negative = under-forecast)."""
    return np.mean(y_pred - y_true)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    metrics = {
        "rmsle": rmsle(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if label:
        print(f"[{label}] RMSLE={metrics['rmsle']:.4f} | WAPE={metrics['wape']:.4f} | Bias={metrics['bias']:.2f}")
    return metrics
