"""
Error analysis for grocery sales forecasting.

Breaks down model performance by store, product family, holiday type,
and promotion period to identify systematic failure modes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from validation import rmsle, wape


def error_by_dimension(
    df_val: pd.DataFrame,
    y_true_col: str = "sales",
    y_pred_col: str = "predicted",
    dimension: str = "family",
) -> pd.DataFrame:
    """
    Compute RMSLE and WAPE for each unique value of a dimension.
    Useful for identifying which stores/families the model struggles with.
    """
    results = []
    for val, group in df_val.groupby(dimension):
        y_true = group[y_true_col].values
        y_pred = group[y_pred_col].values
        results.append({
            dimension: val,
            "rmsle": rmsle(y_true, y_pred),
            "wape": wape(y_true, y_pred),
            "n_rows": len(group),
            "mean_sales": y_true.mean(),
            "total_sales": y_true.sum(),
        })

    return pd.DataFrame(results).sort_values("rmsle", ascending=False)


def error_by_holiday_type(
    df_val: pd.DataFrame,
    y_true_col: str = "sales",
    y_pred_col: str = "predicted",
) -> pd.DataFrame:
    return error_by_dimension(df_val, y_true_col, y_pred_col, "holiday_type")


def error_by_promotion(
    df_val: pd.DataFrame,
    y_true_col: str = "sales",
    y_pred_col: str = "predicted",
) -> pd.DataFrame:
    """Compare RMSLE on promoted vs. non-promoted days."""
    promo_col = "onpromotion"
    if promo_col not in df_val.columns:
        raise ValueError("Need 'onpromotion' column in df_val")

    results = []
    for promo_val, label in [(0, "No Promotion"), (1, "On Promotion")]:
        group = df_val[df_val[promo_col] == promo_val]
        if len(group) == 0:
            continue
        results.append({
            "promotion_status": label,
            "rmsle": rmsle(group[y_true_col].values, group[y_pred_col].values),
            "wape": wape(group[y_true_col].values, group[y_pred_col].values),
            "n_rows": len(group),
        })
    return pd.DataFrame(results)


def plot_error_heatmap(
    error_df: pd.DataFrame,
    dimension: str,
    metric: str = "rmsle",
    figsize: tuple = (12, 8),
    title: str = "",
) -> None:
    top_n = error_df.nlargest(20, metric)
    plt.figure(figsize=figsize)
    sns.barplot(data=top_n, y=dimension, x=metric, palette="Reds_r")
    plt.title(title or f"Top 20 worst {dimension}s by {metric.upper()}")
    plt.xlabel(metric.upper())
    plt.tight_layout()
    plt.savefig(f"outputs/error_by_{dimension}.png", dpi=150)
    plt.show()


def plot_predictions_vs_actuals(
    df_val: pd.DataFrame,
    store_nbr: int,
    family: str,
    y_true_col: str = "sales",
    y_pred_col: str = "predicted",
) -> None:
    subset = df_val[
        (df_val["store_nbr"] == store_nbr) & (df_val["family"] == family)
    ].sort_values("date")

    plt.figure(figsize=(14, 5))
    plt.plot(subset["date"], subset[y_true_col], label="Actual", linewidth=1.5)
    plt.plot(subset["date"], subset[y_pred_col], label="Predicted",
             linestyle="--", linewidth=1.5)
    plt.title(f"Store {store_nbr} | {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/pred_vs_actual_store{store_nbr}_{family}.png", dpi=150)
    plt.show()


def feature_importance_report(
    model,
    feature_cols: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    import lightgbm as lgb
    import xgboost as xgb

    if isinstance(model, lgb.Booster):
        importances = model.feature_importance(importance_type="gain")
    elif isinstance(model, xgb.Booster):
        scores = model.get_score(importance_type="gain")
        importances = np.array([scores.get(f, 0) for f in feature_cols])
    else:
        raise ValueError("Model must be lgb.Booster or xgb.Booster")

    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp, y="feature", x="importance", palette="Blues_r")
    plt.title(f"Top {top_n} Feature Importances (Gain)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.show()

    return df_imp


def full_error_report(
    df_val: pd.DataFrame,
    y_true_col: str = "sales",
    y_pred_col: str = "predicted",
) -> None:
    print("=" * 60)
    print("ERROR ANALYSIS REPORT")
    print("=" * 60)

    overall_rmsle = rmsle(df_val[y_true_col].values, df_val[y_pred_col].values)
    overall_wape = wape(df_val[y_true_col].values, df_val[y_pred_col].values)
    print(f"\nOverall RMSLE: {overall_rmsle:.4f}")
    print(f"Overall WAPE:  {overall_wape:.4f}")

    print("\n--- By Product Family (top 10 worst) ---")
    fam_errors = error_by_dimension(df_val, dimension="family")
    print(fam_errors.head(10).to_string(index=False))

    print("\n--- By Store (top 10 worst) ---")
    store_errors = error_by_dimension(df_val, dimension="store_nbr")
    print(store_errors.head(10).to_string(index=False))

    print("\n--- By Promotion Status ---")
    promo_errors = error_by_promotion(df_val)
    print(promo_errors.to_string(index=False))

    print("\n--- By Holiday Type ---")
    holiday_errors = error_by_holiday_type(df_val)
    print(holiday_errors.to_string(index=False))
