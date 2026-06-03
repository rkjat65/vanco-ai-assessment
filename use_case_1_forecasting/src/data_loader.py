"""
Data loading and merging for Corporacion Favorita sales forecasting.

Handles all 6 source tables, foreign-key joins, and initial cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_all_tables(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all raw CSV tables from the Kaggle competition directory."""
    data_dir = Path(data_dir)
    tables = {}

    tables["train"] = pd.read_csv(
        data_dir / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int8"},
    )
    tables["train"]["onpromotion"] = tables["train"]["onpromotion"].fillna(0).astype("int8")

    tables["test"] = pd.read_csv(
        data_dir / "test.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int8"},
    )
    tables["test"]["onpromotion"] = tables["test"]["onpromotion"].fillna(0).astype("int8")
    tables["stores"] = pd.read_csv(data_dir / "stores.csv", dtype={"store_nbr": "int8"})
    tables["oil"] = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    tables["holidays"] = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    tables["transactions"] = pd.read_csv(
        data_dir / "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int8"},
    )

    print(f"Loaded tables: { {k: v.shape for k, v in tables.items()} }")
    return tables


def build_master_frame(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge all tables into a single training frame and test frame.

    Returns (train_df, test_df) with all joined features.
    """
    train = tables["train"].copy()
    test = tables["test"].copy()
    stores = tables["stores"].copy()
    oil = tables["oil"].copy()
    holidays = tables["holidays"].copy()
    transactions = tables["transactions"].copy()

    # --- Oil: forward-fill gaps (oil prices don't publish on weekends) ---
    date_range = pd.DataFrame(
        {"date": pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")}
    )
    oil = date_range.merge(oil, on="date", how="left")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)

    # --- Holidays: collapse to one row per date with dominant type ---
    # Priority: national > regional > local; transferred > bridge > event > holiday
    holidays["is_national"] = holidays["locale"] == "National"
    holidays["is_regional"] = holidays["locale"] == "Regional"
    holidays_agg = (
        holidays.groupby("date")
        .agg(
            holiday_type=("type", "first"),
            holiday_locale=("locale", "first"),
            transferred=("transferred", "any"),
            is_national=("is_national", "any"),
        )
        .reset_index()
    )
    holidays_agg["is_holiday"] = 1

    # --- Transactions: one row per (date, store_nbr) ---
    # Already correct shape; keep as is

    def merge_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(stores, on="store_nbr", how="left")
        df = df.merge(oil, on="date", how="left")
        df = df.merge(holidays_agg, on="date", how="left")
        df = df.merge(transactions, on=["date", "store_nbr"], how="left")

        df["is_holiday"]  = df["is_holiday"].fillna(0).astype("int8")
        df["transferred"] = df["transferred"].fillna(0).astype("int8")
        df["is_national"] = df["is_national"].fillna(0).astype("int8")
        df["holiday_type"] = df["holiday_type"].fillna("None")
        df["holiday_locale"] = df["holiday_locale"].fillna("None")

        return df

    train = merge_all(train)
    test = merge_all(test)

    print(f"Master train shape: {train.shape}")
    print(f"Master test shape:  {test.shape}")

    return train, test
