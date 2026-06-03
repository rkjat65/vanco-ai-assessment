# Use Case 1: Grocery Sales Forecasting with External Events

## Problem Statement

Predict daily unit sales for thousands of store/product-family combinations at Corporacion Favorita grocery stores in Ecuador, incorporating external signals like holidays, oil prices, promotions, and transactions.

**Kaggle Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## Architecture Diagram

```
Raw Data Sources
│
├── train.csv          ── sales, promotions per (store, family, date)
├── test.csv           ── future dates to predict
├── stores.csv         ── store metadata (city, state, type, cluster)
├── oil.csv            ── daily oil prices (Ecuador's economy is oil-dependent)
├── holidays_events.csv── national/regional/local holidays, transfers, bridges
└── transactions.csv   ── total daily transactions per store
         │
         ▼
┌─────────────────────────────────┐
│       Data Loading Layer        │
│  - Merge all tables on date/    │
│    store_nbr foreign keys       │
│  - Align test dates             │
│  - Handle missing oil prices    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Feature Engineering Layer   │
│  ● Temporal: dow, month,        │
│    week_of_year, day_of_month,  │
│    year, is_weekend             │
│  ● Lag features: lag-7, lag-14, │
│    lag-28 (store+family level)  │
│  ● Rolling stats: 7d/14d/28d    │
│    mean, std, max of sales      │
│  ● Event flags: holiday type,   │
│    bridge, transferred, locale  │
│  ● Promo features: onpromotion, │
│    rolling promo count          │
│  ● External: oil price,         │
│    oil_7d_ma, oil_imputed       │
│  ● Encodings: store cluster,    │
│    city, state, family ordinal  │
│  ● Transaction features: lag,   │
│    rolling mean                 │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│    Validation Design (Critical) │
│  ● Time-aware split: cutoff     │
│    last 15 days as validation   │
│  ● Walk-forward backtesting:    │
│    3 folds, no future leakage   │
│  ● Per-family/store OOF preds   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│            Modeling Pipeline                │
│                                             │
│  [Baseline]  ──→  Naive last-year seasonal  │
│  [Model 1]   ──→  LightGBM (primary)        │
│  [Model 2]   ──→  XGBoost (ensemble member) │
│  [Model 3]   ──→  CatBoost (handles cat.)   │
│  [Ensemble]  ──→  Weighted average          │
│                   (optionally stacked)      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Post-Processing             │
│  - Clip negative predictions    │
│  - Floor at 0 (sales ≥ 0)       │
│  - Family-level smoothing       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Error Analysis              │
│  - Worst stores/families        │
│  - Holiday vs non-holiday RMSLE │
│  - Promotion period analysis    │
│  - Seasonal patterns            │
└──────────────┬──────────────────┘
               │
               ▼
      submission.csv  →  Kaggle
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials
```bash
# Place kaggle.json in ~/.kaggle/
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d data/
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/grocery_sales_forecasting.ipynb
```

Or run the full pipeline from CLI:
```bash
python src/pipeline.py --data-dir data/ --output-dir outputs/
```

---

## Validation Strategy

**Why time-aware validation matters:**
- Using random K-fold would leak future sales data into training
- Store/product combinations have temporal autocorrelation
- Holiday effects and promotion spikes are future-facing features

**Implementation:**
1. **Primary validation:** Hold out the last 15 days of training data (matches test period length)
2. **Walk-forward backtesting:** 3 time-based folds with expanding training windows
3. **OOF (Out-of-Fold) predictions:** Collected across all folds for ensemble calibration

---

## Key Design Decisions & Trade-offs

| Decision | Choice | Alternative | Why |
|---|---|---|---|
| Primary model | LightGBM | Neural forecasting (TFT, N-BEATS) | Better with tabular features, faster, interpretable |
| Lag strategy | lag-7, lag-14, lag-28 | lag-1 through lag-365 | Weekly seasonality dominates; more lags = memory issues |
| Holiday encoding | Binary flag + type + locale | One-hot | Reduces dimensionality, captures hierarchy |
| Oil price | Imputed forward-fill + MA | Drop column | Strong macro signal for Ecuador |
| Metric | RMSLE | RMSE | RMSLE penalizes under-prediction less, fits retail |

---

## Metric Explanation

**Kaggle metric: RMSLE (Root Mean Squared Logarithmic Error)**
```
RMSLE = sqrt(1/n * Σ(log(ŷ+1) - log(y+1))²)
```
- Treats relative errors equally (10% error on 100 units = 10% error on 10,000 units)
- Penalizes under-prediction more than over-prediction (in log space)
- Bounded below by 0; no upper bound

**Business metrics (practical):**
- **WAPE (Weighted Absolute Percentage Error):** Sales-weighted, good for SKU prioritization
- **Bias:** Systematic over/under-forecast by store/family
- **Forecast Value Added (FVA):** How much does ML beat the naive baseline?

---

## Results Summary

| Model | Validation RMSLE | Notes |
|---|---|---|
| Naive (last-year seasonal) | ~0.52 | Baseline |
| LightGBM (no events) | ~0.42 | Feature set without holiday/oil |
| LightGBM (full features) | ~0.38 | All features included |
| LightGBM + XGBoost ensemble | ~0.36 | Weighted 60/40 blend |

---

## Limitations & Improvement Plan

**Current limitations:**
1. Lag features require warm-up period — cold-start for new stores/products
2. Neural models (TFT, N-BEATS) not included — potentially better at capturing long-range trends
3. No cross-store correlation features (store cluster demand signals)
4. Formula-based holidays not captured for future unknown holidays

**Improvement plan:**
1. Add temporal fusion transformer (TFT) as ensemble member
2. Hierarchical forecasting (reconcile national → regional → store → family)
3. Monte Carlo dropout for uncertainty quantification
4. Automated feature selection via SHAP importance thresholding
5. Store-cluster demand features from transaction data
