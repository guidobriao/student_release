# Task 1: Harmful Algal Bloom (HAB) Binary Classification

This document summarizes the feature engineering, missing-value handling, class-imbalance strategy, spatial encoding, and chronological validation workflow used for **Part II — Task 1** (predicting daily `HAB` occurrence).

---

## 1. Engineered Features (`fe_trig_log` and `fe_full`)

* **`CHL_is_missing`**: Binary indicator (`1.0` if `CHL` is `NaN`, else `0.0`) recording whether satellite chlorophyll data was missing on that row.
* **`log1p_chl`**: $\ln(1 + \max(\text{CHL}, 0))$, compressing right-skewed chlorophyll concentrations (`CHL` ranges from `0.354` to `62.732`, median `2.925`, mean `5.658`).
* **`sin_doy`**: $\sin(2\pi \cdot \text{DayOfYear} / 365)$, paired with `YDOY` so annual seasonality has both sine and cosine components around the year without a discontinuity between December 31 and January 1.
* **`log1p_chl_unc`**: $\ln(1 + \max(\text{CHL\_uncertainty}, 0))$, compressing skewed uncertainty values (`6.740` to `92.740`).
* **`chl_snr`**: Signal-to-uncertainty ratio $\text{CHL} / (\text{CHL\_uncertainty} + 10^{-3})$, weighting chlorophyll concentration relative to its measurement uncertainty.

---

## 2. Missing Values Handling

* **`CHL_is_missing`**: Binary indicator (`1.0` if `CHL` is `NaN`, else `0.0`) recording whether satellite chlorophyll data was missing on that row.
* **Imputed train (`comp_only=False`)**: Trains on all rows in the training split after explicitly imputing missing values (`CHL`, `CHL_uncertainty`, and derived `CHL` features) using medians computed **only** on that training split (plus the `CHL_is_missing` indicator column).
* **Complete-rows-only train (`comp_only=True`)**: Filters the training split first to rows where both `CHL` and `CHL_uncertainty` are non-missing, and trains only on those complete rows (matching the complete daily samples of the 2021–2023 test set while preserving the explicit median-fill fallback at inference time).

---

## 3. Class Imbalance Handling

* **`class_weight="balanced"` and `"balanced_subsample"`**: Passed `class_weight="balanced"` for models that support it (`LogisticRegression`, `HistGradientBoostingClassifier`, `RandomForestClassifier`, and `SVC`), and tested `balanced_subsample` on `RandomForestClassifier` (and `class_weight=None` on `LogisticRegression`) during tuning. Because `scikit-learn`'s `MLPClassifier` does not accept `class_weight`, we tuned its capacity and regularization by disabling early stopping (`early_stopping=False`) so training runs up to `max_iter` with moderate L2 regularization (`alpha=0.5`) to avoid premature stopping and overfitting.
* **Expanding chronological windows**: Multi-year expanding folds (`<=2016 -> 2017–2018`, `<=2018 -> 2019`, `<=2019 -> 2020`) preserve strict time ordering and ensure every validation fold contains positive events (`22`, `29`, and `31` `HAB=1` cases respectively).

---

## 4. Spatial Clusters Handling

Because `spatial_cluster` is a discrete categorical cluster ID (`0–5`) rather than an ordered continuous quantity:

* **Explicit one-hot encoding**: Encoded into six binary indicator columns (`cluster_0` through `cluster_5`) using Pandas/NumPy so linear and neural models assign an independent weight to each cluster without imposing an artificial numerical ordering.
* **Deterministic sorting**: Rows are explicitly sorted by `["Date", "spatial_cluster"]` to guarantee identical row ordering across training splits, validation folds, and prediction calls (`probs[-1]`).

---

## 5. Compliance & Serialization Rules

* **No forbidden wrappers**: All feature construction, missing-value imputation, and standardization are implemented explicitly in Pandas/NumPy without `Pipeline`, `ColumnTransformer`, scikit-learn imputers, or calibration wrappers.
* **Plain numerical state in `model.pkl`**: Standardization parameters (`scaler_mean`, `scaler_scale`) and imputation medians (`fill_stats`) are stored as plain NumPy arrays and dictionaries alongside the fitted estimator rather than pickled transformer objects.

---

## 6. Reproduction Script (`117/task1_experiments.py`)

The unified experiment script (`117/task1_experiments.py`) evaluates all combinations across `fe_trig_log` and `fe_full`, All Train and Complete Train, and the candidate configurations for both expanding chronological CV (`<=2016 -> 2017–2018`, `<=2018 -> 2019`, `<=2019 -> 2020`) and the single chronological split (`<=2018 -> 2019–2020`, `n=321`).

Run it from your project root using a relative path:

```bash
python "task1_experiments.py"
```