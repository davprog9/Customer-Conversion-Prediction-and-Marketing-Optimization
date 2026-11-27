# Customer Conversion Prediction & Marketing Optimization

> Predicting which bank customers are most likely to subscribe to a term deposit, so telemarketing effort can be focused where it pays off.

**Status:** 🚧 Work in progress. Exploratory analysis and model benchmarking are complete; final model selection, test-set evaluation and customer ranking are still pending (see [Roadmap](#roadmap)).

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Key EDA Findings](#key-eda-findings)
6. [Modeling](#modeling)
7. [Results](#results)
8. [Known Limitations & Notes](#known-limitations--notes)
9. [Roadmap](#roadmap)
10. [Getting Started](#getting-started)

---

## Business Problem

A bank runs telemarketing campaigns to sell term deposits. Calling every customer is expensive and most calls do not convert. The goal of this project is twofold:

1. **Understand** which customer segments, timing patterns and campaign histories are associated with a higher subscription rate.
2. **Predict** the probability that a given customer will subscribe, so the bank can prioritize who to call, reducing cost and increasing conversion rate.

## Dataset

The project uses the Bank Marketing dataset (`bank.csv`), loaded from `../Data/bank.csv`.

| Property | Value |
|---|---|
| Rows | 11,162 |
| Columns | 17 (16 features + target) |
| Missing values | None (`unknown` is used as a category in several columns) |
| Duplicate rows | None |
| Target | `deposit` (yes/no), mapped to 1/0 |
| Class balance | ~47.4% yes / ~52.6% no (approximately balanced) |

**Features:**

| Group | Columns |
|---|---|
| Demographics | `age`, `job`, `marital`, `education` |
| Financial | `default`, `balance`, `housing`, `loan` |
| Contact details | `contact`, `day`, `month`, `duration` |
| Campaign history | `campaign`, `pdays`, `previous`, `poutcome` |

## Project Structure

```
.
├── Data/
│   └── bank.csv            # Dataset (not included here; place it in this folder)
├── marketing.ipynb         # Full analysis: EDA -> feature prep -> modeling
└── README.md
```

## Methodology

The notebook follows a standard end-to-end workflow:

```
Raw data -> Cleaning & target encoding -> EDA & segmentation
         -> Leakage / multicollinearity checks -> Train/test split
         -> Preprocessing pipelines -> 10-fold CV benchmarking
         -> Hyperparameter tuning -> (pending) final evaluation & ranking
```

### 1. Data cleaning

- Checked data types, nulls and duplicates (none found).
- Encoded the target `deposit` as binary (`yes` = 1, `no` = 0).
- Reviewed categorical distributions (`job`, `education`, `contact`, `poutcome`, `default`, `housing`, `loan`).

### 2. Outlier handling

`balance` is heavily right-skewed (range −6,847 to 81,204; median 550). IQR boundaries flagged anything below −2,257 or above 4,087 as outliers. **Outliers were deliberately kept**: the 228 customers with a balance above 10,000 convert at **59%**, so they carry real signal rather than noise.

### 3. Feature engineering (for EDA)

Binned features were created to analyze segment-level conversion rates:

- `balance_group`: Negative / Low / Medium / High
- `age_group`: 18–30, 30–45, 45–60, 60–100
- `duration_groups`: 10 equal-width bins

### 4. Data leakage prevention

`duration` (call length) is strongly associated with conversion, but it is only known **after** a call ends, so it can't be used to decide whom to call. It is used in EDA to understand behavior but **excluded from every model**, along with the derived binned columns (`duration_groups`, `age_group`, `balance_group`).

### 5. Multicollinearity check

Variance Inflation Factors were computed on the one-hot encoded feature set. All features were below 6.1. The highest was `poutcome_unknown` (≈6.0), which was kept because it's a meaningful predictor and its overlap with the other `poutcome` levels is expected. No features were removed.

### 6. Train/test split

80/20 stratified split (`random_state=42`). **All tuning and cross-validation used the training set only**; the test set has not yet been touched.

### 7. Preprocessing pipelines

Built with scikit-learn `Pipeline` + `ColumnTransformer` so preprocessing is fit inside each CV fold (no leakage across folds):

| Model family | Numeric | Categorical |
|---|---|---|
| Logistic Regression, KNN, SVM | `StandardScaler` | One-hot |
| Random Forest, Gradient Boosting, XGBoost | passthrough | One-hot |

## Key EDA Findings

### Who converts the most?

| Segment | Observation |
|---|---|
| Previous campaign success (`poutcome = success`) | **91%** conversion, regardless of balance |
| Students | **75%** conversion, strong across all balance groups |
| Retired | **66%** conversion |
| High balance (> 10,000) | **59%** conversion (n = 228) |
| Age 60+ | High conversion across Low to High balance |
| Age 18–30 | Converts with Medium to High balance |
| Age 30–60 | Converts mainly with higher balance |
| Tertiary education | 54% vs 39% for primary education |
| No housing loan | 57% vs 37% for those with one |
| Customers with a prior success and fewer than 15 contacts | > 80% conversion |

### Balance

| Balance group | Conversion rate | Customers |
|---|---|---|
| Negative | 30.5% | 688 |
| Low (0 – 1,200) | 44.3% | 6,840 |
| Medium (1,200 – 4,000) | 55.9% | 2,556 |
| High (4,000+) | 57.3% | 1,078 |

Higher balance is consistently associated with higher conversion.

### Seasonality

| Strongest months | Conversion | Weakest months | Conversion |
|---|---|---|---|
| Dec | 90.9% (n = 110) | May | 32.8% (n = 2,824) |
| Mar | 89.9% (n = 276) | Jul | 41.4% (n = 1,514) |
| Sep | 84.3% (n = 319) | Jan | 41.3% (n = 344) |
| Oct | 82.4% (n = 392) | Jun | 44.7% (n = 1,222) |

Note that the best-performing months have far fewer contacts, while May alone accounts for ~25% of all calls with the lowest conversion rate. This suggests a possible reallocation opportunity (subject to the sample-size caveats below).

### Call duration (EDA only)

Conversion climbs from ~32% for calls under ~390 s to ~87–94% for calls over ~780 s. This is a valuable behavioral insight but is excluded from models to avoid leakage.

### Contact attempts and `poutcome`

- Most conversions occur within fewer than 16 contact attempts; a few outliers converted after 21–55.
- `unknown` previous outcome with very few attempts converts at a high rate, hinting that some of these records may actually be prior successes (worth investigating).

## Modeling

Six classifiers were benchmarked with **10-fold stratified cross-validation** on the training set, each with a baseline and a `RandomizedSearchCV`-tuned version optimized for ROC-AUC.

| Model | Tuned hyperparameters |
|---|---|
| Logistic Regression | `C`, `class_weight` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight` |
| K-Nearest Neighbors | `n_neighbors`, `weights`, `p`, `leaf_size` |
| SVM (SVC) | `C`, `kernel`, `class_weight` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_*`, `subsample` |
| XGBoost | `max_depth`, `min_child_weight`, `gamma`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree` |

Metrics tracked: Accuracy, Precision, Recall, F1, ROC-AUC.

## Results

Cross-validated performance on the training set (mean over folds):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (baseline) | ROC-AUC (tuned) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.704 | 0.741 | 0.579 | 0.649 | 0.763 | 0.763 |
| K-Nearest Neighbors | 0.684 | 0.687 | 0.611 | 0.646 | 0.727 | 0.771 |
| Random Forest | 0.728 | 0.746 | 0.647 | 0.693 | 0.782 | 0.789 |
| SVM (RBF) | 0.732 | **0.802** | 0.578 | 0.672 | 0.780 | 0.781 |
| Gradient Boosting | 0.734 | 0.779 | 0.612 | 0.685 | 0.787 | 0.793 |
| XGBoost | **0.738** | 0.785 | 0.616 | **0.690** | 0.793 | **0.795** |

*Accuracy through F1 are baseline-model values. Tuned values for those metrics were not recorded in the notebook.*

**Takeaways**

- **Tree-based boosting (XGBoost, Gradient Boosting) and Random Forest lead** on ROC-AUC, all landing at roughly 0.79–0.795.
- **Tuning gave marginal gains** for most models (≤ 0.01 ROC-AUC); the exception is KNN, which improved from 0.727 to 0.771.
- **SVM is the most conservative** model: highest precision (~80%) but lowest recall (~58%). It is a candidate if the priority is minimizing wasted calls rather than capturing every subscriber.
- **Logistic Regression is a useful interpretable baseline.** Its strongest positive coefficients are `poutcome_success`, `month_mar`, `month_dec`, `month_oct`, `job_student` and `month_sep`, which agrees with the EDA.
- Performance appears to plateau around 0.79 ROC-AUC, which suggests the remaining gains are more likely to come from feature engineering and threshold selection than from more hyperparameter search.

## Known Limitations & Notes

Documented honestly so that they can be addressed before any results are published:

- **No test-set evaluation yet.** All reported numbers are cross-validation scores on the training split.
- **Statistical rigor in EDA is incomplete.** Segment conversion rates (e.g. December, March) are based on small samples; confidence intervals (Wilson) and minimum sample-size rules are still TODO. Observed differences should not yet be treated as firm business conclusions.
- **`contact = unknown` and `poutcome = unknown`** are large categories (21% and 75% of rows) treated as ordinary levels.
- **`age_group` has 8 missing values** because the lowest bin edge (18) excludes customers aged exactly 18. This does not affect modeling since the column is dropped.
- **Notebook text vs. outputs.** A few markdown comments refer to "GridSearchCV" and "5-step cross validation" while the code uses `RandomizedSearchCV` and 10-fold CV. The baseline SVC output shows 5 folds, suggesting it was run with an earlier configuration. Results in this README follow the recorded outputs and these should be re-run for consistency.
- **`balance_group` code comment** mentions `qcut`, but fixed-width `pd.cut` bins are used.
- **Random seeds vary** (42 for the split, 99 for CV, mixed for models); consider centralizing them.
- **Small search budgets.** Several searches sample only 10–25 candidates.

## Roadmap

- [ ] Add Wilson confidence intervals and minimum sample-size thresholds to segment analysis
- [ ] Select a final model (likely XGBoost or Gradient Boosting) and evaluate on the held-out test set
- [ ] Choose a decision threshold based on business costs (cost per call vs. value of a subscription); report precision/recall trade-offs, and add ROC and precision-recall curves
- [ ] Add feature importance / SHAP for the tree-based models
- [ ] **Rank customers and identify the top 10% to call**, with expected lift/gain vs. random calling
- [ ] Investigate whether `poutcome = unknown` with low contact counts hides prior successes
- [ ] Feature engineering (e.g. `pdays = -1` flag, month/season grouping, balance transformations)
- [ ] Persist the final pipeline (`joblib`) and move reusable code out of the notebook
- [ ] Add `requirements.txt`

## Getting Started

### Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `statsmodels`, `xgboost`
- `jupyter`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost jupyter
```

### Run

1. Place `bank.csv` in a `Data/` folder one level above the notebook (path used: `../Data/bank.csv`), or adjust the path in the first data-loading cell.
2. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook marketing.ipynb
   ```

3. Run all cells top to bottom. Note that the hyperparameter searches (especially SVM, Random Forest and Gradient Boosting) can take a while.

---

*This README documents work in progress and will be updated as the project advances.*
