# 50 Startups Profit Prediction using Scikit-Learn (CRISP-DM)

This project solves the classic **50 Startups Multiple Linear Regression** problem using **scikit-learn**, structured strictly around the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

---

## CRISP-DM Framework Phases

```
+-------------------------------------------------------------+
|                     1. Business Understanding               |
|      - Objective: Predict startup profit from spending      |
|      - Goal: Assist VCs & founders in resource allocation   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                       2. Data Understanding                 |
|      - 50 records: R&D Spend, Admin, Marketing, State, Profit|
|      - Correlation analysis, distribution & outlier checks   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                       3. Data Preparation                   |
|      - Feature/target splitting (X, y)                      |
|      - One-Hot Encoding for 'State' (avoid dummy trap)      |
|      - Feature scaling with StandardScaler                  |
|      - Leakage-free ColumnTransformer & Pipeline            |
|      - 80/20 Train-Test split                               |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                         4. Modeling                         |
|      - Multiple Linear Regression (OLS baseline)            |
|      - Ridge Regression (L2 regularization with CV)         |
|      - Lasso Regression (L1 feature selection with CV)      |
|      - 5-Fold Cross-Validation on training data             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                         5. Evaluation                       |
|      - Metrics: R², Adjusted R², MAE, RMSE, MAPE            |
|      - Regression assumptions check:                        |
|          * Linearity & Homoscedasticity                     |
|          * Normality of residuals (Shapiro-Wilk)            |
|          * Autocorrelation (Durbin-Watson)                  |
|      - Feature coefficient economic interpretation          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                         6. Deployment                       |
|      - Pipeline serialization (startup_profit_pipeline.joblib|
|      - Real-time inference function for new startup pitches |
+-------------------------------------------------------------+
```

---

## File Structure

- `50_Startups.csv` : Dataset containing spending and profit metrics for 50 companies.
- `solve_startups_crisp_dm.py` : Comprehensive end-to-end CRISP-DM Python script.
- `artifacts/` :
  - `eda_overview.png` : Exploratory data distributions and scatter plots.
  - `model_diagnostics.png` : Actual vs. Predicted and Residual plots.
  - `startup_profit_pipeline.joblib` : Exported production-ready scikit-learn pipeline.

---

## How to Run

Execute the main script:
```bash
python solve_startups_crisp_dm.py
```
