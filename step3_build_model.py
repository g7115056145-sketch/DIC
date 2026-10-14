"""
Step 3: Build Model for 50 Startups Multiple Linear Regression
=============================================================
Topics covered:
1. Model Instantiation (sklearn LinearRegression)
2. Model Training via Ordinary Least Squares (OLS) fitting
3. Parameter Extraction: Intercept (beta_0) and Coefficients (beta_1 ... beta_p)
4. Formulating the Mathematical Equation
5. Model Predictions on Train and Test Sets (Actual vs Predicted)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# =============================================================================
# 0. PREREQUISITES: DATA PREPARATION (FROM STEP 1 & 2)
# =============================================================================
csv_path = "50_Startups.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} not found.")

df = pd.read_csv(csv_path)

numerical_cols = ["R&D Spend", "Administration", "Marketing Spend"]
for col in numerical_cols:
    df[col] = df[col].astype(np.float64)
df["Profit"] = df["Profit"].astype(np.float64)

X = df.drop(columns=["Profit"])
y = df["Profit"]

# OneHotEncoder with drop='first' on 'State'
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
    ],
    remainder="passthrough"
)

X_encoded = ct.fit_transform(X)
cat_feature_names = list(ct.named_transformers_["cat"].get_feature_names_out(["State"]))
feature_names = cat_feature_names + numerical_cols
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded_df, y, test_size=0.2, random_state=42
)

print("=" * 80)
print("STEP 3: BUILD MODEL (構建多元線性迴歸模型)")
print("=" * 80)

# =============================================================================
# 1. MODEL INSTANTIATION (模型實例化)
# =============================================================================
print("\n[PART 1: MODEL INSTANTIATION]")
print("-" * 80)
print("使用 Scikit-Learn 的 LinearRegression() 類別：")
print("  - 算法原理：最小平方法 (Ordinary Least Squares, OLS)")
print("  - 優化目標：最小化殘差平方和 (Minimize Sum of Squared Residuals):")
print("            min sum((y_i - y_hat_i)^2)")
print("  - 解析解公式 (Normal Equation): beta = (X^T * X)^(-1) * X^T * y\n")

regressor = LinearRegression(fit_intercept=True)
print(f"模型物件初始化完成: {regressor}")

# =============================================================================
# 2. MODEL TRAINING / FITTING (模型訓練 / 擬合)
# =============================================================================
print("\n[PART 2: MODEL TRAINING (FITTING)]")
print("-" * 80)
print("呼叫 regressor.fit(X_train, y_train) 進行權重學習...")
regressor.fit(X_train, y_train)
print(">> 模型擬合完成！已成功從 40 筆訓練樣本中學習到最佳迴歸權重。")

# =============================================================================
# 3. EXTRACT MODEL PARAMETERS (提取模型參數: 截距與迴歸係數)
# =============================================================================
print("\n[PART 3: LEARNED MODEL PARAMETERS]")
print("-" * 80)

intercept = regressor.intercept_
coefficients = regressor.coef_

print(f"截距 (Intercept, beta_0): ${intercept:,.2f}")
print("各特徵迴歸係數 (Coefficients, beta_1 ~ beta_5):")

coef_table = pd.DataFrame({
    "Feature Name": feature_names,
    "Coefficient (beta)": coefficients
})
print(coef_table.to_string(index=False))

# =============================================================================
# 4. MATHEMATICAL FORMULATION (構建數學方程式)
# =============================================================================
print("\n[PART 4: MATHEMATICAL REGRESSION EQUATION]")
print("-" * 80)
print("根據學到的參數，構建多元線性迴歸預測方程式：")
equation_lines = [f"Profit = ${intercept:,.2f}"]
for name, coef in zip(feature_names, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation_lines.append(f"       {sign} {abs(coef):.4f} * [{name}]")
print("\n".join(equation_lines))

print("\n係數意義解析 (Economic Insights):")
print("1. R&D Spend (beta = +0.8056): 每投入 $1.00 於研發，預期增加約 $0.81 獲利，是利潤的核心驅動力。")
print("2. Marketing Spend (beta = +0.0299): 每投入 $1.00 於行銷，預期增加約 $0.03 獲利，具正向輔助效果。")
print("3. Administration (beta = -0.0688): 行政支出每增加 $1.00，利潤微幅減少約 $0.07，呈現負邊際效應。")
print("4. State_Florida (+$938.79) & State_New York (+$6.99): 與基準加州相比，地理位置對獲利影響甚微。")

# =============================================================================
# 5. PREDICTION ON TEST SET (模型預測與測試集驗證)
# =============================================================================
print("\n[PART 5: PREDICTIONS (ACTUAL VS PREDICTED)]")
print("-" * 80)

# 對測試集進行預測
y_pred_test = regressor.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual Profit ($)": y_test.values,
    "Predicted Profit ($)": np.round(y_pred_test, 2),
    "Residual Error ($)": np.round(y_test.values - y_pred_test, 2),
    "Absolute Error (%)": np.round(np.abs((y_test.values - y_pred_test) / y_test.values) * 100, 2)
})

print("測試集 (10 筆新創公司) 實際獲利 vs 預測獲利對照表:")
print(comparison_df.to_string())

print("\n" + "=" * 80)
print("STEP 3: BUILD MODEL COMPLETED SUCCESSFULLY")
print("=" * 80)
