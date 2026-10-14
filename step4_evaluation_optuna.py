"""
Step 4: Evaluation, Overfit/Underfit Diagnostics & Model Optimization via Optuna
==============================================================================
Topics covered:
1. Meaning and Mathematical Formulas of MSE, RMSE, MAE, R^2, and Adjusted R^2
2. Overfitting vs. Underfitting Identification
3. Learning Curve & Model Complexity Curves (Training vs. Validation Curves)
4. Hyperparameter Optimization using Optuna (Bayesian Optimization / TPE)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# 0. PREPARE DATA & CLEAN PIPELINE SPLIT
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

# Train-Test Split (80% Train, 20% Test) on raw features to strictly prevent Data Leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline ColumnTransformer (OneHotEncoder drop='first' on 'State')
baseline_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
    ]
)

baseline_pipeline = Pipeline([
    ("preprocessor", baseline_preprocessor),
    ("regressor", LinearRegression())
])

print("=" * 80)
print("STEP 4: EVALUATION & HYPERPARAMETER OPTIMIZATION (OPTUNA)")
print("=" * 80)

# =============================================================================
# 1. EVALUATION METRICS: MSE, RMSE, MAE, R^2 的意義
# =============================================================================
print("\n[PART 1: EVALUATION METRICS 的意義與計算]")
print("-" * 80)

baseline_pipeline.fit(X_train, y_train)

y_train_pred = baseline_pipeline.predict(X_train)
y_test_pred = baseline_pipeline.predict(X_test)

def calc_all_metrics(y_true, y_pred, p):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2, "Adj R^2": adj_r2, "MAPE(%)": mape}

p = 5 # 3 numerical + 2 one-hot state features
metrics_train = calc_all_metrics(y_train, y_train_pred, p)
metrics_test = calc_all_metrics(y_test, y_test_pred, p)

metrics_summary = pd.DataFrame([metrics_train, metrics_test], index=["Training Set (40)", "Testing Set (10)"])
print("基準模型 (Multiple Linear Regression) 評估指標對照表:")
print(metrics_summary.to_string())

print("""
各指標的核心數學意義與商業價值：
1. MAE (Mean Absolute Error, 平均絕對誤差):
   - 公式: (1/n) * sum(|y_i - y_hat_i|)
   - 本例測試集 MAE = $6,961.48
   - 意義: 對離群值不敏感，具有最直觀的商業解釋：「預測獲利平均偏離實際值約 6,961 美元」。

2. MSE (Mean Squared Error) & RMSE (Root Mean Squared Error):
   - 公式: MSE = (1/n) * sum((y_i - y_hat_i)^2), RMSE = sqrt(MSE)
   - 本例測試集 RMSE = $9,055.96
   - 意義: 透過平方給予「較大預測誤差」更嚴厲的懲罰。
     若 RMSE 顯著大於 MAE，代表存在個別誤差特大的預測樣本。

3. R^2 (決定係數, Coefficient of Determination):
   - 公式: 1 - (SS_res / SS_tot) = 1 - [sum((y_i - y_hat_i)^2) / sum((y_i - y_mean)^2)]
   - 本例測試集 R^2 = 0.8987 (約 89.9%)
   - 意義: 相較於單純猜測平均數，模型能解釋目標變量 (Profit) 約 89.9% 的變異量。

4. Adjusted R^2 (調整後決定係數):
   - 公式: 1 - [(1 - R^2)*(n - 1) / (n - p - 1)]
   - 本例測試集 Adj R^2 = 0.7721
   - 意義: 懲罰多餘特徵。若加入無關特徵，R^2 即使微升，Adj R^2 也會下降，防止模型盲目堆疊特徵。
""")

# =============================================================================
# 2. OVERFIT AND UNDERFIT 的判斷與學習曲線 (LEARNING CURVE)
# =============================================================================
print("\n[PART 2: OVERFIT VS UNDERFIT 的判斷]")
print("-" * 80)
print(f"訓練集 R^2: {metrics_train['R^2']:.4f}  |  測試集 R^2: {metrics_test['R^2']:.4f}")
print(f"訓練集 RMSE: ${metrics_train['RMSE']:,.2f}  |  測試集 RMSE: ${metrics_test['RMSE']:,.2f}")

diff_r2 = metrics_train['R^2'] - metrics_test['R^2']
print(f"\n泛化差距 (Generalization Gap = Train R^2 - Test R^2): {diff_r2:.4f}")

print("""
擬合狀態診斷標準：
  - 欠擬合 (Underfitting / High Bias):
      Train Error 高且 Test Error 亦高 (Train R^2 和 Test R^2 均偏低)。
      表示模型過於簡單，未能捕捉特徵與目標間的規律。
  - 過度擬合 (Overfitting / High Variance):
      Train Error 極低，但 Test Error 明顯很高 (兩者差距巨幅拉大)。
      表示模型死記訓練集雜訊，缺乏泛化至新資料的能力。
  - 良好擬合 (Good Fit):
      Train Error 低，Test Error 亦低，且泛化差距在合理範圍內。
  -> 診斷結果: 本模型 Train R^2=0.9537, Test R^2=0.8987，泛化差距約 5.5%，
     在 N=50 的小數據集上表現相當穩定，無明顯過擬合或欠擬合。
""")

# 計算學習曲線 (Learning Curve)
train_sizes, train_scores, val_scores = learning_curve(
    baseline_pipeline, X_train, y_train,
    train_sizes=np.linspace(0.3, 1.0, 6),
    cv=5, scoring="r2", shuffle=True, random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# 示範典型過擬合：多項式回歸 (Degree 1 vs Degree 2 vs Degree 3)
degrees = [1, 2, 3]
deg_train_rmse = []
deg_test_rmse = []

for d in degrees:
    pipe = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
        ])),
        ("reg", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    deg_train_rmse.append(np.sqrt(mean_squared_error(y_train, pipe.predict(X_train))))
    deg_test_rmse.append(np.sqrt(mean_squared_error(y_test, pipe.predict(X_test))))

# 繪製曲線圖
os.makedirs("artifacts", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 曲線 1: 學習曲線 (Learning Curve)
axes[0].plot(train_sizes, train_mean, 'o-', color="#1f77b4", label="Training Score (R^2)")
axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
axes[0].plot(train_sizes, val_mean, 's--', color="#ff7f0e", label="Cross-Validation Score (R^2)")
axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#ff7f0e")
axes[0].set_title("1. Learning Curve (Training vs. Cross-Validation)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Training Set Size (Samples)")
axes[0].set_ylabel("R^2 Score")
axes[0].set_ylim(0.7, 1.0)
axes[0].legend(loc="lower right")
axes[0].grid(True, linestyle="--", alpha=0.6)

# 曲線 2: 複雜度曲線 (Model Complexity Curve - 觀察過擬合)
axes[1].plot(degrees, deg_train_rmse, 'o-', color="#2ca02c", label="Train RMSE")
axes[1].plot(degrees, deg_test_rmse, 's--', color="#d62728", label="Test RMSE")
axes[1].set_title("2. Model Complexity: Underfitting vs. Overfitting", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Polynomial Degree (Model Complexity)")
axes[1].set_ylabel("RMSE ($)")
axes[1].set_xticks(degrees)
axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
curve_path = os.path.join("artifacts", "step4_curves.png")
plt.savefig(curve_path, dpi=300)
plt.close()
print(f"訓練與測試曲線已儲存至: {curve_path}\n")

# =============================================================================
# 3. 優化模型: OPTUNA (BAYESIAN HYPERPARAMETER OPTIMIZATION)
# =============================================================================
print("\n[PART 3: 優化模型 (HYPERPARAMETER OPTIMIZATION WITH OPTUNA)]")
print("-" * 80)
print("使用 Optuna (Tree-structured Parzen Estimator, TPE 貝氏優化) 尋找最佳超參數組合：")
print("  - 探索空間：正則化算法 (Ridge vs. Lasso vs. ElasticNet)")
print("  - 探索參數：正則化強度 alpha (1e-4 ~ 1e3)、ElasticNet l1_ratio (0.0 ~ 1.0)")
print("  - 評估指標：訓練集 5-Fold 交叉驗證平均 R^2 (防止測試集數據洩漏)\n")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["Ridge", "Lasso", "ElasticNet"])
    alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
        ]
    )
    
    if model_type == "Ridge":
        reg = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        reg = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    else:
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
        
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg)
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    return scores.mean()

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)

print(f">> Optuna 優化完成 (共執行 50 次試驗)")
print(f"最佳試驗編號 (Trial #{study.best_trial.number}):")
print(f"  - 最佳 5-Fold CV R^2: {study.best_value:.4f}")
print("  - 最佳超參數組合 (Best Params):")
for param_name, param_val in study.best_params.items():
    if isinstance(param_val, float):
        print(f"      * {param_name:12s}: {param_val:.5f}")
    else:
        print(f"      * {param_name:12s}: {param_val}")

# =============================================================================
# 4. OPTUNA 最佳模型與基準模型 (BASELINE) 比較
# =============================================================================
print("\n[PART 4: OPTUNA 最佳模型與基準模型 (BASELINE) 比較]")
print("-" * 80)

best_p = study.best_params
best_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
    ]
)

if best_p["model_type"] == "Ridge":
    best_reg = Ridge(alpha=best_p["alpha"])
elif best_p["model_type"] == "Lasso":
    best_reg = Lasso(alpha=best_p["alpha"], random_state=42, max_iter=10000)
else:
    best_reg = ElasticNet(alpha=best_p["alpha"], l1_ratio=best_p["l1_ratio"], random_state=42, max_iter=10000)

best_pipeline = Pipeline([
    ("preprocessor", best_preprocessor),
    ("regressor", best_reg)
])

# 使用訓練集擬合最佳 Pipeline，並在獨立測試集上驗證
best_pipeline.fit(X_train, y_train)
y_opt_pred = best_pipeline.predict(X_test)
metrics_opt = calc_all_metrics(y_test, y_opt_pred, p)

comparison_summary = pd.DataFrame([
    {
        "Model": "Baseline LinearRegression",
        "Test R^2": metrics_test["R^2"],
        "Test Adj R^2": metrics_test["Adj R^2"],
        "Test MAE ($)": metrics_test["MAE"],
        "Test RMSE ($)": metrics_test["RMSE"],
        "Test MAPE (%)": metrics_test["MAPE(%)"]
    },
    {
        "Model": f"Optuna Optimized ({best_p['model_type']})",
        "Test R^2": metrics_opt["R^2"],
        "Test Adj R^2": metrics_opt["Adj R^2"],
        "Test MAE ($)": metrics_opt["MAE"],
        "Test RMSE ($)": metrics_opt["RMSE"],
        "Test MAPE (%)": metrics_opt["MAPE(%)"]
    }
])

print(comparison_summary.to_string(index=False))

print("\n" + "=" * 80)
print("STEP 4 EVALUATION & OPTUNA OPTIMIZATION COMPLETED SUCCESSFULLY")
print("=" * 80)
