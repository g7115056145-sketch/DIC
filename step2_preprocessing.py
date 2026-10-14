"""
Step 2: Preprocessing for 50 Startups Multiple Linear Regression
================================================================
Topics covered:
1. One-Hot Encoding for Nominal Features (and Dummy Variable Trap)
2. Train-Test Split Necessity (泛化能力、防止過擬合)
3. Data Type Conversion and Reshaping (X 2D vs y 1D/2D, reshape(-1, 1))
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

# =============================================================================
# 0. LOAD DATA
# =============================================================================
csv_path = "50_Startups.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} not found.")

df = pd.read_csv(csv_path)

print("=" * 80)
print("STEP 2: PREPROCESSING (數據預處理)")
print("=" * 80)

# =============================================================================
# 1. CHANGE TYPE & RESHAPE (數據型態轉換與形狀重構)
# =============================================================================
print("\n[PART 1: DATA TYPE INSPECTION & CONVERSION / RESHAPE]")
print("-" * 80)

# 1.1 檢查原始數據類型
print("1.1 原始數據類型 (df.dtypes):")
for col, dtype in df.dtypes.items():
    print(f"    - {col:18s}: {dtype}")

# 1.2 強制確保數值特徵為 float64，並分離特徵 X 與目標 y
numerical_cols = ["R&D Spend", "Administration", "Marketing Spend"]
for col in numerical_cols:
    df[col] = df[col].astype(np.float64)
df["Profit"] = df["Profit"].astype(np.float64)

X = df.drop(columns=["Profit"])
y = df["Profit"]

print("\n1.2 特徵矩陣 X 與目標變量 y 的形狀 (Shape) 與維度 (Dimension):")
print(f"    - X Shape: {X.shape} (2D 表格: {X.shape[0]} 個樣本, {X.shape[1]} 個特徵)")
print(f"    - y Shape: {y.shape} (1D 序列/向量: {y.shape[0]} 個目標數值)")

# 1.3 Reshape 解釋：為什麼 Scikit-Learn 需要 2D 矩陣？
print("\n1.3 為什麼需要 Reshape？")
print("    - Scikit-Learn 規定特徵矩陣 X 必須是 2D array: (n_samples, n_features)。")
print("    - 如果只選單一特徵 (如單變量回歸 df['R&D Spend'])，Pandas 會產出 1D Series (50,)。")
print("    - 此時必須使用 .values.reshape(-1, 1) 將 1D 轉為 2D 矩陣 (50, 1)，其中 -1 代表自動推斷樣本數。")

# 示範單變量特徵 reshape
single_feature_1d = df["R&D Spend"].values
single_feature_2d = single_feature_1d.reshape(-1, 1)
print(f"    - 單特徵原始 1D 形狀: {single_feature_1d.shape}")
print(f"    - 經過 reshape(-1, 1) 後的 2D 形狀: {single_feature_2d.shape}")

# =============================================================================
# 2. ONE-HOT ENCODING FOR NOMINAL FEATURES (名義特徵獨熱編碼)
# =============================================================================
print("\n[PART 2: ONE-HOT ENCODING FOR NOMINAL FEATURES & DUMMY VARIABLE TRAP]")
print("-" * 80)

print("2.1 為什麼類別變量 'State' 需要 One-Hot Encoding？")
print("    - 'State' 是名義變量 (Nominal Feature): California, Florida, New York。")
print("    - 名義變量之間沒有大小、等級或順序關係 (California != 1, Florida != 2, New York != 3)。")
print("    - 若使用標籤編碼 (Label Encoding: 0, 1, 2)，線性回歸會錯誤假設 New York (2) 是 California (0) 的兩倍權重。")
print("    - 因此必須使用獨熱編碼 (One-Hot Encoding)，將各州轉換為獨立的二元旗標 (0 或 1)。")

print("\n2.2 虛擬變量陷阱 (Dummy Variable Trap / 完美共線性):")
print("    - 若一特徵有 k 個類別，轉換為 k 個虛擬變量後，變量總和必為 1 (常數項截距)。")
print("    - 即: State_California + State_Florida + State_New York = 1")
print("    - 這會導致完美多重共線性 (Multicollinearity)，使迴歸矩陣 (X^T X) 不可逆，係數估計失真。")
print("    - 解決辦法：設定 drop='first'，只保留 k - 1 個虛擬變量 (以第一個類別為基準)。")

# 示範 OneHotEncoder 與 drop='first' 的差別
ohe_all = OneHotEncoder(sparse_output=False)
state_encoded_all = ohe_all.fit_transform(df[["State"]])
print(f"\n2.3 未刪除基準欄位 (k=3): 欄位數 = {state_encoded_all.shape[1]}")
print("    編碼範例 (前 3 筆):")
print(pd.DataFrame(state_encoded_all[:3], columns=ohe_all.get_feature_names_out(["State"])))

# 使用 ColumnTransformer 進行正式預處理 (drop='first')
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
    ],
    remainder="passthrough"  # 保留未處理的數值特徵
)

X_encoded = ct.fit_transform(X)
cat_feature_names = list(ct.named_transformers_["cat"].get_feature_names_out(["State"]))
all_feature_names = cat_feature_names + numerical_cols

print(f"\n2.4 正式預處理 (drop='first', k-1=2 個虛擬變量):")
print(f"    - 生成的特徵欄位: {all_feature_names}")
X_encoded_df = pd.DataFrame(X_encoded, columns=all_feature_names)
print("    編碼後前 5 筆:")
print(X_encoded_df.head())

# =============================================================================
# 3. TRAIN-TEST SPLIT 的必要性 (訓練集與測試集切分)
# =============================================================================
print("\n[PART 3: TRAIN TEST SPLIT 的必要性]")
print("-" * 80)

print("""
3.1 為什麼必須進行 Train-Test Split？
    1. 評估模型泛化能力 (Generalization Ability):
       - 模型的真正價值在於預測「未見過的新數據 (Unseen Data)」，而非單純記憶過去數據。
    
    2. 防止過度擬合 (Prevent Overfitting):
       - 若把所有數據拿來訓練與評估，訓練誤差 (Training Error) 可能極低，
         但在實際部署或面對新客戶時預測效果極差。
    
    3. 避免資訊洩漏 (Data Leakage):
       - 特徵工程 (如標準化 Scaler 的均值與標準差) 只能從訓練集計算，
         絕不能提前使用測試集的信息，切分確保了嚴格的評估隔離。
    
    4. 小數據集 (N=50) 的切分策略:
       - 採用 80% 訓練集 (40 筆) 與 20% 測試集 (10 筆)。
       - 設置 random_state (如 42) 保證實驗結果可重現 (Reproducibility)。
""")

# 執行切分
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded_df, y, test_size=0.2, random_state=42
)

print("3.2 切分後的數據集維度:")
print(f"    - X_train Shape : {X_train.shape} (40 筆樣本, 5 個特徵)")
print(f"    - y_train Shape : {y_train.shape} (40 筆目標)")
print(f"    - X_test  Shape : {X_test.shape} (10 筆樣本, 5 個特徵)")
print(f"    - y_test  Shape : {y_test.shape} (10 筆目標)")

print("\n3.3 測試集樣本預覽 (X_test & y_test):")
test_preview = X_test.copy()
test_preview["Actual Profit"] = y_test
print(test_preview.head())

print("\n" + "=" * 80)
print("STEP 2 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 80)
