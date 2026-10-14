"""
Step 5: Deployment using Pickle and Gzip
=======================================
Topics covered:
1. Complete Pipeline Encapsulation (ColumnTransformer + Regressor)
2. Model Serialization with `pickle`
3. Compression with `gzip` (.pkl vs .pkl.gz storage comparison)
4. Model Deserialization & Integrity Verification
5. Real-Time Production Inference Simulation (API-ready payload)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

# =============================================================================
# 0. LOAD DATA & TRAIN PRODUCTION PIPELINE
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

# 生產環境全流程 Pipeline (結合前處理與 Lasso 最優回歸器)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State"])
    ]
)

# 使用 Step 4 Optuna 驗證出的最佳模型 (Lasso, alpha=833.75)
production_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Lasso(alpha=833.75, random_state=42, max_iter=10000))
])

# 以全部歷史數據進行最終部署擬合 (Production Retraining)
production_pipeline.fit(X, y)

print("=" * 80)
print("STEP 5: DEPLOYMENT (模型序列化與 GZIP 壓縮部署)")
print("=" * 80)

# =============================================================================
# 1. SERIALIZATION: PICKLE VS. PICKLE + GZIP (序列化與壓縮)
# =============================================================================
print("\n[PART 1: MODEL SERIALIZATION & GZIP COMPRESSION]")
print("-" * 80)

os.makedirs("artifacts", exist_ok=True)
raw_pickle_path = os.path.join("artifacts", "startup_model_raw.pkl")
gzip_pickle_path = os.path.join("artifacts", "startup_model_compressed.pkl.gz")

# 1.1 單純使用 Pickle 序列化
with open(raw_pickle_path, "wb") as f_out:
    pickle.dump(production_pipeline, f_out, protocol=pickle.HIGHEST_PROTOCOL)

# 1.2 結合 Gzip 進行二進制壓縮序列化
with gzip.open(gzip_pickle_path, "wb", compresslevel=9) as f_out_gz:
    pickle.dump(production_pipeline, f_out_gz, protocol=pickle.HIGHEST_PROTOCOL)

# 1.3 檔案大小比較
raw_size = os.path.getsize(raw_pickle_path)
gzip_size = os.path.getsize(gzip_pickle_path)
compression_ratio = (1 - (gzip_size / raw_size)) * 100

print(f"1. 未壓縮 Pickle 檔案: {raw_pickle_path}")
print(f"   - 檔案大小: {raw_size:,} bytes")
print(f"2. Gzip 壓縮 Pickle 檔案: {gzip_pickle_path}")
print(f"   - 檔案大小: {gzip_size:,} bytes")
print(f"3. 儲存空間節省率 (Compression Ratio): {compression_ratio:.2f}% 體積縮減\n")

print("""
為什麼部署推薦結合 Pickle + Gzip？
  1. 體積縮減：大幅降低檔案大小，加速網路傳輸與雲端儲存 (如 AWS S3, GCP Cloud Storage)。
  2. 冷啟動優化：在 Serverless 架構 (如 AWS Lambda) 或容器化微服務 (Docker) 中，
     較小的模型體積能縮短容器映像檔拉取 (image pull) 與載入記憶體的時間。
  3. 標準相容：Python 內建 gzip 與 pickle 模組，不需額外安裝第三方依賴庫。
""")

# =============================================================================
# 2. DESERIALIZATION: LOAD GZIP COMPRESSED MODEL (反序列化載入)
# =============================================================================
print("\n[PART 2: DESERIALIZATION & INTEGRITY CHECK]")
print("-" * 80)

print(f"從磁碟還原解壓模型: {gzip_pickle_path} ...")
with gzip.open(gzip_pickle_path, "rb") as f_in_gz:
    loaded_pipeline = pickle.load(f_in_gz)

print(">> 模型解壓反序列化成功！")
print(f"載入之物件類型: {type(loaded_pipeline)}")
print(f"Pipeline 包含步驟: {list(loaded_pipeline.named_steps.keys())}")
print(f"迴歸算法: {loaded_pipeline.named_steps['regressor']}")

# =============================================================================
# 3. PRODUCTION INFERENCE ENGINE (生產環境即時預測引擎)
# =============================================================================
print("\n[PART 3: REAL-TIME PRODUCTION INFERENCE SIMULATION]")
print("-" * 80)

def predict_startup_profit(input_data):
    """
    生產環境預測函數：
    接收未經處理的原始新創申請數據 (Dict 或 DataFrame)，
    自動調用 Pipeline 內的 OneHotEncoder 與 StandardScaler 進行轉換並輸出獲利預測。
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("輸入格式必須為 Dict, List[Dict] 或 DataFrame")
        
    predictions = loaded_pipeline.predict(df_input)
    df_result = df_input.copy()
    df_result["Predicted_Profit ($)"] = np.round(predictions, 2)
    return df_result

# 模擬生產環境來自前端 API 或 Webhook 的 3 筆投資提案 (Raw JSON / Payload)
new_proposals = [
    {
        "Proposal": "A: 高研發導向新創 (Silicon Valley AI)",
        "R&D Spend": 165000.0,
        "Administration": 135000.0,
        "Marketing Spend": 450000.0,
        "State": "California"
    },
    {
        "Proposal": "B: 穩健型平衡新創 (New York FinTech)",
        "R&D Spend": 90000.0,
        "Administration": 120000.0,
        "Marketing Spend": 250000.0,
        "State": "New York"
    },
    {
        "Proposal": "C: 早期行銷導向新創 (Florida E-Commerce)",
        "R&D Spend": 15000.0,
        "Administration": 80000.0,
        "Marketing Spend": 100000.0,
        "State": "Florida"
    }
]

print("收到 3 筆待審核之新創公司融資企劃 (Raw Input):")
inference_results = predict_startup_profit(new_proposals)
print(inference_results[["Proposal", "R&D Spend", "Marketing Spend", "State", "Predicted_Profit ($)"]].to_string(index=False))

print("\n" + "=" * 80)
print("STEP 5 DEPLOYMENT (PICKLE + GZIP) COMPLETED SUCCESSFULLY")
print("=" * 80)
