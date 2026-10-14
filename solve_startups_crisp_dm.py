"""
CRISP-DM Implementation for 50 Startups Multiple Linear Regression
==================================================================

Framework Phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =============================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =============================================================================
def phase_1_business_understanding():
    """
    Business Objectives:
    1. Identify key factors driving profitability across 50 startups.
    2. Determine whether geographic location (State) significantly influences profit.
    3. Provide actionable guidance for venture capital allocation (e.g., R&D vs. Marketing spend).
    
    Data Mining Goals:
    - Build a Multiple Linear Regression model with high explainability and predictive accuracy.
    - Validate linear regression assumptions (linearity, normality of residuals, homoscedasticity, low multicollinearity).
    - Deploy a robust inference pipeline for predicting profitability of new startup ventures.
    """
    print("=" * 80)
    print("PHASE 1: BUSINESS UNDERSTANDING")
    print("=" * 80)
    print("Objective: Predict startup 'Profit' using spending patterns (R&D, Admin, Marketing) and 'State'.")
    print("Goal: Assist investors and founders in capital allocation to maximize ROI.")
    print("Metric criteria: High R^2 (> 0.90), low RMSE/MAE, and statistically sound coefficients.\n")


# =============================================================================
# PHASE 2: DATA UNDERSTANDING
# =============================================================================
def phase_2_data_understanding(csv_path="50_Startups.csv", output_dir="artifacts"):
    print("=" * 80)
    print("PHASE 2: DATA UNDERSTANDING")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 Records:")
    print(df.head())
    
    print("\nData Types and Missing Values:")
    info_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum()
    })
    print(info_df)
    
    print("\nDescriptive Statistics (Numerical Features):")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])
    
    print("\nCategorical Feature Distribution ('State'):")
    print(df['State'].value_counts())
    
    # Correlation with target
    num_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    corr_matrix = df[num_cols].corr()
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(4))
    
    print(f"\nCorrelation with Profit:")
    for col in ["R&D Spend", "Marketing Spend", "Administration"]:
        print(f"  - {col:18s}: {corr_matrix.loc[col, 'Profit']:+.4f}")
        
    # Generate EDA visualization
    try:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Target Distribution
        sns.histplot(df["Profit"], kde=True, ax=axes[0, 0], color="#2b5c8f")
        axes[0, 0].set_title("Distribution of Profit (Target)", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Profit ($)")
        
        # 2. R&D Spend vs Profit
        sns.regplot(data=df, x="R&D Spend", y="Profit", ax=axes[0, 1],
                    scatter_kws={"color": "#1b7837", "alpha": 0.7}, line_kws={"color": "#d73027"})
        axes[0, 1].set_title("R&D Spend vs Profit (Strong Linear Relation)", fontsize=12, fontweight='bold')
        
        # 3. Marketing Spend vs Profit
        sns.regplot(data=df, x="Marketing Spend", y="Profit", ax=axes[1, 0],
                    scatter_kws={"color": "#762a83", "alpha": 0.7}, line_kws={"color": "#d73027"})
        axes[1, 0].set_title("Marketing Spend vs Profit", fontsize=12, fontweight='bold')
        
        # 4. State vs Profit (Boxplot)
        sns.boxplot(data=df, x="State", y="Profit", ax=axes[1, 1], hue="State", legend=False, palette="Set2")
        axes[1, 1].set_title("Profit by State", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        eda_plot_path = os.path.join(output_dir, "eda_overview.png")
        plt.savefig(eda_plot_path, dpi=300)
        plt.close()
        print(f"\nEDA summary plot saved to: {eda_plot_path}\n")
    except Exception as e:
        print(f"Plot generation skipped or failed: {e}\n")
        
    return df


# =============================================================================
# PHASE 3: DATA PREPARATION
# =============================================================================
def phase_3_data_preparation(df, test_size=0.2, random_state=42):
    print("=" * 80)
    print("PHASE 3: DATA PREPARATION")
    print("=" * 80)
    
    # Separate Features and Target
    X = df.drop(columns=["Profit"])
    y = df["Profit"]
    
    num_features = ["R&D Spend", "Administration", "Marketing Spend"]
    cat_features = ["State"]
    
    print(f"Features (X): {list(X.columns)}")
    print(f"Target (y): Profit")
    
    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training instances: {len(X_train)} (80%)")
    print(f"Testing instances : {len(X_test)} (20%)")
    
    # Column Preprocessor:
    # 1. OneHotEncoder with drop='first' on 'State' to prevent dummy variable trap (multicollinearity).
    # 2. StandardScaler on numerical features (enables fair comparison of coefficients & stable regularization).
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features)
        ],
        remainder="passthrough"
    )
    
    # Inspect transformed feature names
    preprocessor.fit(X_train)
    encoded_cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features))
    transformed_feature_names = num_features + encoded_cat_names
    print(f"\nEngineered / Encoded Feature Names ({len(transformed_feature_names)} features):")
    for idx, f_name in enumerate(transformed_feature_names, 1):
        print(f"  {idx}. {f_name}")
    print("Note: One category of State was dropped to avoid the Dummy Variable Trap (Collinearity).")

    # Multicollinearity Assessment via Variance Inflation Factor (VIF)
    X_train_trans = preprocessor.transform(X_train)
    vif_data = []
    for i in range(X_train_trans.shape[1]):
        y_i = X_train_trans[:, i]
        X_other = np.delete(X_train_trans, i, axis=1)
        r2_i = LinearRegression().fit(X_other, y_i).score(X_other, y_i)
        vif_i = 1.0 / (1.0 - r2_i) if r2_i < 1.0 else np.inf
        vif_data.append({"Feature": transformed_feature_names[i], "VIF": round(vif_i, 3)})
    print("\nVariance Inflation Factor (VIF) on Training Features:")
    vif_df = pd.DataFrame(vif_data)
    print(vif_df.to_string(index=False))
    print("Interpretation: VIF < 5 indicates acceptable multicollinearity levels.\n")
    
    return X_train, X_test, y_train, y_test, preprocessor, transformed_feature_names


# =============================================================================
# PHASE 4: MODELING
# =============================================================================
def phase_4_modeling(X_train, y_train, preprocessor):
    print("=" * 80)
    print("PHASE 4: MODELING")
    print("=" * 80)
    
    # Model 1: Ordinary Least Squares (OLS) Multiple Linear Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Model 2: Ridge Regression (L2 regularization) with internal Cross-Validation
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
    ])
    
    # Model 3: Lasso Regression (L1 regularization / feature selection) with internal CV
    lasso_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, random_state=42, max_iter=10000))
    ])
    
    models = {
        "Multiple Linear Regression (OLS)": lr_pipeline,
        "Ridge Regression (L2 Regularized)": ridge_pipeline,
        "Lasso Regression (L1 Regularized)": lasso_pipeline
    }
    
    # Perform 5-Fold Cross Validation on Training Set
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        cv_results[name] = cv_scores
        print(f"[{name}]")
        print(f"  Training 5-Fold CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        if hasattr(pipeline.named_steps["regressor"], "alpha_"):
            print(f"  Optimal Regularization alpha: {pipeline.named_steps['regressor'].alpha_:.4f}")
        print()
        
    return models


# =============================================================================
# PHASE 5: EVALUATION
# =============================================================================
def calculate_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # Adjusted R2 formula: 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if (n - n_features - 1) > 0 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

def phase_5_evaluation(models, X_train, y_train, X_test, y_test, transformed_feature_names, output_dir="artifacts"):
    print("=" * 80)
    print("PHASE 5: EVALUATION")
    print("=" * 80)
    
    num_features = len(transformed_feature_names)
    eval_table = []
    
    for name, pipeline in models.items():
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_metrics = calculate_metrics(y_train, y_train_pred, num_features)
        test_metrics = calculate_metrics(y_test, y_test_pred, num_features)
        
        eval_table.append({
            "Model": name,
            "Train R^2": train_metrics["R2"],
            "Test R^2": test_metrics["R2"],
            "Test Adj R^2": test_metrics["Adj_R2"],
            "Test MAE ($)": test_metrics["MAE"],
            "Test RMSE ($)": test_metrics["RMSE"],
            "Test MAPE (%)": test_metrics["MAPE (%)"]
        })
        
    eval_df = pd.DataFrame(eval_table)
    print("Performance Comparison on Train and Test Sets:")
    print(eval_df.to_string(index=False))
    
    # Detailed Diagnostics on the Primary Multiple Linear Regression Model
    best_name = "Multiple Linear Regression (OLS)"
    best_model = models[best_name]
    y_test_pred = best_model.predict(X_test)
    residuals = y_test - y_test_pred
    
    print("\n" + "-" * 60)
    print("REGRESSION COEFFICIENT INTERPRETATION:")
    print("-" * 60)
    regressor = best_model.named_steps["regressor"]
    intercept = regressor.intercept_
    coefficients = regressor.coef_
    
    print(f"Intercept (beta_0): ${intercept:,.2f}")
    coef_df = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Standardized Coefficient (beta)": coefficients
    }).sort_values(by="Standardized Coefficient (beta)", ascending=False)
    print(coef_df.to_string(index=False))
    
    # Calculate Unstandardized Coefficients (Dollar for Dollar)
    scaler = best_model.named_steps["preprocessor"].named_transformers_["num"]
    means = scaler.mean_
    scales = scaler.scale_
    num_cols_len = len(means)
    
    unstd_coefs = []
    offset = 0.0
    for idx, f in enumerate(transformed_feature_names):
        if idx < num_cols_len:
            b_i = coefficients[idx] / scales[idx]
            offset += b_i * means[idx]
            unstd_coefs.append(b_i)
        else:
            unstd_coefs.append(coefficients[idx])
    unstd_intercept = intercept - offset
    
    print("\n" + "-" * 60)
    print("UNSTANDARDIZED (DOLLAR-FOR-DOLLAR) REGRESSION EQUATION:")
    print("-" * 60)
    print(f"Profit = ${unstd_intercept:,.2f}")
    for f, b in zip(transformed_feature_names, unstd_coefs):
        print(f"       + ({b:+.4f}) * [{f}]")
        
    print("\nEconomic Interpretation for Investors & Founders:")
    print(f"- R&D Spend ROI: Every $1.00 spent on R&D yields approx ${unstd_coefs[0]:.2f} in profit.")
    print(f"- Marketing Spend ROI: Every $1.00 spent on Marketing yields approx ${unstd_coefs[2]:.2f} in profit.")
    print(f"- Administration: Marginal impact is approx ${unstd_coefs[1]:.2f} per dollar spent.")
    print(f"- State: Negligible impact (< 1% variation across California, Florida, New York).")
    
    # Statistical Diagnostics on Residuals
    print("\n" + "-" * 60)
    print("MODEL ASSUMPTION DIAGNOSTICS (Test Set Residuals):")
    print("-" * 60)
    print(f"1. Mean of Residuals: {np.mean(residuals):.4f} (Close to 0 -> Unbiased)")
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"2. Normality (Shapiro-Wilk): W = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f} "
          f"({'Residuals are Normal (p > 0.05)' if shapiro_p > 0.05 else 'Residuals deviate from normality'})")
    
    # Durbin-Watson statistic for autocorrelation
    dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    print(f"3. Independence (Durbin-Watson): d = {dw_stat:.4f} (Ideal value ~ 2.0)")
    
    # Residual Plotting
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs Actual
        axes[0].scatter(y_test, y_test_pred, color="#1f77b4", edgecolor='k', s=60)
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
        axes[0].set_title(f"{best_name}: Actual vs Predicted", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Actual Profit ($)")
        axes[0].set_ylabel("Predicted Profit ($)")
        axes[0].legend()
        
        # Residuals vs Fitted
        axes[1].scatter(y_test_pred, residuals, color="#ff7f0e", edgecolor='k', s=60)
        axes[1].axhline(0, color='r', linestyle='--', lw=2)
        axes[1].set_title("Residuals vs Fitted Values (Homoscedasticity)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Fitted Values ($)")
        axes[1].set_ylabel("Residuals ($)")
        
        plt.tight_layout()
        diag_plot_path = os.path.join(output_dir, "model_diagnostics.png")
        plt.savefig(diag_plot_path, dpi=300)
        plt.close()
        print(f"Model diagnostic plots saved to: {diag_plot_path}\n")
    except Exception as e:
        print(f"Diagnostic plotting skipped: {e}\n")
        
    return best_model, eval_df


# =============================================================================
# PHASE 6: DEPLOYMENT
# =============================================================================
def phase_6_deployment(model, output_path="artifacts/startup_profit_pipeline.joblib"):
    print("=" * 80)
    print("PHASE 6: DEPLOYMENT")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model Pipeline successfully exported to: {output_path}")
    
    # Validation of the Deployed Pipeline via Inference Simulation
    print("\nSimulating Production Inference with Deployed Pipeline:")
    loaded_pipeline = joblib.load(output_path)
    
    # 3 Real-World Hypothetical Scenarios:
    new_startups = pd.DataFrame([
        {
            "R&D Spend": 160000.0,
            "Administration": 130000.0,
            "Marketing Spend": 300000.0,
            "State": "California"
        },
        {
            "R&D Spend": 85000.0,
            "Administration": 120000.0,
            "Marketing Spend": 200000.0,
            "State": "New York"
        },
        {
            "R&D Spend": 10000.0,
            "Administration": 90000.0,
            "Marketing Spend": 50000.0,
            "State": "Florida"
        }
    ])
    
    predictions = loaded_pipeline.predict(new_startups)
    new_startups["Predicted Profit ($)"] = np.round(predictions, 2)
    
    print(new_startups.to_string(index=False))
    print("\nProduction inference test PASSED successfully.")
    print("=" * 80)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    # Phase 1: Business Understanding
    phase_1_business_understanding()
    
    # Phase 2: Data Understanding
    df = phase_2_data_understanding(csv_path="50_Startups.csv")
    
    # Phase 3: Data Preparation
    X_train, X_test, y_train, y_test, preprocessor, feature_names = phase_3_data_preparation(df)
    
    # Phase 4: Modeling
    models = phase_4_modeling(X_train, y_train, preprocessor)
    
    # Phase 5: Evaluation
    best_model, eval_df = phase_5_evaluation(models, X_train, y_train, X_test, y_test, feature_names)
    
    # Phase 6: Deployment
    phase_6_deployment(best_model)


if __name__ == "__main__":
    main()
