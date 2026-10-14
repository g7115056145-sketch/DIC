"""
Step 1: Load Data + Generate Scatter Plot
=========================================
1. Load the 50_Startups dataset using Pandas.
2. Generate scatter plots:
   - Part A: Actual 50 Startups data (R&D Spend vs Profit colored by State)
   - Part B: Synthetic / Random data scatter plot for baseline comparison
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
csv_path = "50_Startups.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} not found.")

# Read dataset into DataFrame
df = pd.read_csv(csv_path)

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)
print(f"Dataset successfully loaded from '{csv_path}'.")
print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("First 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe().T[["mean", "std", "min", "50%", "max"]])

# -------------------------------------------------------------
# 2. GENERATE RANDOM DATA (FOR COMPARISON / INTUITION)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 50

# Random uniform feature X and random noisy target Y
random_x = np.random.uniform(10000, 180000, size=n_samples)
# Pure uncorrelated random noise
random_y_uncorrelated = np.random.uniform(40000, 200000, size=n_samples)
# Random data with linear correlation + Gaussian noise (y = 0.85*x + 40000 + noise)
noise = np.random.normal(0, 12000, size=n_samples)
random_y_correlated = 0.85 * random_x + 40000 + noise

# -------------------------------------------------------------
# 3. GENERATE SCATTER PLOTS
# -------------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual 50 Startups Data (R&D Spend vs Profit)
sns.scatterplot(
    data=df, 
    x="R&D Spend", 
    y="Profit", 
    hue="State", 
    style="State",
    s=90, 
    palette="tab10", 
    ax=axes[0]
)
axes[0].set_title("1. Actual 50 Startups: R&D Spend vs Profit", fontsize=12, fontweight="bold")
axes[0].set_xlabel("R&D Spend ($)")
axes[0].set_ylabel("Profit ($)")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Plot 2: Synthetic Random Data (Uncorrelated Noise)
axes[1].scatter(random_x, random_y_uncorrelated, color="#e41a1c", alpha=0.7, edgecolors="k", s=70)
axes[1].set_title("2. Random Data (Uncorrelated Baseline)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Random Variable X ($)")
axes[1].set_ylabel("Random Variable Y ($)")
axes[1].grid(True, linestyle="--", alpha=0.6)

# Plot 3: Synthetic Random Data with Linear Trend + Noise
axes[2].scatter(random_x, random_y_correlated, color="#377eb8", alpha=0.7, edgecolors="k", s=70)
# Add regression trend line
m, b = np.polyfit(random_x, random_y_correlated, 1)
axes[2].plot(random_x, m * random_x + b, color="#e41a1c", linestyle="--", lw=2, label=f"Trend line (slope={m:.2f})")
axes[2].set_title("3. Synthetic Random Data with Linear Trend", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Synthetic Feature X ($)")
axes[2].set_ylabel("Synthetic Target Y ($)")
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
output_img = os.path.join("artifacts", "step1_scatter_plots.png")
plt.savefig(output_img, dpi=300)
plt.close()

print("\n" + "=" * 60)
print("SCATTER PLOT GENERATED")
print("=" * 60)
print(f"Scatter plot visualization successfully saved to: {output_img}")
