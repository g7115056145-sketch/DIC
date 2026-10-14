"""
Feature Selection Evaluation Plot
=================================
Reproduces:
1. RMSE by Number of Features
2. R-squared by Number of Features
3. Clean summary table of selected features and performance metrics
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------
# 1. LOAD DATA & DUMMY ENCODING
# -------------------------------------------------------------
csv_path = "50_Startups.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} not found.")

df = pd.read_csv(csv_path)
df_encoded = pd.get_dummies(df, columns=["State"], dtype=float)

# -------------------------------------------------------------
# 2. FEATURE PROGRESSION (FORWARD SELECTION STEPS)
# -------------------------------------------------------------
# Evaluated feature sets with random_state=0 and test_size=0.2
feature_steps = [
    ["R&D Spend"],
    ["R&D Spend", "Marketing Spend"],
    ["R&D Spend", "Marketing Spend", "State_New York"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"]
]

display_features = [
    "[R&D Spend]",
    "[R&D Spend, Marketing Spend]",
    "[R&D Spend, Marketing Spend, State_New York]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida]",
    "[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]"
]

num_features = [1, 2, 3, 4, 5]
rmses = []
r2s = []

for f in feature_steps:
    X_tr, X_te, y_tr, y_te = train_test_split(df_encoded[f], df_encoded["Profit"], test_size=0.2, random_state=0)
    lr = LinearRegression().fit(X_tr, y_tr)
    pred = lr.predict(X_te)
    rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
    r2s.append(r2_score(y_te, pred))

# -------------------------------------------------------------
# 3. PLOT CHARTS AND SUMMARY TABLE
# -------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.5), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.9], hspace=0.35, wspace=0.22)

# Subplot 1: RMSE by Number of Features
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(num_features, rmses, marker='o', color='#1f77b4', linewidth=1.5, markersize=5)
ax1.set_title('RMSE by Number of Features', fontsize=12, pad=10)
ax1.set_xlabel('Number of Features', fontsize=10)
ax1.set_ylabel('RMSE', fontsize=10)
ax1.set_xticks(np.arange(1.0, 5.5, 0.5))
ax1.set_yticks(np.arange(8200, 9200, 200))
ax1.set_xlim(0.8, 5.2)
ax1.set_ylim(8140, 9180)

# Subplot 2: R-squared by Number of Features
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(num_features, r2s, marker='o', color='#1f77b4', linewidth=1.5, markersize=5)
ax2.set_title('R-squared by Number of Features', fontsize=12, pad=10)
ax2.set_xlabel('Number of Features', fontsize=10)
ax2.set_ylabel('R-squared', fontsize=10)
ax2.set_xticks(np.arange(1.0, 5.5, 0.5))
ax2.set_yticks(np.arange(0.936, 0.950, 0.002))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax2.set_xlim(0.8, 5.2)
ax2.set_ylim(0.9345, 0.9485)

# Subplot 3: Table
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

table_data = []
for i in range(len(num_features)):
    table_data.append([
        str(num_features[i]),
        display_features[i],
        f"{rmses[i]:.6f}",
        f"{r2s[i]:.6f}"
    ])

col_labels = ['Number of Features', 'Selected Features', 'RMSE', 'R-squared']
table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left'
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 1.8)

# Format table borders (booktabs style)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='#111111')
        cell.visible_edges = 'TB'
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.5)
        if col in [2, 3]:
            cell.set_text_props(ha='right', weight='bold')
        else:
            cell.set_text_props(ha='left', weight='bold')
    else:
        cell.visible_edges = 'B'
        cell.set_edgecolor('#ebebeb')
        cell.set_linewidth(0.8)
        if row == len(num_features):
            cell.set_edgecolor('#333333')
            cell.set_linewidth(1.2)
        if col in [2, 3]:
            cell.set_text_props(ha='right')
        else:
            cell.set_text_props(ha='left')

table.auto_set_column_width(col=list(range(len(col_labels))))

os.makedirs("artifacts", exist_ok=True)
output_path = os.path.join("artifacts", "feature_selection_chart.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Plot successfully generated and saved to: {output_path}")
