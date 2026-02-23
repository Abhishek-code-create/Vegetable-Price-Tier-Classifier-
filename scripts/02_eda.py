"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 02: Exploratory Data Analysis (EDA)
=============================================================================
Purpose : Analyse cleaned dataset, generate descriptive statistics, and
          produce EDA visualisations including class distribution charts.
Run     : python scripts/02_eda.py
Input   : data/clean/kalimati_clean.csv
Output  : charts/eda_*.png (7 figures)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

PALETTE = ["#2d6a4f", "#40916c", "#74c69d", "#f77f00",
           "#fcbf49", "#e63946", "#457b9d", "#1d3557"]
TIER_COLORS = {"Low": "#27ae60", "Medium": "#f39c12",
               "High": "#e67e22", "Very High": "#e74c3c"}

print("=" * 65)
print("  STW5000CEM — Step 2: Exploratory Data Analysis")
print("=" * 65)

df = pd.read_csv("data/clean/kalimati_clean.csv", low_memory=False)
df["Date"] = pd.to_datetime(df["Date"])
print(f"\nLoaded: {df.shape}  |  Commodities: {df['Commodity'].nunique()}")
print(f"Columns: {df.columns.tolist()}")

# ── Descriptive Stats ─────────────────────────────────────────────────────────
print("\n── Descriptive Statistics ──────────────────────────────────")
print(df[["Minimum", "Average", "Maximum"]].describe().round(2).to_string())

print("\n── Class Distribution ──────────────────────────────────────")
print(df["PriceTier"].value_counts().to_string())

# Figure 1: Price Tier Class Distribution
print("\n[Fig 1] Class distribution ...")
tier_order = ["Low", "Medium", "High", "Very High"]
counts = df["PriceTier"].value_counts().reindex(tier_order)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = [TIER_COLORS[t] for t in tier_order]
axes[0].bar(tier_order, counts.values, color=colors, edgecolor="white", alpha=0.88)
for i, (t, v) in enumerate(zip(tier_order, counts.values)):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Price Tier Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Record Count"); axes[0].set_xlabel("Price Tier")
axes[1].pie(counts.values, labels=tier_order, colors=colors, autopct="%1.1f%%",
            startangle=140, textprops={"fontsize": 11})
axes[1].set_title("Price Tier Proportions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_01_class_distribution.png")

# Figure 2: Average Price by Top 15 Vegetables
print("[Fig 2] Top 15 average prices ...")
top15 = df.groupby("Commodity")["Average"].mean().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top15.index[::-1], top15.values[::-1],
               color=PALETTE[0], edgecolor="white", alpha=0.85)
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"NPR {val:.0f}", va="center", fontsize=9)
ax.set_xlabel("Average Price (NPR/Kg)", fontsize=12)
ax.set_title("Top 15 Vegetables by Average Price", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_02_top_avg_prices.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_02_top_avg_prices.png")

# Figure 3: Price Distribution by Tier (violin plot)
print("[Fig 3] Price distribution by tier ...")
top8 = df["Commodity"].value_counts().head(8).index.tolist()
sub = df[df["Commodity"].isin(top8)].copy()
sub["PriceTier"] = pd.Categorical(sub["PriceTier"], categories=tier_order, ordered=True)
fig, ax = plt.subplots(figsize=(10, 6))
parts = ax.violinplot([sub[sub["PriceTier"] == t]["Average"].values for t in tier_order],
                      positions=[1, 2, 3, 4], showmedians=True)
for i, (pc, t) in enumerate(zip(parts["bodies"], tier_order)):
    pc.set_facecolor(colors[i]); pc.set_alpha(0.75)
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(tier_order, fontsize=11)
ax.set_ylabel("Average Price (NPR/Kg)", fontsize=12)
ax.set_title("Price Distribution by Tier — Violin Plot", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_03_price_by_tier_violin.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_03_price_by_tier_violin.png")

# Figure 4: Monthly Price Trends (Top 5)
print("[Fig 4] Monthly price trends ...")
top5 = df["Commodity"].value_counts().head(5).index.tolist()
fig, ax = plt.subplots(figsize=(13, 6))
for veg, col in zip(top5, PALETTE):
    monthly = (df[df["Commodity"] == veg]
               .set_index("Date").resample("ME")["Average"].mean())
    ax.plot(monthly.index, monthly.values, label=veg, color=col, linewidth=2)
ax.set_xlabel("Date", fontsize=12); ax.set_ylabel("Average Price (NPR/Kg)", fontsize=12)
ax.set_title("Monthly Average Price Trends — Top 5 Vegetables (2013–2023)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("charts/eda_04_price_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_04_price_trends.png")

# Figure 5: Seasonal Heatmap
print("[Fig 5] Seasonal heatmap ...")
top8h = df["Commodity"].value_counts().head(8).index.tolist()
pivot_data = []
for veg in top8h:
    s = df[df["Commodity"] == veg]
    for m in range(1, 13):
        pivot_data.append({"Vegetable": veg, "Month": m,
                           "AvgPrice": s[s["Month"] == m]["Average"].mean()})
pt = (pd.DataFrame(pivot_data)
      .pivot(index="Vegetable", columns="Month", values="AvgPrice"))
pt.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pt, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "NPR/Kg"})
ax.set_title("Seasonal Price Heatmap — Monthly Average Prices (NPR/Kg)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_05_seasonal_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_05_seasonal_heatmap.png")

# Figure 6: Correlation Heatmap of Features
print("[Fig 6] Correlation matrix ...")
num_cols = ["Average", "Year", "Month", "DayOfYear", "Quarter",
            "Lag7", "Lag30", "Roll7Mean", "PriceRange"]
avail = [c for c in num_cols if c in df.columns]
corr = df[avail].dropna().corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Pearson r"})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_06_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_06_correlation_matrix.png")

# Figure 7: Yearly Trend
print("[Fig 7] Yearly average price trend ...")
yearly = df.groupby("Year")["Average"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(yearly.index, yearly.values, color=PALETTE[1], edgecolor="white", alpha=0.85, width=0.6)
ax.plot(yearly.index, yearly.values, "o-", color=PALETTE[3], lw=2, ms=7)
for x, y in zip(yearly.index, yearly.values):
    ax.text(x, y + 1.5, f"{y:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Average Price (NPR/Kg)")
ax.set_title("Yearly Average Vegetable Price Trend at Kalimati Bazar",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/eda_07_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: eda_07_yearly_trend.png")

print("\n  EDA complete ✅  —  All charts saved to charts/")
