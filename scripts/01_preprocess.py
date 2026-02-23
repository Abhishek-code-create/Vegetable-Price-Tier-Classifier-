"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 01: Data Preprocessing Pipeline
=============================================================================
Domain    : CLASSIFICATION — Predict vegetable price tier (Low/Medium/High/Very High)
Algorithm : K-Nearest Neighbours (KNN) + Decision Tree + Naive Bayes
Dataset   : Kalimati Tarkari Dataset (280,862 records, 2013–2023)
Author    : [Your Name / Student ID]
Run       : python scripts/01_preprocess.py
Output    : data/clean/kalimati_clean.csv
            data/clean/kalimati_model.csv
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "raw",   "Kalimati_Tarkari_Dataset.csv")
CLEAN_PATH = os.path.join("data", "clean", "kalimati_clean.csv")
MODEL_PATH = os.path.join("data", "clean", "kalimati_model.csv")
os.makedirs("data/clean", exist_ok=True)

print("=" * 65)
print("  STW5000CEM — Vegetable Price Tier Classifier")
print("  Step 1: Data Preprocessing")
print("=" * 65)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\n[1/7] Loading raw dataset ...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"      Raw shape  : {df.shape}")
print(f"      Columns    : {df.columns.tolist()}")
print(f"      Sample:\n{df.head(3).to_string()}")

# ── 2. Fix Price Types ────────────────────────────────────────────────────────
print("\n[2/7] Cleaning price columns ...")
for col in ["Minimum", "Maximum", "Average"]:
    df[col] = (df[col].astype(str)
               .str.replace("Rs ", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False, errors="coerce")
before = len(df)
df.dropna(subset=["Minimum", "Maximum", "Average", "Date"], inplace=True)
print(f"      Dropped {before - len(df)} unparseable rows. Remaining: {len(df):,}")

# ── 3. Unit Standardisation ───────────────────────────────────────────────────
print("\n[3/7] Standardising units (Kg only) ...")
print(f"      Unique units: {df['Unit'].unique().tolist()}")
df = df[df["Unit"].isin(["Kg", "KG"])].copy()
df["Unit"] = "Kg"
print(f"      After unit filter: {df.shape}")

# ── 4. Outlier Removal (per-commodity IQR) ────────────────────────────────────
print("\n[4/7] IQR outlier removal (5th–95th percentile per commodity) ...")
cleaned = []
for commodity, group in df.groupby("Commodity"):
    q5  = group["Average"].quantile(0.05)
    q95 = group["Average"].quantile(0.95)
    cleaned.append(group[(group["Average"] >= q5) & (group["Average"] <= q95)])
df = pd.concat(cleaned, ignore_index=True)
# Ensure Commodity column is retained (groupby can drop it in some pandas versions)
if "Commodity" not in df.columns:
    raise ValueError("Commodity column lost after groupby — check pandas version.")
print(f"      After outlier removal: {df.shape}")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
print("\n[5/7] Engineering features ...")
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)

df["Year"]       = df["Date"].dt.year
df["Month"]      = df["Date"].dt.month
df["Day"]        = df["Date"].dt.day
df["DayOfYear"]  = df["Date"].dt.dayofyear
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Quarter"]    = df["Date"].dt.quarter
df["DayOfWeek"]  = df["Date"].dt.dayofweek

# Cyclical encodings (prevent Dec→Jan discontinuity)
df["MonthSin"]   = np.sin(2 * np.pi * df["Month"] / 12)
df["MonthCos"]   = np.cos(2 * np.pi * df["Month"] / 12)
df["DaySin"]     = np.sin(2 * np.pi * df["DayOfYear"] / 365)
df["DayCos"]     = np.cos(2 * np.pi * df["DayOfYear"] / 365)

# Lag features (auto-regressive)
df["Lag7"]       = df.groupby("Commodity")["Average"].shift(7)
df["Lag30"]      = df.groupby("Commodity")["Average"].shift(30)
df["Roll7Mean"]  = df.groupby("Commodity")["Average"].transform(
                       lambda x: x.rolling(7, min_periods=1).mean())
df["Roll30Mean"] = df.groupby("Commodity")["Average"].transform(
                       lambda x: x.rolling(30, min_periods=1).mean())
df["PriceRange"] = df["Maximum"] - df["Minimum"]   # daily spread

# ── 6. Create Classification Target Labels ────────────────────────────────────
print("\n[6/7] Creating price-tier classification labels ...")
# Per-commodity quartile-based bins → robust across different vegetables
def assign_tier(group):
    q25 = group["Average"].quantile(0.25)
    q50 = group["Average"].quantile(0.50)
    q75 = group["Average"].quantile(0.75)
    conditions = [
        group["Average"] <= q25,
        (group["Average"] > q25) & (group["Average"] <= q50),
        (group["Average"] > q50) & (group["Average"] <= q75),
        group["Average"] > q75,
    ]
    labels = ["Low", "Medium", "High", "Very High"]
    group = group.copy()
    group["PriceTier"] = np.select(conditions, labels, default="Medium")
    return group

tier_dfs = []
for commodity, group in df.groupby("Commodity"):
    tier_dfs.append(assign_tier(group))
df = pd.concat(tier_dfs, ignore_index=True)
print(f"      Price tier distribution:\n{df['PriceTier'].value_counts().to_string()}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
print("\n[7/7] Saving datasets ...")
df.to_csv(CLEAN_PATH, index=False)
print(f"      Saved full clean: {CLEAN_PATH}  ({len(df):,} rows, {df.shape[1]} cols)")

# Model dataset: top 30 commodities, remove NaN lags
TOP_30 = df["Commodity"].value_counts().head(30).index.tolist()
df_model = (df[df["Commodity"].isin(TOP_30)]
            .dropna(subset=["Lag7", "Lag30"])
            .reset_index(drop=True))
df_model.to_csv(MODEL_PATH, index=False)
print(f"      Saved model data : {MODEL_PATH}  ({len(df_model):,} rows)")

print("\n── Summary ──────────────────────────────────────────────────")
print(f"  Raw records          : 280,862")
print(f"  After cleaning       : {len(df):,}")
print(f"  Model records        : {len(df_model):,}")
print(f"  Date range           : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Unique commodities   : {df['Commodity'].nunique()}")
print(f"  Classification target: PriceTier (Low/Medium/High/Very High)")
print(f"  Feature count        : {df_model.shape[1]}")
print("  Preprocessing complete ✅")
