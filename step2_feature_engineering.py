"""
STEP 2 — Feature Engineering
==============================
What this does:
  - Reads the vessel/weather/tides data from Step 1
  - Builds ~48 features per vessel that the ML models will learn from
  - Groups features into: Temporal, Vessel, Environmental
  - Splits data chronologically: 70% train / 15% val / 15% test
  - Applies StandardScaler (normalises all features to same scale)

Run AFTER step1_generate_data.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data.feature_engineering import (
    build_feature_matrix, get_feature_columns, train_test_split_timeseries
)

print("=" * 55)
print("  STEP 2: Feature Engineering")
print("=" * 55)

# ── Load data saved in Step 1 ────────────────────────────
print("\nLoading data from data/synthetic/ ...")
vessels = pd.read_csv("data/synthetic/vessel_calls.csv")
weather = pd.read_csv("data/synthetic/weather.csv")
tides   = pd.read_csv("data/synthetic/tides.csv")
print(f"  Vessels loaded  : {len(vessels):,} rows")
print(f"  Weather loaded  : {len(weather):,} rows")
print(f"  Tides loaded    : {len(tides):,} rows")

# ── Build feature matrix ─────────────────────────────────
print("\nBuilding feature matrix...")
df = build_feature_matrix(vessels, weather, tides)
feature_cols = get_feature_columns(df)

print("\n" + "-" * 55)
print(f"FEATURES CREATED ({len(feature_cols)} total):")
print("-" * 55)

temporal = [c for c in feature_cols if any(k in c for k in
            ["hour", "day", "week", "month", "quarter", "weekend",
             "sin", "cos", "arrivals", "rolling", "avg_service"])]
vessel   = [c for c in feature_cols if any(k in c for k in
            ["vtype", "length", "draft", "tonnage", "cargo",
             "service", "priority", "reliability", "hist"])]
enviro   = [c for c in feature_cols if any(k in c for k in
            ["wind", "vis", "wave", "precip", "weather",
             "tide", "safe"])]
other    = [c for c in feature_cols if c not in temporal + vessel + enviro]

print(f"\n  Temporal features  ({len(temporal)}): {temporal}")
print(f"\n  Vessel features    ({len(vessel)}):   {vessel}")
print(f"\n  Environmental      ({len(enviro)}):  {enviro}")
if other:
    print(f"\n  Other              ({len(other)}):  {other}")

# ── Train / Validation / Test split ─────────────────────
print("\n" + "-" * 55)
print("CHRONOLOGICAL SPLIT (no data leakage):")
print("-" * 55)
train_df, val_df, test_df = train_test_split_timeseries(df)

fmt = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d")
print(f"\n  Training set  : {len(train_df):,} vessels  "
      f"({fmt(train_df['eta'].min())} to {fmt(train_df['eta'].max())})")
print(f"  Validation set: {len(val_df):,} vessels  "
      f"({fmt(val_df['eta'].min())} to {fmt(val_df['eta'].max())})")
print(f"  Test set      : {len(test_df):,} vessels  "
      f"({fmt(test_df['eta'].min())} to {fmt(test_df['eta'].max())})")

# ── Scale features ───────────────────────────────────────
print("\n" + "-" * 55)
print("SCALING (StandardScaler — zero mean, unit variance):")
print("-" * 55)
scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
X_val   = scaler.transform(val_df[feature_cols].fillna(0))
X_test  = scaler.transform(test_df[feature_cols].fillna(0))

y_train = train_df["target_delay_hours"].values
y_val   = val_df["target_delay_hours"].values
y_test  = test_df["target_delay_hours"].values

print(f"\n  X_train shape : {X_train.shape}   (rows x features)")
print(f"  X_val shape   : {X_val.shape}")
print(f"  X_test shape  : {X_test.shape}")
print(f"\n  y_train delay : min={y_train.min():.1f}h  max={y_train.max():.1f}h  mean={y_train.mean():.2f}h")
print(f"  y_val delay   : min={y_val.min():.1f}h  max={y_val.max():.1f}h  mean={y_val.mean():.2f}h")
print(f"  y_test delay  : min={y_test.min():.1f}h  max={y_test.max():.1f}h  mean={y_test.mean():.2f}h")

# ── Save processed data for next steps ───────────────────
import joblib, os
os.makedirs("data/processed", exist_ok=True)
joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test,
             feature_cols, scaler, train_df, val_df, test_df),
            "data/processed/features.pkl")
print("\n  Processed data saved to data/processed/features.pkl")

print("\n" + "=" * 55)
print("  Step 2 complete.")
print("  Run step3_train_models.py next.")
print("=" * 55)
