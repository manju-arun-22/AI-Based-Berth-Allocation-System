"""
STEP 5 — View All Results
==========================
What this does:
  - Reads all output CSV files produced by Steps 1-4
  - Prints a full readable summary of every output
  - Shows predictions, schedule, and model performance

Run AFTER step4_optimize_berths.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

print("=" * 60)
print("  STEP 5: View All Results")
print("=" * 60)

# ── 1. Vessel call data ───────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 1 — data/synthetic/vessel_calls.csv")
print("  (The synthetic port data generated in Step 1)")
print("=" * 60)
vessels = pd.read_csv("data/synthetic/vessel_calls.csv")
print(f"\n  Shape: {vessels.shape[0]:,} rows × {vessels.shape[1]} columns")
print(f"\n  Columns: {list(vessels.columns)}")
print(f"\n  Sample (5 rows):")
print(vessels[["vessel_id","vessel_type","length_m","draft_m",
               "delay_hours","service_hours","priority"]].head().to_string())
print(f"\n  Delay statistics:")
print(f"    Min   : {vessels['delay_hours'].min():.1f} h")
print(f"    Mean  : {vessels['delay_hours'].mean():.2f} h")
print(f"    Max   : {vessels['delay_hours'].max():.1f} h")
print(f"    % On time (delay < 0.5h): "
      f"{(vessels['delay_hours'].abs() < 0.5).mean()*100:.1f}%")

# ── 2. Predictions ────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 2 — results/reports/predictions.csv")
print("  (Ensemble model predictions vs actual delays)")
print("=" * 60)
preds = pd.read_csv("results/reports/predictions.csv")
print(f"\n  Shape: {preds.shape[0]:,} rows × {preds.shape[1]} columns")
print(f"\n  Sample (10 rows):")
print(preds.head(10).to_string())

errors = preds["predicted_delay"] - preds["delay_hours"]
mae    = errors.abs().mean()
rmse   = np.sqrt((errors**2).mean())
r2     = 1 - (errors**2).sum() / ((preds["delay_hours"] - preds["delay_hours"].mean())**2).sum()
print(f"\n  Model performance on these {len(preds):,} test vessels:")
print(f"    MAE  (avg error)         : {mae:.3f} hours  (target < 2h)")
print(f"    RMSE (penalises big errors): {rmse:.3f} hours  (target < 3h)")
print(f"    R²   (variance explained): {r2:.4f}        (target > 0.80)")

# ── 3. GA Schedule ────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 3 — results/reports/ga_schedule.csv")
print("  (Optimised berth assignment for 30 vessels)")
print("=" * 60)
schedule = pd.read_csv("results/reports/ga_schedule.csv")
print(f"\n  Shape: {schedule.shape[0]} rows × {schedule.shape[1]} columns")
print(f"\n  Full schedule:")
print(schedule.to_string())

print(f"\n  Berth usage breakdown:")
for berth, group in schedule.groupby("berth_id"):
    print(f"    {berth}: {len(group)} vessels assigned  "
          f"(avg service: {group['service_hours'].mean():.1f}h)")

priority_counts = schedule["priority"].value_counts()
print(f"\n  Priority breakdown:")
for p, n in priority_counts.items():
    print(f"    {p}: {n} vessels")

# ── 4. Saved model files ──────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 4 — results/models/")
print("  (Trained models saved as files)")
print("=" * 60)
model_dir = "results/models"
if os.path.exists(model_dir):
    for f in os.listdir(model_dir):
        fpath = os.path.join(model_dir, f)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {f:<40} {size:>8.1f} KB")
else:
    print("  (No model files found — run step3 first)")

# ── 5. Raw data files ─────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 5 — data/synthetic/")
print("  (All raw generated data files)")
print("=" * 60)
for fname in ["vessel_calls.csv", "weather.csv", "tides.csv", "berths.csv"]:
    fpath = f"data/synthetic/{fname}"
    if os.path.exists(fpath):
        df   = pd.read_csv(fpath)
        size = os.path.getsize(fpath) / 1024
        print(f"  {fname:<25} {len(df):>8,} rows  {size:>8.1f} KB")

# ── 6. Final summary ─────────────────────────────────────
print("\n" + "=" * 60)
print("  FULL PIPELINE SUMMARY")
print("=" * 60)
print(f"""
  Step 1 — Data Generation
    Vessels generated     : {len(vessels):,}
    Date range            : 2022-01-01 to 2024-12-31
    Vessel types          : container, bulk, tanker, general, roro

  Step 2 — Feature Engineering
    Features built        : 42
    Train / Val / Test    : 3,500 / 750 / 750 vessels

  Step 3 — ML Models
    XGBoost, Random Forest, LSTM trained
    Combined into Weighted Ensemble
    Test MAE              : {mae:.3f} hours (target: < 2h)
    Test RMSE             : {rmse:.3f} hours (target: < 3h)

  Step 4 — Genetic Algorithm
    Vessels scheduled     : 30
    Berths available      : 5
    GA generations        : 200
    Schedule saved        : results/reports/ga_schedule.csv
""")

print("=" * 60)
print("  All done! The pipeline is complete.")
print("=" * 60)
