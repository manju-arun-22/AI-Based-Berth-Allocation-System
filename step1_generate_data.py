"""
STEP 1 — Generate Synthetic Port Data
======================================
What this does:
  - Creates 5,000 fake vessel arrivals (2022-2024)
  - Creates hourly weather records
  - Creates hourly tidal records
  - Creates 5 berth definitions
  - Saves all of these as CSV files in data/synthetic/

Run this first before any other step.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.data.synthetic_generator import generate_and_save_all

print("=" * 55)
print("  STEP 1: Generate Synthetic Port Data")
print("=" * 55)

# ── Run the generator ────────────────────────────────────
vessels, weather, tides, berths = generate_and_save_all("data/synthetic")

# ── Show what was created ────────────────────────────────
print("\n" + "-" * 55)
print("VESSELS — first 5 rows:")
print("-" * 55)
print(vessels[["vessel_id", "vessel_type", "length_m", "draft_m",
               "eta", "delay_hours", "service_hours", "priority"]].head().to_string())

print("\n" + "-" * 55)
print("VESSEL STATISTICS:")
print("-" * 55)
print(f"  Total vessel calls  : {len(vessels):,}")
print(f"  Vessel types        : {vessels['vessel_type'].value_counts().to_dict()}")
print(f"  Avg arrival delay   : {vessels['delay_hours'].mean():.2f} hours")
print(f"  Avg service time    : {vessels['service_hours'].mean():.2f} hours")
print(f"  High-priority ships : {(vessels['priority']=='high').sum()}")
print(f"  Date range          : {vessels['eta'].min()} to {vessels['eta'].max()}")

print("\n" + "-" * 55)
print("WEATHER — first 3 rows:")
print("-" * 55)
print(weather.head(3).to_string())

print("\n" + "-" * 55)
print("TIDES — first 3 rows:")
print("-" * 55)
print(tides.head(3).to_string())

print("\n" + "-" * 55)
print("BERTHS (port configuration):")
print("-" * 55)
print(berths.to_string())

print("\n" + "=" * 55)
print("  Step 1 complete.")
print("  Files saved in:  data/synthetic/")
print("  vessel_calls.csv, weather.csv, tides.csv, berths.csv")
print("  Run step2_feature_engineering.py next.")
print("=" * 55)
