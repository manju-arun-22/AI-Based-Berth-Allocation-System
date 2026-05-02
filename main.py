"""
main.py — AI Berth Allocation Optimization Pipeline
MSc AI Dissertation - Manju Arun (202433718), University of Hull

Run this script to execute the full project pipeline:
  Step 1: Generate synthetic data
  Step 2: Feature engineering
  Step 3: Train ML forecasting models (XGBoost, RF, LSTM ensemble)
  Step 4: Optimize berth allocation (Genetic Algorithm)
  Step 5: Evaluate and save results
"""

import sys
import os
import logging
import warnings
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.synthetic_generator import generate_and_save_all
from src.data.feature_engineering import (
    build_feature_matrix, get_feature_columns, train_test_split_timeseries
)
from src.models.forecasting import (
    train_xgboost, train_random_forest, train_lstm,
    WeightedEnsemble, evaluate_model
)
from src.optimization.genetic_algorithm import GeneticAlgorithmBAP, Vessel, Berth


def run_pipeline():
    print("=" * 60)
    print("  AI-Driven Berth Allocation Optimization")
    print("  MSc AI Dissertation - Manju Arun (202433718)")
    print("=" * 60)

    # ── STEP 1: DATA GENERATION ──────────────────────────────────────────────
    print("\n[1/5] Generating synthetic port data...")
    vessels_raw, weather, tides, berths_df = generate_and_save_all("data/synthetic")

    # ── STEP 2: FEATURE ENGINEERING ──────────────────────────────────────────
    print("\n[2/5] Building feature matrix...")
    df = build_feature_matrix(vessels_raw, weather, tides)
    feature_cols = get_feature_columns(df)
    print(f"      Using {len(feature_cols)} features")

    train_df, val_df, test_df = train_test_split_timeseries(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
    X_val   = scaler.transform(val_df[feature_cols].fillna(0))
    X_test  = scaler.transform(test_df[feature_cols].fillna(0))

    y_train = train_df["target_delay_hours"].values
    y_val   = val_df["target_delay_hours"].values
    y_test  = test_df["target_delay_hours"].values

    # ── STEP 3: TRAIN FORECASTING MODELS ────────────────────────────────────
    print("\n[3/5] Training ML forecasting models...")

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    rf_model,  rf_metrics  = train_random_forest(X_train, y_train, X_val, y_val)
    lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_val, y_val)

    # Ensemble
    ensemble = WeightedEnsemble()
    ensemble.fit(
        {"xgboost": xgb_model, "random_forest": rf_model, "lstm": lstm_model},
        X_val, y_val
    )

    print("\n-- Final Test Set Evaluation --")
    test_preds = ensemble.predict(X_test)
    final_metrics = evaluate_model(y_test, test_preds, "Ensemble (Test Set)")

    # Save predictions
    os.makedirs("results/reports", exist_ok=True)
    test_df = test_df.copy()
    test_df["predicted_delay"] = test_preds
    test_df[["vessel_id", "eta", "ata", "delay_hours", "predicted_delay"]].to_csv(
        "results/reports/predictions.csv", index=False
    )

    # ── STEP 4: BERTH ALLOCATION OPTIMIZATION ───────────────────────────────
    print("\n[4/5] Running Genetic Algorithm for berth allocation...")

    # Use a sample of vessels for optimization demo (24h window)
    sample = test_df.head(30).copy()
    sample["pred_ata_h"] = (
        pd.to_datetime(sample["ata"]).apply(lambda x: x.timestamp() / 3600)
    )

    vessels_for_ga = [
        Vessel(
            id=row["vessel_id"],
            length=row["length_m"],
            beam=row["beam_m"],
            draft=row["draft_m"],
            predicted_ata=float(row["pred_ata_h"]),
            service_hours=float(row["service_hours"]),
            priority=row["priority"],
            vessel_type=row["vessel_type"],
        )
        for _, row in sample.iterrows()
    ]

    berths_for_ga = [
        Berth(
            id=row["id"],
            length=row["length"],
            depth=row["depth"],
            beam=row["beam"],
            allowed_types=str(row["allowed_types"]).split(","),
        )
        for _, row in berths_df.iterrows()
    ]

    ga = GeneticAlgorithmBAP(
        vessels=vessels_for_ga,
        berths=berths_for_ga,
        pop_size=100,
        generations=200,
    )
    best_schedule, best_fitness = ga.optimize(verbose=True)
    schedule_df = pd.DataFrame(ga.decode_schedule(best_schedule))
    schedule_df.to_csv("results/reports/ga_schedule.csv", index=False)

    # ── STEP 5: SAVE MODELS ──────────────────────────────────────────────────
    print("\n[5/5] Saving models and results...")
    ensemble.save("results/models")

    # Save lightweight API artifacts — avoids DataFrame pickle issues across pandas versions
    _all = pd.concat([train_df, val_df, test_df])
    _hist = _all.groupby("vessel_type")["delay_hours"].agg(["median", "std"]).to_dict("index")
    joblib.dump(
        {"feature_cols": feature_cols, "scaler": scaler, "hist_stats": _hist},
        "data/processed/api_artifacts.pkl"
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Forecast MAE : {final_metrics['mae']:.3f} hours")
    print(f"  Forecast R²  : {final_metrics['r2']:.4f}")
    print(f"  GA Best Cost : {-best_fitness:,.1f}")
    print("  Outputs saved to results/")
    print("=" * 60)

    return final_metrics, best_fitness


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()
