"""
STEP 3 — Train ML Forecasting Models
======================================
What this does:
  - Loads the feature matrix from Step 2
  - Trains 3 individual models: XGBoost, Random Forest, LSTM
  - Combines them into a Weighted Ensemble
  - Evaluates the ensemble on the held-out test set
  - Saves all trained models to results/models/

Run AFTER step2_feature_engineering.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from src.models.forecasting import (
    train_xgboost, train_random_forest, train_lstm,
    WeightedEnsemble, evaluate_model
)

print("=" * 55)
print("  STEP 3: Train ML Forecasting Models")
print("=" * 55)

# ── Load features from Step 2 ────────────────────────────
print("\nLoading processed features from Step 2...")
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 feature_cols, scaler,
 train_df, val_df, test_df) = joblib.load("data/processed/features.pkl")

print(f"  Training samples  : {X_train.shape[0]:,}")
print(f"  Validation samples: {X_val.shape[0]:,}")
print(f"  Test samples      : {X_test.shape[0]:,}")
print(f"  Features per ship : {X_train.shape[1]}")

# ── MODEL 1: XGBoost ─────────────────────────────────────
print("\n" + "-" * 55)
print("MODEL 1 — XGBoost (Gradient Boosted Trees)")
print("-" * 55)
print("  How it works: Builds 300 decision trees one after")
print("  another, each tree correcting the previous one's errors.")
print("  Includes early stopping — stops if no improvement.")
xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)

# ── MODEL 2: Random Forest ───────────────────────────────
print("\n" + "-" * 55)
print("MODEL 2 — Random Forest (Bagged Decision Trees)")
print("-" * 55)
print("  How it works: Builds 150 independent trees on random")
print("  subsets of data. Averages their predictions.")
print("  OOB Score = validation accuracy without a val set.")
rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)

# ── MODEL 3: LSTM ────────────────────────────────────────
print("\n" + "-" * 55)
print("MODEL 3 — LSTM (Long Short-Term Memory Neural Network)")
print("-" * 55)
print("  How it works: A type of deep learning that can learn")
print("  patterns over sequences. Architecture:")
print("  Input -> LSTM(128) -> Dropout -> LSTM(64) -> Dense(1)")
lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_val, y_val)

# ── ENSEMBLE ─────────────────────────────────────────────
print("\n" + "-" * 55)
print("COMBINING INTO WEIGHTED ENSEMBLE")
print("-" * 55)
print("  Weight formula: w = (1/MAE) / sum(1/MAE)")
print("  Better model = lower MAE = higher weight")
ensemble = WeightedEnsemble()
ensemble.fit(
    {"xgboost": xgb_model, "random_forest": rf_model, "lstm": lstm_model},
    X_val, y_val
)

# ── FINAL EVALUATION ON TEST SET ─────────────────────────
print("\n" + "-" * 55)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("-" * 55)
print("  (These vessels were never seen during training)")
test_preds   = ensemble.predict(X_test)
final_metrics = evaluate_model(y_test, test_preds, "Ensemble (Test Set)")

# ── COMPARE ALL MODELS ───────────────────────────────────
print("\n" + "-" * 55)
print("COMPARISON TABLE:")
print("-" * 55)
print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'XGBoost':<20} {xgb_metrics['mae']:>8.3f} {xgb_metrics['rmse']:>8.3f} {xgb_metrics['r2']:>8.4f}")
print(f"  {'Random Forest':<20} {rf_metrics['mae']:>8.3f} {rf_metrics['rmse']:>8.3f} {rf_metrics['r2']:>8.4f}")
if lstm_metrics:
    print(f"  {'LSTM':<20} {lstm_metrics['mae']:>8.3f} {lstm_metrics['rmse']:>8.3f} {lstm_metrics['r2']:>8.4f}")
print(f"  {'Ensemble (test)':<20} {final_metrics['mae']:>8.3f} {final_metrics['rmse']:>8.3f} {final_metrics['r2']:>8.4f}")

# ── SHOW SAMPLE PREDICTIONS ──────────────────────────────
print("\n" + "-" * 55)
print("SAMPLE PREDICTIONS (first 10 test vessels):")
print("-" * 55)
print(f"  {'Actual delay':>14}  {'Predicted':>10}  {'Error':>8}")
for actual, pred in zip(y_test[:10], test_preds[:10]):
    err = pred - actual
    print(f"  {actual:>14.2f}h  {pred:>10.2f}h  {err:>+8.2f}h")

# ── SAVE MODELS AND PREDICTIONS ──────────────────────────
os.makedirs("results/reports", exist_ok=True)
test_df = test_df.copy()
test_df["predicted_delay"] = test_preds
test_df[["vessel_id", "eta", "ata", "delay_hours", "predicted_delay"]].to_csv(
    "results/reports/predictions.csv", index=False
)
ensemble.save("results/models")
joblib.dump(test_df, "data/processed/test_df_with_predictions.pkl")

print("\n  Predictions saved to results/reports/predictions.csv")
print("  Models saved to results/models/")

print("\n" + "=" * 55)
print("  Step 3 complete.")
print("  Run step4_optimize_berths.py next.")
print("=" * 55)
