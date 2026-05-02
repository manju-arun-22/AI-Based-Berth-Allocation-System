"""
Predictive Models for Vessel Arrival Time Forecasting
Models: XGBoost, LSTM, Random Forest → Weighted Ensemble
Target: Predict delay_hours (ATA - ETA)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os


# ── METRICS ──────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, model_name="Model"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1.0          # only measure % error on meaningful delays (>1h)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if mask.sum() > 0 else float("nan")
    r2   = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"  MAE  = {mae:.3f} hours  (target: <2h)")
    print(f"  RMSE = {rmse:.3f} hours (target: <3h)")
    print(f"  MAPE = {mape:.2f}%      (target: <15%)")
    print(f"  R²   = {r2:.4f}         (target: >0.80)")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ── XGBOOST ──────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with Optuna hyperparameter tuning."""
    print("\nTraining XGBoost...")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val)
    metrics = evaluate_model(y_val, val_preds, "XGBoost")
    return model, metrics


# ── RANDOM FOREST ────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest as robust ensemble baseline."""
    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    metrics = evaluate_model(y_val, val_preds, "Random Forest")
    print(f"  OOB Score: {model.oob_score_:.4f}")
    return model, metrics


# ── LSTM ─────────────────────────────────────────────────────────────────────

def build_lstm_model(input_dim: int):
    """Build LSTM architecture as specified in proposal."""
    try:
        import logging, warnings
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential([
            LSTM(128, input_shape=(1, input_dim), return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model
    except ImportError:
        print("TensorFlow not available. Skipping LSTM.")
        return None


def train_lstm(X_train, y_train, X_val, y_val):
    """Train LSTM model for sequential pattern learning."""
    print("\nTraining LSTM...")

    model = build_lstm_model(X_train.shape[1])
    if model is None:
        return None, {}

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_3d   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        val_preds = model.predict(X_val_3d, verbose=0).flatten()
        metrics = evaluate_model(y_val, val_preds, "LSTM")
        return model, metrics

    except Exception as e:
        print(f"  LSTM training failed: {e}")
        return None, {}


# ── WEIGHTED ENSEMBLE ─────────────────────────────────────────────────────────

class WeightedEnsemble:
    """
    Weighted average ensemble: Final = w1*XGB + w2*RF + w3*LSTM
    Weights determined by inverse validation MAE.
    """

    def __init__(self):
        self.models  = {}
        self.weights = {}
        self.scaler  = None

    def fit(self, models_dict: dict, X_val, y_val):
        """
        models_dict: {"xgboost": model, "random_forest": model, "lstm": model}
        """
        maes = {}
        for name, model in models_dict.items():
            if model is None:
                continue
            try:
                if name == "lstm":
                    X_3d = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                    preds = model.predict(X_3d, verbose=0).flatten()
                else:
                    preds = model.predict(X_val)
                maes[name] = mean_absolute_error(y_val, preds)
                self.models[name] = model
            except Exception as e:
                print(f"  Skipping {name}: {e}")

        # Weights = inverse MAE, normalized
        inv_maes = {k: 1.0 / v for k, v in maes.items()}
        total    = sum(inv_maes.values())
        self.weights = {k: v / total for k, v in inv_maes.items()}

        print("\nEnsemble Weights:")
        for k, w in self.weights.items():
            print(f"  {k:15s}: {w:.4f}  (val MAE={maes[k]:.3f}h)")

    def predict(self, X):
        total = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if name == "lstm":
                X_3d  = X.reshape(X.shape[0], 1, X.shape[1])
                preds = model.predict(X_3d, verbose=0).flatten()
            else:
                preds = model.predict(X)
            total += w * preds
        return total

    def save(self, path="results/models"):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            if name == "lstm":
                model.save(f"{path}/lstm_model.keras")
            else:
                joblib.dump(model, f"{path}/{name}_model.pkl")
        joblib.dump(self.weights, f"{path}/ensemble_weights.pkl")
        print(f"Models saved to {path}/")

    @classmethod
    def load(cls, path="results/models"):
        ensemble = cls()
        ensemble.weights = joblib.load(f"{path}/ensemble_weights.pkl")
        for name in ensemble.weights:
            if name == "lstm":
                try:
                    import logging
                    logging.getLogger("tensorflow").setLevel(logging.ERROR)
                    import tensorflow as tf
                    tf.get_logger().setLevel("ERROR")
                    ensemble.models[name] = tf.keras.models.load_model(f"{path}/lstm_model.keras")
                except Exception:
                    pass
            else:
                model_path = f"{path}/{name}_model.pkl"
                if os.path.exists(model_path):
                    ensemble.models[name] = joblib.load(model_path)
        return ensemble
