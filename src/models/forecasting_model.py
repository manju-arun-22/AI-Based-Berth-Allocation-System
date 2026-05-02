"""
ML Ensemble Model for Vessel Arrival Time Prediction
MSc AI Dissertation - Manju Arun (202433718)

Models: XGBoost + Random Forest + LSTM → Weighted Ensemble
Target: predict delay_hours (ATA - ETA)
"""

import numpy as np
import pandas as pd
import pickle, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name="Model"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    print(f"  [{model_name}] MAE={mae:.3f}h  RMSE={rmse:.3f}h  R²={r2:.3f}  MAPE={mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = model.predict(X_val)
    metrics = evaluate(y_val, preds, "XGBoost")
    return model, metrics


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_val, y_val):
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
    preds = model.predict(X_val)
    metrics = evaluate(y_val, preds, "Random Forest")
    return model, metrics


# ── LSTM ─────────────────────────────────────────────────────────────────────

def train_lstm(X_train, y_train, X_val, y_val, seq_len=24, epochs=50):
    """
    Build and train a 2-layer LSTM.
    Reshapes tabular data into sequences of length seq_len.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        tf.random.set_seed(42)
    except ImportError:
        print("  [LSTM] TensorFlow not available — skipping LSTM.")
        return None, {"mae": 99, "rmse": 99, "r2": -1, "mape": 99}

    def make_sequences(X, y, seq=seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq):
            Xs.append(X[i:i+seq])
            ys.append(y[i+seq])
        return np.array(Xs), np.array(ys)

    X_tr_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_tr_arr = y_train.values if hasattr(y_train, "values") else y_train
    X_vl_arr = X_val.values   if hasattr(X_val, "values")   else X_val
    y_vl_arr = y_val.values   if hasattr(y_val, "values")   else y_val

    Xtr_seq, ytr_seq = make_sequences(X_tr_arr, y_tr_arr)
    Xvl_seq, yvl_seq = make_sequences(X_vl_arr, y_vl_arr)

    n_features = Xtr_seq.shape[2]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
    ]
    model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xvl_seq, yvl_seq),
        epochs=epochs, batch_size=32,
        callbacks=callbacks, verbose=0,
    )
    preds = model.predict(Xvl_seq, verbose=0).flatten()
    metrics = evaluate(yvl_seq, preds, "LSTM")
    return model, metrics


# ── Weighted Ensemble ─────────────────────────────────────────────────────────

class EnsembleForecaster:
    """
    Combines XGBoost + Random Forest + LSTM predictions
    using inverse-MAE weighting.
    """

    def __init__(self):
        self.xgb_model = None
        self.rf_model  = None
        self.lstm_model = None
        self.weights   = None
        self.metrics   = {}

    def fit(self, X_train, y_train, X_val, y_val):
        print("\nTraining XGBoost...")
        self.xgb_model,  m_xgb  = train_xgboost(X_train, y_train, X_val, y_val)
        print("Training Random Forest...")
        self.rf_model,   m_rf   = train_random_forest(X_train, y_train, X_val, y_val)
        print("Training LSTM...")
        self.lstm_model, m_lstm = train_lstm(X_train, y_train, X_val, y_val)

        # Inverse-MAE weights
        maes = np.array([m_xgb["mae"], m_rf["mae"], m_lstm["mae"]])
        inv  = 1.0 / (maes + 1e-8)
        self.weights = inv / inv.sum()
        print(f"\nEnsemble weights → XGBoost: {self.weights[0]:.3f} | RF: {self.weights[1]:.3f} | LSTM: {self.weights[2]:.3f}")

        self.metrics = {"xgb": m_xgb, "rf": m_rf, "lstm": m_lstm}
        return self

    def predict(self, X):
        p_xgb  = self.xgb_model.predict(X)
        p_rf   = self.rf_model.predict(X)
        p_lstm = np.zeros(len(X))  # fallback if LSTM unavailable
        if self.lstm_model is not None:
            seq_len = 24
            X_arr = X.values if hasattr(X, "values") else X
            if len(X_arr) > seq_len:
                seqs = np.array([X_arr[i:i+seq_len] for i in range(len(X_arr)-seq_len)])
                p_lstm_raw = self.lstm_model.predict(seqs, verbose=0).flatten()
                p_lstm[seq_len:] = p_lstm_raw
                p_lstm[:seq_len] = p_lstm_raw[0]

        ensemble = (
            self.weights[0] * p_xgb +
            self.weights[1] * p_rf  +
            self.weights[2] * p_lstm
        )
        return ensemble

    def save(self, path="results/models/ensemble_forecaster.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path="results/models/ensemble_forecaster.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Time-Series Cross Validation ──────────────────────────────────────────────

def time_series_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    print(f"\nRunning {n_splits}-fold walk-forward validation...")

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]

        model = EnsembleForecaster()
        model.fit(X_tr, y_tr, X_vl, y_vl)
        preds = model.predict(X_vl)
        m = evaluate(y_vl.values, preds[:len(y_vl)], f"Ensemble Fold {fold}")
        fold_metrics.append(m)

    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    print(f"\nCV Average → MAE={avg['mae']:.3f}h | RMSE={avg['rmse']:.3f}h | R²={avg['r2']:.3f}")
    return fold_metrics


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.synthetic_generator import generate_vessel_calls
    from data.feature_engineering  import build_feature_matrix

    df  = generate_vessel_calls("2022-01-01", "2024-12-31")
    X, y = build_feature_matrix(df)

    split = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    ensemble = EnsembleForecaster()
    ensemble.fit(X_train, y_train, X_val, y_val)
    ensemble.save()

    preds = ensemble.predict(X_val)
    print("\nFinal Ensemble Performance on Test Set:")
    evaluate(y_val.values, preds[:len(y_val)], "Ensemble")
