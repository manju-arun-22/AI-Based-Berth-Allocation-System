"""
Feature Engineering for Vessel Arrival Prediction
Generates the ~48 features described in the project proposal.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def build_feature_matrix(vessels: pd.DataFrame, weather: pd.DataFrame, tides: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full 48-feature matrix for each vessel call by merging
    vessel, weather, and tidal data aligned on ETA timestamp.
    """

    df = vessels.copy()
    df["eta"] = pd.to_datetime(df["eta"])
    df["ata"] = pd.to_datetime(df["ata"])

    # ── TEMPORAL FEATURES (18) ──────────────────────────────────────────────
    df["hour"]         = df["eta"].dt.hour
    df["day_of_week"]  = df["eta"].dt.dayofweek
    df["week_of_year"] = df["eta"].dt.isocalendar().week.astype(int)
    df["month"]        = df["eta"].dt.month
    df["quarter"]      = df["eta"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding — prevents discontinuity at period boundaries
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)

    # Lag / rolling features — sorted chronologically
    df = df.sort_values("eta").reset_index(drop=True)

    # Vectorised O(n log n) rolling arrival count using searchsorted
    eta_vals = df["eta"].values
    window_24 = np.timedelta64(24, "h")
    window_48 = np.timedelta64(48, "h")
    lower_24  = np.searchsorted(eta_vals, eta_vals - window_24, side="left")
    lower_48  = np.searchsorted(eta_vals, eta_vals - window_48, side="left")
    idx       = np.arange(len(df))
    df["arrivals_last_24h"] = idx - lower_24
    df["arrivals_last_48h"] = idx - lower_48

    df["avg_service_last_7d"] = df["service_hours"].rolling(window=50, min_periods=1).mean()
    df["rolling_mean_7d"]     = df["arrivals_last_24h"].rolling(7,  min_periods=1).mean()
    df["rolling_std_7d"]      = df["arrivals_last_24h"].rolling(7,  min_periods=1).std().fillna(0)
    df["rolling_mean_30d"]    = df["arrivals_last_24h"].rolling(30, min_periods=1).mean()

    # ── VESSEL FEATURES (12) ─────────────────────────────────────────────────
    # dtype=int avoids bool columns that pandas 3.x returns by default
    type_dummies = pd.get_dummies(df["vessel_type"], prefix="vtype", dtype=int)
    df = pd.concat([df, type_dummies], axis=1)

    # Historical delay stats per vessel type (computed on full dataset)
    type_stats = df.groupby("vessel_type")["delay_hours"].agg(["median", "std"]).reset_index()
    type_stats.columns = ["vessel_type", "hist_median_delay", "hist_std_delay"]
    df = df.merge(type_stats, on="vessel_type", how="left")

    df["priority_encoded"] = df["priority"].map({"high": 2, "normal": 1, "low": 0})
    df["cargo_complexity"]  = df["service_hours"] / df["length_m"]

    # ── ENVIRONMENTAL FEATURES ───────────────────────────────────────────────
    # Round ETA to nearest hour for merge key
    df["eta_hour"] = df["eta"].dt.floor("h")

    # The vessel CSV already contains per-vessel weather/tide values generated
    # at the same time as the delay — these ARE the signal the model needs.
    # We keep them and merge the hourly timeseries as PORT-WIDE context
    # (prefix "port_") to avoid column name conflicts.

    weather = weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather_hourly = (
        weather.set_index("timestamp")
               .resample("h").mean(numeric_only=True)
               .reset_index()
               .rename(columns={
                   "timestamp":        "eta_hour",
                   "wind_speed_ms":    "port_wind_speed_ms",
                   "visibility_km":    "port_visibility_km",
                   "wave_height_m":    "port_wave_height_m",
                   "precipitation_mm": "port_precipitation_mm",
                   "weather_severity": "port_weather_severity",
               })
    )
    df = df.merge(weather_hourly, on="eta_hour", how="left")

    tides = tides.copy()
    tides["timestamp"] = pd.to_datetime(tides["timestamp"])
    tides_hourly = (
        tides.set_index("timestamp")
             .resample("h").mean(numeric_only=True)
             .reset_index()
             .rename(columns={
                 "timestamp":           "eta_hour",
                 "tide_height_m":       "port_tide_height_m",
                 "time_to_high_tide_h": "port_time_to_high_tide_h",
                 "tide_favorable":      "port_tide_favorable",
             })
    )
    df = df.merge(tides_hourly, on="eta_hour", how="left")

    # Composite safety flag — uses vessel's own weather at ETA
    df["safe_to_berth"] = (
        (df["weather_severity"] < 2) & (df["wind_speed_ms"] < 20)
    ).astype(int)

    # ── TARGET VARIABLE ──────────────────────────────────────────────────────
    df["target_delay_hours"] = df["delay_hours"]

    # ── FILL MISSING ─────────────────────────────────────────────────────────
    df = df.fillna(df.median(numeric_only=True))

    print(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ML-usable feature columns (excludes IDs, timestamps, target)."""
    exclude = {"vessel_id", "eta", "ata", "atd", "eta_hour",
               "target_delay_hours", "delay_hours", "vessel_type",
               "priority", "cargo_type"}
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def train_test_split_timeseries(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """Time-aware split: train / val / test in strict chronological order."""
    df = df.sort_values("eta").reset_index(drop=True)
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Split: Train={len(train):,}  Val={len(val):,}  Test={len(test):,}")
    return train, val, test
