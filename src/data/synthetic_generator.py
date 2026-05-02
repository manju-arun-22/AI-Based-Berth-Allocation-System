"""
Synthetic Port Data Generator
MSc AI Dissertation - Manju Arun (202433718)
Generates realistic vessel arrival and berth operation data
calibrated against industry statistics (Bierwirth & Meisel 2015, UNCTAD)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

PORT_CONFIG = {
    "berths": [
        {"id": "B1", "length": 400, "depth": 16, "beam": 60, "allowed_types": "container,roro"},
        {"id": "B2", "length": 350, "depth": 14, "beam": 55, "allowed_types": "container,general,roro"},
        {"id": "B3", "length": 300, "depth": 15, "beam": 50, "allowed_types": "tanker,bulk"},
        {"id": "B4", "length": 250, "depth": 12, "beam": 45, "allowed_types": "tanker,bulk,general"},
        {"id": "B5", "length": 200, "depth": 10, "beam": 35, "allowed_types": "general,bulk,roro"},
    ],
    "tidal_cycle_hours": 12.4,
}

VESSEL_TYPES = {
    "container": {"prob": 0.30, "length_range": (150, 380), "draft_range": (8, 15),  "service_mean": 18, "service_std": 6},
    "bulk":      {"prob": 0.25, "length_range": (100, 300), "draft_range": (7, 13),  "service_mean": 24, "service_std": 8},
    "tanker":    {"prob": 0.20, "length_range": (120, 330), "draft_range": (9, 14),  "service_mean": 20, "service_std": 7},
    "general":   {"prob": 0.15, "length_range": (80,  220), "draft_range": (5, 10),  "service_mean": 14, "service_std": 5},
    "roro":      {"prob": 0.10, "length_range": (100, 250), "draft_range": (6, 11),  "service_mean": 10, "service_std": 4},
}


def generate_tidal_height(timestamp, base=5.0, amp=3.0):
    t = pd.Timestamp(timestamp).timestamp()
    cycle = PORT_CONFIG["tidal_cycle_hours"] * 3600
    return base + amp * np.sin(2 * np.pi * t / cycle)


def generate_weather(timestamp):
    month = pd.Timestamp(timestamp).month
    base_sev = 0.3 if month in [12, 1, 2] else 0.1
    wind  = max(0, np.random.normal(15 + base_sev * 20, 8))
    vis   = max(0.5, np.random.normal(8 - base_sev * 4, 2))
    prec  = max(0, np.random.exponential(base_sev * 5 + 0.1))
    sev   = min(2, int(wind / 20) + int(vis < 3))
    return round(wind, 1), round(vis, 1), round(prec, 1), sev


def generate_delay(weather_sev, vtype, season=1.0):
    factors = {"container": 1.0, "bulk": 1.2, "tanker": 1.1, "general": 0.9, "roro": 0.8}
    mean = 1.5 * season * (1.0 + weather_sev * 0.8) * factors.get(vtype, 1.0)
    return round(np.clip(np.random.normal(mean, mean * 0.15), -6, 48), 2)


def _priority_label():
    r = np.random.random()
    if r < 0.10:
        return "high"
    elif r < 0.20:
        return "low"
    return "normal"


def generate_vessel_calls(n=5000, start="2022-01-01", end="2024-12-31"):
    print(f"Generating {n} vessel calls ({start} to {end})...")
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    total_h = (e - s).total_seconds() / 3600
    types = list(VESSEL_TYPES.keys())
    probs = [VESSEL_TYPES[t]["prob"] for t in types]
    records = []
    for i in range(n):
        eta   = s + timedelta(hours=np.random.uniform(0, total_h))
        vtype = np.random.choice(types, p=probs)
        vc    = VESSEL_TYPES[vtype]
        length   = round(np.random.uniform(*vc["length_range"]), 1)
        draft    = round(np.random.uniform(*vc["draft_range"]), 1)
        beam     = round(length * np.random.uniform(0.13, 0.17), 1)
        gtonnage = round(length * draft * np.random.uniform(0.6, 0.8) * 100)
        cargo    = round(np.random.uniform(500, 5000))
        svc_h    = round(max(2, np.random.lognormal(np.log(vc["service_mean"]), 0.4)), 1)
        month    = eta.month
        season   = 1.3 if month in [7, 8, 9] else (0.8 if month in [1, 2] else 1.0)
        wind, vis, prec, wsev = generate_weather(eta)
        delay    = generate_delay(wsev, vtype, season)
        ata      = eta + timedelta(hours=delay)
        atd      = ata + timedelta(hours=svc_h)
        tide     = generate_tidal_height(eta)
        wave_h   = round(max(0, wind / 10 + np.random.normal(0, 0.3)), 1)
        records.append({
            "vessel_id":              f"V{1000+i:05d}",
            "vessel_type":            vtype,
            "length_m":               length,
            "beam_m":                 beam,
            "draft_m":                draft,
            "gross_tonnage":          gtonnage,
            "cargo_volume":           cargo,
            "eta":                    eta,
            "ata":                    ata,
            "atd":                    atd,
            "service_hours":          svc_h,        # renamed from service_time_hours
            "delay_hours":            delay,        # renamed from arrival_delay_hours
            "priority":               _priority_label(),
            "historical_reliability": round(np.clip(np.random.normal(0.75, 0.15), 0.3, 1.0), 2),
            "wind_speed_ms":          wind,
            "visibility_km":          vis,
            "wave_height_m":          wave_h,
            "precipitation_mm":       prec,
            "weather_severity":       wsev,
            "tide_height_m":          round(tide, 2),
            "time_to_high_tide_h":    round(abs(np.random.normal(3, 2)), 2),
            "tide_favorable":         int(tide > 5.5),
        })
    df = pd.DataFrame(records).sort_values("eta").reset_index(drop=True)
    print(f"  Done. Avg delay={df['delay_hours'].mean():.2f}h | Avg service={df['service_hours'].mean():.2f}h")
    return df


def generate_weather_series(start="2022-01-01", end="2024-12-31"):
    """Hourly weather timeseries for the full date range."""
    print("Generating hourly weather timeseries...")
    timestamps = pd.date_range(start, end, freq="h")
    rows = []
    for ts in timestamps:
        wind, vis, prec, sev = generate_weather(ts)
        wave_h = round(max(0, wind / 10 + np.random.normal(0, 0.3)), 1)
        rows.append({
            "timestamp":        ts,
            "wind_speed_ms":    wind,
            "visibility_km":    vis,
            "wave_height_m":    wave_h,
            "precipitation_mm": prec,
            "weather_severity": sev,
        })
    df = pd.DataFrame(rows)
    print(f"  Weather: {len(df):,} hourly records")
    return df


def generate_tides_series(start="2022-01-01", end="2024-12-31"):
    """Hourly tidal timeseries for the full date range."""
    print("Generating hourly tidal timeseries...")
    timestamps = pd.date_range(start, end, freq="h")
    T_sec = PORT_CONFIG["tidal_cycle_hours"] * 3600
    rows = []
    for ts in timestamps:
        height = generate_tidal_height(ts)
        phase  = ts.timestamp() % T_sec
        t2high = ((T_sec / 4) - phase) % T_sec / 3600
        rows.append({
            "timestamp":           ts,
            "tide_height_m":       round(height, 2),
            "time_to_high_tide_h": round(t2high, 2),
            "tide_favorable":      int(height > 5.5),
        })
    df = pd.DataFrame(rows)
    print(f"  Tides: {len(df):,} hourly records")
    return df


def generate_and_save_all(output_dir="data/synthetic"):
    """Generate all data, save CSVs, return (vessels, weather, tides, berths)."""
    os.makedirs(output_dir, exist_ok=True)
    start, end = "2022-01-01", "2024-12-31"

    vessels = generate_vessel_calls(5000, start, end)
    vessels.to_csv(f"{output_dir}/vessel_calls.csv", index=False)

    weather = generate_weather_series(start, end)
    weather.to_csv(f"{output_dir}/weather.csv", index=False)

    tides = generate_tides_series(start, end)
    tides.to_csv(f"{output_dir}/tides.csv", index=False)

    berths = pd.DataFrame(PORT_CONFIG["berths"])
    berths.to_csv(f"{output_dir}/berths.csv", index=False)

    print(f"\nAll data saved to {output_dir}/")
    return vessels, weather, tides, berths


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "../../data/synthetic")
    vessels, weather, tides, berths = generate_and_save_all(out)
    print(vessels[["vessel_id", "vessel_type", "eta", "delay_hours",
                   "service_hours", "weather_severity"]].head(8).to_string())
