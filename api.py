# -*- coding: utf-8 -*-
"""
FastAPI backend - AI Berth Allocation System
MSc AI Dissertation - Manju Arun (202433718), University of Hull

Run with:
    venv311/Scripts/uvicorn api:app --reload --port 8000
Then open:  http://localhost:8000
"""

import sys, os, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib

from src.optimization.genetic_algorithm import (
    GeneticAlgorithmBAP, Vessel as GAVessel, Berth as GABerth
)
from src.models.forecasting import WeightedEnsemble

# -- Global state -------------------------------------------------------------
STATE = {
    "ready": False,
    "ensemble": None,
    "scaler": None,
    "feature_cols": None,
    "hist_stats": {},
    "last_schedule": [],
    "last_ga_cost": None,
    "confirmed_allocations": [],
}
BERTHS_DF: Optional[pd.DataFrame] = None


# -- Startup / shutdown -------------------------------------------------------
async def _startup():
    global BERTHS_DF
    try:
        artifacts = joblib.load("data/processed/api_artifacts.pkl")
        STATE["feature_cols"] = artifacts["feature_cols"]
        STATE["scaler"]       = artifacts["scaler"]
        STATE["hist_stats"]   = artifacts["hist_stats"]

        STATE["ensemble"] = WeightedEnsemble.load("results/models")
        STATE["ready"] = True
        print("Models loaded successfully.")
    except Exception as e:
        STATE["error"] = str(e)
        print(f"WARNING: Could not load models - {e}")
        print("Run main.py first to train models.")

    try:
        BERTHS_DF = pd.read_csv("data/synthetic/berths.csv")
    except FileNotFoundError:
        print("WARNING: berths.csv not found. Run main.py first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield


# -- App ----------------------------------------------------------------------
app = FastAPI(
    title="AI Berth Allocation System",
    description="MSc AI Dissertation - Manju Arun (202433718)",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# -- Pydantic schemas ---------------------------------------------------------
class VesselInput(BaseModel):
    vessel_id: str
    vessel_type: str
    length_m: float
    beam_m: float
    draft_m: float
    service_hours: float
    eta: str
    priority: str = "normal"
    wind_speed_ms: float = 12.0
    visibility_km: float = 8.0
    wave_height_m: float = 1.0
    precipitation_mm: float = 0.5
    historical_reliability: float = 0.75


class AllocationRequest(BaseModel):
    vessels: List[VesselInput]


class ConfirmRequest(BaseModel):
    vessel_id: str
    vessel_type: str
    berth_id: str
    eta: str
    predicted_ata: str
    actual_start: str
    end_time: str
    service_hours: float
    predicted_delay_hours: float
    priority: str
    length_m: float
    beam_m: float
    draft_m: float


# -- Feature builder ----------------------------------------------------------
def _build_features(vessel: VesselInput) -> np.ndarray:
    feature_cols = STATE["feature_cols"]
    scaler       = STATE["scaler"]
    hist_stats   = STATE["hist_stats"]

    eta  = pd.Timestamp(vessel.eta)
    wind = vessel.wind_speed_ms
    vis  = vessel.visibility_km
    wave = vessel.wave_height_m
    prec = vessel.precipitation_mm
    wsev = min(2, int(wind / 20) + int(vis < 3))
    tide = 5.0 + 3.0 * np.sin(2 * np.pi * eta.timestamp() / (12.4 * 3600))

    row = {col: 0.0 for col in feature_cols}

    row.update({
        "hour":               eta.hour,
        "day_of_week":        eta.dayofweek,
        "week_of_year":       eta.isocalendar()[1],
        "month":              eta.month,
        "quarter":            eta.quarter,
        "is_weekend":         int(eta.dayofweek >= 5),
        "hour_sin":           np.sin(2 * np.pi * eta.hour / 24),
        "hour_cos":           np.cos(2 * np.pi * eta.hour / 24),
        "dayofweek_sin":      np.sin(2 * np.pi * eta.dayofweek / 7),
        "dayofweek_cos":      np.cos(2 * np.pi * eta.dayofweek / 7),
        "month_sin":          np.sin(2 * np.pi * eta.month / 12),
        "month_cos":          np.cos(2 * np.pi * eta.month / 12),
        "arrivals_last_24h":  5,
        "arrivals_last_48h":  10,
        "avg_service_last_7d": 19.94,
        "rolling_mean_7d":    5.0,
        "rolling_std_7d":     1.5,
        "rolling_mean_30d":   5.0,
    })

    for vtype in ["bulk", "container", "general", "roro", "tanker"]:
        col = f"vtype_{vtype}"
        if col in row:
            row[col] = 1 if vessel.vessel_type == vtype else 0

    stats = hist_stats.get(vessel.vessel_type, {})
    row["hist_median_delay"] = stats.get("median", 2.0)
    row["hist_std_delay"]    = stats.get("std", 1.0)

    row.update({
        "priority_encoded":       {"high": 2, "normal": 1, "low": 0}.get(vessel.priority, 1),
        "length_m":               vessel.length_m,
        "beam_m":                 vessel.beam_m,
        "draft_m":                vessel.draft_m,
        "service_hours":          vessel.service_hours,
        "gross_tonnage":          vessel.length_m * vessel.draft_m * 70,
        "cargo_volume":           2000.0,
        "historical_reliability": vessel.historical_reliability,
        "cargo_complexity":       vessel.service_hours / max(vessel.length_m, 1),
    })

    env = {
        "wind_speed_ms":      wind,
        "visibility_km":      vis,
        "wave_height_m":      wave,
        "precipitation_mm":   prec,
        "weather_severity":   wsev,
        "tide_height_m":      round(tide, 2),
        "time_to_high_tide_h": 3.0,
        "tide_favorable":     int(tide > 5.5),
        "safe_to_berth":      int(wsev < 2 and wind < 20),
    }
    row.update(env)
    for k, v in env.items():
        if k != "safe_to_berth":
            row[f"port_{k}"] = v

    X = np.array([[row.get(col, 0.0) for col in feature_cols]])
    return scaler.transform(X)


# -- Routes -------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status":        "ready" if STATE["ready"] else "not_ready",
        "models_loaded": STATE["ready"],
        "features":      len(STATE["feature_cols"] or []),
        "error":         STATE.get("error"),
    }


@app.get("/berths")
async def get_berths():
    if BERTHS_DF is None:
        raise HTTPException(503, "Berths data not loaded. Run main.py first.")
    records = BERTHS_DF.to_dict("records")
    for r in records:
        if isinstance(r.get("allowed_types"), str):
            r["allowed_types"] = r["allowed_types"].split(",")
    return records


@app.post("/predict")
async def predict_delay(vessel: VesselInput):
    if not STATE["ready"]:
        raise HTTPException(503, "Models not loaded. Run main.py first.")
    X    = _build_features(vessel)
    pred = float(STATE["ensemble"].predict(X)[0])
    eta  = pd.Timestamp(vessel.eta)
    ata  = eta + pd.Timedelta(hours=pred)
    return {
        "vessel_id":             vessel.vessel_id,
        "vessel_type":           vessel.vessel_type,
        "eta":                   vessel.eta,
        "predicted_delay_hours": round(pred, 2),
        "predicted_ata":         ata.isoformat(),
        "priority":              vessel.priority,
    }


@app.post("/allocate")
async def allocate(request: AllocationRequest):
    if not STATE["ready"]:
        raise HTTPException(503, "Models not loaded. Run main.py first.")
    if BERTHS_DF is None:
        raise HTTPException(503, "Berths not loaded.")
    if not request.vessels:
        raise HTTPException(400, "No vessels provided.")

    # Validate vessel dimensions against actual berth limits
    max_loa   = float(BERTHS_DF["length"].max())
    max_beam  = float(BERTHS_DF["beam"].max())
    max_depth = float(BERTHS_DF["depth"].max())
    valid_types = {"container", "bulk", "tanker", "general", "roro"}

    errors = []
    for v in request.vessels:
        if v.length_m > max_loa:
            errors.append(f"{v.vessel_id}: LOA {v.length_m}m exceeds max berth length of {max_loa:.0f}m")
        if v.beam_m > max_beam:
            errors.append(f"{v.vessel_id}: Beam {v.beam_m}m exceeds max berth beam of {max_beam:.0f}m")
        if v.draft_m > max_depth:
            errors.append(f"{v.vessel_id}: Draft {v.draft_m}m exceeds max berth depth of {max_depth:.0f}m")
        if v.service_hours <= 0:
            errors.append(f"{v.vessel_id}: Service hours must be > 0")
        if v.vessel_type not in valid_types:
            errors.append(f"{v.vessel_id}: Unknown vessel type '{v.vessel_type}'")
    if errors:
        raise HTTPException(422, detail={"validation_errors": errors})

    predictions = []
    for v in request.vessels:
        X     = _build_features(v)
        pred  = float(STATE["ensemble"].predict(X)[0])
        ata_h = (pd.Timestamp(v.eta) + pd.Timedelta(hours=pred)).timestamp() / 3600
        predictions.append({"vessel": v, "delay": pred, "ata_h": ata_h})

    ga_vessels = [
        GAVessel(
            id=p["vessel"].vessel_id,
            length=p["vessel"].length_m,
            beam=p["vessel"].beam_m,
            draft=p["vessel"].draft_m,
            predicted_ata=p["ata_h"],
            service_hours=p["vessel"].service_hours,
            priority=p["vessel"].priority,
            vessel_type=p["vessel"].vessel_type,
        )
        for p in predictions
    ]
    ga_berths = [
        GABerth(
            id=str(row["id"]),
            length=float(row["length"]),
            depth=float(row["depth"]),
            beam=float(row["beam"]),
            allowed_types=str(row["allowed_types"]).split(","),
        )
        for _, row in BERTHS_DF.iterrows()
    ]

    ga = GeneticAlgorithmBAP(
        vessels=ga_vessels, berths=ga_berths,
        pop_size=100, generations=200,
    )
    best_chrom, best_fitness = ga.optimize(verbose=False)
    schedule = ga.decode_schedule(best_chrom)

    vmap = {p["vessel"].vessel_id: p for p in predictions}
    result = []
    for item in schedule:
        p = vmap.get(item["vessel_id"], {})
        v = p.get("vessel")
        result.append({
            **item,
            "vessel_type":           v.vessel_type if v else "",
            "length_m":              v.length_m    if v else 0,
            "beam_m":                v.beam_m      if v else 0,
            "draft_m":               v.draft_m     if v else 0,
            "eta":                   v.eta         if v else "",
            "predicted_delay_hours": round(p.get("delay", 0), 2),
            "predicted_ata":         pd.Timestamp(
                p["ata_h"] * 3600, unit="s"
            ).isoformat() if p else "",
        })

    STATE["last_schedule"] = result
    STATE["last_ga_cost"]  = round(-best_fitness, 2)

    return {
        "schedule":          result,
        "ga_cost":           round(-best_fitness, 2),
        "vessels_scheduled": len(result),
        "berths_used":       len(set(r["berth_id"] for r in result)),
        "convergence":       [round(c, 1) for c in ga.best_fitness_history[::5]],
    }


@app.get("/schedule")
async def get_schedule():
    return {
        "schedule": STATE["last_schedule"],
        "count":    len(STATE["last_schedule"]),
        "ga_cost":  STATE["last_ga_cost"],
    }


# -- Single-vessel check (no commit) ------------------------------------------
@app.post("/check")
async def check_allocation(vessel: VesselInput):
    if not STATE["ready"]:
        raise HTTPException(503, "Models not loaded.")
    if BERTHS_DF is None:
        raise HTTPException(503, "Berths not loaded.")

    # Validate dimensions
    max_loa  = float(BERTHS_DF["length"].max())
    max_beam = float(BERTHS_DF["beam"].max())
    max_dep  = float(BERTHS_DF["depth"].max())
    errs = []
    if vessel.length_m > max_loa:  errs.append(f"LOA {vessel.length_m}m > max {max_loa:.0f}m")
    if vessel.beam_m   > max_beam: errs.append(f"Beam {vessel.beam_m}m > max {max_beam:.0f}m")
    if vessel.draft_m  > max_dep:  errs.append(f"Draft {vessel.draft_m}m > max {max_dep:.0f}m")
    if errs:
        raise HTTPException(422, detail={"validation_errors": errs})

    # Predict delay
    X         = _build_features(vessel)
    pred_delay = float(STATE["ensemble"].predict(X)[0])
    eta        = pd.Timestamp(vessel.eta)
    pred_ata   = eta + pd.Timedelta(hours=pred_delay)
    pred_ata_h = pred_ata.timestamp() / 3600

    # Build berth occupancy from confirmed allocations
    berth_free: dict = {}
    for alloc in STATE["confirmed_allocations"]:
        bid = alloc["berth_id"]
        berth_free[bid] = max(berth_free.get(bid, 0.0), alloc["end_time_h"] + 1.0)

    # Score each compatible berth
    options = []
    for _, row in BERTHS_DF.iterrows():
        bid      = str(row["id"])
        allowed  = str(row["allowed_types"]).split(",")
        compat   = {
            "loa":       vessel.length_m <= float(row["length"]),
            "beam":      vessel.beam_m   <= float(row["beam"]),
            "draft":     vessel.draft_m  <= float(row["depth"]),
            "cargo":     vessel.vessel_type in allowed,
        }
        if not all(compat.values()):
            continue
        earliest   = max(pred_ata_h, berth_free.get(bid, 0.0))
        wait_h     = round(earliest - pred_ata_h, 2)
        end_h      = earliest + vessel.service_hours
        options.append({
            "berth_id":    bid,
            "berth_length": float(row["length"]),
            "berth_beam":   float(row["beam"]),
            "berth_depth":  float(row["depth"]),
            "allowed_types": allowed,
            "wait_hours":  wait_h,
            "start_h":     earliest,
            "end_h":       end_h,
            "compat":      compat,
        })

    if not options:
        raise HTTPException(409, "No compatible berth available. All berths are occupied or incompatible.")

    options.sort(key=lambda x: (x["wait_hours"], x["berth_id"]))
    best = options[0]
    fmt  = lambda h: pd.Timestamp(h * 3600, unit="s").isoformat(timespec="milliseconds")

    return {
        "vessel_id":             vessel.vessel_id,
        "vessel_type":           vessel.vessel_type,
        "eta":                   vessel.eta,
        "predicted_delay_hours": round(pred_delay, 2),
        "predicted_ata":         pred_ata.isoformat(),
        "recommended_berth":     best["berth_id"],
        "actual_start":          fmt(best["start_h"]),
        "end_time":              fmt(best["end_h"]),
        "wait_hours":            best["wait_hours"],
        "service_hours":         vessel.service_hours,
        "priority":              vessel.priority,
        "length_m":              vessel.length_m,
        "beam_m":                vessel.beam_m,
        "draft_m":               vessel.draft_m,
        "compatibility":         best["compat"],
        "alternatives": [
            {"berth_id": o["berth_id"], "wait_hours": o["wait_hours"],
             "actual_start": fmt(o["start_h"])}
            for o in options[1:4]
        ],
    }


# -- Confirm and save a booking -----------------------------------------------
@app.post("/confirm")
async def confirm_booking(req: ConfirmRequest):
    booking_id = f"BK-{len(STATE['confirmed_allocations'])+1:04d}"
    alloc = {
        "booking_id":            booking_id,
        "vessel_id":             req.vessel_id,
        "vessel_type":           req.vessel_type,
        "berth_id":              req.berth_id,
        "eta":                   req.eta,
        "predicted_ata":         req.predicted_ata,
        "actual_start":          req.actual_start,
        "end_time":              req.end_time,
        "end_time_h":            pd.Timestamp(req.end_time).timestamp() / 3600,
        "service_hours":         req.service_hours,
        "predicted_delay_hours": req.predicted_delay_hours,
        "priority":              req.priority,
        "length_m":              req.length_m,
        "beam_m":                req.beam_m,
        "draft_m":               req.draft_m,
        "confirmed_at":          pd.Timestamp.now().isoformat(),
        "status":                "confirmed",
    }
    STATE["confirmed_allocations"].append(alloc)
    return {"booking_id": booking_id, "status": "confirmed", "allocation": alloc}


# -- Get confirmed allocations with optional date filter ----------------------
@app.get("/confirmed")
async def get_confirmed(date_from: Optional[str] = None, date_to: Optional[str] = None):
    allocs = STATE["confirmed_allocations"]
    if date_from:
        dt = pd.Timestamp(date_from)
        allocs = [a for a in allocs if pd.Timestamp(a["actual_start"]) >= dt]
    if date_to:
        dt = pd.Timestamp(date_to) + pd.Timedelta(days=1)
        allocs = [a for a in allocs if pd.Timestamp(a["actual_start"]) < dt]
    return allocs
