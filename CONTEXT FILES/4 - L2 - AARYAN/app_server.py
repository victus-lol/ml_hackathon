"""Laptop 1 FastAPI service for server-side trip loading and map matching."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from pipeline_contract import MatchEngine, load_engine


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("pathsnap.api")
DEFAULT_DATA_ROOT = Path(__file__).parent / "data" / "evaluator_test"
DATA_ROOT = Path(os.environ.get("PATHSNAP_EVALUATOR_DIR", DEFAULT_DATA_ROOT)).resolve()
TRIP_ID_PATTERN = re.compile(r"^\d{8}$")

app = FastAPI(title="PathSnap API", version="1.0.0")
_engine: MatchEngine | None = None
_engine_error: str | None = None


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str


def get_engine() -> MatchEngine | None:
    global _engine, _engine_error
    if _engine is None and _engine_error is None:
        try:
            _engine = load_engine()
        except Exception as error:
            _engine_error = str(error)
            LOGGER.warning("Matcher not ready: %s", _engine_error)
    return _engine


def trip_directory(trip_id: str) -> Path:
    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise HTTPException(status_code=400, detail="trip_id must contain eight digits.")
    path = (DATA_ROOT / trip_id).resolve()
    if path.parent != DATA_ROOT:
        raise HTTPException(status_code=400, detail="Invalid trip_id.")
    return path


def read_trip(trip_id: str) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
    folder = trip_directory(trip_id)
    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"
    if not all(path.is_file() for path in (track_path, nodes_path, arcs_path)):
        raise HTTPException(status_code=404, detail="Trip data is unavailable on the API server.")

    try:
        track = np.loadtxt(track_path, dtype=np.float64, ndmin=2)
        nodes = np.loadtxt(nodes_path, dtype=np.float64, ndmin=2)
        arcs = np.loadtxt(arcs_path, dtype=np.int64, ndmin=2)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Could not read trip data: {error}") from error

    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise HTTPException(status_code=422, detail="Trip files have unexpected column counts.")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise HTTPException(status_code=422, detail="Trip coordinates or timestamps are invalid.")
    if np.any(np.abs(track[:, 0]) > 180) or np.any(np.abs(nodes[:, 0]) > 180):
        raise HTTPException(status_code=422, detail="Longitude values are outside valid bounds.")
    if np.any(np.abs(track[:, 1]) > 90) or np.any(np.abs(nodes[:, 1]) > 90):
        raise HTTPException(status_code=422, detail="Latitude values are outside valid bounds.")
    if arcs.size and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise HTTPException(status_code=422, detail="Trip network references a missing node.")
    return track.tolist(), nodes.tolist(), arcs.tolist()


def list_trip_ids() -> list[str]:
    if not DATA_ROOT.is_dir():
        return []
    trip_ids: list[str] = []
    for folder in sorted(DATA_ROOT.iterdir()):
        trip_id = folder.name
        if folder.is_dir() and TRIP_ID_PATTERN.fullmatch(trip_id):
            names = [folder / f"{trip_id}{suffix}" for suffix in (".track", ".nodes", ".arcs")]
            if all(path.is_file() for path in names):
                trip_ids.append(trip_id)
    return trip_ids


def clean_metrics(value: Any) -> dict[str, float]:
    allowed_names = {
        "accuracy", "top1_accuracy", "precision", "recall", "f1", "macro_f1",
        "log_loss", "route_continuity", "continuity_score", "candidate_miss_rate",
    }
    if not isinstance(value, dict):
        raise HTTPException(status_code=500, detail="Matcher returned invalid metrics.")
    clean: dict[str, float] = {}
    for name, metric in value.items():
        if name in allowed_names and isinstance(metric, (int, float, np.integer, np.floating)):
            if not isinstance(metric, (bool, np.bool_)) and np.isfinite(metric):
                clean[name] = float(metric)
    return clean


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    model_ready = bool(engine and engine.is_ready())
    data_ready = DATA_ROOT.is_dir()
    available_trip_count = len(list_trip_ids())
    process_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "service": "PathSnap Model & HMM API",
        "wire_protocol": "No ground-truth labels are accepted or returned.",
        "ready": model_ready and data_ready and available_trip_count == 10,
        "model_ready": model_ready,
        "data_directory_ready": data_ready,
        "available_trip_count": available_trip_count,
        "memory_usage_mb": round(process_memory_mb, 1),
        "matcher_error": _engine_error,
    }


@app.get("/api/v1/trips")
def trips() -> dict[str, list[str]]:
    """Return IDs only; the UI never reads trip files directly."""
    return {"trip_ids": list_trip_ids()}


@app.post("/api/v1/match")
def match_trip(request: MatchRequest) -> dict[str, Any]:
    engine = get_engine()
    if engine is None or not engine.is_ready():
        raise HTTPException(status_code=503, detail="Matcher is not ready; no prediction was produced.")

    track, nodes, arcs = read_trip(request.trip_id)
    start = time.perf_counter()
    LOGGER.info("Matching trip_id=%s", request.trip_id)
    try:
        result = engine.match_trip(request.trip_id, track, nodes, arcs)
    except Exception as error:
        LOGGER.exception("Matching failed for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=500, detail="Matching failed; see API logs.") from error

    predicted_route = result.get("predicted_route")
    predicted_coordinates = result.get("predicted_coordinates")
    if not isinstance(predicted_route, list) or not predicted_route or not isinstance(predicted_coordinates, list) or not predicted_coordinates:
        raise HTTPException(status_code=500, detail="Matcher returned an invalid result.")
    if any(not isinstance(arc_id, (int, np.integer)) or arc_id < 0 for arc_id in predicted_route):
        raise HTTPException(status_code=500, detail="Matcher returned invalid predicted arc IDs.")
    clean_coordinates: list[list[float]] = []
    for point in predicted_coordinates:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise HTTPException(status_code=500, detail="Matcher returned invalid route coordinates.")
        longitude, latitude = float(point[0]), float(point[1])
        if not np.isfinite([longitude, latitude]).all() or abs(longitude) > 180 or abs(latitude) > 90:
            raise HTTPException(status_code=500, detail="Matcher returned invalid route coordinates.")
        clean_coordinates.append([longitude, latitude])

    allowed_metrics = clean_metrics(result.get("metrics", {}))
    latency_ms = (time.perf_counter() - start) * 1000.0
    LOGGER.info("Completed trip_id=%s latency_ms=%.2f", request.trip_id, latency_ms)
    return {
        "trip_id": request.trip_id,
        "track_coordinates": [[point[0], point[1]] for point in track],
        "predicted_route": [int(arc_id) for arc_id in predicted_route],
        "predicted_coordinates": clean_coordinates,
        "predicted_route_geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": clean_coordinates},
            "properties": {"arc_ids": [int(arc_id) for arc_id in predicted_route]},
        },
        "metrics": allowed_metrics,
        "latency_ms": round(latency_ms, 2),
    }
