"""Laptop 1 prediction API. The inference path never opens reference route files."""
from __future__ import annotations

import os
import logging
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from decoder import bridge_route, decode
from pathsnap import FEATURES, _candidate_frame, load_trip, trip_ids

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_DIR", "data/evaluator_test")).resolve()
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", "artifacts/experiments/final_model.joblib")).resolve()
DATA_KIND = os.environ.get("PATHSNAP_DATA_KIND", "evaluator")
HMM_BETA_M = float(os.environ.get("PATHSNAP_HMM_BETA_M", "50"))
HMM_SIGMA_M = float(os.environ.get("PATHSNAP_HMM_SIGMA_M", "25"))
ID_RE = re.compile(r"^\d{8}$")
app = FastAPI(title="PathSnap model and HMM API", version="1.0")
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
LOGGER=logging.getLogger("pathsnap.api")
_bundle: dict[str, Any] | None = None
_model_error: str | None = None


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str


def get_bundle():
    global _bundle, _model_error
    if _bundle is None and _model_error is None:
        try:
            loaded = joblib.load(MODEL_PATH)
            if not isinstance(loaded, dict) or "model" not in loaded:
                raise ValueError("Model artifact must be a dict with a 'model' entry")
            _bundle = loaded
        except Exception as exc:
            _model_error = str(exc)
    return _bundle


def valid_trip_path(tid: str) -> Path:
    if not ID_RE.fullmatch(tid):
        raise HTTPException(400, "trip_id must be eight digits")
    folder = (DATA_ROOT / tid).resolve()
    if folder.parent != DATA_ROOT:
        raise HTTPException(400, "Invalid trip_id")
    return folder


def discover_ids() -> list[str]:
    if not DATA_ROOT.is_dir():
        return []
    return [tid for tid in trip_ids(DATA_ROOT) if ID_RE.fullmatch(tid) and all(
        (DATA_ROOT / tid / f"{tid}{ext}").is_file() for ext in (".track", ".nodes", ".arcs")
    )]


@app.get("/api/v1/health")
def health():
    bundle = get_bundle()
    ids = discover_ids()
    return {"status":"healthy", "ready":bool(bundle and len(ids)==10 and DATA_KIND=="evaluator"),
        "model_ready":bundle is not None, "evaluator_trip_count":len(ids),
        "configured_data_kind":DATA_KIND,
        "memory_usage_mb":round(psutil.Process().memory_info().rss/(1024*1024),1),
        "model_error":_model_error,
        "wire_protocol":"trip_id request; no reference labels accepted or returned"}


@app.get("/api/v1/trips")
def trips():
    return {"trip_ids":discover_ids()}


@app.post("/api/v1/match")
def match(request: MatchRequest):
    bundle=get_bundle()
    if bundle is None:
        raise HTTPException(503,"Prediction model is not loaded")
    folder=valid_trip_path(request.trip_id)
    if not all((folder/f"{request.trip_id}{ext}").is_file() for ext in (".track",".nodes",".arcs")):
        raise HTTPException(404,"Trip inputs unavailable on API server")
    try:
        data=load_trip(folder,include_route=False)
    except (OSError,ValueError) as exc:
        raise HTTPException(422,f"Trip input could not be parsed: {exc}") from exc
    t0=time.perf_counter()
    LOGGER.info("Match request trip_id=%s",request.trip_id)
    frame=_candidate_frame(data,labeled=False)
    if frame.empty:
        raise HTTPException(422,"No road candidates could be generated")
    feature_names=bundle.get("features",FEATURES)
    if feature_names != FEATURES:
        raise HTTPException(500,"Model feature schema does not match runtime feature code")
    probs=bundle["model"].predict_proba(frame[feature_names].to_numpy(np.float32))
    classes=list(bundle["model"].classes_)
    p=probs[:,classes.index(1)] if 1 in classes else np.zeros(len(frame))
    if bundle.get("calibrator") is not None:
        p=bundle["calibrator"].predict(p)
    frame["_p"]=p
    candidate_sets=[]
    for _,g in frame.groupby("observation_id",sort=True):
        candidate_sets.append([{"arc_id":int(r["candidate_arc_id"]),"p":max(float(r["_p"]),1e-9),
            "distance_m":float(r["distance_m"]),"fraction":float(r["projection_fraction"])} for r in g.to_dict("records")])
    obs_ids=sorted(frame.observation_id.unique())
    pred=decode(candidate_sets,data["arcs"],data["lengths"],data["gps_xy"][obs_ids],data["track"][obs_ids,2],
                beta_m=HMM_BETA_M,sigma_m=HMM_SIGMA_M)
    route=bridge_route(pred,data["arcs"],data["lengths"])
    route_coords=[]
    for arc_id in route:
        u,v=data["arcs"][arc_id]
        a=data["nodes"][u].tolist(); b=data["nodes"][v].tolist()
        if route_coords and route_coords[-1]==a:
            route_coords.append(b)
        else:
            route_coords.extend([a,b])
    latency=(time.perf_counter()-t0)*1000
    LOGGER.info("Completed trip_id=%s latency_ms=%.2f candidate_count=%d",request.trip_id,latency,len(frame))
    return {"trip_id":request.trip_id,
        "track_coordinates":data["track"][:,0:2].tolist(),
        "predicted_route":route,
        "predicted_coordinates":route_coords,
        "metrics":{},"latency_ms":round(latency,2)}
