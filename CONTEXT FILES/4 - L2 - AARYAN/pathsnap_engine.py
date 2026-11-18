"""
CONTEXT FILES/4 - L2 - AARYAN/pathsnap_engine.py
------------------------------------------------
Production MatchEngine implementation for PathSnap Two-Laptop Distributed Deployment.
Conforms to pipeline_contract.MatchEngine protocol.

Flow:
1. Receives raw track, nodes, arcs over wire or local discovery
2. Projects coordinates to metric equirectangular plane (meters)
3. Queries candidate arcs via shapely.strtree.STRtree (50m radius + 150m fallback)
4. Computes 14 invariant physical features
5. Predicts calibrated emission probabilities using production LightGBM model
6. Runs log-space Viterbi decoding over networkx.MultiDiGraph
7. Applies contiguous run-length deduplication & Dijkstra topological bridge splicing
8. Returns evaluator-safe route prediction and Folium overlay coordinates (strictly zero .route)
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

# Add teammate 3 folder for HMM decoder
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
HMM_DIR = REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP"
if str(HMM_DIR) not in sys.path:
    sys.path.insert(0, str(HMM_DIR))

from hmm_viterbi_decoder import PathSnapHMMDecoder, build_road_multidigraph, measure_route_continuity

FEATURE_COLUMNS = [
    "perp_dist_m", "heading_delta_rad", "speed_mps", "proj_ratio", "arc_len_m",
    "dist_to_start_m", "dist_to_end_m", "speed_ratio", "temporal_dt",
    "u_degree", "v_degree", "sin_heading_delta", "cos_heading_delta", "dist_norm"
]

class PathSnapProductionEngine:
    def __init__(self, model_path: Path | None = None):
        if model_path is None:
            model_path = REPO_ROOT / "models" / "best_model_lightgbm.joblib"
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                self.model = None
                print(f"[Warning] Failed to load model from {self.model_path}: {e}")
        else:
            self.model = None

    def is_ready(self) -> bool:
        if self.model is None and self.model_path.exists():
            self._load_model()
        return self.model is not None

    def match_trip(
        self,
        trip_id: str,
        track: list[list[float]],
        nodes: list[list[float]],
        arcs: list[list[int]],
    ) -> dict[str, Any]:
        t0 = time.time()
        if not self.is_ready():
            raise RuntimeError("PathSnap Engine is not ready: Trained model is not loaded.")

        track_df = pd.DataFrame(track, columns=["lon", "lat", "t"])
        nodes_df = pd.DataFrame(nodes, columns=["lon", "lat"])
        arcs_df = pd.DataFrame(arcs, columns=["u", "v"])

        # 1. Metric Projection
        lat0 = math.radians(track_df["lat"].mean())
        lon0 = math.radians(track_df["lon"].mean())
        R = 6371000.0

        node_lats = np.radians(nodes_df["lat"].values)
        node_lons = np.radians(nodes_df["lon"].values)
        nodes_x = R * (node_lons - lon0) * math.cos(lat0)
        nodes_y = R * (node_lats - lat0)
        nodes_m = np.column_stack([nodes_x, nodes_y])

        track_lats = np.radians(track_df["lat"].values)
        track_lons = np.radians(track_df["lon"].values)
        track_x = R * (track_lons - lon0) * math.cos(lat0)
        track_y = R * (track_lats - lat0)

        # 2. Build STRtree & Road Graph
        u_idx = arcs_df["u"].values
        v_idx = arcs_df["v"].values
        arc_lines, arc_lens, arc_heads = [], [], []

        for i in range(len(arcs_df)):
            u_pt, v_pt = nodes_m[u_idx[i]], nodes_m[v_idx[i]]
            dx, dy = v_pt[0] - u_pt[0], v_pt[1] - u_pt[1]
            line = LineString([u_pt, v_pt])
            arc_lines.append(line)
            arc_lens.append(max(math.hypot(dx, dy), 1e-3))
            arc_heads.append(math.atan2(dy, dx))

        tree = STRtree(arc_lines)
        u_deg = pd.Series(u_idx).value_counts()
        v_deg = pd.Series(v_idx).value_counts()

        road_graph = build_road_multidigraph(nodes_df, arcs_df, lat0, lon0)
        decoder = PathSnapHMMDecoder(road_graph, arcs_df)

        # 3. Candidate Query & 14-Feature Extraction
        candidate_arcs_per_t = []
        perp_dists_per_t = []
        features_flat = []
        index_mapping = []

        linear_dists = []
        for i in range(len(track_df) - 1):
            dx = track_x[i+1] - track_x[i]
            dy = track_y[i+1] - track_y[i]
            linear_dists.append(math.hypot(dx, dy))

        for i in range(len(track_df)):
            px, py = track_x[i], track_y[i]
            pt = Point(px, py)
            t_sec = track_df["t"].iloc[i]

            if i < len(track_df) - 1:
                dt = max(track_df["t"].iloc[i+1] - t_sec, 1.0)
                vx = (track_x[i+1] - px) / dt
                vy = (track_y[i+1] - py) / dt
            elif i > 0:
                dt = max(t_sec - track_df["t"].iloc[i-1], 1.0)
                vx = (px - track_x[i-1]) / dt
                vy = (py - track_y[i-1]) / dt
            else:
                dt, vx, vy = 1.0, 0.0, 0.0

            gps_speed = math.hypot(vx, vy)
            gps_heading = math.atan2(vy, vx)

            matches = list(tree.query(pt.buffer(50.0)))
            if len(matches) == 0:
                matches = list(tree.query(pt.buffer(150.0)))
            if len(matches) == 0:
                # Nearest line fallback
                nearest_arc = min(range(len(arc_lines)), key=lambda a: arc_lines[a].distance(pt))
                matches = [nearest_arc]

            candidate_arcs_per_t.append(matches)
            dists_t = []

            for arc_id in matches:
                line = arc_lines[arc_id]
                perp_dist = line.distance(pt)
                proj_dist = line.project(pt)
                arc_len = arc_lens[arc_id]
                proj_ratio = min(max(proj_dist / arc_len, 0.0), 1.0)

                head_delta = abs(gps_heading - arc_heads[arc_id])
                head_delta = min(head_delta, 2 * math.pi - head_delta)

                u_node = u_idx[arc_id]
                v_node = v_idx[arc_id]
                dist_u = math.hypot(px - nodes_m[u_node][0], py - nodes_m[u_node][1])
                dist_v = math.hypot(px - nodes_m[v_node][0], py - nodes_m[v_node][1])

                dists_t.append(perp_dist)
                features_flat.append([
                    perp_dist, head_delta, gps_speed, proj_ratio, arc_len,
                    dist_u, dist_v, gps_speed / max(arc_len, 5.0), dt,
                    u_deg.get(u_node, 1), v_deg.get(v_node, 1),
                    math.sin(head_delta), math.cos(head_delta), perp_dist / 50.0
                ])
                index_mapping.append((i, len(dists_t) - 1))

            perp_dists_per_t.append(dists_t)

        # 4. Predict Calibrated Probabilities
        X_eval = np.array(features_flat, dtype=float)
        probs_flat = self.model.predict_proba(X_eval)[:, 1]

        # Reshape probabilities back to per-t
        ml_probs_per_t = []
        cursor = 0
        for matches in candidate_arcs_per_t:
            k = len(matches)
            ml_probs_per_t.append(list(probs_flat[cursor : cursor + k]))
            cursor += k

        # 5. Log-Space Viterbi Sequence Decoding
        raw_viterbi_seq = decoder.viterbi_decode(
            candidate_arcs_per_t,
            ml_probs_per_t,
            perp_dists_per_t,
            linear_dists
        )

        # 6. Two-Stage Post-Processing (Deduplication + Dijkstra Bridge Splicing)
        final_route = decoder.post_process_and_bridge_route(raw_viterbi_seq)
        continuity = measure_route_continuity(final_route, arcs_df)
        total_time_ms = (time.time() - t0) * 1000

        # 7. Generate Coordinates for Evaluator UI / GeoJSON
        route_coords = []
        for arc_id in final_route:
            u_pt = nodes_df.iloc[u_idx[arc_id]]
            v_pt = nodes_df.iloc[v_idx[arc_id]]
            route_coords.append([float(u_pt["lon"]), float(u_pt["lat"])])
            route_coords.append([float(v_pt["lon"]), float(v_pt["lat"])])

        return {
            "predicted_route": [int(a) for a in final_route],
            "predicted_coordinates": route_coords,
            "metrics": {
                "route_continuity": round(float(continuity), 4),
                "continuity_score": round(float(continuity), 4),
            },
            "status": "success"
        }

def create_engine() -> PathSnapProductionEngine:
    return PathSnapProductionEngine()
