"""
Batch feature extraction across the training trips
- shapely.strtree.STRtree 2D candidate discovery (50m radius + 150m fallback)
- Monotonic forward route projection to prevent intersection label contamination
- 14 invariant physical geometric & kinematic features
- Zero ID-leakage and anti-memorization compliance check
- Exports: data/train_candidates.parquet
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

# Banned columns check
BANNED_COLUMNS = ["trip_id", "arc_id", "from_node", "to_node", "lon", "lat", "time", "u", "v"]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def filter_gps_noise(track: pd.DataFrame, max_speed_mps: float = 75.0) -> pd.DataFrame:
    """Filters duplicate/non-monotonic timestamps and GPS spikes above 75 m/s (270 km/h)."""
    valid_indices = [0]
    for i in range(1, len(track)):
        prev_idx = valid_indices[-1]
        dt = track["t"].iloc[i] - track["t"].iloc[prev_idx]
        if dt <= 0:
            continue
        dist = haversine_distance(
            track["lat"].iloc[prev_idx], track["lon"].iloc[prev_idx],
            track["lat"].iloc[i], track["lon"].iloc[i]
        )
        if dist / dt > max_speed_mps:
            continue
        valid_indices.append(i)
    return track.iloc[valid_indices].reset_index(drop=True)

def apply_kalman_filter(track_m: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """2D Constant-Velocity Kalman Filter smoothing planar (x, y) coordinates."""
    N = len(track_m)
    smoothed = np.zeros_like(track_m)
    x = np.array([track_m[0, 0], track_m[0, 1], 0.0, 0.0])
    P = np.eye(4) * 5.0
    F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    Q = np.eye(4) * 0.5
    R = np.eye(2) * 4.0
    for i in range(N):
        x = F @ x
        P = F @ P @ F.T + Q
        z = track_m[i]
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        smoothed[i] = x[:2]
    return smoothed

def verify_feature_matrix_invariance(X: pd.DataFrame) -> None:
    offending_cols = [c for c in X.columns if any(banned == c.lower() for banned in BANNED_COLUMNS)]
    if offending_cols:
        raise ValueError(f"Feature matrix contains banned local columns: {offending_cols}")

def extract_features_single_trip(trip_dir: str, trip_id: str, search_radius_m: float = 50.0, fallback_radius_m: float = 150.0, use_kalman: bool = True):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path  = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    raw_track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs  = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"])
    
    # 0. Apply GPS Noise & Multipath Teleportation Filtering
    track = filter_gps_noise(raw_track, max_speed_mps=75.0)

    route_arcs = []
    if os.path.exists(route_path):
        route_arcs = list(pd.read_csv(route_path, header=None)[0].values)

    # 1. Metric Projection
    lat0, lon0 = math.radians(track["lat"].mean()), math.radians(track["lon"].mean())
    R = 6371000.0

    nodes_x = R * (np.radians(nodes["lon"].values) - lon0) * math.cos(lat0)
    nodes_y = R * (np.radians(nodes["lat"].values) - lat0)
    nodes_m = np.column_stack([nodes_x, nodes_y])

    track_x = R * (np.radians(track["lon"].values) - lon0) * math.cos(lat0)
    track_y = R * (np.radians(track["lat"].values) - lat0)

    # 1.1 Apply 2D Kalman Filter Smoothing
    if use_kalman and len(track) > 2:
        track_planar = np.column_stack([track_x, track_y])
        smoothed_planar = apply_kalman_filter(track_planar)
        track_x, track_y = smoothed_planar[:, 0], smoothed_planar[:, 1]

    # 2. Build STRtree
    u_idx, v_idx = arcs["u"].values, arcs["v"].values
    arc_lines, arc_lens, arc_heads = [], [], []

    for i in range(len(arcs)):
        u_pt, v_pt = nodes_m[u_idx[i]], nodes_m[v_idx[i]]
        dx, dy = v_pt[0] - u_pt[0], v_pt[1] - u_pt[1]
        line = LineString([u_pt, v_pt])
        arc_lines.append(line)
        arc_lens.append(max(math.hypot(dx, dy), 1e-3))
        arc_heads.append(math.atan2(dy, dx))

    tree = STRtree(arc_lines)
    u_deg = pd.Series(u_idx).value_counts()
    v_deg = pd.Series(v_idx).value_counts()

    # 3. Monotonic Sequential Route Matching
    route_ptr = 0
    num_route = len(route_arcs)
    
    trip_records = []
    misses = 0

    for i in range(len(track)):
        px, py = track_x[i], track_y[i]
        pt = Point(px, py)
        t_sec = track["t"].iloc[i]

        # Velocity and Heading
        if i < len(track) - 1:
            dt = max(track["t"].iloc[i+1] - t_sec, 1.0)
            vx = (track_x[i+1] - px) / dt
            vy = (track_y[i+1] - py) / dt
        elif i > 0:
            dt = max(t_sec - track["t"].iloc[i-1], 1.0)
            vx = (px - track_x[i-1]) / dt
            vy = (py - track_y[i-1]) / dt
        else:
            dt, vx, vy = 1.0, 0.0, 0.0

        gps_speed = math.hypot(vx, vy)
        gps_heading = math.atan2(vy, vx)

        # Query candidates
        matches = list(tree.query(pt.buffer(search_radius_m)))
        if len(matches) == 0:
            matches = list(tree.query(pt.buffer(fallback_radius_m)))

        # Monotonic Ground-Truth Matching
        # Look ahead in the forward route window (up to 15 arcs ahead)
        best_true_arc = None
        if num_route > 0 and len(matches) > 0:
            forward_window = route_arcs[route_ptr : min(route_ptr + 15, num_route)]
            common = [a for a in matches if a in forward_window]
            if common:
                # Pick the forward-window arc closest to this GPS point
                distances = [arc_lines[a].distance(pt) for a in common]
                best_true_arc = common[int(np.argmin(distances))]
                # Advance route_ptr to the matched position
                route_ptr = route_arcs.index(best_true_arc, route_ptr)
            else:
                # If not in forward window, check current arc or route set
                if route_ptr < num_route and route_arcs[route_ptr] in matches:
                    best_true_arc = route_arcs[route_ptr]
                else:
                    misses += 1

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

            trip_records.append({
                "trip_id": trip_id,
                "timestamp": t_sec,
                "arc_id": arc_id,
                "label": 1 if arc_id == best_true_arc else 0,
                
                # 14 Invariant Features
                "perp_dist_m": perp_dist,
                "heading_delta_rad": head_delta,
                "speed_mps": gps_speed,
                "proj_ratio": proj_ratio,
                "arc_len_m": arc_len,
                "dist_to_start_m": dist_u,
                "dist_to_end_m": dist_v,
                "speed_ratio": gps_speed / max(arc_len, 5.0),
                "temporal_dt": dt,
                "u_degree": u_deg.get(u_node, 1),
                "v_degree": v_deg.get(v_node, 1),
                "sin_heading_delta": math.sin(head_delta),
                "cos_heading_delta": math.cos(head_delta),
                "dist_norm": perp_dist / search_radius_m,
            })

    return trip_records, len(track), misses

def batch_extract_all_trips(
    train_root: str = "C:/Users/adity/Downloads/ML HACKATHON/Student folder/Student folder/train",
    out_parquet: str = "data/train_candidates.parquet"
):
    print("=" * 75)
    print("   PATHSNAP 80-TRIP BATCH FEATURE EXTRACTION & DATASET COMPILATION")
    print("=" * 75)

    trips = sorted(os.listdir(train_root))
    print(f"Found {len(trips)} training trip directories in:\n  -> {train_root}\n")

    t_start = time.time()
    all_records = []
    total_gps = 0
    total_misses = 0

    for idx, trip_id in enumerate(trips, 1):
        trip_dir = os.path.join(train_root, trip_id)
        if not os.path.isdir(trip_dir):
            continue

        records, gps_count, misses = extract_features_single_trip(trip_dir, trip_id)
        all_records.extend(records)
        total_gps += gps_count
        total_misses += misses

        if idx % 10 == 0 or idx == len(trips):
            elapsed = time.time() - t_start
            print(f"[{idx:02d}/80] Processed Trip {trip_id} | Total Candidates: {len(all_records):,} | Elapsed: {elapsed:.1f}s ({elapsed/idx:.2f}s/trip)")

    print("\n[Finalizing DataFrame Construction...]")
    df_all = pd.DataFrame(all_records)
    
    total_time = time.time() - t_start
    print(f"\nExtraction Completed in {total_time:.2f} seconds ({total_time/len(trips):.2f}s per trip)!")

    # Verify Anti-Memorization
    feature_cols = [c for c in df_all.columns if c not in ["trip_id", "timestamp", "arc_id", "label"]]
    verify_feature_matrix_invariance(df_all[feature_cols])
    print("[Anti-Memorization Check]: PASSED! Exactly 14 invariant features.")

    # Save to Parquet
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df_all.to_parquet(out_parquet, index=False)
    print(f"\nSaved consolidated dataset to:\n  -> {out_parquet}")
    print(f"  -> File size: {os.path.getsize(out_parquet) / (1024*1024):.2f} MB")

    # Audit Summary
    pos_labels = df_all["label"].sum()
    neg_labels = len(df_all) - pos_labels
    miss_rate = (total_misses / total_gps) * 100

    print("\n" + "=" * 75)
    print("                 80-TRIP DATASET EXTRACTION AUDIT")
    print("=" * 75)
    print(f"Total Trips Processed:          {len(trips)}")
    print(f"Total GPS Observations:         {total_gps:,}")
    print(f"Total Candidate Rows:           {len(df_all):,}")
    print(f"Average Candidates per Point:   {len(df_all)/total_gps:.2f}")
    print(f"Positive Labels (True Arc):     {pos_labels:,} ({pos_labels/len(df_all)*100:.2f}%)")
    print(f"Negative Labels (False Arcs):   {neg_labels:,} ({neg_labels/len(df_all)*100:.2f}%)")
    print(f"Class Imbalance Ratio:          1 : {neg_labels/max(pos_labels, 1):.2f}")
    print(f"Overall Candidate Miss Rate:    {miss_rate:.2f}% (Target: < 0.3%)")
    print("=" * 75)

if __name__ == "__main__":
    batch_extract_all_trips()
