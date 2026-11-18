"""
CONTEXT FILES/1 - AKSHAY/generate_map.py
----------------------------------------
Interactive Multi-Layer Folium Visualization for Exploratory Data Analysis
Displays:
1. Road Network Arcs (Grey)
2. Raw GPS Observations (Red Circles)
3. Filtered Outlier Spikes & Teleportation (Red/Black X Markers with speed annotations)
4. Kalman-Smoothed Trajectory (Thick Blue PolyLine)
5. Ground-Truth Reference Route (Bright Green PolyLine)
Interactive Layer Control allows toggling each layer independently.
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
import folium

SCRIPT_DIR = Path(__file__).parent.resolve()
SAMPLE_DIR = SCRIPT_DIR / "sample_trip_00000000"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def apply_kalman(track_m, dt=1.0):
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

def main():
    track_path = SAMPLE_DIR / "00000000.track"
    nodes_path = SAMPLE_DIR / "00000000.nodes"
    arcs_path  = SAMPLE_DIR / "00000000.arcs"
    route_path = SAMPLE_DIR / "00000000.route"

    raw_track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs  = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(route_path, header=None)[0].values if route_path.exists() else []

    # Detect outliers (> 75 m/s)
    valid_idx = [0]
    dropped = []
    for i in range(1, len(raw_track)):
        prev = valid_idx[-1]
        dt = raw_track["t"].iloc[i] - raw_track["t"].iloc[prev]
        if dt <= 0:
            dropped.append((i, raw_track.iloc[i], 0.0, "Duplicate timestamp"))
            continue
        d = haversine(raw_track["lat"].iloc[prev], raw_track["lon"].iloc[prev],
                      raw_track["lat"].iloc[i], raw_track["lon"].iloc[i])
        v = d / dt
        if v > 75.0:
            dropped.append((i, raw_track.iloc[i], v * 3.6, f"Speed spike ({v*3.6:.1f} km/h)"))
            continue
        valid_idx.append(i)

    clean_track = raw_track.iloc[valid_idx].reset_index(drop=True)

    # Kalman smoothing
    lat0, lon0 = math.radians(clean_track["lat"].mean()), math.radians(clean_track["lon"].mean())
    R = 6371000.0
    track_x = R * (np.radians(clean_track["lon"].values) - lon0) * math.cos(lat0)
    track_y = R * (np.radians(clean_track["lat"].values) - lat0)
    smoothed_m = apply_kalman(np.column_stack([track_x, track_y]))
    smoothed_lon = np.degrees(smoothed_m[:, 0] / (R * math.cos(lat0)) + lon0)
    smoothed_lat = np.degrees(smoothed_m[:, 1] / R + lat0)

    # Base Folium Map
    center_lat = clean_track["lat"].mean()
    center_lon = clean_track["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Layer 1: Road Network Arcs
    fg_arcs = folium.FeatureGroup(name="1. Road Network Arcs (Grey)", show=True)
    for _, arc in arcs.iterrows():
        u_idx, v_idx = int(arc["u"]), int(arc["v"])
        if u_idx < len(nodes) and v_idx < len(nodes):
            u_pt = nodes.iloc[u_idx]
            v_pt = nodes.iloc[v_idx]
            folium.PolyLine(
                [(u_pt["lat"], u_pt["lon"]), (v_pt["lat"], v_pt["lon"])],
                color="#555555", weight=1.2, opacity=0.5
            ).add_to(fg_arcs)
    fg_arcs.add_to(m)

    # Layer 2: Raw GPS Observations
    fg_raw = folium.FeatureGroup(name="2. Raw GPS Points (Red/Orange)", show=False)
    for _, row in raw_track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2.0, color="#E65100", fill=True, fill_color="#FF9800", fill_opacity=0.6,
            tooltip=f"Raw Ping: t={int(row['t'])}s"
        ).add_to(fg_raw)
    fg_raw.add_to(m)

    # Layer 3: Removed Outliers / Teleportation Spikes
    fg_spikes = folium.FeatureGroup(name=f"3. Removed GPS Spikes ({len(dropped)} outliers)", show=True)
    for idx, row, spd, reason in dropped:
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="red", icon="remove", prefix="fa"),
            tooltip=f"DROPPED: {reason} at t={int(row['t'])}s"
        ).add_to(fg_spikes)
    fg_spikes.add_to(m)

    # Layer 4: Kalman-Smoothed Trajectory
    fg_kalman = folium.FeatureGroup(name="4. Kalman-Smoothed Trajectory (Blue)", show=True)
    folium.PolyLine(
        list(zip(smoothed_lat, smoothed_lon)),
        color="#1E88E5", weight=3.5, opacity=0.85,
        tooltip="Kalman Smoothed Path"
    ).add_to(fg_kalman)
    fg_kalman.add_to(m)

    # Layer 5: Ground-Truth Route
    if len(route) > 0:
        fg_route = folium.FeatureGroup(name="5. Ground-Truth Route (Green)", show=True)
        route_lines = []
        for arc_id in route:
            if arc_id < len(arcs):
                u_idx, v_idx = int(arcs.iloc[arc_id]["u"]), int(arcs.iloc[arc_id]["v"])
                route_lines.append([(nodes.iloc[u_idx]["lat"], nodes.iloc[u_idx]["lon"]),
                                    (nodes.iloc[v_idx]["lat"], nodes.iloc[v_idx]["lon"])])
        for line in route_lines:
            folium.PolyLine(line, color="#00C853", weight=4.0, opacity=0.9).add_to(fg_route)
        fg_route.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_file = SCRIPT_DIR / "trip_00000000_eda.html"
    m.save(str(out_file))
    print(f"Saved complete multi-layer interactive Folium map to:\n  -> {out_file}")

if __name__ == "__main__":
    main()
