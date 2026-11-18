"""Dataset loading, projected geometry, candidate features, and offline map export."""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

FEATURES = [
    "distance_m", "projection_fraction", "arc_length_m", "heading_delta_sin",
    "heading_delta_cos", "speed_mps", "time_gap_s", "distance_to_start_m",
    "distance_to_end_m", "from_out_degree", "to_in_degree", "local_displacement_m",
    "from_in_degree", "to_out_degree",
]


def trip_ids(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_trip(folder: Path, include_route: bool = False) -> dict:
    tid = folder.name
    track = np.loadtxt(folder / f"{tid}.track", dtype=float, ndmin=2)
    nodes = np.loadtxt(folder / f"{tid}.nodes", dtype=float, ndmin=2)
    arcs = np.loadtxt(folder / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected input dimensions in trip {tid}")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Non-finite coordinate or timestamp in trip {tid}")
    if np.any(np.abs(track[:, 0]) > 180) or np.any(np.abs(nodes[:, 0]) > 180):
        raise ValueError(f"Invalid longitude in trip {tid}")
    if np.any(np.abs(track[:, 1]) > 90) or np.any(np.abs(nodes[:, 1]) > 90):
        raise ValueError(f"Invalid latitude in trip {tid}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Out-of-range node ID in trip {tid}")
    # The supplied trips span more than one UTM zone and hemisphere. Select a
    # local UTM CRS per trip instead of forcing all coordinates into one zone.
    lon0=float(np.mean(track[:,0])); lat0=float(np.mean(track[:,1]))
    zone=max(1,min(60,int((lon0+180)//6)+1))
    epsg=(32600 if lat0>=0 else 32700)+zone
    crs=f"EPSG:{epsg}"
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    node_xy = np.column_stack(to_xy.transform(nodes[:, 0], nodes[:, 1]))
    gps_xy = np.column_stack(to_xy.transform(track[:, 0], track[:, 1]))
    if not np.isfinite(node_xy).all() or not np.isfinite(gps_xy).all():
        raise ValueError(f"Projection produced invalid metric coordinates for trip {tid} ({crs})")
    lengths = np.linalg.norm(node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]], axis=1)
    geometries = [LineString([node_xy[u], node_xy[v]]) for u, v in arcs]
    in_degree = np.bincount(arcs[:, 1], minlength=len(nodes))
    out_degree = np.bincount(arcs[:, 0], minlength=len(nodes))
    to_out_degree = np.bincount(arcs[:, 0], minlength=len(nodes))
    result = dict(trip_id=tid, track=track, gps_xy=gps_xy, nodes=nodes, metric_crs=crs,
                  node_xy=node_xy, arcs=arcs, geometries=geometries,
                  lengths=lengths, in_degree=in_degree, out_degree=out_degree,
                  to_out_degree=to_out_degree)
    if include_route:
        route_path = folder / f"{tid}.route"
        result["route"] = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
    return result


def _route_alignment(data: dict) -> np.ndarray:
    """Monotonic nearest-route-arc alignment using the supplied offline labels."""
    route = data["route"]
    if len(route) == 0 or np.any(route < 0) or np.any(route >= len(data["arcs"])):
        raise ValueError(f"Invalid reference route in trip {data['trip_id']}")
    geoms = [data["geometries"][int(i)] for i in route]
    gps = data["gps_xy"]
    # Shapely 2 vectorizes this route-by-observation distance matrix in native code.
    route_shapes = np.asarray(geoms, dtype=object)[:, None]
    observations = shapely.points(gps[:, 0], gps[:, 1])[None, :]
    cost = np.asarray(shapely.distance(route_shapes, observations).T, dtype=np.float32)
    # Prefix-min recurrence enforces chronological nondecreasing route progress.
    back = np.zeros(cost.shape, dtype=np.int32)
    prev = cost[0].copy()
    for t in range(1, len(gps)):
        prefix = np.minimum.accumulate(prev)
        # Stable prefix argmin for backtracking.
        best_idx = np.empty(len(prev), dtype=np.int32)
        running_idx = 0
        running_val = prev[0]
        for j in range(len(prev)):
            if prev[j] < running_val:
                running_idx, running_val = j, prev[j]
            best_idx[j] = running_idx
        back[t] = best_idx
        prev = cost[t] + prefix
    idx = np.empty(len(gps), dtype=np.int32)
    idx[-1] = int(np.argmin(prev))
    for t in range(len(gps) - 1, 0, -1):
        idx[t - 1] = back[t, idx[t]]
    return route[idx]


def _candidate_frame(data: dict, labeled: bool, radius_m: float = 60.0, max_candidates: int = 32) -> pd.DataFrame:
    geoms = data["geometries"]
    tree = STRtree(geoms)
    track, gps = data["track"], data["gps_xy"]
    truth = _route_alignment(data) if labeled else None
    rows = []
    prev_gps_heading = 0.0
    for obs, ((lon, lat, ts), (x, y)) in enumerate(zip(track, gps)):
        point = Point(x, y)
        if obs:
            dx, dy = gps[obs] - gps[obs - 1]
            dt = max(float(ts - track[obs - 1, 2]), 1e-3)
            speed = float(np.clip(np.hypot(dx, dy) / dt, 0.0, 60.0))
            gps_heading = float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 1e-8 else prev_gps_heading
            displacement = float(np.clip(np.hypot(dx, dy), 0.0, 1000.0))
        else:
            dt, speed, gps_heading, displacement = 1.0, 0.0, 0.0, 0.0
        prev_gps_heading = gps_heading
        found = np.asarray(tree.query(point.buffer(radius_m)), dtype=np.int64)
        if len(found) == 0:
            found = np.asarray(tree.query_nearest(point, all_matches=True), dtype=np.int64).reshape(-1)
        scored = []
        for arc_id in found:
            geom = geoms[int(arc_id)]
            dist = float(geom.distance(point))
            if dist <= radius_m or len(found) <= max_candidates:
                scored.append((dist, int(arc_id)))
        scored.sort()
        chosen = scored[:max_candidates]
        if not chosen:
            continue
        for dist, arc_id in chosen:
            u, v = data["arcs"][arc_id]
            geom = geoms[arc_id]
            fraction = float(np.clip(geom.project(point) / max(geom.length, 1e-6), 0, 1))
            a = data["node_xy"][u]
            b = data["node_xy"][v]
            angle = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            delta = angle - gps_heading
            rows.append({
                "trip_id": data["trip_id"], "observation_id": obs, "candidate_arc_id": arc_id,
                "label": int(arc_id == truth[obs]) if labeled else -1,
                "distance_m": dist, "projection_fraction": fraction,
                "arc_length_m": float(data["lengths"][arc_id]),
                "heading_delta_sin": float(np.sin(delta)), "heading_delta_cos": float(np.cos(delta)),
                "speed_mps": speed, "time_gap_s": float(np.clip(dt, 1.0, 120.0)),
                "distance_to_start_m": float(np.hypot(x-a[0], y-a[1])),
                "distance_to_end_m": float(np.hypot(x-b[0], y-b[1])),
                "from_out_degree": float(data["out_degree"][u]), "to_in_degree": float(data["in_degree"][v]),
                "local_displacement_m": displacement,
                "from_in_degree": float(data["in_degree"][u]), "to_out_degree": float(data["out_degree"][v]),
            })
    frame=pd.DataFrame(rows)
    if labeled and not frame.empty:
        # Preserve candidate-generation failures as explicit missing labels. They
        # are reported as misses and excluded from classifier fitting/scoring.
        positive_obs=set(frame.loc[frame.label==1,"observation_id"].unique())
        frame.loc[~frame.observation_id.isin(positive_obs),"label"]=-1
    return frame


def build_partition(dataset_root: Path, partition: str, output: Path) -> pd.DataFrame:
    root = dataset_root / partition
    dfs = []
    expected = 80 if partition == "train" else 10
    ids = trip_ids(root)
    if len(ids) != expected:
        raise ValueError(f"Expected {expected} trips in {root}; found {len(ids)}")
    for i, tid in enumerate(ids, 1):
        data = load_trip(root / tid, include_route=True)
        frame = _candidate_frame(data, labeled=True, max_candidates=8)
        dfs.append(frame)
        print(f"{partition}: {i}/{len(ids)} trip {tid}: {len(data['track'])} observations, {len(frame)} candidates")
    all_rows = pd.concat(dfs, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output, index=False, compression="gzip")
    print(f"Saved {len(all_rows)} candidate rows to {output}")
    return all_rows


def create_map(dataset_root: Path, trip_id: str, partition: str, output: Path) -> None:
    data = load_trip(dataset_root / partition / trip_id, include_route=True)
    center = [float(np.mean(data["track"][:, 1])), float(np.mean(data["track"][:, 0]))]
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    network = folium.FeatureGroup("Road network", show=False)
    for u, v in data["arcs"]:
        folium.PolyLine([[data["nodes"][u, 1], data["nodes"][u, 0]],
                         [data["nodes"][v, 1], data["nodes"][v, 0]]],
                        color="#999999", weight=1, opacity=0.45).add_to(network)
    network.add_to(m)
    gps = folium.FeatureGroup("GPS observations", show=True)
    gps_points = [[float(lat), float(lon)] for lon, lat, _ in data["track"]]
    folium.PolyLine(gps_points, color="blue", weight=3, tooltip="Raw GPS").add_to(gps)
    for loc in gps_points:
        folium.CircleMarker(loc, radius=2, color="blue", fill=True).add_to(gps)
    gps.add_to(m)
    route = folium.FeatureGroup("Reference route (offline development only)", show=True)
    for arc_id in data["route"]:
        u, v = data["arcs"][int(arc_id)]
        folium.PolyLine([[data["nodes"][u, 1], data["nodes"][u, 0]],
                         [data["nodes"][v, 1], data["nodes"][v, 0]]],
                        color="green", weight=4).add_to(route)
    route.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(gps_points)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f"Saved offline {partition} map for {trip_id} to {output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("map")
    m.add_argument("--dataset-root", type=Path, required=True)
    m.add_argument("--partition", choices=["train", "student_test"], required=True)
    m.add_argument("--trip-id", required=True)
    m.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "build":
        build_partition(args.dataset_root, "train", args.output / "train_candidates.csv.gz")
        build_partition(args.dataset_root, "student_test", args.output / "student_test_candidates.csv.gz")
    else:
        create_map(args.dataset_root, args.trip_id, args.partition, args.output)


if __name__ == "__main__":
    main()
