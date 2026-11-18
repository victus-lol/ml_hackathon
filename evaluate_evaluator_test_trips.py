"""
evaluate_evaluator_test_trips.py
---------------------------------
Comprehensive Inference & Benchmark on the Brand-New Evaluator Test Dataset
Evaluates:
- Route Continuity Ratio (%)
- Route Jaccard Accuracy & True Arc Recall (%)
- Latency per Observation (ms/point)
- Generates interactive Folium maps for all 10 evaluator trips
"""

import os
import sys
import time
import folium
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN"))
sys.path.insert(0, str(REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP"))

from pathsnap_engine import create_engine

def main():
    print("\n" + "=" * 80)
    print("      PATHSNAP: EVALUATING BRAND-NEW EVALUATOR TEST DATASET")
    print("      Model: Champion LightGBM GBDT + HMM Viterbi Decoder")
    print("=" * 80)

    test_root = REPO_ROOT / "evaluator_test"
    if not test_root.exists():
        raise FileNotFoundError(f"evaluator_test directory not found at: {test_root}")

    trip_ids = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    print(f"Found {len(trip_ids)} Evaluator Test trips in:\n  -> {test_root}\n")

    engine = create_engine()
    if not engine.is_ready():
        raise RuntimeError("Production engine model is not loaded. Train models first.")

    maps_dir = REPO_ROOT / "outputs" / "evaluator_test_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = []

    t_all_start = time.perf_counter()
    for idx, trip_id in enumerate(trip_ids, 1):
        trip_dir = test_root / trip_id
        track = pd.read_csv(trip_dir / f"{trip_id}.track", sep=r"\s+", header=None).values.tolist()
        nodes = pd.read_csv(trip_dir / f"{trip_id}.nodes", sep=r"\s+", header=None).values.tolist()
        arcs  = pd.read_csv(trip_dir / f"{trip_id}.arcs", sep=r"\s+", header=None).values.tolist()

        route_file = trip_dir / f"{trip_id}.route"
        true_route = []
        if route_file.exists():
            true_route = list(pd.read_csv(route_file, header=None)[0].values)

        t_start = time.perf_counter()
        result = engine.match_trip(trip_id, track, nodes, arcs)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        pred_route = result["predicted_route"]
        continuity = result["metrics"]["route_continuity"]

        # Jaccard & Recall
        pred_set = set(pred_route)
        true_set = set(true_route)
        
        jaccard = len(pred_set.intersection(true_set)) / len(pred_set.union(true_set)) if true_set else 0.0
        recall = len(pred_set.intersection(true_set)) / len(true_set) if true_set else 0.0

        benchmark_rows.append({
            "trip_id": trip_id,
            "gps_points": len(track),
            "road_arcs": len(arcs),
            "pred_route_len": len(pred_route),
            "true_route_len": len(true_route),
            "route_continuity_pct": round(continuity * 100, 2),
            "jaccard_accuracy_pct": round(jaccard * 100, 2),
            "route_recall_pct": round(recall * 100, 2),
            "latency_ms": round(latency_ms, 1),
            "latency_ms_per_pt": round(latency_ms / len(track), 2)
        })

        # Save interactive Folium map
        m = folium.Map(location=[track[0][1], track[0][0]], zoom_start=15, tiles="CartoDB positron")
        gps_coords = [[pt[1], pt[0]] for pt in track]
        folium.PolyLine(gps_coords, color="#2196F3", weight=2.5, opacity=0.6, tooltip="Raw GPS").add_to(m)
        route_coords = [[pt[1], pt[0]] for pt in result["predicted_coordinates"]]
        folium.PolyLine(route_coords, color="#D32F2F", weight=4.5, opacity=0.9, tooltip="Predicted Route").add_to(m)
        m.save(str(maps_dir / f"trip_{trip_id}_prediction.html"))

        print(f"[{idx:02d}/10] Trip {trip_id} | Continuity: {continuity*100:.1f}% | Recall: {recall*100:.1f}% | Latency: {latency_ms/len(track):.2f} ms/pt")

    total_time = time.perf_counter() - t_all_start
    df_bench = pd.DataFrame(benchmark_rows)
    out_table = REPO_ROOT / "outputs" / "evaluator_test_benchmark_table.md"
    df_bench.to_markdown(out_table, index=False)

    print("\n" + "=" * 80)
    print(f"           EVALUATOR TEST BENCHMARK COMPLETE (Total Time: {total_time:.1f}s)!")
    print("=" * 80)
    print(f"Mean Route Continuity : {df_bench['route_continuity_pct'].mean():.2f}%")
    print(f"Mean Route Recall     : {df_bench['route_recall_pct'].mean():.2f}%")
    print(f"Mean Sequence Jaccard : {df_bench['jaccard_accuracy_pct'].mean():.2f}%")
    print(f"Mean Latency          : {df_bench['latency_ms_per_pt'].mean():.2f} ms / point")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
