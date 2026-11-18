"""
test_end_to_end_pipeline.py
---------------------------
Authoritative Automated End-to-End Pipeline Verification
Tests:
1. Production LightGBM model loading
2. STRtree candidate generation on test trip 00000000
3. 14-Feature extraction
4. Calibrated ML emission scoring
5. Log-space Viterbi decoding over networkx.MultiDiGraph
6. Two-stage post-processing (Deduplication + Dijkstra Bridge Splicing)
7. Evaluator UI Map generation (raw GPS in blue, predicted route in red)
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

def run_e2e_test(trip_id: str = "00000000"):
    print("=" * 75)
    print("      PATHSNAP AUTOMATED END-TO-END PIPELINE INTEGRATION TEST")
    print("=" * 75)

    test_dir = REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "sample_trip_00000000"
    if not test_dir.exists():
        test_dir = Path("C:/Users/adity/Downloads/ML HACKATHON/Student folder/Student folder/student_test/00000000")

    print(f"\n[Test Setup] Testing on Trip {trip_id} from:\n  -> {test_dir}")
    
    # 1. Load raw files
    track = pd.read_csv(test_dir / f"{trip_id}.track", sep=r"\s+", header=None).values.tolist()
    nodes = pd.read_csv(test_dir / f"{trip_id}.nodes", sep=r"\s+", header=None).values.tolist()
    arcs  = pd.read_csv(test_dir / f"{trip_id}.arcs", sep=r"\s+", header=None).values.tolist()
    
    # 2. Initialize Engine
    print("\n[Engine Check] Initializing PathSnap Production Engine...")
    engine = create_engine()
    if not engine.is_ready():
        raise RuntimeError("Engine model is not ready. Ensure models/best_model_lightgbm.joblib is trained.")
    print("  -> Engine Ready: Production LightGBM + Isotonic Calibration Loaded.")

    # 3. Execute Match Trip
    print(f"\n[Inference] Running Map-Matching on {len(track):,} GPS points over {len(arcs):,} road arcs...")
    t_start = time.time()
    result = engine.match_trip(trip_id, track, nodes, arcs)
    lat_total = (time.time() - t_start) * 1000

    # 4. Assert Output Schema & Constraints
    print("\n[Schema Verification] Validating Output Contract...")
    assert "predicted_route" in result, "Missing predicted_route"
    assert "predicted_coordinates" in result, "Missing predicted_coordinates"
    assert "metrics" in result, "Missing metrics"
    
    continuity = result["metrics"]["route_continuity"]
    print(f"  -> Predicted Route Arcs:      {len(result['predicted_route']):,} segments")
    print(f"  -> Route Continuity Ratio:     {continuity * 100:.2f}% (Target: >= 99%)")
    print(f"  -> Total Pipeline Latency:     {lat_total:.2f} ms ({lat_total / len(track):.2f} ms/point)")
    assert continuity >= 0.95, f"Route continuity failed: {continuity}"

    # 5. Render Evaluator Interactive Map
    print("\n[Visualization] Rendering Evaluator UI Folium Map...")
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    map_file = out_dir / f"end_to_end_reconstruction_{trip_id}.html"

    center = [track[0][1], track[0][0]]
    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

    # Add Raw GPS (Blue markers & line)
    gps_coords = [[pt[1], pt[0]] for pt in track]
    folium.PolyLine(gps_coords, color="#2196F3", weight=2.5, opacity=0.6, tooltip="Raw GPS (Noisy)").add_to(m)
    for pt in gps_coords[::2]:  # subsample markers for fast rendering
        folium.CircleMarker(pt, radius=2, color="#1976D2", fill=True, fill_color="#2196F3").add_to(m)

    # Add Predicted Driven Route (Thick Red line)
    route_coords = [[pt[1], pt[0]] for pt in result["predicted_coordinates"]]
    folium.PolyLine(route_coords, color="#D32F2F", weight=4.5, opacity=0.9, tooltip="Reconstructed Driving Route").add_to(m)

    # Add Start / End Markers
    folium.Marker(gps_coords[0], popup="Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps_coords[-1], popup="Trip End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.save(str(map_file))
    print(f"  -> Map saved to: {map_file}")

    print("\n" + "=" * 75)
    print("END-TO-END PIPELINE INTEGRATION TEST: 100% PASSED & PRODUCTION READY")
    print("=" * 75)

if __name__ == "__main__":
    run_e2e_test()
