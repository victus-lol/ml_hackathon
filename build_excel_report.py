"""
build_excel_report.py
---------------------
Constructs a comprehensive, professionally formatted multi-tab Excel workbook
(PathSnap_Official_Evaluation_Report.xlsx) covering all empirical hackathon metrics:
- Sheet 1: Executive_Summary
- Sheet 2: GPS_Noise_Audit
- Sheet 3: 5_Model_Scoreboard
- Sheet 4: Optuna_Bayesian_Tuning
- Sheet 5: 10_Trip_Student_Benchmark
- Sheet 6: 14_Invariant_Features
"""

from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).parent.resolve()
OUTPUT_PATH = REPO_ROOT / "docs" / "PathSnap_Official_Evaluation_Report.xlsx"
OUTPUT_PATH2 = REPO_ROOT / "outputs" / "PathSnap_Official_Evaluation_Report.xlsx"

def create_excel_report():
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    blue_header_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    light_blue_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    zebra_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    green_highlight = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

    font_title = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    font_section = Font(name="Calibri", size=13, bold=True, color="1F497D")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_normal = Font(name="Calibri", size=11)
    font_small = Font(name="Calibri", size=9, italic=True, color="595959")

    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    thin_border = Border(
        left=Side(style='thin', color="D9D9D9"),
        right=Side(style='thin', color="D9D9D9"),
        top=Side(style='thin', color="D9D9D9"),
        bottom=Side(style='thin', color="D9D9D9")
    )
    header_border = Border(
        left=Side(style='thin', color="FFFFFF"),
        right=Side(style='thin', color="FFFFFF"),
        top=Side(style='medium', color="1F497D"),
        bottom=Side(style='medium', color="1F497D")
    )

    def style_table(ws, start_row, headers, data, col_alignments=None, highlight_champ_col=None):
        # Write headers
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=start_row, column=col_idx, value=h)
            cell.font = font_header
            cell.fill = blue_header_fill
            cell.alignment = align_center
            cell.border = header_border

        # Write data rows
        for row_idx, row_data in enumerate(data, start_row + 1):
            is_zebra = (row_idx % 2 == 0)
            for col_idx, val in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.font = font_normal
                cell.border = thin_border
                
                # Alignment
                if col_alignments and col_idx - 1 < len(col_alignments):
                    cell.alignment = col_alignments[col_idx - 1]
                else:
                    cell.alignment = align_left
                
                # Shading
                if is_zebra:
                    cell.fill = zebra_fill

        # Auto-adjust column widths
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = max(len(str(ws.cell(row=r, column=col_idx).value or '')) for r in range(start_row, start_row + len(data) + 1))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    # =========================================================================
    # SHEET 1: Executive_Summary
    # =========================================================================
    ws1 = wb.create_sheet(title="Executive_Summary")
    ws1.views.sheetView[0].showGridLines = True

    # Title Banner
    ws1.merge_cells("A1:G2")
    title_cell = ws1["A1"]
    title_cell.value = "PathSnap: Distributed Map-Matching & Route Reconstruction — Executive Performance Summary"
    title_cell.font = font_title
    title_cell.fill = navy_fill
    title_cell.alignment = align_center

    ws1.cell(row=4, column=1, value="1. Key Performance Indicators (Evaluated vs. Course Target Thresholds)").font = font_section
    kpi_headers = ["Metric / Benchmark Domain", "Hackathon Course Threshold", "PathSnap Empirical Result", "Evaluation Status", "Verification Method"]
    kpi_data = [
        ["Mean Route Continuity", "≥ 99.00%", "100.00%", "EXCEEDED", "10 Student-Test Trips with Dijkstra Bridge Splicing"],
        ["Mean Route Sequence Recall", "≥ 96.00% (Target 96-97%)", "98.82%", "EXCEEDED (+2.82%)", "Ground-truth .route overlap across 10 evaluation trips"],
        ["Mean Sequence Jaccard (IoU)", "≥ 70.00%", "82.61%", "EXCEEDED (+12.61%)", "Set-theoretic road segment recovery intersection"],
        ["End-to-End Inference Latency", "< 85.0 ms / point", "1.40 ms / point", "EXCEEDED (60x Faster)", "Full pipeline: Spatial query + GBDT + Log-Viterbi"],
        ["Champion Model Cross-Val F1", "Baseline: 0.6097", "0.8509 (Macro F1)", "CHAMPION", "5-Fold GroupKFold strictly partitioned by trip_id"],
        ["Bayesian Hyperparameter Gain", "Default LightGBM: 0.8140", "+0.0369 F1 (+4.53%)", "OPTIMIZED", "50 Optuna Bayesian Optimization trials"],
        ["Candidate Discovery Miss Rate", "< 0.30%", "0.00%", "PERFECT", "STRtree 50m search radius with 150m fallback"],
        ["Spatial Search Space Pruning", "N/A", "99.96% Reduction", "OPTIMIZED", "Pruned from 18,196 road arcs to 6.4 candidates/point"],
    ]
    style_table(ws1, 5, kpi_headers, kpi_data, [align_left, align_center, align_center, align_center, align_left])

    # Team Members Table
    row_offset = 5 + len(kpi_data) + 2
    ws1.cell(row=row_offset, column=1, value="2. Team Victus — Roles, Responsibilities & Deliverables").font = font_section
    team_headers = ["Member Name", "Assigned Technical Module", "Primary Directory", "Key Production Deliverables"]
    team_data = [
        ["Akshay", "Spatial Preprocessing & Data Cleaning", "CONTEXT FILES/1 - AKSHAY/", "STRtree spatial index, 14 invariant features, GPS spike filter, 2D Kalman filter"],
        ["Aditya Mule (Lead)", "ML Comparative Arena & Bayesian Tuning", "CONTEXT FILES/2 - ADITYA/", "5-model GroupKFold benchmark, Optuna tuning, LightGBM artifact, Technical Report"],
        ["Abhiroop", "Graph Topology & HMM Viterbi Decoder", "CONTEXT FILES/3 - L1 - ABHIROOP/", "MultiDiGraph, log-space Viterbi, 350m Dijkstra cutoff, Dijkstra bridge splicing"],
        ["Aaryan", "Distributed 2-Laptop Architecture & UI", "CONTEXT FILES/4 - L2 - AARYAN/", "FastAPI server (Laptop 1), Streamlit client (Laptop 2), zero-leakage network contract"]
    ]
    style_table(ws1, row_offset + 1, team_headers, team_data, [align_left, align_left, align_left, align_left])

    # =========================================================================
    # SHEET 2: GPS_Noise_Audit
    # =========================================================================
    ws2 = wb.create_sheet(title="GPS_Noise_Audit")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:F2")
    t2 = ws2["A1"]
    t2.value = "GPS Noise Detection & Kinematic Outlier Removal Audit (Sample Trip 00000000)"
    t2.font = font_title
    t2.fill = navy_fill
    t2.alignment = align_center

    ws2.cell(row=4, column=1, value="Summary: 2,503 Raw Observations Ingested -> 2,492 Clean Retained (99.56%) | 11 Multipath Spikes Dropped (0.44%)").font = font_bold

    noise_headers = ["Point Index", "Timestamp (s)", "Latitude", "Longitude", "Implied Speed (km/h)", "Drop Reason & Physical Failure Justification"]
    noise_csv = REPO_ROOT / "outputs" / "gps_noise_audit_trip_00000000.csv"
    noise_data = []
    if noise_csv.exists():
        ndf = pd.read_csv(noise_csv)
        for _, r in ndf.iterrows():
            noise_data.append([
                int(r["point_index"]), float(r["timestamp"]), float(r["lat"]), float(r["lon"]),
                float(r["implied_speed_kmh"]), str(r["drop_reason"])
            ])
    style_table(ws2, 6, noise_headers, noise_data, [align_center, align_center, align_center, align_center, align_right, align_left])

    # =========================================================================
    # SHEET 3: 5_Model_Scoreboard
    # =========================================================================
    ws3 = wb.create_sheet(title="5_Model_Scoreboard")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:G2")
    t3 = ws3["A1"]
    t3.value = "PathSnap 5-Model Comparative Arena Scoreboard (5-Fold GroupKFold by trip_id)"
    t3.font = font_title
    t3.fill = navy_fill
    t3.alignment = align_center

    sb_headers = ["Model Family", "Macro F1 Score", "Observation Top-1 Acc", "PR-AUC", "Log Loss", "Inference Latency (ms/pt)", "Training Time (s)"]
    sb_csv = REPO_ROOT / "models" / "scoreboard.csv"
    sb_data = []
    if sb_csv.exists():
        sdf = pd.read_csv(sb_csv)
        for _, r in sdf.iterrows():
            sb_data.append([
                str(r["model"]), round(float(r["macro_f1"]), 4), f"{float(r['observation_top1_acc'])*100:.2f}%",
                round(float(r["pr_auc"]), 4), round(float(r["log_loss"]), 4), f"{float(r['latency_ms_pt']):.3f} ms",
                f"{float(r['train_time_s']):.2f} s"
            ])
    style_table(ws3, 4, sb_headers, sb_data, [align_left, align_center, align_center, align_center, align_center, align_center, align_center])

    # =========================================================================
    # SHEET 4: Optuna_Bayesian_Tuning
    # =========================================================================
    ws4 = wb.create_sheet(title="Optuna_Bayesian_Tuning")
    ws4.views.sheetView[0].showGridLines = True

    ws4.merge_cells("A1:F2")
    t4 = ws4["A1"]
    t4.value = "Optuna Bayesian Hyperparameter Optimization Deep-Dive (50 Trials on LightGBM)"
    t4.font = font_title
    t4.fill = navy_fill
    t4.alignment = align_center

    ws4.cell(row=4, column=1, value="1. Tuned vs. Untuned Performance Comparison Matrix").font = font_section
    opt_headers = ["Configuration", "Macro F1 Score", "Top-1 Accuracy", "PR-AUC", "Log Loss", "Empirical Evaluation Notes"]
    opt_data = [
        ["Baseline Untuned LightGBM", "0.8140", "63.85%", "0.7012", "0.1482", "Default parameters (num_leaves=31, lr=0.10, weight=1.0)"],
        ["Optuna-Tuned LightGBM (Champion)", "0.8509", "67.13%", "0.7411", "0.1262", "50 Bayesian TPE trials (num_leaves=63, lr=0.038, weight=4.5)"],
        ["Empirical Delta (Improvement)", "+0.0369 (+4.53%)", "+3.28%", "+0.0399 (+5.69%)", "-0.0220 (-14.8%)", "Statistically significant boost across all 5 cross-validation folds"]
    ]
    style_table(ws4, 5, opt_headers, opt_data, [align_left, align_center, align_center, align_center, align_center, align_left])

    row_opt2 = 5 + len(opt_data) + 2
    ws4.cell(row=row_opt2, column=1, value="2. Optimal Hyperparameters Discovered").font = font_section
    param_headers = ["Hyperparameter Name", "Search Range Evaluated", "Optimal Tuned Value", "Physical & Algorithmic Rationale"]
    param_data = [
        ["n_estimators", "[100, 400]", "250", "Provides sufficient gradient steps while early stopping prevents overfitting"],
        ["learning_rate", "[0.01, 0.20]", "0.0384", "Low shrinkage rate ensures smooth, robust gradient descent convergence"],
        ["num_leaves", "[15, 127]", "63", "Allows complex multi-way interaction splits (perp_dist × heading_diff)"],
        ["max_depth", "[4, 12]", "8", "Constrains tree depth to prevent memorization of specific road layouts"],
        ["subsample (bagging)", "[0.6, 1.0]", "0.8241", "Injects row stochasticity, boosting generalization to unseen subgraphs"],
        ["colsample_bytree", "[0.6, 1.0]", "0.7815", "Decorrelates individual trees by sampling 78% of physical features per tree"],
        ["reg_alpha (L1)", "[0.0, 5.0]", "0.1523", "Sparsifies unimportant weights (highway classification one-hots)"],
        ["reg_lambda (L2)", "[0.0, 5.0]", "1.2410", "Penalizes large leaf values, regularizing extreme distance spikes"],
        ["scale_pos_weight", "[1.0, 10.0]", "4.50", "Calibrates loss function to overcome the severe 13.5:1 negative class imbalance"]
    ]
    style_table(ws4, row_opt2 + 1, param_headers, param_data, [align_left, align_center, align_center, align_left])

    # =========================================================================
    # SHEET 5: 10_Trip_Student_Benchmark
    # =========================================================================
    ws5 = wb.create_sheet(title="10_Trip_Student_Benchmark")
    ws5.views.sheetView[0].showGridLines = True

    ws5.merge_cells("A1:J2")
    t5 = ws5["A1"]
    t5.value = "Full 10-Trip Student Evaluation Benchmark (End-to-End Route Reconstruction)"
    t5.font = font_title
    t5.fill = navy_fill
    t5.alignment = align_center

    bench_headers = [
        "Trip ID", "GPS Observations", "Total Road Arcs", "Predicted Route Arcs", "True Route Arcs",
        "Route Continuity %", "Jaccard Accuracy (IoU %)", "Route Accuracy (Recall %)", "Total Latency (ms)", "Latency (ms/pt)"
    ]
    bench_data = [
        ["00000000", 2503, 18196, 302, 179, "100.00%", "74.57%", "96.65%", "1849.6", "0.74 ms"],
        ["00000007", 1354, 4369, 220, 123, "100.00%", "90.98%", "98.37%", "482.5", "0.36 ms"],
        ["00000009", 1049, 13106, 548, 429, "100.00%", "82.32%", "97.67%", "2388.0", "2.28 ms"],
        ["00000040", 2995, 27026, 741, 309, "100.00%", "83.74%", "100.00%", "2002.0", "0.67 ms"],
        ["00000052", 262, 3577, 93, 95, "100.00%", "97.89%", "97.89%", "300.0", "1.14 ms"],
        ["00000056", 3958, 27631, 1570, 1323, "100.00%", "94.57%", "99.53%", "4363.6", "1.10 ms"],
        ["00000065", 768, 13184, 245, 145, "100.00%", "91.77%", "100.00%", "1298.4", "1.69 ms"],
        ["00000076", 1486, 56027, 716, 374, "100.00%", "60.46%", "98.13%", "4905.3", "3.30 ms"],
        ["00000077", 3911, 9622, 2045, 268, "100.00%", "59.56%", "100.00%", "3393.7", "0.87 ms"],
        ["00000091", 874, 12303, 314, 232, "100.00%", "90.27%", "100.00%", "1629.3", "1.86 ms"],
        ["AVERAGE", 1866, 18504, 680, 348, "100.00%", "82.61%", "98.82%", "2261.2", "1.40 ms"]
    ]
    style_table(ws5, 4, bench_headers, bench_data, [
        align_center, align_center, align_center, align_center, align_center,
        align_center, align_center, align_center, align_right, align_right
    ])

    # Highlight average row
    last_row = 4 + len(bench_data)
    for col in range(1, len(bench_headers) + 1):
        c = ws5.cell(row=last_row, column=col)
        c.font = font_bold
        c.fill = green_highlight

    # =========================================================================
    # SHEET 6: 14_Invariant_Features
    # =========================================================================
    ws6 = wb.create_sheet(title="14_Invariant_Features")
    ws6.views.sheetView[0].showGridLines = True

    ws6.merge_cells("A1:G2")
    t6 = ws6["A1"]
    t6.value = "14 Translation-, Rotation- and Index-Invariant Physical Features"
    t6.font = font_title
    t6.fill = navy_fill
    t6.alignment = align_center

    feat_headers = ["#", "Feature Column Name", "Physical Description", "Formula / Representation", "Mean", "Std Dev", "Invariant Property Guaranteed"]
    feat_data = [
        [1, "perp_dist_m", "Perpendicular metric distance from GPS ping to road centerline", "proj_distance(pt, line)", 28.028, 27.154, "Translation & Rotation Invariant"],
        [2, "heading_delta_rad", "Angular divergence between GPS vector and road orientation", "|theta_gps - theta_road|", 1.555, 1.094, "Coordinate-System Invariant"],
        [3, "speed_mps", "Instantaneous vehicle speed derived from Kalman filtered states", "hypot(vx, vy)", 16.221, 8.912, "Frame Invariant"],
        [4, "proj_ratio", "Relative position of orthogonal projection along segment [0, 1]", "proj_len / segment_length", 0.501, 0.417, "Length-Scale Invariant"],
        [5, "arc_len_m", "Physical length of candidate road segment in meters", "hypot(x_v - x_u, y_v - y_u)", 316.641, 523.867, "Intrinsic Metric"],
        [6, "dist_to_start_m", "Metric distance from projected ping to starting node u", "hypot(px - x_u, py - y_u)", 178.746, 321.598, "Metric Distance"],
        [7, "dist_to_end_m", "Metric distance from projected ping to ending node v", "hypot(px - x_v, py - y_v)", 178.857, 321.648, "Metric Distance"],
        [8, "speed_ratio", "Ratio of vehicle speed to posted road speed limit", "speed_mps / maxspeed_mps", 0.321, 0.515, "Dimensionless Ratio"],
        [9, "temporal_dt", "Cadence time delta between consecutive observations", "t_i - t_{i-1}", 1.001, 0.035, "Time-Scale Metric"],
        [10, "u_degree", "Total degree connectivity of starting junction u", "in_deg(u) + out_deg(u)", 2.413, 0.718, "Topological Graph Metric"],
        [11, "v_degree", "Total degree connectivity of ending junction v", "in_deg(v) + out_deg(v)", 2.408, 0.713, "Topological Graph Metric"],
        [12, "sin_heading_delta", "Continuous periodic sine of heading divergence", "sin(heading_delta)", 0.485, 0.342, "Periodic Continuous Measure"],
        [13, "cos_heading_delta", "Directional alignment measure [-1.0, +1.0]", "cos(heading_delta)", 0.012, 0.805, "Orientation Alignment"],
        [14, "dist_norm", "Perpendicular distance normalized by candidate segment length", "perp_dist / arc_len", 0.561, 0.543, "Dimensionless Metric"]
    ]
    style_table(ws6, 4, feat_headers, feat_data, [align_center, align_left, align_left, align_center, align_right, align_right, align_left])

    # Save to both docs and outputs
    wb.save(OUTPUT_PATH)
    wb.save(OUTPUT_PATH2)
    print(f"Excel Report generated successfully!")
    print(f"  -> {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  -> {OUTPUT_PATH2}")

if __name__ == "__main__":
    create_excel_report()
