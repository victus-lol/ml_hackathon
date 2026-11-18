"""
generate_all_trips_preprocessing_report.py
------------------------------------------
Generates the authoritative Trip-Level Preprocessing & Data Cleaning Report
across all 90 trips (80 training trips + 10 test trips).
Format contains:
- trip_id
- partition
- raw_gps_points
- missing_values (0s)
- duplicate_timestamps (0s)
- dropped_speed_spikes (mostly 0s, with trips having numbers above 0)
- total_dropped_records (a lot of 0s, and dropped ones with numbers above 0)
- clean_gps_points
- retention_rate_pct
Outputs:
- outputs/preprocessing_report.csv
- CONTEXT FILES/1 - AKSHAY/preprocessing_report.csv
- docs/PathSnap_Official_Evaluation_Report.xlsx (Sheet: Preprocessing_Report)
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = REPO_ROOT / "outputs"
AKSHAY_DIR = REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY"
DOCS_DIR = REPO_ROOT / "docs"

TRAIN_DIR = Path("c:/Users/adity/Downloads/ML HACKATHON/Student folder/Student folder/train")
TEST_DIR = Path("c:/Users/adity/Downloads/ML HACKATHON/Student folder/Student folder/student_test")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def audit_trip(trip_path, trip_id, partition):
    track_file = trip_path / f"{trip_id}.track"
    if not track_file.exists():
        return None
    
    df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    raw_count = len(df)
    
    # 1. Missing values
    null_count = int(df.isnull().sum().sum())
    
    # 2. Duplicate / Non-monotonic dt
    dt = df["t"].diff()
    dup_dt = int((dt <= 0).sum())
    
    # 3. Kinematic Speed Spikes (> 45 m/s = 162 km/h)
    lat = df["lat"].values
    lon = df["lon"].values
    t = df["t"].values
    
    spike_count = 0
    valid_idx = [0]
    for i in range(1, raw_count):
        prev = valid_idx[-1]
        dt_s = t[i] - t[prev]
        if dt_s <= 0:
            continue
        d = haversine(lat[prev], lon[prev], lat[i], lon[i])
        v = d / dt_s
        if v > 45.0:
            spike_count += 1
            continue
        valid_idx.append(i)
        
    total_dropped = dup_dt + spike_count + null_count
    clean_count = raw_count - total_dropped
    retention_pct = round((clean_count / raw_count) * 100, 2)
    
    return {
        "trip_id": trip_id,
        "partition": partition,
        "raw_gps_points": raw_count,
        "missing_values": null_count,
        "duplicate_timestamps": dup_dt,
        "dropped_speed_spikes": spike_count,
        "total_dropped_records": total_dropped,
        "clean_gps_points": clean_count,
        "retention_rate_pct": retention_pct
    }

def main():
    print("=" * 70)
    print("  SCANNING ALL 90 TRIPS FOR DATASET PREPROCESSING & CLEANING REPORT")
    print("=" * 70)

    results = []
    # 1. Student Test (10 trips)
    for d in sorted(TEST_DIR.iterdir()):
        if d.is_dir():
            r = audit_trip(d, d.name, "student_test")
            if r: results.append(r)

    # 2. Student Train (80 trips)
    for d in sorted(TRAIN_DIR.iterdir()):
        if d.is_dir():
            r = audit_trip(d, d.name, "student_train")
            if r: results.append(r)

    df = pd.DataFrame(results)
    print(f"Audited {len(df)} total trips ({len(df[df['partition']=='student_train'])} train, {len(df[df['partition']=='student_test'])} test).")
    print(f"Trips with total_dropped_records == 0: {(df['total_dropped_records'] == 0).sum()}")
    print(f"Trips with total_dropped_records > 0 : {(df['total_dropped_records'] > 0).sum()}")

    # Save CSVs
    csv_out1 = OUTPUTS_DIR / "preprocessing_report.csv"
    csv_out2 = AKSHAY_DIR / "preprocessing_report.csv"
    df.to_csv(csv_out1, index=False)
    df.to_csv(csv_out2, index=False)
    print(f"Saved: {csv_out1}")
    print(f"Saved: {csv_out2}")

    # Update Excel Report with a dedicated Preprocessing_Report tab
    excel_path = DOCS_DIR / "PathSnap_Official_Evaluation_Report.xlsx"
    if excel_path.exists():
        wb = openpyxl.load_workbook(excel_path)
        sheet_name = "Preprocessing_Report"
        if sheet_name in wb.sheetnames:
            wb.remove(wb[sheet_name])
        
        ws = wb.create_sheet(title=sheet_name, index=1)
        ws.views.sheetView[0].showGridLines = True

        # Styles
        navy_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        blue_header_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
        zebra_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
        orange_highlight = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

        font_title = Font(name="Calibri", size=15, bold=True, color="FFFFFF")
        font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        font_normal = Font(name="Calibri", size=10)
        font_bold = Font(name="Calibri", size=10, bold=True)

        align_center = Alignment(horizontal="center", vertical="center")
        align_left = Alignment(horizontal="left", vertical="center")
        align_right = Alignment(horizontal="right", vertical="center")

        thin_border = Border(
            left=Side(style='thin', color="D9D9D9"), right=Side(style='thin', color="D9D9D9"),
            top=Side(style='thin', color="D9D9D9"), bottom=Side(style='thin', color="D9D9D9")
        )

        # Title
        ws.merge_cells("A1:I2")
        t = ws["A1"]
        t.value = "PathSnap: Preprocessing & Data Cleaning Audit across All 90 Trips (80 Train + 10 Test)"
        t.font = font_title
        t.fill = navy_fill
        t.alignment = align_center

        headers = [
            "Trip ID", "Dataset Partition", "Raw GPS Points", "Missing Values",
            "Duplicate Timestamps", "Dropped Speed Spikes", "Total Dropped Records",
            "Clean GPS Points", "Retention Rate %"
        ]

        # Write Headers
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=4, column=col_idx, value=h)
            cell.font = font_header
            cell.fill = blue_header_fill
            cell.alignment = align_center

        # Write Data Rows
        for row_idx, r in enumerate(df.itertuples(index=False), 5):
            is_zebra = (row_idx % 2 == 0)
            has_drops = (r.total_dropped_records > 0)

            ws.cell(row=row_idx, column=1, value=str(r.trip_id)).alignment = align_center
            ws.cell(row=row_idx, column=2, value=str(r.partition)).alignment = align_center
            ws.cell(row=row_idx, column=3, value=int(r.raw_gps_points)).alignment = align_right
            ws.cell(row=row_idx, column=4, value=int(r.missing_values)).alignment = align_center
            ws.cell(row=row_idx, column=5, value=int(r.duplicate_timestamps)).alignment = align_center
            ws.cell(row=row_idx, column=6, value=int(r.dropped_speed_spikes)).alignment = align_center
            ws.cell(row=row_idx, column=7, value=int(r.total_dropped_records)).alignment = align_center
            ws.cell(row=row_idx, column=8, value=int(r.clean_gps_points)).alignment = align_right
            ws.cell(row=row_idx, column=9, value=f"{r.retention_rate_pct:.2f}%").alignment = align_center

            for c_idx in range(1, 10):
                c = ws.cell(row=row_idx, column=c_idx)
                c.font = font_normal
                c.border = thin_border
                if has_drops and c_idx in [6, 7]:
                    c.fill = orange_highlight
                    c.font = font_bold
                elif is_zebra:
                    c.fill = zebra_fill

        # Auto width
        for col_idx in range(1, 10):
            col_letter = get_column_letter(col_idx)
            max_len = max(len(str(ws.cell(row=r, column=col_idx).value or '')) for r in range(4, 5 + len(df)))
            ws.column_dimensions[col_letter].width = max(max_len + 3, 14)

        wb.save(excel_path)
        wb.save(OUTPUTS_DIR / "PathSnap_Official_Evaluation_Report.xlsx")
        print(f"Updated Excel Report: {excel_path}")

if __name__ == "__main__":
    main()
