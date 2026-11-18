"""
package_teammate_zips.py
------------------------
Packages standalone, self-contained ZIP bundles for each teammate
and an all-in-one master offline archive.
"""

import os
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = REPO_ROOT / "OFFLINE_PACKAGES"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_zip(zip_path: Path, files_to_include: list[tuple[Path, str]]):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_name in files_to_include:
            if not src_path.exists():
                print(f"Warning: {src_path} does not exist, skipping.")
                continue
            if src_path.is_file():
                zf.write(src_path, arcname=arc_name)
            elif src_path.is_dir():
                for root, _, files in os.walk(src_path):
                    for f in files:
                        if "__pycache__" in root or f.endswith(".pyc"):
                            continue
                        full_f = Path(root) / f
                        rel_f = full_f.relative_to(src_path)
                        zf.write(full_f, arcname=f"{arc_name}/{rel_f}")
    print(f"Created: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

def build_all_packages():
    print("=" * 70)
    print("      BUILDING STANDALONE OFFLINE TEAMMATE ZIP PACKAGES")
    print("=" * 70)

    # 1. Akshay Package (Spatial Data & Preprocessing)
    akshay_files = [
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "preprocess_pipeline.py", "preprocess_pipeline.py"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "extract_all_training_features.py", "extract_all_training_features.py"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "generate_preprocessing_report.py", "generate_preprocessing_report.py"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "train_akshay_models.py", "train_akshay_models.py"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "REPORT_SECTIONS_1_AND_2.md", "REPORT_SECTIONS_1_AND_2.md"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "GPS_CLEANING_AND_PREPROCESSING_REPORT.md", "GPS_CLEANING_AND_PREPROCESSING_REPORT.md"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "GPS_CLEANING_AND_PREPROCESSING_REPORT.html", "GPS_CLEANING_AND_PREPROCESSING_REPORT.html"),
        (REPO_ROOT / "docs" / "PathSnap_GPS_Cleaning_and_Preprocessing_Report.pdf", "PathSnap_GPS_Cleaning_and_Preprocessing_Report.pdf"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "preprocessing_report.csv", "preprocessing_report.csv"),
        (REPO_ROOT / "outputs" / "gps_noise_audit_trip_00000000.csv", "gps_noise_audit_trip_00000000.csv"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "HOW_TO_RUN.md", "HOW_TO_RUN.md"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "README.md", "README.md"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "generate_map.py", "generate_map.py"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "trip_00000000_eda.html", "trip_00000000_eda.html"),
        (REPO_ROOT / "CONTEXT FILES" / "1 - AKSHAY" / "sample_trip_00000000", "sample_trip_00000000"),
    ]
    create_zip(OUTPUT_DIR / "PathSnap_1_AKSHAY_Spatial_EDA.zip", akshay_files)

    # 2. Abhiroop Package (Graph MultiDiGraph & HMM Viterbi Decoder)
    abhiroop_files = [
        (REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP" / "hmm_viterbi_decoder.py", "hmm_viterbi_decoder.py"),
        (REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP" / "HOW_TO_RUN.md", "HOW_TO_RUN.md"),
        (REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP" / "README.md", "README.md"),
        (REPO_ROOT / "src" / "graph_hmm" / "graph_audit.py", "graph_audit.py"),
        (REPO_ROOT / "outputs" / "graph_audit_trip_00000000.txt", "graph_audit_trip_00000000.txt"),
        (REPO_ROOT / "CONTEXT FILES" / "3 - L1 - ABHIROOP" / "sample_trip_00000000", "sample_trip_00000000"),
    ]
    create_zip(OUTPUT_DIR / "PathSnap_3_ABHIROOP_Graph_HMM.zip", abhiroop_files)

    # 3. Aaryan Package (Two-Laptop Distributed Deployment & UI)
    aaryan_files = [
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "train_mlp.py", "train_mlp.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "server.py", "server.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "app_server.py", "app_server.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "ui_client.py", "ui_client.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "pipeline_contract.py", "pipeline_contract.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "pathsnap_engine.py", "pathsnap_engine.py"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "requirements.txt", "requirements.txt"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "HOW_TO_RUN.md", "HOW_TO_RUN.md"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "README.md", "README.md"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "AARYAN_WORK_HANDOUT.md", "AARYAN_WORK_HANDOUT.md"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "REPORT_SECTION_6_DRAFT.md", "REPORT_SECTION_6_DRAFT.md"),
        (REPO_ROOT / "CONTEXT FILES" / "4 - L2 - AARYAN" / "VIVA_FIRST_REVIEW.md", "VIVA_FIRST_REVIEW.md"),
    ]
    create_zip(OUTPUT_DIR / "PathSnap_4_AARYAN_Distributed_UI.zip", aaryan_files)

    # 4. Aditya Package (Machine Learning Arena & Bayesian Tuning)
    aditya_files = [
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "PathSnap_Model_Training_and_Tuning.ipynb", "PathSnap_Model_Training_and_Tuning.ipynb"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "train_aditya_models.py", "train_aditya_models.py"),
        (REPO_ROOT / "docs" / "PathSnap_Official_Evaluation_Report.xlsx", "PathSnap_Official_Evaluation_Report.xlsx"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "evaluate.py", "evaluate.py"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "baseline_models.py", "baseline_models.py"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "optuna_tune.py", "optuna_tune.py"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "train_5_model_arena.py", "train_5_model_arena.py"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "HOW_TO_RUN.md", "HOW_TO_RUN.md"),
        (REPO_ROOT / "CONTEXT FILES" / "2 - ADITYA" / "README.md", "README.md"),
        (REPO_ROOT / "models" / "scoreboard.csv", "scoreboard.csv"),
        (REPO_ROOT / "models" / "scoreboard.json", "scoreboard.json"),
        (REPO_ROOT / "models" / "best_model_lightgbm.joblib", "best_model_lightgbm.joblib"),
    ]
    create_zip(OUTPUT_DIR / "PathSnap_2_ADITYA_ML_Arena.zip", aditya_files)

    # 5. Master All-In-One Package
    master_files = [
        (REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.md", "docs/PathSnap_Consolidated_Technical_Report.md"),
        (REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.html", "docs/PathSnap_Consolidated_Technical_Report.html"),
        (REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.pdf", "docs/PathSnap_Consolidated_Technical_Report.pdf"),
        (REPO_ROOT / "docs" / "PathSnap_GPS_Cleaning_and_Preprocessing_Report.pdf", "docs/PathSnap_GPS_Cleaning_and_Preprocessing_Report.pdf"),
        (REPO_ROOT / "docs" / "PathSnap_Official_Evaluation_Report.xlsx", "docs/PathSnap_Official_Evaluation_Report.xlsx"),
        (REPO_ROOT / "notebooks" / "PathSnap_Model_Training_and_Tuning.ipynb", "notebooks/PathSnap_Model_Training_and_Tuning.ipynb"),
        (REPO_ROOT / "test_end_to_end_pipeline.py", "test_end_to_end_pipeline.py"),
        (REPO_ROOT / "evaluate_10_student_test_trips.py", "evaluate_10_student_test_trips.py"),
        (REPO_ROOT / "outputs" / "student_test_benchmark_table.md", "outputs/student_test_benchmark_table.md"),
        (REPO_ROOT / "outputs" / "preprocessing_report.csv", "outputs/preprocessing_report.csv"),
        (REPO_ROOT / "outputs" / "gps_noise_audit_trip_00000000.csv", "outputs/gps_noise_audit_trip_00000000.csv"),
        (REPO_ROOT / "outputs" / "end_to_end_reconstruction_00000000.html", "outputs/end_to_end_reconstruction_00000000.html"),
        (REPO_ROOT / "outputs" / "student_test_maps", "outputs/student_test_maps"),
        (REPO_ROOT / "CONTEXT FILES", "CONTEXT FILES"),
    ]
    if (REPO_ROOT / "models").exists():
        master_files.append((REPO_ROOT / "models", "models"))
    create_zip(OUTPUT_DIR / "PathSnap_MASTER_ALL_IN_ONE.zip", master_files)

    print("=" * 70)
    print("ALL 5 OFFLINE PACKAGES GENERATED IN: OFFLINE_PACKAGES/")

if __name__ == "__main__":
    build_all_packages()
