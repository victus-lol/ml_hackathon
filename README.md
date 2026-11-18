# PathSnap: Distributed Map-Matching & Route Reconstruction
## PES University Semester 5 CSE — UE24CS352A Machine Learning Hackathon

PathSnap is an enterprise-grade, distributed spatial machine learning system engineered to solve large-scale map-matching, topological ambiguity resolution, and end-to-end vehicle route reconstruction from sparse, noisy GPS trajectories.

---

## 🏆 Key Performance Benchmark Highlights

| Metric | Target / Baseline | PathSnap Achieved | Verification Protocol |
| :--- | :--- | :--- | :--- |
| **Mean Route Continuity** | $\ge 99.0\%$ | **100.00%** | 10 Student Test Trips (Dijkstra Bridge Splicing) |
| **Mean Sequence Recall** | $\ge 75.0\%$ | **98.82%** | Ground-truth arc overlap across test trips |
| **Mean Sequence Jaccard**| $\ge 70.0\%$ | **82.61%** | Set-theoretic road segment recovery |
| **Mean Inference Latency**| $< 85.0\text{ ms/pt}$ | **1.40 ms / observation** | End-to-end spatial query + GBDT + Viterbi |
| **Champion Model CV F1** | Baseline: $0.6097$ | **0.8509 (Macro F1)** | 5-Fold `GroupKFold` across 80 training trips |
| **Bayesian Tuning Gain** | Untuned: $0.8140$ | **+0.0369 F1 (+4.5%)** | 50-Trial Optuna Optimization |

---

## 👥 Team Roles & Responsibilities

| Member | Module / Role | Primary Directory | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Akshay** | Spatial Preprocessing & Feature Engineering | `CONTEXT FILES/1 - AKSHAY/` | STRtree spatial query, 14 invariant features, 2.73M candidate parquet, Folium EDA |
| **Aditya (Lead)** | ML Arena & Bayesian Tuning | `CONTEXT FILES/2 - ADITYA/` | 5-model arena, GroupKFold CV, Optuna tuning, LightGBM champion artifact |
| **Abhiroop** | Graph Topology & HMM Viterbi Decoder | `CONTEXT FILES/3 - L1 - ABHIROOP/` | MultiDiGraph, log-space Viterbi, bounded Dijkstra ($350\text{ m}$ cutoff), bridge splicing |
| **Aaryan** | Distributed 2-Laptop API & UI | `CONTEXT FILES/4 - L2 - AARYAN/` | FastAPI server (Laptop 1), Streamlit client (Laptop 2), zero-leakage wire contract |

---

## 📂 Repository Architecture

```text
ml_hackathon/
├── data/
│   └── train_candidates.parquet          # 161.58 MB (2,734,105 candidate training pairs)
├── models/
│   ├── best_model_lightgbm.joblib         # Calibrated production LightGBM classifier
│   ├── scoreboard.csv                    # 5-model cross-validation comparison matrix
│   └── scoreboard.json                   # Machine-readable evaluation metrics
├── outputs/
│   ├── student_test_benchmark_table.md   # Detailed per-trip metrics table across 10 test trips
│   ├── end_to_end_reconstruction_00000000.html # Interactive single-trip verification map
│   └── student_test_maps/                # 10 interactive Folium maps for student test trips
├── docs/
│   ├── PathSnap_Consolidated_Technical_Report.md   # Official 6-section technical report
│   └── PathSnap_Consolidated_Technical_Report.html # Styled HTML report (print-to-PDF ready)
├── OFFLINE_PACKAGES/                     # Standalone offline teammate ZIP packages
│   ├── PathSnap_1_AKSHAY_Spatial_EDA.zip
│   ├── PathSnap_2_ADITYA_ML_Arena.zip
│   ├── PathSnap_3_ABHIROOP_Graph_HMM.zip
│   ├── PathSnap_4_AARYAN_Distributed_UI.zip
│   └── PathSnap_MASTER_ALL_IN_ONE.zip
├── CONTEXT FILES/
│   ├── 1 - AKSHAY/                       # Preprocessing & spatial indexing scripts
│   ├── 2 - ADITYA/                       # 5-Model arena & Optuna tuning scripts
│   ├── 3 - L1 - ABHIROOP/                # HMM Viterbi decoder & graph topology
│   └── 4 - L2 - AARYAN/                  # Distributed server & UI client
├── evaluate_10_student_test_trips.py     # Comprehensive 10-trip benchmark runner
├── test_end_to_end_pipeline.py           # Single-trip end-to-end integration test
├── package_teammate_zips.py              # Teammate offline packaging script
└── convert_report_to_html.py             # Markdown-to-HTML report compiler
```

---

## 🚀 Quick Start & Reproduction

### 1. Verification of Full 10-Trip Benchmark
Run the comprehensive student test benchmark across all 10 evaluation trips:
```bash
python evaluate_10_student_test_trips.py
```
*Output*: Generates per-trip performance metrics and saves all 10 interactive Folium route reconstruction maps into `outputs/student_test_maps/`.

### 2. Single-Trip End-to-End Pipeline Smoke Test
```bash
python test_end_to_end_pipeline.py
```

### 3. Distributed 2-Laptop Live Demonstration
#### On Laptop 1 (Model & HMM Server):
```bash
cd "CONTEXT FILES/4 - L2 - AARYAN"
uvicorn server:app --host 0.0.0.0 --port 8000
```
*Health Check*: Open `http://localhost:8000/api/v1/health` in browser to confirm `{"status": "healthy"}`.

#### On Laptop 2 (Client UI):
Connect Laptop 2 to the same Wi-Fi / hotspot as Laptop 1:
```bash
cd "CONTEXT FILES/4 - L2 - AARYAN"
streamlit run ui_client.py
```
Enter Laptop 1's local IP address (e.g., `http://192.168.1.15:8000`) and select any test trip to view live reconstruction.

---

## 🔒 Evaluation Integrity & Compliance

1. **Zero Ground-Truth UI Exposure**: Ground-truth `.route` files are strictly isolated from the inference pipeline and network wire protocol. The UI renders raw GPS and model-predicted routes only.
2. **Strict Group-Level Isolation**: All 5 models and Bayesian hyperparameters were tuned strictly using 5-fold `GroupKFold` grouped by `trip_id`. No observation or trip data leaked across folds.
3. **Physical Spatial Invariance**: The 14 features depend strictly on metric projections, relative headings, speed ratios, and road metadata. Arbitrary graph IDs (`arc_id`, `u`, `v`) and absolute coordinates are strictly excluded.
