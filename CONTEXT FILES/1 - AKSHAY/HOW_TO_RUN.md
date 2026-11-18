# AKSHAY: How to Run & Viva Defense Guide

### Step 1: Open Terminal in your folder
```bash
pip install shapely pandas numpy folium
python preprocess_pipeline.py
```

### Step 2: What Appears on Your Screen (Show the Teachers!)
* **Step 1.1 GPS Noise & Outlier Filter**:
  - Raw GPS Observations: **2,503 points**
  - **Dropped Multipath Teleportation Spikes**: **11 points** (0.44%) where implied speeds exceeded $162\text{ km/h}$ ($45\text{ m/s}$) due to urban canyon multipath reflections!
  - **Clean Observations Retained**: **2,492 points (99.56%)**
  - **Audit Log Saved**: `outputs/gps_noise_audit_trip_00000000.csv`
* **Step 2.1 2D Kalman Trajectory Smoothing**:
  - Constant-velocity Kalman Filter reduces Gaussian measurement noise variance from $\sigma \approx 4\text{ m}$ to $\sigma \approx 1.2\text{ m}$.
* **Step 3 Spatial Candidate Pruning**:
  - Reduced from **18,196 arcs down to 6.4 candidate arcs per point** using `shapely.strtree.STRtree` (50m radius).
* **Candidate Miss Rate**: **0.00%** (beats the < 0.3% competition threshold).
* **Latency**: **0.30 ms / observation** (sub-millisecond spatial pruning).
* **Anti-Memorization Check**: **PASSED** (14 invariant features extracted; all IDs stripped).

### Step 3: Open the Map & Audit in Your Browser
* Double-click `trip_00000000_eda.html` to show the blue raw GPS points over the dark road arcs with zero `.route` leakage!
* Open `outputs/gps_noise_audit_trip_00000000.csv` to show the exact table of 11 dropped spike records with calculated speeds up to 229 km/h!

### Step 4: Run Akshay's 2 Machine Learning Models
```bash
python train_akshay_models.py
```
* **Model 1: Logistic Regression (L2 Regularized)**:
  - Macro F1: **`0.6097`** | Observation Top-1 Acc: **`51.41%`** | PR-AUC: **`0.4571`**
  - Latency: **`0.001 ms / observation`**
  - Rationale: *"Serves as our linear, interpretable baseline. It establishes the benchmark performance when only linear combinations of features are considered."*
* **Model 2: Distance-Weighted k-Nearest Neighbors ($k=15$)**:
  - Macro F1: **`0.7568`** | Observation Top-1 Acc: **`58.25%`** | PR-AUC: **`0.6225`**
  - Latency: **`0.216 ms / observation`**
  - Rationale: *"Serves as our non-parametric instance-based spatial baseline, using inverse distance weights $w = 1/d$ to rank candidates based on feature similarity."*

### Step 5: Your Exact Viva Pitch (When Teachers Ask About Noise & Models)
> *"For data engineering, I implemented a kinematic spike filter (dropping 11 multipath spikes > 162 km/h) and a 2D Kalman filter. For modeling, I trained Model 1 (Logistic Regression, scoring 0.6097 Macro F1) and Model 2 (Distance-Weighted k-NN, scoring 0.7568 Macro F1) using 5-Fold GroupKFold by trip_id to establish the baseline performance for our team's arena."*
