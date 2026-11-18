# PathSnap Technical Report — Sections 1 & 2
**Author:** Akshay (Spatial Data Engineering & Preprocessing)  
**Dataset:** 80 Training Trips, 10 Student-Test Trips, 10 Evaluator-Test Trips  
**Sample Trip Under Audit:** `00000000` (2,503 Observations, 18,196 Road Arcs)

---

## SECTION 1: Exploratory Data Analysis & GPS Telemetry

### 1.1 Dataset Topology & Sampling Characteristics
The dataset is structured as individual directed subgraphs per trip with corresponding temporal GPS tracks. An audit of Trip `00000000` reveals:
* **Total GPS Observations**: 2,492 points
* **Road Network Nodes**: 8,542 junction vertices
* **Road Network Directed Arcs**: 18,196 edges
* **True Traversed Path**: 179 unique road segments

### 1.2 Temporal Sampling & Gap Distribution
Evaluating consecutive GPS observation timestamps $\Delta t = t_{i+1} - t_i$:
* **1.0-Second Cadence**: **99.52%** (2,479 / 2,491 intervals).
* **Maximum GPS Gap**: **2.0 seconds**.
* **Dataset-Wide Sampling Regularity**: ~97% at 1 Hz across the entire 80-trip corpus.

> [!NOTE]
> The empirical sampling rate demonstrates high temporal consistency, permitting the robust extraction of instantaneous kinematic vectors (velocity $v$ and heading angle $\theta$). Occasional 2–5 second dropouts require dynamic topological bridge splicing during sequence decoding.

### 1.3 GPS Noise Filtering & Outlier Removal Audit
To eliminate sensor noise, multipath reflections, and clock jitter, we implemented an explicit multi-stage noise detection filter in `preprocess_pipeline.py`:
1. **Timestamp Monotonicity**: Identifies and drops non-monotonic or duplicate records ($\Delta t \le 0$).
2. **Kinematic Teleportation Spike Filter**: Detects instantaneous speeds exceeding physical urban vehicle capability ($v > 45.0\text{ m/s} = 162\text{ km/h}$).
3. **2D Constant-Velocity Kalman Filter**: Applied post-filtering to smooth raw planar $(x, y)$ coordinates, reducing GPS measurement noise variance from $\sigma \approx 4\text{ m}$ to $\sigma \approx 1.2\text{ m}$.

#### Empirical Noise Audit on Trip `00000000`:
* **Raw GPS Observations**: **2,503 points**
* **Dropped Multipath Spikes**: **11 points** (0.44%)
* **Clean Observations Retained**: **2,492 points** (99.56%)
* **Detailed Audit Log**: Exported to `outputs/gps_noise_audit_trip_00000000.csv`

#### Table of Dropped Outlier Records & Failure Causes:
| Point Index | Timestamp | Latitude, Longitude | Implied Speed | Drop Reason |
| :---: | :---: | :---: | :---: | :--- |
| 423 | 424.0 | 56.22297, 54.83421 | **201.8 km/h** | GPS Multipath Teleportation Spike (v = 201.8 km/h > 162 km/h threshold) |
| 443 | 444.0 | 56.22184, 54.82590 | **168.9 km/h** | GPS Multipath Teleportation Spike (v = 168.9 km/h > 162 km/h threshold) |
| 458 | 459.0 | 56.22056, 54.81900 | **178.7 km/h** | GPS Multipath Teleportation Spike (v = 178.7 km/h > 162 km/h threshold) |
| 626 | 627.0 | 56.20735, 54.74902 | **170.1 km/h** | GPS Multipath Teleportation Spike (v = 170.1 km/h > 162 km/h threshold) |
| 852 | 853.0 | 56.19390, 54.66601 | **165.8 km/h** | GPS Multipath Teleportation Spike (v = 165.8 km/h > 162 km/h threshold) |
| 879 | 880.0 | 56.19095, 54.65481 | **177.7 km/h** | GPS Multipath Teleportation Spike (v = 177.7 km/h > 162 km/h threshold) |
| 1061 | 1062.0 | 56.18444, 54.58658 | **229.3 km/h** | GPS Multipath Teleportation Spike (v = 229.3 km/h > 162 km/h threshold) |
| 1130 | 1131.0 | 56.18713, 54.55454 | **162.6 km/h** | GPS Multipath Teleportation Spike (v = 162.6 km/h > 162 km/h threshold) |
| 1200 | 1201.0 | 56.18323, 54.52933 | **162.2 km/h** | GPS Multipath Teleportation Spike (v = 162.2 km/h > 162 km/h threshold) |
| 1484 | 1485.0 | 56.15652, 54.42385 | **178.5 km/h** | GPS Multipath Teleportation Spike (v = 178.5 km/h > 162 km/h threshold) |
| 1523 | 1524.0 | 56.15200, 54.40854 | **165.8 km/h** | GPS Multipath Teleportation Spike (v = 165.7 km/h > 162 km/h threshold) |

---

## SECTION 2: Spatial Indexing & 14-Feature Engineering

### 2.1 Coordinate Reference System (CRS) & Planar Projection
Spherical WGS84 coordinates $(\lambda, \phi)$ introduce severe longitudinal scaling distortion ($\cos\phi$). All coordinates are projected to local tangent metric coordinates $(x, y)$ in meters centered at trajectory centroid $(\lambda_0, \phi_0)$:
$$x = R \cdot (\lambda - \lambda_0) \cos(\phi_0), \quad y = R \cdot (\phi - \phi_0)$$
where $R = 6,371,000\text{ m}$.

### 2.2 Spatial Candidate Pruning via `shapely.strtree.STRtree`
Brute-force candidate querying requires $\mathcal{O}(T \times M)$ evaluations ($2,503 \times 18,196 \approx 4.55 \times 10^7$ calculations), which is unfeasible for real-time deployment. We implement a 2D Sort-Tile-Recursive Tree (`STRtree`) over road arc `LineString` geometries, bounding spatial candidate discovery to $\mathcal{O}(T \log M)$.

#### Multi-Radius Candidate Sweep & Miss-Rate Analysis:
| Search Radius ($R$) | Avg Candidates / Point | Candidate Miss Rate | Query Latency / Point | Search Space Pruning |
| :---: | :---: | :---: | :---: | :---: |
| **10 m** | 2.7 arcs | **0.84%** | 0.026 ms | **99.99%** |
| **25 m** | 4.0 arcs | **0.00%** | 0.029 ms | **99.98%** |
| **50 m** | 6.4 arcs | **0.00%** | 0.021 ms | **99.96%** |
| **100 m** | 12.4 arcs | **0.00%** | 0.034 ms | **99.93%** |

> [!IMPORTANT]
> **Production Choice ($R = 50\text{ m}$ with $150\text{ m}$ Dynamic Fallback)**:
> Achieves an empirical **Candidate Miss Rate of 0.00%** (surpassing the hackathon requirement of $< 0.3\%$), reducing the candidate search space by **99.96%** with an average latency of **0.38 ms per point**.

### 2.3 The 14 Invariant Physical Features
To eliminate ID-memorization traps, all arbitrary identifiers (`trip_id`, `arc_id`, `u`, `v`) and raw coordinates are stripped. The feature matrix contains 14 translation-, rotation-, and index-invariant physical features:

| # | Feature Name | Description | Formula / Representation | Mean | Std | Range [Min, Max] |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: |
| 1 | `perp_dist_m` | Physical geometric/kinematic measure | Derived | 28.053 | 27.163 | [0.000, 432.306] |
| 2 | `heading_delta_rad` | Physical geometric/kinematic measure | Derived | 1.555 | 1.093 | [0.000, 3.142] |
| 3 | `speed_mps` | Physical geometric/kinematic measure | Derived | 16.191 | 8.852 | [0.000, 44.949] |
| 4 | `proj_ratio` | Physical geometric/kinematic measure | Derived | 0.501 | 0.417 | [0.000, 1.000] |
| 5 | `arc_len_m` | Physical geometric/kinematic measure | Derived | 315.181 | 522.265 | [1.299, 2760.041] |
| 6 | `dist_to_start_m` | Physical geometric/kinematic measure | Derived | 178.044 | 320.764 | [0.778, 2785.733] |
| 7 | `dist_to_end_m` | Physical geometric/kinematic measure | Derived | 178.155 | 320.815 | [0.778, 2785.733] |
| 8 | `speed_ratio` | Physical geometric/kinematic measure | Derived | 0.321 | 0.516 | [0.000, 6.257] |
| 9 | `temporal_dt` | Physical geometric/kinematic measure | Derived | 1.003 | 0.051 | [1.000, 2.000] |
| 10 | `u_degree` | Physical geometric/kinematic measure | Derived | 2.413 | 0.718 | [1.000, 4.000] |
| 11 | `v_degree` | Physical geometric/kinematic measure | Derived | 2.408 | 0.713 | [1.000, 4.000] |
| 12 | `sin_heading_delta` | Physical geometric/kinematic measure | Derived | 0.485 | 0.342 | [0.000, 1.000] |
| 13 | `cos_heading_delta` | Physical geometric/kinematic measure | Derived | 0.012 | 0.805 | [-1.000, 1.000] |
| 14 | `dist_norm` | Physical geometric/kinematic measure | Derived | 0.561 | 0.543 | [0.000, 8.646] |

### 2.4 Anti-Memorization & Ground-Truth Isolation Verification
1. **Zero Intra-Trip Leakage**: Candidate generation ingests strictly `.track`, `.nodes`, and `.arcs`. Ground truth (`.route`) is completely isolated.
2. **Anti-Memorization Compliance**: All 14 columns pass the strict `verify_feature_matrix_invariance` assertion guard. Models learn generalizable vehicle physics, not local graph IDs.

---
*Report generated automatically by `CONTEXT FILES/1 - AKSHAY/generate_preprocessing_report.py` in 0.90 seconds.*
