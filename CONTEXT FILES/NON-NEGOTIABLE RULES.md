# PATHSNAP: NON-NEGOTIABLE GROUND RULES & COMPLIANCE DIRECTIVES
## PES University CSE — UE24CS352A Machine Learning Hackathon
**Mandatory Operational & Technical Protocol for All Team Members**

---

### SECTION 1: TEAM WORKSPACE ISOLATION & OPERATIONAL RULES

#### 1. Strict Workspace Isolation Mandate (Zero Cross-Editing Rule)
To prevent merge conflicts, accidental overwrites, and code regression, the `CONTEXT FILES/` folder enforces strict workspace boundaries:
* **`1 - AKSHAY/`**: Akshay writes and edits **EXCLUSIVELY** here. This folder is strictly **READ-ONLY** for Aditya, Abhiroop, and Aaryan.
* **`2 - ADITYA/`**: Aditya writes and edits **EXCLUSIVELY** here. This folder is strictly **READ-ONLY** for Akshay, Abhiroop, and Aaryan.
* **`3 - L1 - ABHIROOP/`**: Abhiroop writes and edits **EXCLUSIVELY** here. This folder is strictly **READ-ONLY** for Akshay, Aditya, and Aaryan.
* **`4 - L2 - AARYAN/`**: Aaryan writes and edits **EXCLUSIVELY** here. This folder is strictly **READ-ONLY** for Akshay, Aditya, and Abhiroop.
* **Universal Rule**: Every teammate must work strictly within their designated folder. You may inspect and read other folders for context, but **NEVER** edit, overwrite, or delete files belonging to another member.

#### 2. Continuous Task Closure & Execution Tracking
* Whenever a task, script, or pipeline phase completes:
  1. Provide a concise **Summary of What Was Done**.
  2. Clearly declare the **Next Upcoming Tasks** to maintain synchronized team progress.

---

### SECTION 2: OFFICIAL COURSE & EVALUATION GUIDELINES (PES UNIVERSITY)

#### A. Data Integrity & Anti-Leakage Rules (Zero-Tolerance Disqualification Traps)
1. **Zero Ground-Truth UI Exposure**:
   - Evaluator ground-truth labels must stay server-side and **never enter the prediction pipeline or client UI**.
   - The evaluator UI on Laptop 2 must display **only** raw GPS points (blue), the PathSnap predicted route (red), and server-computed accuracy numbers. Showing ground truth on evaluator trips is treated as data leakage.
2. **Untouched Evaluator Partition**:
   - The 10 evaluator-test trips must remain completely untouched during development.
   - Evaluator data must **never** be used for feature selection, scaling, threshold tuning, or hyperparameter choices.
3. **Strict Trip-Level Isolation (GroupKFold)**:
   - No observations from the same trip may appear across both training and validation splits.
   - Cross-validation must strictly use `GroupKFold(n_splits=5)` grouped by `trip_id` (standard random $k$-fold is strictly prohibited).
4. **No Hardcoding & No Local Coordinate Memorization**:
   - Hardcoding route segments or altering predictions based on known paths is strictly prohibited (repositories will be inspected for hardcoded logic).
   - Local arbitrary IDs (`arc_id`, `from_node`, `to_node`) and raw `(lon, lat)` coordinates must not be fed into models as features to prevent arbitrary subgraph memorization.

#### B. Machine Learning & Modeling Mandates
5. **Train at least 5 Distinct Models**:
   - Teams must train and evaluate **at least five different models from multiple algorithm families** (e.g., Linear/Logistic Regression, Instance-based $k$-NN, Tree/Random Forest, Boosting/LightGBM, Neural Network/MLP). Directly picking one model without comparative experiments results in heavy grade deductions.
   - Models must never be selected based on training accuracy alone.
6. **Mandatory Bayesian Hyperparameter Optimization**:
   - Must perform Bayesian optimization (Optuna / BayesSearchCV) on the champion model using group-aware CV on the 80 training trips.
   - Must log and report search ranges, objective functions, trial histories, parameter importances, and a formal **Tuned vs. Untuned comparison**.
7. **Mandatory HMM & Viterbi Route Reconstruction**:
   - Must implement a Hidden Markov Model (HMM) where hidden states represent candidate road segments.
   - Emission probabilities must model observation likelihood; transition probabilities must model topological road connectivity and shortest-path driving distance.
   - Must decode optimal routes via log-space Viterbi and demonstrate that HMM decoding improves **Route Continuity** and sequence accuracy over greedy classifier predictions.

#### C. Spatial & Algorithmic Math Rules
8. **Metric Coordinate Projections**:
   - Angular degree coordinates ($\text{lat}, \text{lon}$) cannot be treated as flat Euclidean planes. Spatial distances must be computed via planar metric projection (local UTM) or geodesic Haversine math.
9. **Arc-Geometry Spatial Indexing**:
   - Candidate road segments must be queried against the **arc line geometry** (using `shapely.strtree.STRtree` or dense interpolated points along segments), **not** a KD-tree over junction nodes alone (which misses long roads).

#### D. Distributed 2-Laptop Deployment Rules
10. **Hardware & Architecture Separation**:
    - **Laptop 1 (Model & API Server)**: Hosts FastAPI, trained model weights, road graph, and the HMM/Viterbi decoder.
    - **Laptop 2 (User Interface)**: Hosts the frontend client, displays evaluation trip options, sends requests to Laptop 1, and renders the interactive Folium map.
    - The two laptops must communicate over a local network (Wi-Fi hotspot / LAN) with zero `.route` ground-truth data passing over the wire.

#### E. Submission & Academic Integrity Rules
11. **Public GitHub Repository**:
    - Must be public and well-structured (`data/`, `models/`, `api/`, `ui/`, `artifacts/`, `README.md`).
    - Granular commit histories from all teammates across development (no single-day monolithic commit).
    - Code must reproduce all results reported in the PDF.
12. **Consolidated Technical Report (PDF)**:
    - Exactly one consolidated report covering all 6 mandatory sections:
      1. *Dataset Understanding & Preprocessing* (missing values, GPS plots, Folium EDA map).
      2. *Feature Engineering & Labeling* (14 features, mathematical definitions).
      3. *Overall Model Comparison* (consolidated matrix covering CV/Test F1, Top-1, Log Loss, Latency + **compulsory 9 questions for EVERY model**).
      4. *Bayesian Optimization Deep-Dive* (Optuna ranges, curves, tuned vs. untuned).
      5. *HMM & Viterbi Route Reconstruction* (emission/transition equations, bridge splicing, continuity gains).
      6. *2-Laptop Distributed Deployment & Folium Maps* (architecture diagram, latency tables, live Folium maps).
13. **Anti-Plagiarism & Viva Defense**:
    - Architectures, weights, and configurations will be compared across all teams and sections using automated similarity scanners.
    - Every team member must be able to independently explain and defend all code, math, and design decisions in the live viva demonstration.
