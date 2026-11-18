# 2 - ADITYA: Core Machine Learning & Bayesian Optimization

## Strict Workspace Isolation Ground Rule
> **MANDATORY**: Antigravity is strictly authorized to write and edit ONLY inside this folder (`CONTEXT FILES/2 - ADITYA/`).
> All other teammate folders (`1 - AKSHAY/`, `3 - L1 - ABHIROOP/`, `4 - L2 - AARYAN/`) are strictly **READ-ONLY** for Aditya and Antigravity to inspect progress and read context. Never edit or overwrite teammate folders.

---

### Primary Deliverables & Ownership
1. **Zero-Leakage Partitioning**: Implement 5-fold `GroupKFold(n_splits=5)` grouped strictly by `trip_id` on the 80 training trips (zero intra-trip leakage).
2. **Feature Sanitization (Check 1 Audit)**: Strip all local arbitrary identifiers (`arc_id`, `from_node`, `to_node`, `trip_id`, raw lon/lat coordinates) to prevent models memorizing local subgraph IDs. Input feature vector must be strictly translation-, rotation-, and index-invariant.
3. **Core Model Implementations**:
   - **Model 3**: Random Forest Classifier (balanced class weights, depth tuning).
   - **Model 4**: LightGBM / XGBoost Classifier (handling class imbalance via `scale_pos_weight` tuned via Optuna).
4. **Bayesian Hyperparameter Optimization (Optuna)**:
   - Run 30–50 trials on champion model optimizing 5-fold GroupKFold Macro F1.
   - Record parameter importance plots and optimization histories.
5. **Probability Calibration**:
   - Fit Isotonic Regression / Platt scaling on out-of-fold predictions to ensure `predict_proba` produces well-behaved emission scores for Abhiroop's HMM.
   - Export production model: `best_model.joblib`.
6. **Unified Scoreboard & Report Ownership**:
   - Aggregate CV and Student-Test metrics (Macro F1, Top-1 Accuracy, Log Loss, Latency) across all 5 models.
   - Author Report Sections 3 (Model Comparison & Methodology) & 4 (Bayesian Optimization & Calibration).
