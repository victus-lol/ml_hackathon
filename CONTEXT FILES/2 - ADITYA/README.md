# 2 - ADITYA: Core Machine Learning & Bayesian Optimization

### Primary Deliverables
1. **Zero-Leakage Partitioning**: Implement 5-fold `GroupKFold(n_splits=5)` grouped strictly by `trip_id` on the 80 training trips (zero intra-trip leakage).
2. **Feature Sanitization**: Strip all local arbitrary identifiers (`arc_id`, `from_node`, `to_node`, `trip_id`, raw lon/lat coordinates) to prevent models memorizing local subgraph IDs.
3. **Core Model Implementations**:
   - **Model 3**: Random Forest Classifier (balanced class weights, depth tuning).
   - **Model 4**: LightGBM / XGBoost Classifier (handling class imbalance via `scale_pos_weight`).
4. **Bayesian Hyperparameter Optimization (Optuna)**:
   - Run 30–50 trials on champion model optimizing 5-fold GroupKFold Macro F1.
   - Record parameter importance plots and optimization histories.
5. **Probability Calibration**:
   - Fit Isotonic Regression / Platt scaling on out-of-fold predictions to ensure `predict_proba` produces well-behaved emission scores for the HMM.
   - Export production model: `best_model.joblib`.
6. **Unified Scoreboard**:
   - Aggregate CV and Student-Test metrics (Macro F1, Top-1 Accuracy, Log Loss, Latency) across all 5 models.
   - Author Report Sections 3 & 4.
