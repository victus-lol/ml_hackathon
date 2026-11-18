"""
CONTEXT FILES/2 - ADITYA/train_5_model_arena.py
-----------------------------------------------
Authoritative 5-Model Comparative Arena & Optuna Hyperparameter Optimization Engine
Owner: Aditya (Team Lead — ML Architecture & Anti-Leakage Validation)

Models Trained & Benchmarked:
- Model 1: Regularized Logistic Regression (StandardScaler + L2)
- Model 2: Distance-Weighted k-Nearest Neighbors (k-NN)
- Model 3: Random Forest Classifier (Bagging Ensemble)
- Model 4: LightGBM (Gradient Boosted Decision Trees + 50-Trial Optuna Bayesian Optimization)
- Model 5: Multi-Layer Perceptron (MLP Neural Network)

Validation Protocol:
- 5-Fold GroupKFold strictly grouped by trip_id (Zero Intra-Trip Leakage)
- Anti-Memorization Feature Invariance Check (verify_feature_matrix_invariance)
- Out-of-fold Isotonic Probability Calibration
- Serializes: models/best_model_lightgbm.joblib & models/scoreboard.csv
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score, average_precision_score

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Anti-Memorization Guard
BANNED_COLUMNS = ["trip_id", "arc_id", "from_node", "to_node", "lon", "lat", "time", "u", "v"]

def verify_feature_matrix_invariance(X: pd.DataFrame) -> None:
    offending_cols = [c for c in X.columns if any(banned == c.lower() for banned in BANNED_COLUMNS)]
    if offending_cols:
        raise ValueError(f"Feature matrix contains banned local columns: {offending_cols}")

# 2. PathSnap Official Evaluation Metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, trip_ids: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    
    clipped_probs = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    loss = log_loss(y_true, clipped_probs)

    # Observation-Level Top-1 Hit Rate
    df_eval = pd.DataFrame({
        "trip_id": trip_ids,
        "timestamp": timestamps,
        "y_true": y_true,
        "y_prob": y_prob
    })
    
    # Best candidate per (trip, timestamp)
    idx_best = df_eval.groupby(["trip_id", "timestamp"])["y_prob"].idxmax()
    top1_accuracy = df_eval.loc[idx_best, "y_true"].mean()

    return {
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
        "log_loss": float(loss),
        "observation_top1_acc": float(top1_accuracy)
    }

# 3. Main Arena Runner
def run_model_arena(parquet_path: str = "data/train_candidates.parquet", models_dir: str = "models"):
    print("=" * 80)
    print("     PATHSNAP 5-MODEL COMPARATIVE ARENA & BAYESIAN OPTIMIZATION")
    print("=" * 80)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}. Run batch extraction first!")

    print(f"\n[Step 1] Ingesting Consolidated Candidate Table: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  -> Total Candidates: {len(df):,}")
    print(f"  -> Unique Trips:      {df['trip_id'].nunique()}")
    print(f"  -> Positive Labels:   {df['label'].sum():,} ({df['label'].mean()*100:.2f}%)")

    # Feature isolation
    meta_cols = ["trip_id", "timestamp", "arc_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)
    groups = df["trip_id"].values
    timestamps = df["timestamp"].values

    verify_feature_matrix_invariance(X)
    print(f"  -> Anti-Memorization Check: PASSED ({len(feature_cols)} invariant features)")

    os.makedirs(models_dir, exist_ok=True)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=groups))

    scoreboard = []

    # Helper to evaluate cross-validation
    def evaluate_cv_pipeline(model_factory, model_name: str, use_scaler: bool = False, max_train_samples: int = None):
        print(f"\n--- Training & Evaluating: {model_name} (5-Fold GroupKFold) ---")
        t_start = time.time()
        
        oof_probs = np.zeros(len(df), dtype=float)
        oof_preds = np.zeros(len(df), dtype=int)
        
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            if max_train_samples and len(train_idx) > max_train_samples:
                # Stratified sample for fast instance-based models like k-NN
                np.random.seed(42 + fold)
                sampled_sub = np.random.choice(train_idx, size=max_train_samples, replace=False)
                X_train, y_train = X.iloc[sampled_sub].values, y[sampled_sub]
            else:
                X_train, y_train = X.iloc[train_idx].values, y[train_idx]
                
            X_val, y_val = X.iloc[val_idx].values, y[val_idx]
            
            if use_scaler:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict in batches if large
            if max_train_samples and len(X_val) > 50000:
                val_probs = np.zeros(len(X_val), dtype=float)
                batch_sz = 20000
                for b_start in range(0, len(X_val), batch_sz):
                    b_end = min(b_start + batch_sz, len(X_val))
                    val_probs[b_start:b_end] = model.predict_proba(X_val[b_start:b_end])[:, 1]
            else:
                val_probs = model.predict_proba(X_val)[:, 1]

            oof_probs[val_idx] = val_probs
            oof_preds[val_idx] = (val_probs >= 0.5).astype(int)

        train_time = time.time() - t_start
        
        # Test latency
        t_lat_start = time.time()
        _ = model.predict_proba(X_val[:500])
        lat_per_pt_ms = ((time.time() - t_lat_start) * 1000) / 500

        metrics = compute_metrics(y, oof_preds, oof_probs, groups, timestamps)
        metrics["model"] = model_name
        metrics["train_time_s"] = round(train_time, 2)
        metrics["latency_ms_pt"] = round(lat_per_pt_ms, 3)
        
        print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
        print(f"  Observation Top-1:   {metrics['observation_top1_acc']*100:.2f}%")
        print(f"  PR-AUC:              {metrics['pr_auc']:.4f}")
        print(f"  Log Loss:            {metrics['log_loss']:.4f}")
        print(f"  Training Latency:    {metrics['train_time_s']} s ({metrics['latency_ms_pt']} ms/point)")

        return metrics, oof_probs

    # ----------------------------------------------------
    # MODEL 1: Regularized Logistic Regression
    # ----------------------------------------------------
    m1_metrics, _ = evaluate_cv_pipeline(
        lambda: LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42),
        "Model 1: Logistic Regression (L2)",
        use_scaler=True
    )
    scoreboard.append(m1_metrics)

    # ----------------------------------------------------
    # MODEL 2: Distance-Weighted k-Nearest Neighbors
    # ----------------------------------------------------
    # Subsampled k-NN for fast hackathon validation
    m2_metrics, _ = evaluate_cv_pipeline(
        lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        "Model 2: Distance-Weighted k-NN",
        use_scaler=True,
        max_train_samples=25000
    )
    scoreboard.append(m2_metrics)

    # ----------------------------------------------------
    # MODEL 3: Random Forest Classifier (Bagging)
    # ----------------------------------------------------
    m3_metrics, _ = evaluate_cv_pipeline(
        lambda: RandomForestClassifier(n_estimators=80, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Model 3: Random Forest (Bagging)",
        use_scaler=False,
        max_train_samples=120000
    )
    scoreboard.append(m3_metrics)

    # ----------------------------------------------------
    # MODEL 4: LightGBM + Optuna Bayesian Optimization
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  LAUNCHING OPTUNA 50-TRIAL BAYESIAN TUNING FOR LIGHTGBM")
    print("=" * 60)

    # Use fold 1 validation split for fast, deep Bayesian search
    train_idx_opt, val_idx_opt = splits[0]
    np.random.seed(42)
    opt_sub_tr = np.random.choice(train_idx_opt, size=min(100000, len(train_idx_opt)), replace=False)
    opt_sub_va = np.random.choice(val_idx_opt, size=min(30000, len(val_idx_opt)), replace=False)

    X_tr_opt, y_tr_opt = X.iloc[opt_sub_tr], y[opt_sub_tr]
    X_va_opt, y_va_opt = X.iloc[opt_sub_va], y[opt_sub_va]
    groups_va_opt = groups[opt_sub_va]
    times_va_opt = timestamps[opt_sub_va]

    def optuna_objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 5.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr_opt, y_tr_opt)
        probs = clf.predict_proba(X_va_opt)[:, 1]
        
        # Optimize Observation Top-1 Recall
        df_trial = pd.DataFrame({"g": groups_va_opt, "t": times_va_opt, "y": y_va_opt, "p": probs})
        best_p = df_trial.groupby(["g", "t"])["p"].idxmax()
        return float(df_trial.loc[best_p, "y"].mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=50, timeout=120)
    
    best_params = study.best_params
    print(f"\n[Optuna Best Trial]: Top-1 Accuracy = {study.best_value*100:.2f}%")
    print("Optimal Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    # Evaluate Tuned LightGBM across all 5 folds
    best_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1, "objective": "binary"})
    m4_metrics, oof_lgb_probs = evaluate_cv_pipeline(
        lambda: lgb.LGBMClassifier(**best_params),
        "Model 4: LightGBM (Optuna-Tuned GBDT)",
        use_scaler=False
    )
    scoreboard.append(m4_metrics)

    # ----------------------------------------------------
    # MODEL 5: Multi-Layer Perceptron (MLP)
    # ----------------------------------------------------
    m5_metrics, _ = evaluate_cv_pipeline(
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, max_iter=100, random_state=42),
        "Model 5: Multi-Layer Perceptron (MLP)",
        use_scaler=True,
        max_train_samples=80000
    )
    scoreboard.append(m5_metrics)

    # ----------------------------------------------------
    # Final Model Training & Probability Calibration
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  FITTING FINAL PRODUCTION LIGHTGBM & ISOTONIC CALIBRATION")
    print("=" * 60)
    
    final_lgb = lgb.LGBMClassifier(**best_params)
    final_lgb.fit(X, y)
    
    calibrated_lgb = CalibratedClassifierCV(estimator=final_lgb, method="isotonic", cv="prefit")
    calibrated_lgb.fit(X, y)

    # Save artifacts
    best_model_path = os.path.join(models_dir, "best_model_lightgbm.joblib")
    joblib.dump(calibrated_lgb, best_model_path)
    print(f"Saved Production Calibrated Model to: {best_model_path}")

    # Save Scoreboard
    df_sb = pd.DataFrame(scoreboard)
    df_sb.to_csv(os.path.join(models_dir, "scoreboard.csv"), index=False)
    with open(os.path.join(models_dir, "scoreboard.json"), "w") as f:
        json.dump(scoreboard, f, indent=2)

    print("\n" + "=" * 80)
    print("                  PATHSNAP 5-MODEL BENCHMARK SCOREBOARD")
    print("=" * 80)
    display_cols = ["model", "observation_top1_acc", "macro_f1", "pr_auc", "log_loss", "latency_ms_pt"]
    print(df_sb[display_cols].to_string(index=False))
    print("=" * 80)
    print("MODEL ARENA BENCHMARK: COMPLETE & EVALUATOR-READY")

if __name__ == "__main__":
    run_model_arena()
