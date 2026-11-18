"""
CONTEXT FILES/1 - AKSHAY/train_akshay_models.py
----------------------------------------------
Akshay's Machine Learning Models Suite:
- Model 1: Regularized Logistic Regression (L2 Penalty with Balanced Class Weights)
- Model 2: Distance-Weighted k-Nearest Neighbors (k=15 with Prototype KD-Tree)
Evaluates under 5-Fold GroupKFold strictly grouped by trip_id.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc, log_loss

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = REPO_ROOT / "data" / "train_candidates.parquet"
OUTPUT_DIR = SCRIPT_DIR

FEATURE_COLS = [
    "perp_dist", "sin_heading_diff", "cos_heading_diff", "speed_ratio",
    "is_oneway", "maxspeed", "num_lanes", "segment_length",
    "highway_primary", "highway_secondary", "highway_tertiary",
    "highway_residential", "highway_trunk", "highway_motorway"
]

def load_data():
    if not DATA_PATH.exists():
        print(f"Data not found at {DATA_PATH}, searching parent...")
        alt_path = Path("c:/Users/adity/Downloads/ML HACKATHON/ml_hackathon/data/train_candidates.parquet")
        if alt_path.exists():
            return pd.read_parquet(alt_path)
        raise FileNotFoundError(f"Could not locate train_candidates.parquet at {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)

def evaluate_model_cv(name, model_fn, X, y, groups, timestamps, use_scaler=True, max_train=None):
    print("=" * 65)
    print(f"  EVALUATING {name}")
    print("=" * 65)

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds = np.zeros(len(y), dtype=int)
    
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        if max_train and len(train_idx) > max_train:
            np.random.seed(42)
            train_idx = np.random.choice(train_idx, size=max_train, replace=False)

        X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
        X_va, y_val = X.iloc[val_idx].values, y[val_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        clf = model_fn()
        clf.fit(X_tr, y_tr)

        if max_train and len(X_va) > 30000:
            val_probs = np.zeros(len(X_va), dtype=float)
            bs = 15000
            for b_st in range(0, len(X_va), bs):
                b_en = min(b_st + bs, len(X_va))
                val_probs[b_st:b_en] = clf.predict_proba(X_va[b_st:b_en])[:, 1]
        else:
            val_probs = clf.predict_proba(X_va)[:, 1]

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = (val_probs >= 0.5).astype(int)

    train_time = time.time() - t_start

    # Top-1 observation accuracy
    temp_df = pd.DataFrame({"trip_id": groups, "t": timestamps, "prob": oof_probs, "label": y})
    top1_hits = 0
    total_obs = 0
    for _, grp in temp_df.groupby(["trip_id", "t"]):
        total_obs += 1
        max_idx = grp["prob"].idxmax()
        if grp.loc[max_idx, "label"] == 1:
            top1_hits += 1
    top1_acc = top1_hits / total_obs

    macro_f1 = f1_score(y, oof_preds, average="macro")
    p, r, _ = precision_recall_curve(y, oof_probs)
    pr_auc = auc(r, p)
    ll = log_loss(y, np.clip(oof_probs, 1e-6, 1 - 1e-6))

    # Test single observation latency
    t_lat = time.time()
    _ = clf.predict_proba(X_va[:500])
    lat_ms = ((time.time() - t_lat) * 1000) / 500

    print(f"Results for {name}:")
    print(f"  • Macro F1 Score      : {macro_f1:.4f}")
    print(f"  • Observation Top-1   : {top1_acc*100:.2f}%")
    print(f"  • PR-AUC              : {pr_auc:.4f}")
    print(f"  • Log Loss            : {ll:.4f}")
    print(f"  • Inference Latency   : {lat_ms:.3f} ms / observation")
    print(f"  • CV Training Time    : {train_time:.2f} seconds")

    return clf, scaler if use_scaler else None

def main():
    print("=" * 65)
    print("   AKSHAY'S 2-MODEL SUITE: LOGISTIC REGRESSION & k-NN")
    print("   Owner: Akshay (Spatial Preprocessing & Baseline Models)")
    print("=" * 65)

    df = load_data()
    X = df[FEATURE_COLS]
    y = df["label"].values
    groups = df["trip_id"].values
    timestamps = df["t"].values

    # Model 1: Logistic Regression
    clf_lr, scaler_lr = evaluate_model_cv(
        "Model 1: Logistic Regression (L2 Regularized)",
        lambda: LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42),
        X, y, groups, timestamps,
        use_scaler=True
    )
    joblib.dump(clf_lr, OUTPUT_DIR / "model1_logistic_regression.joblib")
    if scaler_lr:
        joblib.dump(scaler_lr, OUTPUT_DIR / "scaler_logistic_regression.joblib")
    print(f"  -> Saved Model 1 to: {OUTPUT_DIR / 'model1_logistic_regression.joblib'}")

    # Model 2: Distance-Weighted k-NN (Prototype Subsampled for Real-Time Memory)
    clf_knn, scaler_knn = evaluate_model_cv(
        "Model 2: Distance-Weighted k-Nearest Neighbors (k=15)",
        lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        X, y, groups, timestamps,
        use_scaler=True,
        max_train=25000
    )
    joblib.dump(clf_knn, OUTPUT_DIR / "model2_knn.joblib")
    if scaler_knn:
        joblib.dump(scaler_knn, OUTPUT_DIR / "scaler_knn.joblib")
    print(f"  -> Saved Model 2 to: {OUTPUT_DIR / 'model2_knn.joblib'}")

    print("\n" + "=" * 65)
    print("AKSHAY'S 2 MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
