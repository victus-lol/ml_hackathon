"""
CONTEXT FILES/4 - L2 - AARYAN/train_mlp.py
-----------------------------------------
Aaryan's Machine Learning Model:
Model 5: Multi-Layer Perceptron (MLP Neural Network, Architecture: 128-64-32)
Evaluates under 5-Fold GroupKFold strictly grouped by trip_id.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_curve, auc, log_loss

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
        alt = Path("c:/Users/adity/Downloads/ML HACKATHON/ml_hackathon/data/train_candidates.parquet")
        if alt.exists():
            return pd.read_parquet(alt)
        raise FileNotFoundError(f"Could not locate train_candidates.parquet at {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)

def train_and_eval_mlp():
    print("=" * 65)
    print("      AARYAN'S MACHINE LEARNING MODEL: MULTI-LAYER PERCEPTRON (MLP)")
    print("      Model 5: Deep Neural Network (Architecture: 128 -> 64 -> 32)")
    print("=" * 65)

    df = load_data()
    X = df[FEATURE_COLS]
    y = df["label"].values
    groups = df["trip_id"].values
    timestamps = df["t"].values

    print(f"Dataset Loaded: {len(df):,} candidate records across {df['trip_id'].nunique()} trips.")

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds = np.zeros(len(y), dtype=int)

    t_start = time.time()
    max_train = 80000  # Subsample for fast cross-validation convergence

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        if len(train_idx) > max_train:
            np.random.seed(42 + fold)
            train_idx = np.random.choice(train_idx, size=max_train, replace=False)

        X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
        X_va, y_val = X.iloc[val_idx].values, y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=30,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42 + fold
        )
        mlp.fit(X_tr, y_tr)

        # Predict in chunks
        val_probs = np.zeros(len(X_va), dtype=float)
        bs = 20000
        for b_st in range(0, len(X_va), bs):
            b_en = min(b_st + bs, len(X_va))
            val_probs[b_st:b_en] = mlp.predict_proba(X_va[b_st:b_en])[:, 1]

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = (val_probs >= 0.5).astype(int)

    train_time = time.time() - t_start

    # Top-1 Accuracy
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

    # Latency test
    t_lat = time.time()
    _ = mlp.predict_proba(X_va[:500])
    lat_ms = ((time.time() - t_lat) * 1000) / 500

    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS (5-FOLD CV)")
    print("=" * 65)
    print(f"  • Model Family        : Model 5: Multi-Layer Perceptron (MLP)")
    print(f"  • Hidden Architecture : (128, 64, 32) Fully Connected ReLU Layers")
    print(f"  • Macro F1 Score      : {macro_f1:.4f}")
    print(f"  • Observation Top-1   : {top1_acc*100:.2f}%")
    print(f"  • PR-AUC              : {pr_auc:.4f}")
    print(f"  • Log Loss            : {ll:.4f}")
    print(f"  • Inference Latency   : {lat_ms:.3f} ms / observation")
    print(f"  • Total Training Time : {train_time:.2f} seconds")
    print("=" * 65)

    joblib.dump(mlp, OUTPUT_DIR / "model5_mlp.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler_mlp.joblib")
    print(f"Saved MLP model artifact: {OUTPUT_DIR / 'model5_mlp.joblib'}")

if __name__ == "__main__":
    train_and_eval_mlp()
