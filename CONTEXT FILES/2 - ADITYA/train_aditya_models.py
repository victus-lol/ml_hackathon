"""
CONTEXT FILES/2 - ADITYA/train_aditya_models.py
----------------------------------------------
Aditya's Machine Learning Suite (Team Lead):
- Model 3: Random Forest Classifier (Bagging Ensemble of 100 Decision Trees)
- Model 4: LightGBM Gradient Boosted Decision Trees (Optuna Bayesian Tuned Champion)
Evaluates under 5-Fold GroupKFold strictly partitioned by trip_id.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
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

def evaluate_model_cv(name, model_fn, X, y, groups, timestamps, max_train=None):
    print("=" * 65)
    print(f"  EVALUATING {name}")
    print("=" * 65)

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds = np.zeros(len(y), dtype=int)

    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        if max_train and len(train_idx) > max_train:
            np.random.seed(42 + fold)
            train_idx = np.random.choice(train_idx, size=max_train, replace=False)

        X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
        X_va, y_val = X.iloc[val_idx].values, y[val_idx]

        clf = model_fn()
        clf.fit(X_tr, y_tr)

        if max_train and len(X_va) > 50000:
            val_probs = np.zeros(len(X_va), dtype=float)
            bs = 20000
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

    # Latency test
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

    return clf

def main():
    print("=" * 65)
    print("   ADITYA'S 2-MODEL SUITE: RANDOM FOREST & OPTUNA LIGHTGBM")
    print("   Owner: Aditya Mule (Team Lead / ML Comparative Arena)")
    print("=" * 65)

    df = load_data()
    X = df[FEATURE_COLS]
    y = df["label"].values
    groups = df["trip_id"].values
    timestamps = df["t"].values

    # Model 3: Random Forest (Bagging)
    clf_rf = evaluate_model_cv(
        "Model 3: Random Forest Classifier (Bagging - 80 Trees)",
        lambda: RandomForestClassifier(n_estimators=80, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        X, y, groups, timestamps,
        max_train=120000
    )
    joblib.dump(clf_rf, OUTPUT_DIR / "model3_random_forest.joblib")
    print(f"  -> Saved Model 3 to: {OUTPUT_DIR / 'model3_random_forest.joblib'}")

    # Model 4: LightGBM GBDT (Optuna-Tuned Champion)
    best_params = {
        "n_estimators": 250,
        "learning_rate": 0.0384,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 45,
        "subsample": 0.824,
        "colsample_bytree": 0.781,
        "reg_alpha": 0.152,
        "reg_lambda": 1.241,
        "scale_pos_weight": 4.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    clf_lgb = evaluate_model_cv(
        "Model 4: LightGBM (Optuna-Tuned Champion GBDT)",
        lambda: lgb.LGBMClassifier(**best_params),
        X, y, groups, timestamps
    )
    joblib.dump(clf_lgb, OUTPUT_DIR / "best_model_lightgbm.joblib")
    joblib.dump(clf_lgb, REPO_ROOT / "models" / "best_model_lightgbm.joblib")
    print(f"  -> Saved Champion Model 4 to: {OUTPUT_DIR / 'best_model_lightgbm.joblib'}")

    print("\n" + "=" * 65)
    print("ADITYA'S 2 MODELS (RANDOM FOREST & CHAMPION LIGHTGBM) READY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
