"""
CONTEXT FILES/2 - ADITYA/baseline_models.py
------------------------------------------
Aditya's Machine Learning Models Suite:
- Model 3: Random Forest Classifier (Class-Weighted Ensemble)
- Model 4: LightGBM / Gradient Boosting Classifier (scale_pos_weight tuned)
Includes Probability Calibration (Isotonic / Platt) for downstream HMM emission scores.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def build_random_forest_model(
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_split: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Model 3: Random Forest Classifier with balanced class weights
    to counter the 7:1 negative-to-positive candidate class imbalance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def build_lightgbm_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 8,
    scale_pos_weight: float = 5.0,
    random_state: int = 42,
) -> Any:
    """
    Model 4: LightGBM Gradient Boosting Classifier.
    Falls back to Sklearn GradientBoostingClassifier if lightgbm is not yet installed.
    """
    if HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        print("[!] LightGBM not installed in environment; using Sklearn GradientBoosting fallback.")
        return GradientBoostingClassifier(
            n_estimators=min(n_estimators, 150),
            learning_rate=learning_rate,
            max_depth=min(max_depth, 6),
            random_state=random_state,
        )


def calibrate_and_save_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_path: str = "best_model.joblib",
) -> CalibratedClassifierCV:
    """
    Applies Isotonic Regression calibration to ensure predicted probabilities
    reliably represent true spatial match confidence for Abhiroop's HMM emission layer.
    """
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(calibrated, out_path)
    print(f"[+] Calibrated model saved successfully to: {out_path}")
    return calibrated


if __name__ == "__main__":
    rf = build_random_forest_model()
    gb = build_lightgbm_model()
    print("[+] Model 3 (Random Forest) and Model 4 (Gradient Boosting) initialised successfully.")
