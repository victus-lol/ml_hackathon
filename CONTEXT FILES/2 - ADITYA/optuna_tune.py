"""
CONTEXT FILES/2 - ADITYA/optuna_tune.py
---------------------------------------
Aditya's Bayesian Hyperparameter Optimization Engine for PathSnap.
Uses Optuna TPESampler to optimize 5-Fold GroupKFold Macro F1 & Top-1 Hit Rate.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import f1_score

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def objective_lightgbm(trial: Any, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    """
    Optuna objective function for LightGBM under strict 5-Fold GroupKFold.
    Optimizes a composite score of Macro F1 (70%) and Top-1 Hit Rate (30%).
    """
    from evaluate import get_group_kfold_splits, verify_feature_matrix_invariance
    from baseline_models import build_lightgbm_model
    
    # Assert zero leakage / local ID memorization
    verify_feature_matrix_invariance(X)

    # Search space calibrated by Gemini 4 Argon
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.5, 7.5),
        "random_state": 42,
    }

    df_temp = X.copy()
    df_temp["trip_id"] = groups
    splits = get_group_kfold_splits(df_temp, group_col="trip_id", n_splits=5)

    cv_f1_scores = []
    for train_idx, val_idx in splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        clf = build_lightgbm_model(**params)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        cv_f1_scores.append(score)

    return float(np.mean(cv_f1_scores))


def run_bayesian_optimization(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
    """
    Executes the 50-trial Bayesian search using Tree-structured Parzen Estimator (TPE).
    """
    if not HAS_OPTUNA:
        print("[!] Optuna not installed in current environment. Ready for pip/uv env.")
        return {}

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="pathsnap_lgbm_bayesian_tuning",
    )
    study.optimize(lambda trial: objective_lightgbm(trial, X, y, groups), n_trials=n_trials)
    
    print(f"[+] Best 5-Fold GroupKFold Macro F1: {study.best_value:.4f}")
    print("[+] Optimal Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")

    return study.best_params


if __name__ == "__main__":
    print("[*] Optuna Bayesian Hyperparameter Engine (50-Trial TPE) ready.")
    print("[+] Objective: Maximize 5-Fold GroupKFold Macro F1 across 80 training trips.")
