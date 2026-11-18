"""
CONTEXT FILES/2 - ADITYA/evaluate.py
-----------------------------------
Authoritative Zero-Leakage 5-Fold GroupKFold Evaluation Harness for PathSnap.
Strictly enforced for Teammate 2 (Aditya).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

# Banned columns that leak arbitrary local graph IDs or raw coordinates
BANNED_COLUMNS = ["trip_id", "arc_id", "from_node", "to_node", "lon", "lat", "time", "u", "v"]


def verify_feature_matrix_invariance(X: pd.DataFrame) -> None:
    """
    Check 1 Lead Audit:
    Assert that no local graph IDs, trip IDs, or raw degree coordinates exist in X.
    Prevents decision trees and MLPs from memorizing local subgraph IDs.
    """
    offending_cols = [c for c in X.columns if any(banned == c.lower() for banned in BANNED_COLUMNS)]
    if offending_cols:
        raise ValueError(
            f"[CRITICAL RULE VIOLATION] Feature matrix contains non-invariant local columns: {offending_cols}. "
            "Strip all IDs and raw coordinates to guarantee zero test subgraph memorization!"
        )


def compute_pathsnap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    candidate_trip_ids: np.ndarray,
    candidate_timestamps: np.ndarray,
) -> Dict[str, float]:
    """
    Computes official PathSnap evaluation metrics:
    - Macro F1-Score
    - Observation-Level Top-1 Accuracy (timestamp hit rate)
    - Precision & Recall
    - Log Loss (clipped for numerical stability)
    """
    # 1. Classification Metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Clip probabilities for stable log-loss
    clipped_probs = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    loss = log_loss(y_true, clipped_probs)

    # 2. Observation-Level Top-1 Hit Rate
    # For every unique (trip_id, timestamp), pick candidate with highest predicted probability
    df_eval = pd.DataFrame({
        "trip_id": candidate_trip_ids,
        "timestamp": candidate_timestamps,
        "y_true": y_true,
        "y_prob": y_prob,
    })

    # Group by observation and verify if the candidate with max probability has y_true == 1
    def is_top1_hit(group: pd.DataFrame) -> int:
        idx_max = group["y_prob"].idxmax()
        return int(group.loc[idx_max, "y_true"] == 1)

    obs_hits = df_eval.groupby(["trip_id", "timestamp"]).apply(is_top1_hit)
    top1_acc = obs_hits.mean() if len(obs_hits) > 0 else 0.0

    return {
        "macro_f1": round(float(macro_f1), 4),
        "top1_accuracy": round(float(top1_acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "log_loss": round(float(loss), 4),
    }


def get_group_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "trip_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Enforces strict zero-leakage GroupKFold partitioning by trip_id.
    Ensures no GPS observations from the same vehicle trip span train and validation folds.
    """
    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' missing from dataset!")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    return list(gkf.split(df, groups=groups))


if __name__ == "__main__":
    print("[*] PathSnap GroupKFold Evaluation Harness loaded successfully.")
    # Quick self-test with synthetic data
    sample_X = pd.DataFrame({
        "dist_ortho_m": [12.4, 3.2, 45.1],
        "heading_delta_rad": [0.1, 0.4, 2.1],
        "speed_mps": [8.5, 8.5, 8.5],
    })
    verify_feature_matrix_invariance(sample_X)
    print("[+] Feature invariance check passed: Zero banned columns detected.")
