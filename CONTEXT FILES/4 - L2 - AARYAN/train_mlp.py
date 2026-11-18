"""Train and evaluate Aaryan's MLP on Akshay's prepared candidate table."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {"trip_id", "observation_id", "label"}
BANNED_FEATURE_NAMES = {
    "trip_id", "observation_id", "point_id", "candidate_id", "arc_id", "candidate_arc_id",
    "from_node", "to_node", "u", "v", "lon", "lat", "time", "timestamp",
    "longitude", "latitude",
}


def make_model(seed: int, hidden: tuple[int, ...], max_iter: int) -> Pipeline:
    """Keep scaling inside the estimator so each fold fits it on training trips only."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=seed,
            verbose=False,
        )),
    ])


def score_predictions(frame: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(np.int8)
    grouped = pd.DataFrame({
        "trip_id": frame["trip_id"].to_numpy(),
        "observation_id": frame["observation_id"].to_numpy(),
        "label": labels,
        "probability": probabilities,
    })
    top1_hits: list[int] = []
    missed_observations = 0
    observation_count = 0
    for _, group in grouped.groupby(["trip_id", "observation_id"], sort=False):
        observation_count += 1
        best_index = group["probability"].to_numpy().argmax()
        group_labels = group["label"].to_numpy()
        top1_hits.append(int(group_labels[best_index] == 1))
        if group_labels.sum() == 0:
            missed_observations += 1

    return {
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "log_loss": float(log_loss(labels, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "candidate_miss_rate": missed_observations / observation_count if observation_count else 0.0,
        "observation_count": int(observation_count),
    }


def load_candidate_table(path: Path, training_root: Path, feature_names: list[str]) -> pd.DataFrame:
    if not training_root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {training_root}")
    frame = pd.read_csv(path, dtype={"trip_id": "string"})
    missing = REQUIRED_COLUMNS.union(feature_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if len(feature_names) != 14:
        raise ValueError("The agreed candidate feature list must contain exactly 14 columns.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature column names must be unique.")
    bad_names = [name for name in feature_names if name.lower() in BANNED_FEATURE_NAMES]
    if bad_names:
        raise ValueError(f"Identifiers or raw coordinate/time fields cannot be model features: {bad_names}")
    if frame["label"].isna().any() or not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0/1 values and no missing entries.")
    if frame[feature_names].isna().any().any():
        raise ValueError("Feature columns contain missing values; resolve them in the training pipeline first.")
    if not np.isfinite(frame[feature_names].to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature columns contain non-finite values.")
    expected_trip_ids = {
        folder.name
        for folder in training_root.iterdir()
        if folder.is_dir() and folder.name.isdigit() and len(folder.name) == 8
        and all((folder / f"{folder.name}{suffix}").is_file() for suffix in (".track", ".nodes", ".arcs"))
    }
    actual_trip_ids = set(frame["trip_id"].astype(str).unique())
    if len(expected_trip_ids) != 80:
        raise ValueError(f"Training directory must contain the 80 training trips; found {len(expected_trip_ids)}.")
    if actual_trip_ids != expected_trip_ids:
        raise ValueError("Candidate CSV trip IDs must exactly match the supplied 80-trip training directory.")
    return frame


def run_cv(frame: pd.DataFrame, feature_names: list[str], seed: int, hidden: tuple[int, ...], max_iter: int) -> tuple[dict, np.ndarray]:
    x_values = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
    y_values = frame["label"].to_numpy(dtype=np.int8, copy=True)
    groups = frame["trip_id"].to_numpy()
    oof_probabilities = np.full(len(frame), np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []
    total_fit_seconds = 0.0
    total_inference_seconds = 0.0

    splitter = GroupKFold(n_splits=5)
    for fold_number, (train_indices, valid_indices) in enumerate(splitter.split(x_values, y_values, groups), start=1):
        model = make_model(seed + fold_number - 1, hidden, max_iter)
        start = time.perf_counter()
        model.fit(x_values[train_indices], y_values[train_indices])
        total_fit_seconds += time.perf_counter() - start

        start = time.perf_counter()
        probabilities = model.predict_proba(x_values[valid_indices])[:, 1]
        total_inference_seconds += time.perf_counter() - start
        oof_probabilities[valid_indices] = probabilities
        fold_score = score_predictions(frame.iloc[valid_indices], y_values[valid_indices], probabilities)
        fold_score["fold"] = fold_number
        fold_metrics.append(fold_score)
        print(f"Fold {fold_number}/5: macro_f1={fold_score['macro_f1']:.4f}, top1={fold_score['top1_accuracy']:.4f}")

    overall = score_predictions(frame, y_values, oof_probabilities)
    overall["fold_macro_f1_mean"] = float(np.mean([item["macro_f1"] for item in fold_metrics]))
    overall["fold_macro_f1_std"] = float(np.std([item["macro_f1"] for item in fold_metrics]))
    overall["cv_fit_seconds"] = total_fit_seconds
    overall["cv_inference_seconds"] = total_inference_seconds
    overall["folds"] = fold_metrics
    return overall, oof_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MLP using a prepared candidate feature CSV.")
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--training-root", required=True, type=Path, help="Folder containing the 80 training trip directories.")
    parser.add_argument("--features", required=True, help="Comma-separated, agreed 14 feature column names from Akshay.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", default="64,32", help="Hidden layer widths; default is 64,32.")
    parser.add_argument("--max-iter", type=int, default=150)
    args = parser.parse_args()

    feature_names = [name.strip() for name in args.features.split(",") if name.strip()]
    hidden = tuple(int(width) for width in args.hidden.split(",") if width.strip())
    if not hidden or args.max_iter < 1:
        raise ValueError("Provide positive hidden layer widths and max-iter.")

    frame = load_candidate_table(args.candidate_csv, args.training_root, feature_names)
    report, oof_probabilities = run_cv(frame, feature_names, args.seed, hidden, args.max_iter)

    final_model = make_model(args.seed, hidden, args.max_iter)
    x_values = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
    y_values = frame["label"].to_numpy(dtype=np.int8, copy=True)
    start = time.perf_counter()
    final_model.fit(x_values, y_values)
    report["final_fit_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    inference_rows = min(2048, len(x_values))
    final_model.predict_proba(x_values[:inference_rows])
    inference_seconds = time.perf_counter() - start
    report["final_inference_seconds"] = inference_seconds
    report["final_inference_rows"] = inference_rows
    report["final_inference_ms_per_row"] = inference_seconds * 1000.0 / inference_rows
    report["seed"] = args.seed
    report["features"] = feature_names
    report["hidden_layers"] = list(hidden)
    report["max_iter"] = args.max_iter
    report["training_trip_count"] = int(frame["trip_id"].nunique())
    report["candidate_row_count"] = int(len(frame))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.output_dir / "aaryan_mlp.joblib")
    out_of_fold = frame[["trip_id", "observation_id"]].copy()
    if "candidate_arc_id" in frame.columns:
        out_of_fold["candidate_arc_id"] = frame["candidate_arc_id"]
    out_of_fold["label"] = y_values
    out_of_fold["oof_probability"] = oof_probabilities
    out_of_fold.to_csv(args.output_dir / "aaryan_mlp_oof.csv", index=False)
    (args.output_dir / "aaryan_mlp_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model and real CV outputs under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
