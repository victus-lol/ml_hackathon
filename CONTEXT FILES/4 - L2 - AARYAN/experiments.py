"""Run five model families with trip-grouped CV and a held-out student-test comparison."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decoder import bridge_route, continuity, decode
from pathsnap import FEATURES, load_trip


def make_models(seed: int = 42) -> dict:
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)),
        "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=20, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=.08, max_leaf_nodes=23, l2_regularization=.1, random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", alpha=.001, batch_size=1024, learning_rate_init=.001, max_iter=35, early_stopping=False, random_state=seed)),
    }


MODEL_CONFIGS={
    "LogisticRegression":{"family":"linear probabilistic","scaler":"StandardScaler per fit","C":1.0,"class_weight":"balanced","max_iter":300,"objective":"log loss"},
    "GaussianNB":{"family":"probabilistic naive Bayes","scaler":"StandardScaler per fit","var_smoothing":1e-9,"objective":"Gaussian class likelihood"},
    "RandomForest":{"family":"bagged decision trees","n_estimators":60,"max_depth":20,"min_samples_leaf":3,"class_weight":"balanced_subsample","objective":"Gini impurity"},
    "HistGradientBoosting":{"family":"boosted trees","max_iter":80,"learning_rate":0.08,"max_leaf_nodes":23,"l2_regularization":0.1,"objective":"binary log loss"},
    "MLP":{"family":"neural network","hidden_layer_sizes":[32],"activation":"relu","solver":"adam","alpha":0.001,"batch_size":1024,"learning_rate_init":0.001,"max_iter":35,"early_stopping":False,"objective":"binary cross entropy"},
}


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(x))


def downsample_training(frame: pd.DataFrame, seed: int = 42, negatives_per_observation: int = 2) -> pd.DataFrame:
    """Keep every positive and a deterministic sample of negatives for runtime control."""
    keep=frame.label.to_numpy()==1
    neg_mask=~keep
    neg=frame.loc[neg_mask,["trip_id","observation_id"]].copy()
    neg["_position"]=np.flatnonzero(neg_mask)
    rng=np.random.default_rng(seed)
    neg["_random_order"]=rng.random(len(neg))
    rank=neg.groupby(["trip_id","observation_id"],sort=False)["_random_order"].rank(method="first")
    keep[neg["_position"].to_numpy()]=rank.to_numpy()<=negatives_per_observation
    return frame.loc[keep].sort_index()


def metrics(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, full_frame: pd.DataFrame | None = None) -> dict:
    pred = (p >= .5).astype(np.int8)
    groups = frame[["trip_id", "observation_id"]].copy()
    groups["y"] = y
    groups["p"] = p
    hits, misses = [], 0
    for _, g in groups.groupby(["trip_id", "observation_id"], sort=False):
        labels = g.y.to_numpy()
        hits.append(int(labels[int(np.argmax(g.p.to_numpy()))] == 1))
        misses += int(labels.sum() == 0)
    all_frame=full_frame if full_frame is not None else frame
    all_groups=all_frame.groupby(["trip_id","observation_id"],sort=False).label.sum()
    nobs = len(all_groups)
    misses=int((all_groups<0).sum())
    return {
        "candidate_macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "candidate_accuracy": float(accuracy_score(y, pred)),
        "top1_accuracy": float(np.sum(hits)/nobs) if nobs else 0.,
        "top1_accuracy_on_candidate_covered_observations": float(np.mean(hits)) if hits else 0.,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
        "candidate_miss_rate": misses / nobs if nobs else 0.,
        "observation_count": int(nobs),"observations_with_candidate_truth":int(len(hits)),
        "candidate_row_count": int(len(frame)),
    }


def grouped_rows(frame: pd.DataFrame, probs: np.ndarray) -> list[tuple[str, pd.DataFrame, np.ndarray]]:
    tmp = frame.copy()
    tmp["_p"] = probs
    results = []
    for tid, part in tmp.groupby("trip_id", sort=True):
        original_idx = part.index.to_numpy()
        results.append((tid, part.reset_index(drop=True), original_idx))
    return results


def evaluate_routes(frame: pd.DataFrame, probs: np.ndarray, dataset_root: Path, partition: str,
                    beta_m: float=50.0,sigma_m:float=25.0) -> dict:
    output = {}
    for tid, part, _ in grouped_rows(frame, probs):
        data = load_trip(dataset_root / partition / tid, include_route=True)
        chosen_rows = []
        candidate_sets = []
        y_by_obs = []
        for obs, g in part.groupby("observation_id", sort=True):
            rows = g.to_dict("records")
            candidate_sets.append([{
                "arc_id": int(r["candidate_arc_id"]), "p": max(float(r["_p"]), 1e-9),
                "distance_m": float(r["distance_m"]), "fraction": float(r["projection_fraction"]),
            } for r in rows])
            best = rows[int(np.argmax([r["_p"] for r in rows]))]
            chosen_rows.append(int(best["candidate_arc_id"]))
            y_by_obs.append(int(any(int(r["label"]) == 1 for r in rows)))
        # Route quality is computed on observations with at least one candidate row.
        obs_ids = np.array(sorted(part.observation_id.unique()), dtype=int)
        greedy_acc = float(np.mean([chosen_rows[i] == int(_truth_for_obs(data, obs)) for i, obs in enumerate(obs_ids)]))
        greedy_cont = continuity(chosen_rows, data["arcs"])
        viterbi = decode(candidate_sets, data["arcs"], data["lengths"], data["gps_xy"][obs_ids], data["track"][obs_ids, 2],
                         beta_m=beta_m,sigma_m=sigma_m)
        decoded_acc = float(np.mean([viterbi[i] == int(_truth_for_obs(data, obs)) for i, obs in enumerate(obs_ids)]))
        sequence_exact = bool(all(viterbi[i] == int(_truth_for_obs(data, obs)) for i, obs in enumerate(obs_ids)))
        viterbi_cont = continuity(viterbi, data["arcs"])
        bridged = bridge_route(viterbi, data["arcs"], data["lengths"])
        output[tid] = {
            "observations_scored": int(len(obs_ids)),
            "classifier_top1_accuracy": greedy_acc,
            "classifier_route_continuity": greedy_cont,
            "viterbi_top1_accuracy": decoded_acc,
            "viterbi_sequence_exact_match": sequence_exact,
            "viterbi_route_continuity": viterbi_cont,
            "bridged_route_continuity": continuity(bridged, data["arcs"]),
            "viterbi_observation_arc_count": int(len(viterbi)),
            "bridged_route_arc_count": int(len(bridged)),
        }
    return output


def _truth_for_obs(data: dict, obs: int) -> int:
    # Use the same monotonic route alignment used to build candidate labels.
    from pathsnap import _route_alignment
    if "_truth" not in data:
        data["_truth"] = _route_alignment(data)
    return int(data["_truth"][obs])


def run(args):
    train_all = pd.read_csv(args.candidate_dir / "train_candidates.csv.gz",dtype={"trip_id":str})
    test_all = pd.read_csv(args.candidate_dir / "student_test_candidates.csv.gz",dtype={"trip_id":str})
    train = train_all[train_all.label>=0].copy().reset_index(drop=True)
    test = test_all[test_all.label>=0].copy().reset_index(drop=True)
    x, y = train[FEATURES].to_numpy(np.float32), train.label.to_numpy(np.int8)
    groups = train.trip_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    report = {"feature_names": FEATURES, "seed": args.seed, "model_configs":MODEL_CONFIGS,
        "cv_split":"GroupKFold(n_splits=5), grouped by trip_id", "cv": {}, "student_test": {}, "route_diagnostics": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"models").mkdir(exist_ok=True)
    model_defs = make_models(args.seed)
    champion_name = None
    champion_score = -1.
    for name, prototype in model_defs.items():
        print(f"\n=== {name} ===", flush=True)
        oof = np.full(len(train), np.nan, dtype=np.float32)
        fold_scores = []
        fit_s = infer_s = 0.
        for fold, (itr, iva) in enumerate(splitter.split(x, y, groups), 1):
            model = __import__("sklearn.base", fromlist=["clone"]).clone(prototype)
            fit_idx=downsample_training(train.iloc[itr],args.seed+fold).index.to_numpy()
            t0 = time.perf_counter(); model.fit(x[fit_idx], y[fit_idx]); fit_s += time.perf_counter()-t0
            t0 = time.perf_counter(); p = positive_probability(model, x[iva]); infer_s += time.perf_counter()-t0
            oof[iva] = p
            val_rows=train.iloc[iva].reset_index(drop=True)
            val_trips=set(val_rows.trip_id.astype(str))
            val_all=train_all[train_all.trip_id.astype(str).isin(val_trips)]
            score = metrics(val_rows, y[iva], p,full_frame=val_all)
            score["fold"] = fold
            fold_scores.append(score)
            print(f"fold {fold}: macro-F1={score['candidate_macro_f1']:.4f}, top1={score['top1_accuracy']:.4f}", flush=True)
        cv = metrics(train.reset_index(drop=True), y, oof,full_frame=train_all)
        cv["fold_macro_f1_mean"] = float(np.mean([z["candidate_macro_f1"] for z in fold_scores]))
        cv["fold_macro_f1_std"] = float(np.std([z["candidate_macro_f1"] for z in fold_scores]))
        cv["folds"] = fold_scores
        cv["fit_seconds"] = fit_s
        cv["inference_seconds"] = infer_s
        # Calibrate final-model emissions using only out-of-fold training predictions.
        calibrator=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
        final = __import__("sklearn.base", fromlist=["clone"]).clone(prototype)
        fit_idx=downsample_training(train,args.seed).index.to_numpy()
        t0=time.perf_counter(); final.fit(x[fit_idx], y[fit_idx]); cv["final_fit_seconds"] = time.perf_counter()-t0
        cv["sampled_training_rows"] = int(len(fit_idx))
        tx = test[FEATURES].to_numpy(np.float32)
        t0=time.perf_counter(); ptrain_raw=positive_probability(final,x); cv["full_train_inference_seconds"] = time.perf_counter()-t0
        ptrain=calibrator.predict(ptrain_raw)
        train_score=metrics(train.reset_index(drop=True),y,ptrain,full_frame=train_all)
        test_all_x=test_all[FEATURES].to_numpy(np.float32)
        t0=time.perf_counter(); ptest_raw=positive_probability(final, test_all_x); cv["student_test_inference_seconds"] = time.perf_counter()-t0
        ptest_all=calibrator.predict(ptest_raw)
        ptest=ptest_all[test_all.label.to_numpy()>=0]
        tst = metrics(test.reset_index(drop=True), test.label.to_numpy(np.int8), ptest,full_frame=test_all)
        tst["inference_seconds"] = cv["student_test_inference_seconds"]
        route_diag = None
        report["cv"][name] = cv
        report.setdefault("train_fit",{})[name]=train_score
        report["student_test"][name] = tst
        report["route_diagnostics"][name] = {}
        model_bundle={"model":final,"calibrator":calibrator,"features":FEATURES,"model_name":name,
                      "calibrator_method":"IsotonicRegression fitted on training-only OOF predictions"}
        joblib.dump(model_bundle,args.output_dir/"models"/f"{name.lower()}.joblib")
        if cv["fold_macro_f1_mean"] > champion_score:
            champion_name, champion_score = name, cv["fold_macro_f1_mean"]
            joblib.dump(model_bundle, args.output_dir / "final_model.joblib")
        with open(args.output_dir / f"{name.lower()}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"cv": cv, "train_fit":train_score, "student_test": tst, "route_diagnostics": route_diag}, f, indent=2)
        if name == "MLP":
            estimator=final.steps[-1][1]
            pd.DataFrame({"epoch":np.arange(1,len(estimator.loss_curve_)+1),"training_loss":estimator.loss_curve_}).to_csv(args.output_dir/"mlp_training_loss.csv",index=False)
            fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(estimator.loss_curve_)+1),estimator.loss_curve_)
            ax.set_title("MLP training loss (actual fitted model)");ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.grid(alpha=.25)
            fig.tight_layout();fig.savefig(args.output_dir/"mlp_training_loss.png",dpi=160);plt.close(fig)
    report["selected_model_by_cv_macro_f1"] = champion_name
    hmm_rows=[]
    selected_bundle=joblib.load(args.output_dir/"final_model.joblib")
    test_all_x=test_all[FEATURES].to_numpy(np.float32)
    selected_p=positive_probability(selected_bundle["model"],test_all_x)
    if selected_bundle.get("calibrator") is not None:
        selected_p=selected_bundle["calibrator"].predict(selected_p)
    selected_route_diag=evaluate_routes(test_all.reset_index(drop=True),selected_p,args.dataset_root,"student_test")
    report["route_diagnostics"]={champion_name:selected_route_diag}
    for model_name,by_trip in report["route_diagnostics"].items():
        values=list(by_trip.values())
        hmm_rows.append({"model":model_name,
            "classifier_top1_mean":float(np.mean([v["classifier_top1_accuracy"] for v in values])),
            "viterbi_top1_mean":float(np.mean([v["viterbi_top1_accuracy"] for v in values])),
            "classifier_continuity_mean":float(np.mean([v["classifier_route_continuity"] for v in values])),
            "viterbi_continuity_mean":float(np.mean([v["viterbi_route_continuity"] for v in values])),
            "bridged_continuity_mean":float(np.mean([v["bridged_route_continuity"] for v in values])),
            "viterbi_sequence_exact_fraction":float(np.mean([v["viterbi_sequence_exact_match"] for v in values]))})
    report["hmm_comparison"]=hmm_rows
    pd.DataFrame(hmm_rows).to_csv(args.output_dir/"hmm_comparison.csv",index=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [{"model": n, "cv_macro_f1": report["cv"][n]["fold_macro_f1_mean"],
             "student_test_top1": report["student_test"][n]["top1_accuracy"],
             "student_test_macro_f1": report["student_test"][n]["candidate_macro_f1"],
             "student_test_log_loss": report["student_test"][n]["log_loss"],
             "inference_seconds": report["student_test"][n]["inference_seconds"]} for n in model_defs]
    pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
    _plots(pd.DataFrame(rows), args.output_dir)
    hdf=pd.DataFrame(hmm_rows)
    fig,ax=plt.subplots(figsize=(9,4.5));xloc=np.arange(len(hdf));width=.25
    ax.bar(xloc-width,hdf.classifier_top1_mean,width,label="Classifier Top-1")
    ax.bar(xloc,hdf.viterbi_top1_mean,width,label="Viterbi Top-1")
    ax.bar(xloc+width,hdf.bridged_continuity_mean,width,label="Bridged continuity")
    ax.set_xticks(xloc,hdf.model,rotation=20);ax.set_ylim(0,1);ax.set_ylabel("Score");ax.set_title("Student-test route decoding comparison (trip means)")
    ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(args.output_dir/"hmm_route_comparison.png",dpi=160);plt.close(fig)
    print(f"Selected by 5-fold grouped CV Macro-F1: {champion_name}; outputs: {args.output_dir}")


def _plots(df: pd.DataFrame, out: Path):
    for metric, title, filename in [
        ("cv_macro_f1", "Five-fold GroupKFold Macro F1", "cv_macro_f1.png"),
        ("student_test_top1", "Student-test observation Top-1 accuracy", "student_test_top1.png"),
        ("student_test_macro_f1", "Student-test candidate Macro F1", "student_test_macro_f1.png"),
        ("student_test_log_loss", "Student-test log loss", "student_test_log_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(df.model, df[metric], color="#3178a8")
        ax.set_title(title); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42); run(p.parse_args())


if __name__ == "__main__": main()
