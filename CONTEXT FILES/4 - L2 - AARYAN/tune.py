"""Optuna TPE Bayesian tuning on training-trip-only GroupKFold CV."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import downsample_training, make_models, metrics, positive_probability
from pathsnap import FEATURES


def trial_model(name: str, trial: optuna.Trial, seed: int):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 240, step=40),
            max_depth=trial.suggest_int("max_depth", 10, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            n_jobs=-1, random_state=seed,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 60, 200, step=20),
            learning_rate=trial.suggest_float("learning_rate", 0.025, 0.2, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63, step=8),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 2.0, log=True),
            random_state=seed,
        )
    if name == "GaussianNB":
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=trial.suggest_float("var_smoothing",1e-12,1e-7,log=True)))
    if name == "MLP":
        hidden=trial.suggest_categorical("hidden_layer_sizes",[(16,),(32,),(64,),(64,32)])
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=hidden,
            alpha=trial.suggest_float("alpha",1e-5,1e-1,log=True),batch_size=1024,
            learning_rate_init=trial.suggest_float("learning_rate_init",1e-4,5e-3,log=True),
            max_iter=35,early_stopping=False,random_state=seed))
    if name == "LogisticRegression":
        # Keep preprocessing inside every CV fold.
        if name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            return make_pipeline(StandardScaler(), LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 1e2, log=True), max_iter=400,
                class_weight="balanced", random_state=seed))
    raise ValueError(f"No tuning search space for {name}")


def run(args):
    train_all = pd.read_csv(args.candidate_dir / "train_candidates.csv.gz",dtype={"trip_id":str})
    test_all = pd.read_csv(args.candidate_dir / "student_test_candidates.csv.gz",dtype={"trip_id":str})
    train = train_all[train_all.label>=0].copy().reset_index(drop=True)
    test = test_all[test_all.label>=0].copy().reset_index(drop=True)
    scores = json.loads((args.experiment_dir / "comparison.json").read_text(encoding="utf-8"))
    name = scores["selected_model_by_cv_macro_f1"]
    x, y = train[FEATURES].to_numpy(np.float32), train.label.to_numpy(np.int8)
    groups = train.trip_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objective_name=getattr(args,"objective","macro_f1")
    fold_specs=[]
    for fold,(itr,iva) in enumerate(splitter.split(x,y,groups),1):
        fit_idx=downsample_training(train.iloc[itr],args.seed+fold).index.to_numpy()
        fold_specs.append((itr,iva,fit_idx))
        print(f"Prepared grouped fold {fold}/5: fit_rows={len(fit_idx)}, validation_rows={len(iva)}",flush=True)

    def objective(trial):
        model = trial_model(name, trial, args.seed)
        fold_scores = []
        for fold, (itr, iva, fit_idx) in enumerate(fold_specs, 1):
            current = clone(model)
            current.fit(x[fit_idx], y[fit_idx])
            p = positive_probability(current, x[iva])
            macro=float(f1_score(y[iva],(p>=.5).astype(np.int8),average="macro",labels=[0,1],zero_division=0))
            val=train.iloc[iva].copy();val["_p"]=p
            best_rows=val.loc[val.groupby(["trip_id","observation_id"],sort=False)["_p"].idxmax()]
            val_trips=set(val.trip_id.astype(str)); total_obs=train_all[train_all.trip_id.astype(str).isin(val_trips)].groupby(["trip_id","observation_id"]).ngroups
            top1=float((best_rows.label==1).sum()/total_obs) if total_obs else 0.
            score=macro if objective_name=="macro_f1" else top1
            fold_scores.append(score)
            trial.set_user_attr(f"fold_{fold}_macro_f1",macro);trial.set_user_attr(f"fold_{fold}_top1",top1)
        trial.set_user_attr("cv_objective_std", float(np.std(fold_scores)))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(8, args.trials))
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"PathSnap_{name}")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(args.output_dir / "trials.csv", index=False)
    params = study.best_params
    (args.output_dir / "best_params.json").write_text(json.dumps({
        "selected_model": name, "best_objective_score": study.best_value,
        "best_value_cv_macro_f1": study.best_value if objective_name=="macro_f1" else None,
        "best_params": params, "trial_count": len(study.trials),
        "objective": f"mean {objective_name} over five GroupKFold folds grouped by trip_id",
        "seed": args.seed,
        "search_ranges": {
            "RandomForest": {"n_estimators": "80..240 step 40", "max_depth": "10..40",
                "min_samples_leaf": "1..8", "max_features": ["sqrt", .5, .8]},
            "HistGradientBoosting": {"max_iter": "60..200 step 20", "learning_rate": "log-uniform 0.025..0.2",
                "max_leaf_nodes": "15..63 step 8", "l2_regularization": "log-uniform 1e-4..2"},
            "LogisticRegression": {"C": "log-uniform 1e-3..1e2"},
            "GaussianNB": {"var_smoothing":"log-uniform 1e-12..1e-7"},
            "MLP":{"hidden_layer_sizes":[[16],[32],[64],[64,32]],"alpha":"log-uniform 1e-5..1e-1","learning_rate_init":"log-uniform 1e-4..5e-3"},
        }[name]
    }, indent=2), encoding="utf-8")
    try:
        importances = optuna.importance.get_param_importances(study)
        (args.output_dir / "parameter_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")
    except Exception as e:
        (args.output_dir / "parameter_importance_unavailable.txt").write_text(str(e), encoding="utf-8")
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        ax = plot_optimization_history(study); ax.figure.tight_layout(); ax.figure.savefig(args.output_dir / "optimization_history.png", dpi=160); plt.close(ax.figure)
        ax = plot_param_importances(study); ax.figure.tight_layout(); ax.figure.savefig(args.output_dir / "parameter_importance.png", dpi=160); plt.close(ax.figure)
    except Exception as e:
        (args.output_dir / "optuna_plot_notice.txt").write_text(str(e), encoding="utf-8")

    best_trial = optuna.trial.FixedTrial(params)
    tuned_proto = trial_model(name, best_trial, args.seed)
    tuned_oof=np.full(len(train),np.nan,dtype=np.float32);tuned_fold_results=[]
    for fold,(itr,iva,fit_idx) in enumerate(fold_specs,1):
        current=clone(tuned_proto)
        current.fit(x[fit_idx],y[fit_idx])
        fold_p=positive_probability(current,x[iva]);tuned_oof[iva]=fold_p
        val=train.iloc[iva].reset_index(drop=True);val_trips=set(val.trip_id.astype(str))
        val_all=train_all[train_all.trip_id.astype(str).isin(val_trips)]
        tuned_fold_results.append(metrics(val,y[iva],fold_p,full_frame=val_all))
    tuned_calibrator=IsotonicRegression(out_of_bounds="clip").fit(tuned_oof,y)
    tuned = clone(tuned_proto)
    fit_idx=downsample_training(train,args.seed).index.to_numpy()
    start=time.perf_counter(); tuned.fit(x[fit_idx],y[fit_idx]); fit_s=time.perf_counter()-start
    tx=test_all[FEATURES].to_numpy(np.float32)
    start=time.perf_counter(); ptest_all=tuned_calibrator.predict(positive_probability(tuned,tx)); infer_s=time.perf_counter()-start
    baseline_bundle=joblib.load(args.experiment_dir/"models"/f"{name.lower()}.joblib")
    baseline=baseline_bundle["model"]
    baseline_fit_s=scores["cv"][name]["final_fit_seconds"]
    start=time.perf_counter(); pbase_all=positive_probability(baseline,tx); pbase_all=baseline_bundle["calibrator"].predict(pbase_all); baseline_infer_s=time.perf_counter()-start
    valid_mask=test_all.label.to_numpy()>=0
    base_metrics=metrics(test.reset_index(drop=True),test.label.to_numpy(np.int8),pbase_all[valid_mask],full_frame=test_all)
    tuned_metrics=metrics(test.reset_index(drop=True),test.label.to_numpy(np.int8),ptest_all[valid_mask],full_frame=test_all)
    comparison={"selected_model":name,"optimization_objective":objective_name,
        "untuned_cv_macro_f1":scores["cv"][name]["fold_macro_f1_mean"],
        "untuned_cv_top1":scores["cv"][name]["top1_accuracy"],
        "tuned_cv_macro_f1":float(np.mean([q["candidate_macro_f1"] for q in tuned_fold_results])),
        "tuned_cv_top1":float(np.mean([q["top1_accuracy"] for q in tuned_fold_results])),
        "tuned_cv_objective_score":study.best_value,"untuned_student_test":base_metrics,
        "tuned_student_test":tuned_metrics,"untuned_fit_seconds":baseline_fit_s,
        "tuned_fit_seconds":fit_s,"untuned_inference_seconds":baseline_infer_s,
        "tuned_inference_seconds":infer_s,"tuning_trial_count":len(study.trials),
        "same_partitions":True}
    (args.output_dir/"tuned_comparison.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    joblib.dump({"model":tuned,"calibrator":tuned_calibrator,"features":FEATURES,"model_name":name,
                 "calibrator_method":"IsotonicRegression fitted on training-only OOF predictions",
                 "tuned_params":params},args.output_dir/"tuned_model.joblib")
    print(json.dumps(comparison,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--candidate-dir",type=Path,required=True)
    p.add_argument("--experiment-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--trials",type=int,default=25); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--objective",choices=["macro_f1","top1"],default="macro_f1")
    run(p.parse_args())


if __name__ == "__main__": main()
