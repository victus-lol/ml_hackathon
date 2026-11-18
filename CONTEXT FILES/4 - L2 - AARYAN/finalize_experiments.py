"""Assemble comparison outputs from saved real model runs without retraining."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import MODEL_CONFIGS,evaluate_routes, positive_probability
from pathsnap import FEATURES


def finalize(dataset_root:Path,candidate_dir:Path,out:Path,model_path:Path|None=None,beta_m:float=50.0,sigma_m:float=25.0):
    names=["LogisticRegression","GaussianNB","RandomForest","HistGradientBoosting","MLP"]
    cv={};train_fit={};test={};configs={}
    for name in names:
        result=json.loads((out/f"{name.lower()}_metrics.json").read_text(encoding="utf-8"))
        cv[name]=result["cv"];train_fit[name]=result["train_fit"];test[name]=result["student_test"]
        configs[name]=MODEL_CONFIGS[name]
    champion=max(names,key=lambda n:cv[n]["fold_macro_f1_mean"])
    bundle=joblib.load(model_path if model_path else out/"models"/f"{champion.lower()}.joblib")
    joblib.dump(bundle,out/"final_model.joblib")
    all_test=pd.read_csv(candidate_dir/"student_test_candidates.csv.gz",dtype={"trip_id":str})
    x=all_test[FEATURES].to_numpy(np.float32)
    p=positive_probability(bundle["model"],x)
    if bundle.get("calibrator") is not None:p=bundle["calibrator"].predict(p)
    route=evaluate_routes(all_test,p,dataset_root,"student_test",beta_m=beta_m,sigma_m=sigma_m)
    hmm=[]
    vals=list(route.values())
    hmm.append({"model":champion,
        "classifier_top1_mean":float(np.mean([v["classifier_top1_accuracy"] for v in vals])),
        "viterbi_top1_mean":float(np.mean([v["viterbi_top1_accuracy"] for v in vals])),
        "classifier_continuity_mean":float(np.mean([v["classifier_route_continuity"] for v in vals])),
        "viterbi_continuity_mean":float(np.mean([v["viterbi_route_continuity"] for v in vals])),
        "bridged_continuity_mean":float(np.mean([v["bridged_route_continuity"] for v in vals])),
        "viterbi_sequence_exact_fraction":float(np.mean([v["viterbi_sequence_exact_match"] for v in vals]))})
    report={"feature_names":FEATURES,"seed":42,"model_configs":configs,
        "cv_split":"GroupKFold(n_splits=5), grouped by trip_id",
        "probability_calibration":"IsotonicRegression fitted on training-only OOF probabilities; CV model-selection scores remain uncalibrated OOF scores",
        "cv":cv,"train_fit":train_fit,"student_test":test,
        "selected_model_by_cv_macro_f1":champion,
        "route_diagnostics":{champion:route},"hmm_comparison":hmm,
        "decoder":{"emission":"calibrated candidate probability times Gaussian distance likelihood","transition":"directed shortest-path distance penalty against GPS displacement","beta_m":beta_m,"sigma_m":sigma_m,"gap_reset_s":10.0,"disconnected_transition_penalty":9.0,
            "model_artifact":str(model_path) if model_path else str(out/"models"/f"{champion.lower()}.joblib")}}
    (out/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    rows=[]
    for n in names:
        rows.append({"model":n,"family":cv[n].get("family",n),
            "cv_macro_f1":cv[n]["fold_macro_f1_mean"],"cv_macro_f1_std":cv[n]["fold_macro_f1_std"],
            "train_macro_f1":train_fit[n]["candidate_macro_f1"],
            "student_test_top1":test[n]["top1_accuracy"],"student_test_macro_f1":test[n]["candidate_macro_f1"],
            "student_test_log_loss":test[n]["log_loss"],"student_test_candidate_miss_rate":test[n]["candidate_miss_rate"],
            "training_seconds":cv[n]["final_fit_seconds"],"student_test_inference_seconds":test[n]["inference_seconds"]})
    df=pd.DataFrame(rows);df.to_csv(out/"comparison.csv",index=False)
    _plot_models(df,out)
    hdf=pd.DataFrame(hmm);hdf.to_csv(out/"hmm_comparison.csv",index=False);_plot_hmm(hdf,out)
    mlp=joblib.load(out/"models"/"mlp.joblib")["model"].steps[-1][1]
    curve=pd.DataFrame({"epoch":np.arange(1,len(mlp.loss_curve_)+1),"training_loss":mlp.loss_curve_})
    curve.to_csv(out/"mlp_training_loss.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(curve.epoch,curve.training_loss)
    ax.set_title("MLP training loss (actual fitted estimator)");ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/"mlp_training_loss.png",dpi=160);plt.close(fig)
    print("Champion:",champion)
    print(df.to_string(index=False))
    print("HMM:",hdf.to_dict(orient="records"))


def _plot_models(df,out):
    for metric,title,file in [("cv_macro_f1","GroupKFold Macro F1","cv_macro_f1.png"),
        ("student_test_top1","Student-test observation Top-1 accuracy","student_test_top1.png"),
        ("student_test_macro_f1","Student-test candidate Macro F1","student_test_macro_f1.png"),
        ("student_test_log_loss","Student-test log loss","student_test_log_loss.png")]:
        fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(df.model,df[metric],color="#3178a8")
        ax.set_title(title);ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.25)
        fig.tight_layout();fig.savefig(out/file,dpi=160);plt.close(fig)


def _plot_hmm(df,out):
    fig,ax=plt.subplots(figsize=(7,4.5));x=np.arange(len(df));w=.25
    ax.bar(x-w,df.classifier_top1_mean,w,label="Classifier Top-1")
    ax.bar(x,df.viterbi_top1_mean,w,label="Viterbi Top-1")
    ax.bar(x+w,df.bridged_continuity_mean,w,label="Bridged continuity")
    ax.set_xticks(x,df.model,rotation=15);ax.set_ylim(0,1);ax.set_ylabel("Score")
    ax.set_title("Selected model: student-test classifier and decoder")
    ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/"hmm_route_comparison.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--candidate-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument('--model',type=Path,default=None);p.add_argument('--beta-m',type=float,default=50.0);p.add_argument('--sigma-m',type=float,default=25.0)
    a=p.parse_args();finalize(a.dataset_root,a.candidate_dir,a.output_dir,a.model,a.beta_m,a.sigma_m)

if __name__=="__main__":main()
