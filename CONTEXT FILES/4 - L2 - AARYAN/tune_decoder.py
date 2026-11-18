"""Development-only HMM transition sweep on the permitted student-test trips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments import evaluate_routes,positive_probability
from pathsnap import FEATURES


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--candidate-dir",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--betas",default="50,100,200,500,1000,2000")
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(a.model)
    frame=pd.read_csv(a.candidate_dir/"student_test_candidates.csv.gz",dtype={"trip_id":str})
    p_raw=positive_probability(bundle["model"],frame[FEATURES].to_numpy(np.float32))
    probs=bundle["calibrator"].predict(p_raw) if bundle.get("calibrator") is not None else p_raw
    rows=[]
    for beta in [float(v) for v in a.betas.split(",")]:
        print(f"Evaluating decoder beta={beta:g} m",flush=True)
        result=evaluate_routes(frame,probs,a.dataset_root,"student_test",beta_m=beta)
        vals=list(result.values())
        row={"beta_m":beta,
            "classifier_top1_mean":float(np.mean([v["classifier_top1_accuracy"] for v in vals])),
            "classifier_continuity_mean":float(np.mean([v["classifier_route_continuity"] for v in vals])),
            "viterbi_top1_mean":float(np.mean([v["viterbi_top1_accuracy"] for v in vals])),
            "viterbi_continuity_mean":float(np.mean([v["viterbi_route_continuity"] for v in vals])),
            "bridged_continuity_mean":float(np.mean([v["bridged_route_continuity"] for v in vals])),
            "combined_accuracy_continuity_score":float(np.mean([v["viterbi_top1_accuracy"]+v["viterbi_route_continuity"] for v in vals]))}
        rows.append(row)
        print(row,flush=True)
    df=pd.DataFrame(rows);df.to_csv(a.output_dir/"decoder_beta_sweep.csv",index=False)
    # Choose beta by the mean of pointwise accuracy and topological continuity,
    # while reporting each component separately so a tradeoff cannot be hidden.
    best=df.sort_values("combined_accuracy_continuity_score",ascending=False).iloc[0].to_dict()
    (a.output_dir/"decoder_config.json").write_text(json.dumps({
        "beta_m":best["beta_m"],"sigma_m":25.0,"gap_reset_s":10.0,
        "selected_on":"student_test development partition",
        "objective":"maximize mean across trips of Viterbi pointwise Top-1 plus pre-bridge route continuity",
        "best_sweep_row":best,
        "limitation":"student-test results are development-stage tuning results, not an independent final evaluation"},indent=2),encoding="utf-8")
    print("Selected decoder config:",best)

if __name__=="__main__":main()
