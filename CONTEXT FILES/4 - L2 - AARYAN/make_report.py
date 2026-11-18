"""Build a locally reviewable consolidated report from recorded experiment outputs."""
from __future__ import annotations

import argparse,json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image,PageBreak,Paragraph,SimpleDocTemplate,Spacer,
    Table,TableStyle,KeepTogether,Preformatted)


def ptxt(s,style):return Paragraph(str(s),style)
def fmt(v,nd=3):
    if isinstance(v,(int,float)):return f"{v:.{nd}f}"
    return str(v)
def readj(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def build(args):
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    eda=readj(args.eda_summary); comp=readj(args.comparison)
    tuning=readj(args.tuned_comparison) if Path(args.tuned_comparison).exists() else None
    search=readj(args.best_params) if Path(args.best_params).exists() else None
    decoder=readj(args.decoder_config) if Path(args.decoder_config).exists() else comp.get("decoder",{})
    inventory=pd.read_csv(args.inventory) if Path(args.inventory).exists() else pd.DataFrame()
    coverage=pd.read_csv(args.candidate_coverage) if Path(args.candidate_coverage).exists() else pd.DataFrame()
    alignment=pd.read_csv(args.alignment_quality) if Path(args.alignment_quality).exists() else pd.DataFrame()
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2",parent=styles["Title"],fontSize=25,leading=30,textColor=colors.HexColor("#143d59"),alignment=TA_CENTER,spaceAfter=18))
    styles.add(ParagraphStyle(name="H1x",parent=styles["Heading1"],fontSize=16,leading=20,textColor=colors.HexColor("#143d59"),spaceBefore=10,spaceAfter=8))
    styles.add(ParagraphStyle(name="H2x",parent=styles["Heading2"],fontSize=12,leading=15,textColor=colors.HexColor("#1c6e8c"),spaceBefore=8,spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx",parent=styles["BodyText"],fontSize=8.5,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name="Smallx",parent=styles["BodyText"],fontSize=7,leading=9,spaceAfter=3))
    styles.add(ParagraphStyle(name="Callout",parent=styles["BodyText"],fontSize=9,leading=12,borderColor=colors.HexColor("#d07b16"),borderWidth=.7,borderPadding=6,backColor=colors.HexColor("#fff5e6"),spaceBefore=5,spaceAfter=8))
    styles.add(ParagraphStyle(name="TableHead",parent=styles["BodyText"],fontSize=6.8,leading=8,textColor=colors.white,alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TableCell",parent=styles["BodyText"],fontSize=6.8,leading=8))
    story=[]
    story += [Spacer(1,.8*inch),ptxt("PathSnap",styles["Title2"]),
        ptxt("Distributed Map-Matching and Route Reconstruction",ParagraphStyle("Sub",parent=styles["Heading2"],alignment=TA_CENTER,textColor=colors.HexColor("#287271"))),
        Spacer(1,.3*inch),ptxt("Consolidated technical report — local implementation draft",ParagraphStyle("Sub2",parent=styles["BodyText"],alignment=TA_CENTER,fontSize=12)),
        Spacer(1,.2*inch),ptxt("Development evidence: 80 training trips and 10 student-test trips",ParagraphStyle("Sub3",parent=styles["BodyText"],alignment=TA_CENTER,fontSize=10)),
        Spacer(1,.45*inch),ptxt("Status: evaluator data were not present in the local materials. Student-test results are development results, not final evaluator results. The report was prepared with AI assistance and requires team review and disclosure before submission.",styles["Callout"]),
        Spacer(1,.5*inch),ptxt("18 November 2026",ParagraphStyle("Date",parent=styles["BodyText"],alignment=TA_CENTER)),PageBreak()]

    story += [ptxt("Executive summary",styles["H1x"]),ptxt(
        f"Five candidate classifiers were trained and compared using five-fold GroupKFold by trip_id. {comp['selected_model_by_cv_macro_f1']} had the highest mean cross-validation candidate Macro F1 ({fmt(comp['cv'][comp['selected_model_by_cv_macro_f1']]['fold_macro_f1_mean'])}). Student-test Top-1 for that model was {fmt(comp['student_test'][comp['selected_model_by_cv_macro_f1']]['top1_accuracy'])}. The required 96% target was not reached in the current measured results.",styles["Bodyx"]),
        ptxt("The production path keeps .route labels out of model inference, the API request/response, and the UI. Training and offline development evaluation use the permitted reference routes. The local API and UI still require deployment verification with the missing evaluator inputs and a second laptop.",styles["Bodyx"]),
        ptxt("The current implementation’s primary limitation is candidate coverage: among the nearest eight road arcs within 60 m, 11.30% of training observations and 0.41% of student-test observations had no candidate matching the monotonic route-aligned label. Missed observations were excluded from classifier loss and counted as incorrect in overall Top-1.",styles["Bodyx"])]

    story += [ptxt("1. Dataset understanding and exploratory analysis",styles["H1x"]),
        ptxt("The files are whitespace-separated with no headers. .track contains longitude, latitude, timestamp seconds; .nodes contains longitude and latitude; .arcs contains from_node_id and to_node_id, with the row index as arc_id; .route contains an ordered reference arc sequence. Each trip uses local node and arc indices.",styles["Bodyx"])]
    data=[["Partition","Trips","GPS observations","Nodes","Directed arcs","Raw data size","Missing/non-finite GPS","Gaps > 2 s","Max gap"]]
    for part in ["train","student_test"]:
        d=eda["partitions"][part]
        data.append([part,d["trip_count"],d["total_observations"],d["total_nodes"],d["total_arcs"],
            f"{d['total_raw_bytes']/1e6:.2f} MB",d["missing_track_values"],d["gaps_over_2s"],f"{d['max_gap_s']:.0f} s"])
    story.append(_table(data,[.8*inch,.45*inch,.8*inch,.65*inch,.72*inch,.72*inch,.75*inch,.65*inch,.6*inch],styles))
    if not inventory.empty:
        ext=(inventory.longitude_min.min(),inventory.longitude_max.max(),inventory.latitude_min.min(),inventory.latitude_max.max())
        story.append(ptxt(f"Across inspected trips, longitude ranged from {ext[0]:.3f} to {ext[1]:.3f} and latitude from {ext[2]:.3f} to {ext[3]:.3f}. No non-finite raw GPS values or nonpositive timestamp gaps were found. Both UTM zones and hemispheres are present; therefore coordinates are projected to the local UTM zone/hemisphere for each trip. Forcing all trips into UTM 32N produced invalid coordinates for the provided trip near longitude −78°, latitude 0°.",styles["Bodyx"]))
    story += [ptxt("The training partition had a median of " + fmt(eda['partitions']['train']['observations_per_trip']['median'],0) + " observations per trip. The largest recorded timestamp gap was exceptionally long, so the decoder resets its Viterbi sequence after gaps over 10 seconds. The road graph’s median weakly connected component count in the training trips was " + fmt(eda['partitions']['train']['weak_components_median'],0) + ".",styles["Bodyx"])]
    if not coverage.empty:
        bad=coverage[coverage.partition=="train"].sort_values("candidate_miss_rate",ascending=False).head(3)
        align_bad=alignment[alignment.partition=="train"] if not alignment.empty else pd.DataFrame()
        if not align_bad.empty:bad=bad.merge(align_bad[["trip_id","distance_p50_m","fraction_within_60m"]],on="trip_id",how="left")
        anomaly=[["Training trip","Observations","Candidate misses","Miss rate","Route-GPS median distance"]]
        for _,r in bad.iterrows():anomaly.append([str(r.trip_id),str(int(r.observations)),str(int(r.candidate_misses)),fmt(r.candidate_miss_rate),fmt(r.get("distance_p50_m",float("nan")),1)+" m"])
        story.append(ptxt("Candidate/label alignment anomalies (training):",styles["H2x"]))
        story.append(_table(anomaly,[1.2*inch,1.0*inch,1.0*inch,1.0*inch,2.0*inch],styles))
    _add_image(story,args.eda_observations,6.7*inch,3.3*inch,styles)
    _add_image(story,args.eda_network,6.7*inch,3.3*inch,styles)
    story.append(ptxt("The interactive Folium map and map-generation code are included alongside this report. It uses a student-test trip and is an offline development diagnostic, not an evaluator screen.",styles["Bodyx"]))
    _add_image(story,args.folium_map,6.6*inch,4.2*inch,styles)

    story += [ptxt("2. Feature engineering and labeling",styles["H1x"]),
        ptxt("For each trip, the candidate generator projects GPS and node coordinates to the local UTM zone and hemisphere, constructs directed straight-line arc geometries from each node pair, and indexes the arc lines with Shapely STRtree. Up to eight candidate arcs within 60 m are retained per observation; if none intersects the search radius, the nearest arc is retained. The resulting miss rate is measured, not hidden.",styles["Bodyx"]),
        ptxt("Fourteen model features are used: perpendicular distance, projection fraction, arc length, sine/cosine of the relative heading delta, clipped speed, clipped time gap, distance to each arc endpoint, in/out degrees at each endpoint, and local GPS displacement magnitude. Absolute coordinates, absolute heading, timestamps, trip IDs, node IDs, arc IDs, and candidate IDs are excluded from X.",styles["Bodyx"]),
        ptxt("Labels are created offline by projecting each GPS observation to the nearest reference-route arc subject to nondecreasing route order. Candidate arcs matching that aligned arc receive label 1; other candidates receive 0. If the aligned arc is absent from the candidate set, those candidates are marked unlabeled (−1), excluded from training/loss, and the observation is counted as a candidate miss. This alignment is an assumption: supplied routes are treated as reference labels, not independently verified physical truth.",styles["Bodyx"]),
        ptxt("Negative candidate rows are deterministically sampled at up to two per observation for fitting to keep compute practical; all candidate rows are retained for CV validation and student-test scoring. This sampling changes the training class prevalence, so the reported log loss is interpreted with that limitation.",styles["Bodyx"])]

    story += [ptxt("3. Model experimentation and comparison",styles["H1x"]),
        ptxt("All candidate models use the same 14 numeric features and the same training/student-test partition. CV folds group by trip_id. Final preprocessing is inside each model pipeline where scaling is needed. Model choice is based on mean five-fold CV candidate Macro F1, not training score or student-test score. Isotonic probability calibration is fitted from training-only out-of-fold probabilities for each final model. The CV model-selection scores below are raw OOF scores; student-test probabilities use the fitted calibration.",styles["Bodyx"])]
    mdata=[["Model / family","CV Macro F1\nmean ± SD","Train Macro F1","Student Top-1","Student Macro F1","Log loss","Miss rate","Fit s","Inference s"]]
    for n in comp["cv"]:
        c=comp["cv"][n];tr=comp["train_fit"][n];te=comp["student_test"][n]
        mdata.append([n,f"{c['fold_macro_f1_mean']:.3f} ± {c['fold_macro_f1_std']:.3f}",fmt(tr['candidate_macro_f1']),
            fmt(te['top1_accuracy']),fmt(te['candidate_macro_f1']),fmt(te['log_loss']),fmt(te['candidate_miss_rate']),
            fmt(c['final_fit_seconds'],1),fmt(te['inference_seconds'],2)])
    story.append(_table(mdata,[1.1*inch,.95*inch,.8*inch,.85*inch,.85*inch,.65*inch,.7*inch,.55*inch,.7*inch],styles))
    for f,title,file in [(args.plot_cv,"Cross-validation Macro F1",""),(args.plot_top1,"Student-test observation Top-1",""),
        (args.plot_testf1,"Student-test candidate Macro F1",""),(args.plot_logloss,"Student-test Log Loss","")]:
        _add_image(story,f,6.5*inch,3.25*inch,styles)
    descriptions={
      "LogisticRegression":("A regularized linear probability model is a useful low-complexity baseline for candidate geometry and local-motion features. It trained quickly and provides interpretable linear effects; its student-test Top-1 was lower than the tree models, indicating nonlinear interactions may matter. It can underfit nonlinear road/candidate relationships.","Log loss with L2 regularization; C=1.0, class_weight=balanced, max_iter=300; StandardScaler inside each fit."),
      "GaussianNB":("A probabilistic baseline estimates class likelihoods from each feature and is computationally cheap. It is fast but assumes conditional feature independence, which is implausible for distance, projection fraction, and endpoint distances. It had the lowest Top-1 among the five models in this run.","Gaussian class-conditional likelihood with var_smoothing=1e-9; StandardScaler inside each fit."),
      "RandomForest":("Bagged trees can model nonlinear thresholds and feature interactions without feature scaling. It gave strong CV and student-test performance, but the CV/train gap and 22 s final fit show greater complexity and overfitting risk than the boosted model. It is larger and slower to train.","Gini impurity; 60 trees, max_depth=20, min_samples_leaf=3, class_weight=balanced_subsample, n_jobs=-1."),
      "HistGradientBoosting":("Boosted trees can combine nonlinear candidate geometry and kinematic interactions efficiently. It had the strongest mean grouped CV Macro F1 and nearly tied the best student-test candidate Macro F1, with low training time. It remains sensitive to feature/label quality and did not reach the external 96% target.","Binary log loss; 80 iterations, learning_rate=0.08, max_leaf_nodes=23, L2=0.1."),
      "MLP":("A shallow neural network can learn nonlinear combinations among the 14 scaled numeric features. Its CV and student-test results were competitive but below HistGradientBoosting in this run; convergence warnings at 35 iterations indicate the fixed training budget may limit fitting. Scaling is required and the model is less interpretable.","Binary cross-entropy with Adam; hidden layer (32,), ReLU, alpha=0.001, batch size 1024, learning_rate_init=0.001, max_iter=35, early_stopping=False; StandardScaler inside each fit."),
    }
    for name in comp["cv"]:
        c=comp["cv"][name];tr=comp["train_fit"][name];te=comp["student_test"][name]
        desc,config=descriptions[name]
        story += [ptxt(name+" — "+comp.get("model_configs",{}).get(name,{}).get("family",name),styles["H2x"]),
            ptxt("1. Suitability, strengths, and limitations. "+desc,styles["Bodyx"]),
            ptxt("2. Training configuration and objective. "+config+" Fitting used every training trip’s labeled candidate rows with up to two sampled negatives per observation; held-out folds were whole trips. The primary comparison score is candidate-row Macro F1.",styles["Bodyx"]),
            ptxt("3. Student-test results. Observation Top-1 = "+fmt(te["top1_accuracy"])+"; candidate Macro F1 = "+fmt(te["candidate_macro_f1"])+"; precision = "+fmt(te["precision"])+"; recall = "+fmt(te["recall"])+"; log loss = "+fmt(te["log_loss"])+"; candidate miss rate = "+fmt(te["candidate_miss_rate"])+"; inference time = "+fmt(te["inference_seconds"],3)+" s for the scored candidates.",styles["Bodyx"]),
            ptxt("4. Generalization / overfit evidence. Grouped CV Macro F1 mean = "+fmt(c["fold_macro_f1_mean"])+" (SD "+fmt(c["fold_macro_f1_std"])+"); fitted-training candidate Macro F1 = "+fmt(tr["candidate_macro_f1"])+"; student-test candidate Macro F1 = "+fmt(te["candidate_macro_f1"])+". The train-to-test gap is "+fmt(tr["candidate_macro_f1"]-te["candidate_macro_f1"])+". Interpret it with the different trip partitions and the training negative-row sampling; do not use training score alone to select a model.",styles["Bodyx"])]
        if name=="MLP":
            story.append(ptxt("The plotted loss is the actual final estimator’s training loss curve. No artificial validation-loss curve is shown: early_stopping is disabled to avoid a row-level internal split that could mix trips; grouped CV supplies the validation evidence.",styles["Smallx"]))
        story.append(Spacer(1,5))
    if Path(args.mlp_loss).exists():_add_image(story,args.mlp_loss,5.9*inch,3.0*inch,styles)

    story += [PageBreak(),ptxt("4. Bayesian hyperparameter optimization",styles["H1x"])]
    if tuning and search:
        story.append(ptxt(f"Optuna TPE optimized {search['objective']} for the CV-selected {search['selected_model']}. The study ran {search['trial_count']} trials; the best raw CV Macro F1 was {fmt(search['best_value_cv_macro_f1'])}. Search ranges and the full per-trial record are included in artifacts/tuning. The untuned baseline and tuned model use identical training and student-test partitions, and both student-test probability outputs use training-only OOF isotonic calibration.",styles["Bodyx"]))
        tdata=[["Metric","Untuned","Tuned"]]
        tdata += [["CV Macro F1",fmt(tuning["untuned_cv_macro_f1"]),fmt(tuning["tuned_cv_macro_f1"])]]
        for key,label in [("top1_accuracy","Student-test Top-1"),("candidate_macro_f1","Student-test Macro F1"),("log_loss","Student-test Log Loss")]:
            tdata.append([label,fmt(tuning["untuned_student_test"][key]),fmt(tuning["tuned_student_test"][key])])
        tdata += [["Inference seconds",fmt(tuning["untuned_inference_seconds"],3),fmt(tuning["tuned_inference_seconds"],3)],
                  ["Training seconds",fmt(tuning["untuned_fit_seconds"],2),fmt(tuning["tuned_fit_seconds"],2)]]
        story.append(_table(tdata,[2.2*inch,1.2*inch,1.2*inch],styles))
        story.append(ptxt("Best hyperparameters: "+json.dumps(search["best_params"],sort_keys=True),styles["Bodyx"]))
        for f in [args.optuna_history,args.optuna_importance]:_add_image(story,f,6.5*inch,3.2*inch,styles)
    else:
        story.append(ptxt("Bayesian tuning results were not found when this report was generated. This section is incomplete until a real Optuna run is completed.",styles["Callout"]))

    story += [ptxt("5. HMM and Viterbi route reconstruction",styles["H1x"]),
        ptxt("Hidden states are candidate directed road arcs for each GPS observation. The emission score combines the calibrated classifier probability with a Gaussian penalty for perpendicular GPS-to-arc distance. The transition score compares directed shortest-path distance between candidate arc endpoints with observed GPS displacement using an exponential absolute-distance penalty. Viterbi uses log-space dynamic programming. Sequence decoding resets after timestamp gaps over 10 seconds; disconnected transitions receive a finite penalty. Postprocessing deduplicates adjacent same-arc states and inserts a shortest-path connector only when it is at most 1,000 m; longer breaks remain explicit.",styles["Bodyx"])]
    h=comp.get("hmm_comparison",[])
    if h:
        x=h[0]
        hdata=[["Measure (mean over 10 student-test trips)","Classifier-only","Viterbi","Bridged route"]]
        hdata += [["Observation Top-1",fmt(x["classifier_top1_mean"]),fmt(x["viterbi_top1_mean"]),"N/A — route-level"],
            ["Consecutive directed-arc continuity",fmt(x["classifier_continuity_mean"]),fmt(x["viterbi_continuity_mean"]),fmt(x["bridged_continuity_mean"])],
            ["Exact full-trip sequence fraction","N/A",fmt(x["viterbi_sequence_exact_fraction"]),"N/A"]]
        story.append(_table(hdata,[2.8*inch,1.2*inch,1.2*inch,1.2*inch],styles))
        story.append(ptxt("The first decoder setting did not improve pointwise accuracy: the continuity gain came with lower Top-1. Decoder parameters are being tuned on the permitted student-test development partition; use the final sweep in this table before describing the tradeoff. Bridged continuity is a graph property of the reconstructed route and does not mean each inserted bridge arc matches the reference labels.",styles["Callout"]))
        _add_image(story,args.hmm_plot,6.3*inch,3.1*inch,styles)
    story.append(ptxt("No claim is made that HMM improves sequence accuracy unless the recorded final results show it. A full-trip exact match is reported separately and should not be inferred from continuity.",styles["Bodyx"]))

    story += [ptxt("6. API, UI, and deployment",styles["H1x"]),
        ptxt("Laptop 1 loads the trained model and trip-local road network, then runs candidate generation, classification, and Viterbi. Laptop 2 asks for available trip IDs, submits only the selected trip ID, and renders the server response. The API prediction loader opens only .track, .nodes, and .arcs; it never reads .route. The live UI draws blue raw GPS and red predicted route, with no reference-route layer or label-derived metric.",styles["Bodyx"])]
    arch=[[ptxt("Laptop 2 — Streamlit/Folium",styles["TableHead"]),ptxt("LAN: trip_id / prediction response",styles["TableHead"]),ptxt("Laptop 1 — FastAPI",styles["TableHead"])],
          ["Trip selector + blue GPS / red route","GET /api/v1/trips; POST /api/v1/match","Server-side raw input loader"],
          ["No local route labels","No .route on wire","Candidate model + HMM + decoder"]]
    story.append(_table(arch,[2.0*inch,2.2*inch,2.0*inch],styles))
    story.append(ptxt("Endpoints: GET /api/v1/health reports model/data readiness and memory; GET /api/v1/trips discovers IDs from server-side files; POST /api/v1/match loads one trip and returns the prediction. Health readiness remains false when configured with student-test data, even though development match requests can run. The local repository lacked all 10 evaluator trip files, so evaluator readiness, every-trip inference, two-laptop networking, evaluator API/UI screenshots, and final latency remain pending.",styles["Bodyx"]))
    story.append(ptxt("The interactive Folium HTML and code are local artifacts. Do not display the offline student-test reference route in the evaluator UI. The current API returns no accuracy fields because it does not have evaluator labels in the prediction path.",styles["Bodyx"]))
    story += [ptxt("Reproducibility and academic integrity",styles["H1x"]),
        ptxt("All reported model scores and graphs are generated from the saved code and outputs in this local project. Evaluator-testing data were not available and were not used. The student-test set was used for development-stage comparisons and decoder tuning; results on that set are not an independent final evaluation. AI assistance was used to draft and review implementation/report text. Team members must verify the math, assumptions, results, and wording, make the required disclosure, and be able to explain each design choice before submission.",styles["Bodyx"]),
        ptxt("Current measured target status: 96% Top-1 has not been reached. No reported value is raised to meet a target without measured evidence.",styles["Callout"])]

    doc=SimpleDocTemplate(str(out),pagesize=landscape(A4),rightMargin=.45*inch,leftMargin=.45*inch,topMargin=.5*inch,bottomMargin=.45*inch,title="PathSnap Technical Report")
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor("#51616b"))
        canvas.drawString(.45*inch,.22*inch,"PathSnap — development evidence; evaluator partition unavailable")
        canvas.drawRightString(landscape(A4)[0]-.45*inch,.22*inch,f"Page {doc.page}")
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out.resolve())


def _table(rows,widths,styles):
    cooked=[]
    for i,row in enumerate(rows):
        if i==0:
            cooked.append([ptxt(x,styles["TableHead"]) if isinstance(x,str) else x for x in row])
        else:
            cooked.append([ptxt(x,styles["TableCell"]) if isinstance(x,str) else x for x in row])
    t=Table(cooked,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c6e8c")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#aab6bc")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef4f6")]),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    return t

def _add_image(story,path,width,height,styles):
    path=Path(path)
    if path.exists():
        img=Image(str(path));img._restrictSize(width,height);story += [Spacer(1,4),img,Spacer(1,5)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--eda-summary',required=True);p.add_argument('--inventory',required=True)
    p.add_argument('--candidate-coverage',required=True);p.add_argument('--alignment-quality',required=True);p.add_argument('--folium-map',required=True)
    p.add_argument('--comparison',required=True);p.add_argument('--tuned-comparison',required=True);p.add_argument('--best-params',required=True)
    p.add_argument('--decoder-config',required=True);p.add_argument('--eda-observations',required=True);p.add_argument('--eda-network',required=True)
    p.add_argument('--plot-cv',required=True);p.add_argument('--plot-top1',required=True);p.add_argument('--plot-testf1',required=True);p.add_argument('--plot-logloss',required=True)
    p.add_argument('--mlp-loss',required=True);p.add_argument('--optuna-history',required=True);p.add_argument('--optuna-importance',required=True)
    p.add_argument('--hmm-plot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a)
if __name__=='__main__':main()
