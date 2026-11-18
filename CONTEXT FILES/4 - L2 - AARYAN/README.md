# PathSnap distributed inference and training

This folder contains Aaryan's PathSnap work: a FastAPI prediction service for Laptop 1, a Streamlit/Folium client for Laptop 2, the candidate feature pipeline, a calibrated HistGradientBoosting model, and a log-space Viterbi decoder.

## Runtime flow

- `server.py` exposes the FastAPI app in `app_server.py`.
- The API reads `.track`, `.nodes`, and `.arcs` on the server. It never asks the client for route labels and calls `load_trip(..., include_route=False)` during inference.
- `/api/v1/trips` returns trip IDs only. `/api/v1/match` accepts only `trip_id`.
- `ui_client.py` requests IDs and prediction results, then renders blue raw GPS and a red predicted route.
- `pathsnap.py` creates metric features using a suitable UTM zone and hemisphere for each trip. `decoder.py` performs Viterbi decoding and route bridging.
- `artifacts/experiments/final_model.joblib` is the default classifier. `artifacts/tuning/tuned_model.joblib` is the Optuna-tuned alternative.

## Install and run

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for the local student-test preview and two-laptop evaluator setup. The student-test preview is development-only; its labels are not read by the API and it does not establish readiness for the organizer's evaluator partition.

## Training and reproducibility

`pathsnap.py build` constructs candidate tables from the 80 training trips and 10 student-test trips. The training and tuning scripts are `experiments.py`, `tune.py`, `tune_decoder.py`, and `finalize_experiments.py`. Candidate tables are generated locally and are not committed. The checked-in result summaries are under `artifacts/eda`, `artifacts/experiments`, and `artifacts/tuning`.

Five-fold GroupKFold splits are grouped by trip. Student-test figures are development evaluation only and must not be described as organizer-evaluator results. The 135 MB RandomForest artifact and intermediate candidate tables are omitted; the shipped runtime models are the final HistGradientBoosting artifacts.

## Data integrity

- Ground-truth `.route` data is used only by offline labeling and evaluation code when `include_route=True`.
- The API passes `include_route=False`; the route is not returned in API JSON or rendered in the client.
- Keep student-test and organizer evaluator partitions distinct. Set `PATHSNAP_DATA_KIND=evaluator` only for organizer-provided evaluator trips.
- Do not treat classifier candidate metrics or route-continuity development results as end-to-end evaluator metrics.
- Record live two-laptop latency only after measuring it on the actual setup.
