# Aaryan: API, UI, MLP, and First Review

This folder contains Aaryan's work-in-progress only. Keep all edits in this folder. The deployment code uses a small adapter until Akshay's feature interface and Abhiroop's decoder interface are available.

## Current files

- `app_server.py`: Laptop 1 FastAPI service. It provides health, server-side trip discovery, and a trip-ID match endpoint.
- `pipeline_contract.py`: required interface for the final team matcher. No dummy prediction is used when the engine is missing.
- `ui_client.py`: Laptop 2 Streamlit/Folium client. It fetches IDs from the API, submits the selected ID, and draws the API response.
- `train_mlp.py`: five-fold GroupKFold MLP workflow for Akshay's prepared candidate table. Scalers are fitted inside each fold.
- `TEAM_REVIEW_HANDOUT.md`: reviewed interfaces, data boundaries, and open decisions.
- `AARYAN_WORK_HANDOUT.md`: summary of my assigned work, current status, and blockers.
- `VIVA_FIRST_REVIEW.md`: first-review questions and truthful answers.
- `REPORT_SECTION_6_DRAFT.md`: deployment report draft with measured-vs-pending fields.
- `reference_materials/`: the five college PDFs and the available 80-trip training plus 10-trip student-test data.
- `requirements.txt`: dependencies for this folder's code.

## Safety and data rules

- The UI request includes `trip_id`; it does not open trip files or use cached predictions.
- The API reads only `.track`, `.nodes`, and `.arcs` from its configured data folder. It does not load the reference route file.
- The API reports `ready: true` only when the matcher is ready and exactly 10 trips are configured. A live health response with `ready: false` means the API process is up but the full demo is not ready.
- The trip list is discovered from files on the server; trip IDs are not hardcoded.
- Do not configure the student-test directory as the live evaluator directory. Student-test labels are for offline development analysis only.
- No metrics, maps, model scores, or route predictions should be fabricated. The match endpoint returns 503 until the real feature/model/decoder integration is configured.
- Use `(longitude, latitude)` coordinate order and UTM Zone 32N (EPSG:32632) for projected distance calculations in the pipeline.

## Laptop 1: install and start the API

Run PowerShell from this folder:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_EVALUATOR_DIR = 'C:\path\to\server-side\evaluator_trips'
$env:PATHSNAP_ENGINE_MODULE = 'pathsnap_runtime'
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

`PATHSNAP_EVALUATOR_DIR` must contain one directory per trip ID, with matching `.track`, `.nodes`, and `.arcs` files in each directory. It must contain all 10 permitted evaluator trips for `ready` to become true. The engine module setting remains unset until the team integration module exists; without it, health is available and matching returns 503 rather than a made-up result.

On Laptop 1, check `http://127.0.0.1:8000/api/v1/health`. From Laptop 2 on the same trusted Wi-Fi/hotspot, open `http://<laptop1-ip>:8000/api/v1/health`. Use the actual Laptop 1 address shown by Windows network settings; do not commit that address or credentials.

## Laptop 2: start the UI

Run PowerShell from this folder on Laptop 2:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = 'http://<laptop1-ip>:8000'
python -m streamlit run ui_client.py
```

The map uses blue for the raw GPS trace and red for the predicted route. Student-test reference maps, if produced for the report, belong in a separate offline analysis path and must never appear in the evaluator UI.

## Train Model 5 after the feature schema is frozen

The candidate CSV must be built from the 80 training trips only and contain `trip_id`, `observation_id`, `label`, and the agreed 14 numeric feature columns. Candidate IDs and raw coordinates/time are not model features. Ask Akshay for the final feature names and order before running the script.

```powershell
python train_mlp.py path\to\training_candidates.csv --training-root 'reference_materials\dataset\train' --features feature_01,feature_02,feature_03,feature_04,feature_05,feature_06,feature_07,feature_08,feature_09,feature_10,feature_11,feature_12,feature_13,feature_14 --output-dir artifacts
```

The script checks that the feature table's trip IDs exactly match the 80 directories in `--training-root`, so a 10-trip test partition cannot accidentally be used as training input.

The output contains the trained pipeline, out-of-fold probabilities, and metrics from that real run. The current repo does not yet contain Akshay's candidate table or frozen feature-column names, so no MLP scores or weights are claimed here.

## First review screen and viva

Keep the API terminal visible on Laptop 1 and open the health URL from Laptop 2. The honest status is `healthy` when the process answers; `ready` remains false until the real engine and all 10 permitted trips are configured.

Suggested explanation: “Laptop 1 hosts the FastAPI service; Laptop 2 calls it and displays the API's route on a Folium map, with no ground-truth labels sent over the network.” Only claim that model inference is live after the actual engine is integrated and the API reports it ready.

## Pending team handoffs

- Akshay: candidate table path, final 14 feature names/order, and the preprocessing/model-input contract.
- Aditya: MLP evaluation columns expected by the common scoreboard and model artifact-loading contract.
- Abhiroop: decoder callable, returned route coordinates, and agreed server metrics.
- Team: whether Laptop 1 owns evaluator input files or Laptop 2 sends the raw arrays with the required `trip_id`; the payload must never include label data.

The repo requires a public, well-structured root project (`data/`, `models/`, `api/`, `ui/`, `artifacts/`) and one consolidated six-section PDF. Those shared root paths are outside this folder; coordinate with Aditya before changing them.
