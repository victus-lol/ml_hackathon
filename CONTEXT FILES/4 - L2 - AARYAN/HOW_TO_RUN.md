# PathSnap distributed API and UI run guide

## Local preview using the student-test inputs

This preview uses the 10 checked-in student-test trips to demonstrate prediction and map rendering. The API inference path reads only `.track`, `.nodes`, and `.arcs`; it does not read `.route`. This is development data, not the organizer's hidden evaluator partition. Health intentionally reports `ready: false` for this data kind even though prediction requests can be demonstrated.

In PowerShell, open this directory on Laptop 1:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_DATA_DIR = (Join-Path (Get-Location) 'reference_materials\dataset\student_test')
$env:PATHSNAP_DATA_KIND = 'student_test'
$env:PATHSNAP_MODEL = 'artifacts\experiments\final_model.joblib'
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/api/v1/health` on Laptop 1. `/api/v1/trips` should list the 10 development trip IDs. Prediction is available at `/api/v1/match` when the model loads.

On Laptop 2, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = 'http://<LAPTOP1-IP>:8000'
python -m streamlit run ui_client.py
```

Choose trip `00000000` (if listed) and click **Match trip**. Only the trip ID crosses the network; the UI receives raw GPS and the predicted route from the API. Both laptops must be on the same trusted Wi-Fi or hotspot.

## Organizer evaluator setup

When the organizer provides the permitted evaluator partition, copy its raw `.track`, `.nodes`, and `.arcs` files to Laptop 1 only. Keep reference `.route` labels out of the API input directory. Start the server with:

```powershell
$env:PATHSNAP_DATA_DIR = 'C:\path\to\organizer-provided-evaluator-trips'
$env:PATHSNAP_DATA_KIND = 'evaluator'
$env:PATHSNAP_MODEL = 'artifacts\experiments\final_model.joblib'
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Health reports `ready: true` only when the model loads and exactly 10 evaluator trips are available. Set the Laptop 2 `PATHSNAP_API_URL` to Laptop 1's address before starting Streamlit. Check actual health, prediction, network access, and latency during the post-commit test session; none of those two-laptop checks are claimed as completed here.

## Optional tuned model

To use the model from Optuna tuning, set `PATHSNAP_MODEL` to `artifacts\tuning\tuned_model.joblib`. Keep the selected model and reported metrics aligned in any presentation.
