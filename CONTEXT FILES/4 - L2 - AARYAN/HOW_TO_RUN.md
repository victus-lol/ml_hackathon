# PathSnap distributed API and UI run guide

## Current readiness

The FastAPI API and Streamlit client are present. This checkout does not include the trained production model, the HMM decoder module expected by `pathsnap_engine.py`, or the 10 organizer-provided evaluator trips. The two-laptop prediction demo is therefore not ready in this checkout. The health endpoint can report `healthy` while `ready` is false; the match endpoint will not produce a route until the real engine and evaluator inputs are installed on Laptop 1.

Do not report the 1.40 ms latency, 100% continuity, or a ready status from the handoff ZIP as measured results. They are not verified by this repository state.

## Laptop 1: API

From this folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_EVALUATOR_DIR = 'C:\path\to\organizer-provided-evaluator-trips'
$env:PATHSNAP_ENGINE_MODULE = 'pathsnap_runtime'
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

The evaluator folder must contain exactly the permitted evaluator trips, each with matching `.track`, `.nodes`, and `.arcs` files. Keep `.route` labels out of this directory. `pathsnap_runtime` must be the team's completed integration module exposing `create_engine()`, `is_ready()`, and `match_trip()`; it is not included in this checkout.

Check `http://127.0.0.1:8000/api/v1/health` on Laptop 1. A `ready: false` response means prediction is not available yet.

## Laptop 2: evaluator UI

Connect both laptops to the same trusted Wi-Fi or hotspot. From this folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = 'http://<LAPTOP1-IP>:8000'
python -m streamlit run ui_client.py
```

Enter Laptop 1's address in the UI. The UI fetches trip IDs and prediction results from the API; it does not read trip files or route labels locally. The API request contains a trip ID only.

## Before the live demo

1. Obtain the permitted evaluator inputs and completed model/decoder integration from the team.
2. Configure Laptop 1's evaluator directory and engine module.
3. Confirm health reports `ready: true` and 10 available trips.
4. Run the UI from Laptop 2 and confirm a selected trip returns a predicted route.
5. Measure latency and continuity from an actual run before presenting those figures.
