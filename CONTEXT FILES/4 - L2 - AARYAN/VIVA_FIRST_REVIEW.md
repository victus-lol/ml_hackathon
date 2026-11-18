# PathSnap — Aaryan Viva Notes

Use these notes as a speaking guide. Separate the local development preview from the not-yet-tested organizer evaluator demo.

## Short introduction

“My part is the distributed FastAPI and Streamlit/Folium flow, the candidate feature pipeline, and the model/decoder integration. Laptop 1 reads trip inputs and predicts a route; Laptop 2 displays the API response. I have committed the code and model artifacts. The two-laptop setup still needs testing.”

## What does the API do?

“Laptop 1 exposes `/api/v1/health`, `/api/v1/trips`, and `/api/v1/match`. Trip IDs are discovered from the configured data folder. A match request contains only `trip_id`; the server reads the raw track and road network, creates candidates, scores them, decodes the path, and returns predicted geometry.”

## Is the model pipeline integrated?

“The local runtime includes a saved HistGradientBoosting model and the Viterbi decoder. The checked-in student-test data can be used for a development preview. Health remains `ready: false` for that partition. Evaluator readiness requires the organizer's 10 trips, the model to load, and `PATHSNAP_DATA_KIND=evaluator`.”

## How does the UI connect to the server?

“Laptop 2 fetches trip IDs, posts the selected ID, and draws blue GPS observations and the red predicted route from the response. It has no local trip-file loading or reference-route layer.”

## How do you prevent ground-truth leakage?

“The API prediction path calls `load_trip(..., include_route=False)`. It reads `.track`, `.nodes`, and `.arcs`; it does not read `.route`. The client sends only the trip ID and receives raw GPS and predicted geometry.”

## What did the model evaluation show?

“On the saved development runs, HistGradientBoosting reached 0.8546 candidate macro F1 in five-fold GroupKFold on the 80 training trips. Tuning improved CV candidate macro F1 to 0.8567. The tuned model's student-test candidate macro F1 was 0.8785. These are development candidate metrics, not hidden evaluator results or decoded route accuracy.”

## What about route continuity and latency?

“The offline student-test decoder comparison recorded 0.9996 bridged continuity. This uses the student-test reference routes and is development evidence only. I have no verified two-laptop end-to-end latency yet; I will measure that during testing.”

## What remains to verify?

“After committing, I need to run the API and UI locally, check the map output, and then test Laptop 2 against Laptop 1 over the shared network. The organizer evaluator data is still required for final readiness and evaluator results.”

## Explain GroupKFold

“Observations from the same trip are related. GroupKFold keeps all rows from a trip in one fold, so validation trips are unseen during that fold's training.”

## Safe status sentence

“The model-backed API and client are committed. I have not yet tested the committed version or the two-laptop connection; the included student-test preview is development-only, and organizer evaluator readiness is pending.”

Disclose AI assistance according to the event policy and be ready to explain the feature columns, model, data split, and API boundary.
