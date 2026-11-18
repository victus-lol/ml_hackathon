# PathSnap: Aaryan Integration Plan — Team Review

**Status: DRAFT — please review before implementation is treated as agreed**  
**Owner:** Aaryan (Laptop 2 / deployment integration)  
**Reviewers:** Akshay, Aditya, Abhiroop

This note proposes the integration contract for Aaryan's work. Please comment on this file or the pull request with **Approved** or **Changes requested**, and add the relevant details. It does not report measured performance or model results.

## Scope I own

- FastAPI integration and health/status endpoints.
- The evaluator-facing UI and Folium display.
- The Laptop 2 to Laptop 1 connection and demo instructions.
- Model 5 (MLP), using the feature schema and preprocessing agreed with Akshay and Aditya.
- My report contribution on deployment, interface behavior, and measured system performance.

Per the workspace rule, my work in `CONTEXT FILES` stays inside `4 - L2 - AARYAN`. This handout does not authorize edits to another teammate's folder. Please agree where final, runnable deployment code should live before that work is started.

## Non-negotiable data and evaluation rules

- Treat coordinates as `(longitude, latitude)` and use **UTM Zone 32N (EPSG:32632)** for projected distance calculations.
- Load the correct `.nodes` and `.arcs` for each trip. Node IDs and arc IDs are local to that trip.
- Use the 80 training trips for fitting and group-aware cross-validation. Use the 10 student-test trips only for development evaluation.
- Do not use evaluator-test data or labels for fitting, feature selection, tuning, thresholds, or model choice.
- The evaluator UI must not read, transmit, display, or cache evaluator `.route` labels. The inference path must work from raw GPS and the matching network only.
- Report only metrics produced by a real run, with the run configuration and data split recorded.

## Proposed request and response

**Proposed UI request:** `POST /api/v1/match` with a JSON body containing `trip_id` only. The API loads that trip's permitted raw `.track`, `.nodes`, and `.arcs` from the server-side evaluator data directory, runs candidate generation, the selected classifier, and HMM/Viterbi, then returns the result. The UI renders the response and does not compute or substitute predictions.

**Proposed response:**

- `trip_id`
- raw GPS coordinates needed to draw the input trace
- predicted ordered arc IDs and/or predicted route geometry needed to draw the route
- API-computed metrics that are allowed for the evaluator view
- inference latency and a clear error if matching fails

The response must never include ground-truth arc IDs, evaluator `.route` contents, or label-derived overlays. A separate health endpoint can report whether the server and model are ready. A trip-list endpoint may return evaluator trip IDs only, if the UI needs it.

**Please confirm the raw-data location.** This proposal assumes Laptop 1 can load evaluator `.track`, `.nodes`, and `.arcs` by `trip_id`. If the evaluator supplies those files only on Laptop 2, agree on a compliant transfer contract before implementation; the payload must still exclude `.route` and labels. The UI must send the selected `trip_id` and render the API's result.

## Interfaces to agree before integration

Please confirm these with the respective owner so I can connect modules without guessing:

- **Akshay:** feature column names and order, candidate-row grouping by observation, candidate arc IDs, coordinate units, and the fitted preprocessing interface.
- **Aditya:** model artifact format, feature order, `predict_proba` input/output shape, and how the MLP participates in the same comparison.
- **Abhiroop:** decoder function inputs and outputs, candidate/emission representation, network object or graph input, behavior for no candidates and disconnected transitions, and the meaning of the returned continuity metric.
- **All:** which server-side metrics may be returned to the evaluator UI, and which are for offline student-test analysis only.

## Demo checklist

- Laptop 1 starts the API and reports healthy/model-ready status.
- Laptop 2 can reach Laptop 1 over the agreed LAN or hotspot address.
- Selecting each evaluator trip ID sends a request to the API and displays that response.
- The API loads the matching trip network; no global or cross-trip ID assumption is made.
- Evaluator `.route` labels are absent from UI code, request/response schemas, and inference inputs.
- Raw GPS and predicted route are distinguishable on the Folium map, with a legend and layer control.
- API request logs are visible during the demonstration.
- Errors are shown clearly; the UI does not fall back to cached or fabricated predictions.
- Latency and system metrics are measured and recorded during an actual run before they are written in the report.

## Points in the existing master plan that need review

- It specifies equirectangular projection in one section, which conflicts with the required UTM Zone 32N distance calculations above.
- Phrases such as “zero candidate miss guarantee,” candidate miss-rate figures, continuity thresholds, latency budgets, and expected HMM improvement are hypotheses or targets until measured; they must not be presented as achieved results without evidence.
- The master plan proposes transmitting raw arrays from Laptop 2, while this proposal follows the requirement that the UI send `trip_id` to the API. Please resolve where raw evaluator files will be available to the API.
- Proposed feature restrictions should be documented as model-design choices and checked for validity; they should not be described as event rules unless the official handout says so.

## Team review

Reply in the pull request or add a dated note below:

- **Akshay:** Approved / Changes requested — notes:
- **Aditya:** Approved / Changes requested — notes:
- **Abhiroop:** Approved / Changes requested — notes:
- **Aaryan:** Approved / Changes requested — notes:

**Implementation status:** Hold integration decisions above open until the owners have reviewed them. Update this note with the agreed interface and decisions before treating them as final.
