# PathSnap — Aaryan First-Review Viva Notes

Use these answers as a speaking guide, not as claims to memorize. Be explicit about what is already running and what still depends on the team handoffs.

## 20-second introduction

“PathSnap matches noisy GPS observations to directed road segments and reconstructs a continuous route. My part is the FastAPI service, Laptop 2 client and Folium view, and Model 5. The API health service is running, but the full matcher is not integrated yet, so I’m reporting that honestly rather than showing a fabricated prediction.”

## Questions you may get

### What does your API do?

“Laptop 1 exposes `/api/v1/health`, `/api/v1/trips`, and `/api/v1/match`. The trip list is discovered from the configured data directory. A match request includes a `trip_id`; the server loads that trip's permitted GPS and network files and passes them to the team matcher.”

### Is the model pipeline ready right now?

“The API process is healthy and its health endpoint returns HTTP 200. It currently reports `ready: false` because the matcher module and evaluator data are not configured. I have not claimed that HMM inference is live.”

### What does `ready` mean?

“It is true only when a matcher reports that its model and decoder are loaded, the configured data directory exists, and all 10 evaluator trips are available. A healthy process alone does not mean the model is ready.”

### How does the UI connect to the server?

“The Laptop 2 UI fetches trip IDs from the API, posts the selected ID to `/api/v1/match`, and renders the returned route. It does not read local trip files or cache predictions.”

### How do you prevent ground-truth leakage?

“The live API reads only raw track, nodes, and arcs for prediction. The UI receives raw GPS, the predicted route, and approved server-computed metrics. Reference labels are not in the request or response and are not used by the client.”

### What appears on your map?

“The Folium view draws the raw GPS trace in blue and the predicted route in red, with a layer control. The live evaluator view has no reference-route layer.”

### How will you train Model 5?

“I prepared an MLP trainer for the 14 agreed numeric candidate features. It uses five-fold GroupKFold grouped by trip ID, and the StandardScaler is fit inside each training fold. It checks that the input table matches the 80 training-trip directories. I’m waiting for Akshay's final feature names and candidate table before running it.”

### Why GroupKFold by trip?

“Rows from one trip are related. GroupKFold keeps every observation from a trip together, so validation trips are unseen during that fold's training.”

### What are your model results?

“I don’t have MLP results yet because the prepared candidate table and frozen feature schema have not been handed over. I will report the actual out-of-fold results after the run; I won’t estimate or invent metrics.”

### What remains before the two-laptop demo?

“We need the real feature/model/decoder integration, the evaluator inputs on Laptop 1, and a trusted shared local network. I verified health locally, but have not verified a request from the second laptop.”

### Who owns the HMM and candidate features?

“Akshay owns spatial candidates and features; Abhiroop owns the graph and HMM/Viterbi; Aditya owns the cross-validation scoreboard and tuning. My API uses an adapter contract so those pieces can be connected after their interfaces are agreed.”

## Recorded sample-trip audit (not model performance)

The repo's saved graph audit for sample trip `00000000` reports 8,542 nodes, 18,196 directed arcs, 100 weakly connected components, a largest component of 7,466 nodes, and 0 isolated nodes. It reports that the sample reference route's 179 arcs chain for 178 of 178 transitions and lie in one component. These are graph/reference-file audit values for that one sample trip; they are not full-dataset results and do not demonstrate model accuracy.

## Safe one-sentence status for evaluators

“Laptop 1's FastAPI health service is running; the model and decoder are not integrated yet, so it reports healthy but not ready. Once they are integrated, Laptop 2 will request a route and display the API result without receiving ground-truth labels.”

## Before presenting

- Keep the health endpoint visible and say `ready: false` accurately.
- Do not say the two-laptop connection has passed; only the local endpoint was verified.
- Do not claim MLP scores, miss rate, route continuity, or latency results that have not been produced by a real run.
- Disclose AI assistance according to the event policy, and explain the API, fold split, feature inputs, and data boundaries in your own words.
