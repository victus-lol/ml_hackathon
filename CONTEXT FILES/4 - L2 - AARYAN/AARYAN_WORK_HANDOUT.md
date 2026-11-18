# PathSnap — Aaryan Work Handout

**Updated:** 18 November 2026
**Owner:** Aaryan (Laptop 1 API / Laptop 2 UI integration)
**Scope:** Work requested for the PathSnap hackathon, recorded without claiming unfinished integration or unmeasured results.

## My assigned work

- Build the FastAPI service and connect the two-laptop flow.
- Build the evaluator UI with Folium maps.
- Train and compare Model 5 (MLP) after Akshay freezes the feature schema.
- Maintain my README/dependency/run instructions and contribute the deployment section to the consolidated report.
- Keep all edits in `CONTEXT FILES/4 - L2 - AARYAN/`. The other three teammate folders remain read-only for me.

## Work completed so far

### Repo and rules review

- Pulled the latest `origin/main` before starting implementation.
- Read the updated team handoffs, global rules, both master plans, all teammate READMEs, and Aditya's model/evaluation code.
- Checked the local hackathon data layout earlier: 80 training trips and 10 student-test trips. The repo does not contain the 10 evaluator trips.
- Created the integration review handout. [PR #1](https://github.com/victus-lol/ml_hackathon/pull/1) was merged.
- Added the current API, UI, MLP workflow, and setup notes. [PR #3](https://github.com/victus-lol/ml_hackathon/pull/3) was merged.
- Copied the five college handouts and the available local development data into `reference_materials/` inside my folder: 80 training trips and 10 student-test trips. The evaluator partition was not present and is not included.
- Prepared a first-review viva guide in `VIVA_FIRST_REVIEW.md`.
- Drafted Report Section 6 in `REPORT_SECTION_6_DRAFT.md`; measured performance fields remain explicitly marked as not measured.

### API and UI code

- `server.py` starts the FastAPI app with `uvicorn server:app --host 0.0.0.0 --port 8000`.
- `app_server.py` provides `GET /api/v1/health`, `GET /api/v1/trips`, and `POST /api/v1/match`.
- Trip IDs are discovered from the configured server-side directory; the list is not hardcoded.
- The match request contains the selected `trip_id`. The API reads only that trip's raw `.track`, `.nodes`, and `.arcs` files. It does not load the reference route file.
- `pipeline_contract.py` defines the integration interface for the team matcher. The API returns 503 when that real matcher is missing, instead of creating a dummy prediction.
- `ui_client.py` requests trip IDs and results from the API and draws blue raw GPS and a red predicted route. It does not load trip files or cache predictions.
- `requirements.txt` and the Aaryan README provide dependencies and run instructions.

### MLP work

- `train_mlp.py` implements five-fold `GroupKFold` by `trip_id`, with `StandardScaler` inside each fold.
- It requires the feature list to contain 14 agreed numeric columns and checks that input trip IDs exactly match the 80 directories in the supplied training root.
- It records real out-of-fold probabilities, macro F1, top-1 accuracy, precision, recall, log loss, candidate miss rate, training time, and inference time when run.
- The script has not been run: Akshay's frozen feature names and candidate table are not present yet. No MLP metrics or weights are claimed.

## What has actually been verified

- Uvicorn started from this folder and the health endpoint returned HTTP 200 locally.
- The response reported `status: healthy` and `ready: false`.
- `ready` is false because the matcher module and evaluator data directory are not configured; the service reports zero available evaluator trips. This is the accurate state, not a demo failure hidden by a hardcoded `ready: true`.
- Python syntax was checked for the API, adapter, UI, MLP script, and Uvicorn entry point.
- The code changes in PR #3 stayed inside Aaryan's folder.
- The reference bundle contains 5 PDFs and 90 trips (360 trip files), about 90.58 MB total. The largest file is about 3.68 MB.
- The latest repo's saved graph audit for sample trip `00000000` reports 8,542 nodes, 18,196 arcs, 100 weakly connected components, and 0 isolated nodes. Those are sample-trip audit values, not model results.

## Still pending

- **Akshay:** provide the final 14 feature names/order and training candidate table.
- **Aditya:** agree the MLP artifact and scoreboard interface.
- **Abhiroop:** provide the decoder integration and route geometry/metrics contract.
- **Team:** resolve where evaluator raw files live. Repo notes describe Laptop 2 sending raw arrays, while the project brief requires sending a `trip_id`; any chosen contract must keep ground-truth labels off the wire and out of the UI.
- Configure Laptop 1 with the real matcher and exactly 10 permitted evaluator trips. Then the API can report `ready: true` based on actual state.
- Run the UI after installing its dependencies and integrating the real matcher.
- Complete a cross-laptop check on a trusted shared hotspot/private network. The current Wi-Fi profile was marked Public, so no firewall exception was added and remote Laptop 2 access was not verified.
- Train Model 5 after the feature handoff, then provide its real outputs to Aditya's scoreboard.
- Assemble the final one-PDF report from actual team results; include the deployment section and Model 5's nine required report answers.
- Publish this handout, viva notes, and reference bundle on the shared repository after this change is committed.

## Review notes from the repo

- The teammate's sample health code hardcodes `ready: true`; that would overstate readiness without a loaded model and data.
- Its sample trip list hardcodes only five IDs. The implementation instead discovers trip IDs from the configured data directory.
- The repo's API descriptions conflict about raw arrays versus trip-ID-only requests. Resolve the exact request schema with the team before wiring in the final matcher.
- Some plan figures are targets, not measured results. Do not present latency, miss rate, continuity, or model improvement as achieved until a real run produces them.
- AI assistance was used in drafting/reviewing code and documentation; disclose it according to the event policy and be ready to explain every design choice.

## Accurate first-review explanation

“The FastAPI health service is running on Laptop 1. The matcher and evaluator data are not integrated yet, so the service reports healthy but not ready. Laptop 2 will call the API and render its prediction once that integration is complete, with no ground-truth labels sent over the network.”
