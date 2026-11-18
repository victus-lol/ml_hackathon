# PathSnap Reference Materials

This folder contains the five college-provided PathSnap PDFs and the local development dataset that was available in the hackathon workspace.

## Contents

- `college_handouts/`: assessment and integrity policy, dataset exploration, deliverables, main guidelines, and model/training guidelines.
- `dataset/train/`: 80 training trips.
- `dataset/student_test/`: 10 development-evaluation trips.
- Each trip folder includes its `.track`, `.nodes`, `.arcs`, and `.route` files.

The copied dataset is about 89.75 MB. The evaluator-test partition was not present in the local workspace and is not included here.

## Data-use boundaries

- Use only `train/` to fit models and learned preprocessing.
- `student_test/` is for offline development evaluation only. Its reference files must not be read by the evaluator UI or sent over the API.
- Do not configure `student_test/` as `PATHSNAP_EVALUATOR_DIR`.
- The live evaluator directory must contain only the permitted evaluator inputs and remain separate from this development bundle.
- Coordinates in the files are longitude then latitude. Metric calculations use UTM Zone 32N (EPSG:32632), as required by the project brief.

## Example local paths

From the repository root:

```text
CONTEXT FILES/4 - L2 - AARYAN/reference_materials/dataset/train
CONTEXT FILES/4 - L2 - AARYAN/reference_materials/dataset/student_test
```

The MLP trainer's `--training-root` can point to the `train` path above after Akshay provides the prepared candidate feature table.
