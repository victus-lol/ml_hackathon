# ADITYA: How to Run & Viva Defense Guide

### Step 1: Files to Keep Visible on Screen
1. `evaluate.py`: Anti-Memorization Feature Invariance Check (`verify_feature_matrix_invariance`) & 5-fold `GroupKFold`.
2. `optuna_tune.py`: 50-Trial Bayesian Hyperparameter Optimization Engine.
3. `models/scoreboard.csv`: Live 5-model benchmark scoreboard.

### Step 2: The 5-Model Benchmark Scoreboard (Quote These Numbers!)
| Model | Macro F1 | Observation Top-1 Acc | PR-AUC | Log Loss | Latency (ms/pt) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 4: LightGBM (Optuna GBDT)** 🏆 | **0.8509** | **67.13%** | **0.7411** | **0.1262** | **0.006 ms** |
| **Model 3: Random Forest (Bagging)** | 0.8265 | 66.92% | 0.7428 | 0.1541 | 0.039 ms |
| **Model 5: Multi-Layer Perceptron (MLP)** | 0.8325 | 65.59% | 0.7318 | 0.1273 | 0.002 ms |
| **Model 2: Distance-Weighted k-NN** | 0.7568 | 58.25% | 0.6225 | 0.2086 | 0.216 ms |
| **Model 1: Logistic Regression (L2)** | 0.6097 | 51.41% | 0.4571 | 0.4721 | 0.001 ms |

### Step 3: Your Exact 1-Sentence Viva Pitch
> *"We enforce 5-fold GroupKFold by trip_id so no trip leaks across splits, strip all local road identifiers to prevent subgraph memorization, and use Optuna Bayesian optimization to achieve an 0.8509 Macro F1 score on our production LightGBM model."*
