# PathSnap Baseline Models: 9-Question Evaluation Report

**Course:** UE24CS352A – Machine Learning Hackathon  
**Author:** Akshay  
**Section:** Compulsory evaluation questions for Model 1 and Model 2

---

## Model 1: Regularized Logistic Regression (L2)

### A. Model Understanding

**1. Why was this algorithm suitable for the PathSnap problem?**

- Logistic Regression provides a simple and interpretable baseline for deciding whether a road candidate matches a GPS observation.
- It can learn how features such as perpendicular distance, heading difference and speed ratio relate to the probability of a correct road match.
- It has a small memory footprint and very low prediction latency.
- It helps establish whether the engineered features can separate correct road candidates from incorrect candidates using a linear decision boundary.

**2. What are the major strengths and limitations of this algorithm for this task?**

**Strengths**
- **Fast predictions:** Approximately 0.001 ms per observation.
- **Interpretability:** The learned weights indicate how features such as perpendicular distance and heading difference affect predictions.
- **Probability output:** It produces probability scores that can be used by later stages of the map-matching pipeline.

**Limitations**
- **Linear decision boundary:** It may not capture complex relationships involving curved roads, junctions and parallel roads.
- **Class imbalance:** The dataset contains many more negative candidates than positive candidates, so balanced class weights are important.

**3. What preprocessing and assumptions are required?**

- Features are standardized using `StandardScaler` so that measurements in metres do not dominate angular or normalized features.
- The model assumes that the relationship between the input features and the log-odds of a positive prediction is approximately linear.

**4. How robust is the model to spatial GPS noise?**

- Extreme GPS outliers can affect the learned model and its predictions.
- GPS noise filtering and Kalman smoothing before feature extraction help reduce the effect of such observations.

### B. Training and Configuration

**5. What were the important hyperparameters and their initial values?**

- **Regularization:** L2 penalty with `C = 1.0`.
- **Class weighting:** `class_weight='balanced'`.
- **Solver:** `lbfgs`.
- **Maximum iterations:** `max_iter=500`.
- **Tolerance:** `tol=1e-4`.
- **Random state:** `42`.

**6. What loss function or optimisation objective was used?**

- The model uses **binary cross-entropy (log loss)** to measure prediction error.
- L2 regularization penalizes large model weights and helps reduce overfitting.
- The optimization is performed using the `lbfgs` solver.

### C. Experimental Results

**7. What was the model's performance on the student-testing dataset?**

- **Student-test trips:** 10.
- **GPS observations:** 19,160.
- **Cross-validation:** 5-fold `GroupKFold`, grouped by `trip_id`.
- **Cross-validation Macro F1:** 0.6097.
- **Out-of-fold observation Top-1 Accuracy:** 51.41%.
- **Student-testing Macro F1:** 0.6082.
- **Student-testing Top-1 Accuracy:** 51.28%.
- **Log Loss:** 0.4721.
- **Single-observation latency:** 0.001 ms.

**8. Summarise the Top-1 Accuracy, Macro F1-score and Log Loss.**

- **Top-1 Accuracy:** 51.41% of GPS observations had the correct road candidate ranked first.
- **Macro F1-score:** 0.6097, measuring the balance between performance on the positive and negative candidate classes.
- **PR-AUC:** 0.4571.
- **Log Loss:** 0.4721.

**9. Was the model overfitting or underfitting? What evidence supports the conclusion?**

**Conclusion: Underfitting**

- Training Macro F1 was 0.6112, while cross-validation Macro F1 was 0.6097.
- The small difference indicates that the model generalizes consistently, with little evidence of overfitting.
- However, Top-1 Accuracy was approximately 51.4%, leaving substantial room for improvement.
- The linear model may not be flexible enough to represent complex road layouts, intersections and parallel service roads.

---

## Model 2: Distance-Weighted k-Nearest Neighbors (k-NN)

### A. Model Understanding

**1. Why was this algorithm suitable for the PathSnap problem?**

- k-NN is a non-parametric, instance-based learning algorithm.
- It does not require a linear relationship between the input features and the target.
- Road candidates with similar geometric and movement features may have similar matching labels.
- Distance weighting gives more influence to nearby training examples and allows the model to learn non-linear decision boundaries.

**2. What are the major strengths and limitations of this algorithm for this task?**

**Strengths**
- **Non-linear predictions:** It can model more complex decision boundaries than Logistic Regression.
- **Simple training:** The model mainly stores the training examples rather than learning a large set of parameters.
- **Distance-based confidence:** Nearby examples have more influence on the prediction.

**Limitations**
- **Prediction cost:** Finding nearest neighbours can become expensive as the training set grows.
- **Memory usage:** Storing many training examples requires additional memory, so a representative subset is used.
- **Feature scaling:** Features must be standardized so that measurements with larger numerical ranges do not dominate distance calculations.
- **High-dimensional data:** Distances can become less informative when many noisy or irrelevant features are used.

**3. What preprocessing and assumptions are required?**

- `StandardScaler` is used to normalize the input features.
- The model assumes that examples close to one another in the feature space are likely to have similar road-matching labels.
- The model uses the 14 engineered features.

**4. How robust is the model to spatial GPS noise?**

- Distance weighting reduces the influence of training examples that are far away in feature space.
- However, unfiltered GPS outliers can still affect the nearest-neighbour set.
- Upstream GPS cleaning is therefore important.

### B. Training and Configuration

**5. What were the important hyperparameters and their initial values?**

- **Number of neighbours:** `k = 15`.
- **Weighting:** `weights='distance'`.
- **Maximum training samples:** `max_train = 25,000`.
- **Distance metric:** Euclidean distance over standardized features.

**6. What loss function or optimisation objective was used?**

- k-NN does not optimize a conventional training loss in the same way as Logistic Regression.
- It estimates the positive-class probability by combining the labels of the nearest neighbours.
- Neighbours closer to the query point receive greater weight.
- Performance is evaluated using Macro F1 and observation-level Top-1 Accuracy.

### C. Experimental Results

**7. What was the model's performance on the student-testing dataset?**

- **Student-test trips:** 10.
- **Cross-validation:** 5-fold `GroupKFold`, grouped by `trip_id`.
- **Cross-validation Macro F1:** 0.7568.
- **Out-of-fold observation Top-1 Accuracy:** 58.25%.
- **Student-testing Macro F1:** 0.7541.
- **Student-testing Top-1 Accuracy:** 58.10%.
- **Log Loss:** 0.2086.
- **Single-observation latency:** 0.216 ms.

**8. Summarise the Top-1 Accuracy, Macro F1-score and Log Loss.**

- **Top-1 Accuracy:** 58.25%, an improvement of 6.84 percentage points over Logistic Regression.
- **Macro F1-score:** 0.7568, an improvement of 0.1471 over Logistic Regression.
- **PR-AUC:** 0.6225.
- **Log Loss:** 0.2086.

**9. Was the model overfitting or underfitting? What evidence supports the conclusion?**

**Conclusion: Slight underfitting with higher prediction latency**

- Prototype training accuracy was 62.1%, compared with validation accuracy of 58.3%.
- The difference is approximately 3.8 percentage points, suggesting reasonable generalization.
- The model may struggle when road segments have few similar examples in the training data.
- Its prediction latency was 0.216 ms per observation, making it slower than Logistic Regression.
- The results indicate that non-linear models can improve matching performance, but prediction speed must also be considered.
