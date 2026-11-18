# PathSnap Baseline Models: Compulsory 9-Question Evaluation Report
## Subject: UE24CS352A Machine Learning Hackathon
**Author / Module Lead**: Akshay (Spatial Data Engineering, Preprocessing & Baseline Models)  
**Deliverable Section**: 3.2 Compulsory Questions for EVERY Model (Model 1 & Model 2)

---

## Model 1: Regularized Logistic Regression (L2 Ridge)

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Logistic Regression provides the mathematically rigorous linear baseline for binary candidate discrimination ($P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$).
   - In spatial map-matching, first-order physical features (such as perpendicular distance $d_{\perp}$, relative heading alignment $|\Delta\theta|$, and normalized speed ratio) exhibit monotonic, quasi-convex relationships with the probability of being the true ground-truth road segment.
   - It requires negligible memory footprint and executes in sub-microsecond latency ($0.001\text{ ms/point}$), making it ideal for edge deployment and validating whether linear hyperplanes can separate true road candidates from ambient false candidates.

2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**:
     - *Inference Speed*: $\approx 0.001\text{ ms}$ per candidate query—by far the fastest model in the arena.
     - *Interpretability*: Model weights $\mathbf{w}$ directly expose physical feature importances (e.g., negative weight on perpendicular distance and heading error).
     - *Well-Calibrated Probabilities*: Directly minimizes log-loss to output calibrated posterior probabilities suitable for HMM emission priors without temperature scaling.
   - **Limitations**:
     - *Linear Decision Boundary*: Incapable of capturing non-linear interactions between high-speed highway curves and road junction degrees.
     - *Class Imbalance Sensitivity*: Highly vulnerable to the severe $1:27$ positive-to-negative candidate imbalance unless explicit balanced class weighting is enforced.

3. **Data Preprocessing & Structural Assumptions**:
   - Requires zero-mean, unit-variance standardization via `StandardScaler` to prevent features with large physical units (e.g., $d_{\perp} \in [0, 50]\text{ m}$) from dominating angular features ($\Delta\theta \in [0, \pi]\text{ rad}$).
   - Assumes conditional independence between features and log-odds linearity.

4. **Robustness to Spatial GPS Noise**:
   - Linear hyperplanes are vulnerable to extreme multipath outliers; upstream kinematic thresholding ($v > 75\text{ m/s}$) and 2D Kalman smoothing are essential to prevent leverage points from skewing regression coefficients.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Regularization Strength: Inverse penalty $C = 1.0$ (L2 Ridge regularization).
   - Class Weight: `class_weight='balanced'` (dynamically penalizes false negatives inversely proportional to class frequencies).
   - Optimization Solver: `solver='lbfgs'`, `max_iter=500`, `tol=1e-4`.
   - Random State: Seed fixed to 42 for strict reproducibility.

6. **What loss function or optimisation objective was used?**
   - Binary Cross-Entropy with L2 Weight Regularization (Ridge Log-Loss):
     $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N} \left[ w_{\text{pos}} y_i \log(\hat{p}_i) + w_{\text{neg}} (1 - y_i) \log(1 - \hat{p}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
   - Solved via the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) quasi-Newton algorithm.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Evaluated across 10 student-test trips (19,160 GPS observations) and 5-fold `GroupKFold` grouped by `trip_id`.
   - Cross-Validation Macro F1: **0.6097**
   - Out-of-Fold Observation Top-1 Accuracy: **51.41%**
   - Student-Testing Macro F1: **0.6082**
   - Student-Testing Top-1 Accuracy: **51.28%**
   - Log Loss: **0.4721**
   - Single-Observation Latency: **0.001 ms / observation**

8. **Observation-level Top-1 Accuracy, Macro F1-score, and Log Loss Summary**:
   - **Top-1 Accuracy**: $51.41\%$ (in $51.41\%$ of GPS pings, the candidate with the highest logistic score is the exact true arc).
   - **Macro F1-score**: $0.6097$ (balanced harmonic mean across true and false candidate classes).
   - **PR-AUC**: $0.4571$ (precision-recall area under the curve).
   - **Log Loss**: $0.4721$.

9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - **Conclusion**: **Underfitting**.
   - **Evidence**:
     - The training set Macro F1 ($0.6112$) and cross-validation Macro F1 ($0.6097$) are virtually identical with zero generalization gap ($< 0.002$).
     - However, the overall Top-1 accuracy is capped at $\approx 51.4\%$, leaving $\approx 48.6\%$ of points unresolved.
     - A linear model lacks the VC-dimension and feature capacity to model complex topological road crossings and parallel service lanes, establishing that non-linear decision surfaces are strictly necessary.

---

## Model 2: Distance-Weighted $k$-Nearest Neighbors ($k$-NN)

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - $k$-NN is a non-parametric, instance-based learning algorithm that makes zero assumptions regarding the underlying distribution or linearity of the feature space.
   - Road selection in geographic networks often clusters into localized manifold topologies (e.g., highway exits exhibit specific clusters of speed ratios and heading divergences).
   - By weighting candidate neighbors inversely by Euclidean distance ($w_i = 1 / d_i$), the algorithm forms highly flexible, non-linear local decision boundaries around authentic road geometries.

2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**:
     - *Non-Linear Expressiveness*: Naturally adapts to arbitrary decision boundary shapes without explicit kernel engineering.
     - *Zero Training Time*: Instantaneous model initialization ($\mathcal{O}(1)$ fit time).
     - *Continuous Confidence Scoring*: Inverse-distance weights provide smooth, continuous emission likelihoods.
   - **Limitations**:
     - *Inference Latency & Memory Bottleneck*: Evaluating nearest neighbors across millions of candidates requires intensive tree traversal ($\mathcal{O}(K \cdot N \log M)$). Subsampling prototype sets is mandatory for sub-second execution.
     - *Curse of Dimensionality*: Euclidean distances in unweighted 14D space can become equidistant if noisy or unscaled features are present.

3. **Data Preprocessing & Structural Assumptions**:
   - Strict `StandardScaler` normalization is mandatory so that spatial metric dimensions (meters) do not numerically dwarf trigonometric differences.
   - Assumes that candidates close in 14-dimensional kinematic feature space share identical binary road-matching labels.

4. **Robustness to Spatial GPS Noise**:
   - Distance weighting provides inherent smoothing against isolated candidate outliers, but unpruned teleportation spikes can contaminate nearest-neighbor neighborhoods; upstream cleaning is critical.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Neighborhood Size: $k = 15$ neighbors (empirically calibrated: $k=5$ overfit to noise, while $k=30$ over-smoothed acute turn intersections).
   - Weighting Scheme: `weights='distance'` (weights inverse to Euclidean distance $1/d$).
   - Prototype Sample Size: `max_train = 25,000` balanced representative prototypes to maintain sub-millisecond inference.
   - Metric: Standard Euclidean distance ($\ell_2$ norm) over scaled features.

6. **What loss function or optimisation objective was used?**
   - Non-parametric posterior voting objective:
     $$\hat{P}(y=1|\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} \frac{y_i}{\|\mathbf{x} - \mathbf{x}_i\|_2 + \epsilon}}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} \frac{1}{\|\mathbf{x} - \mathbf{x}_i\|_2 + \epsilon}}$$
   - Evaluated under Macro F1 score and Top-1 observation accuracy.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Evaluated across 10 student-test trips and 5-fold `GroupKFold`.
   - Cross-Validation Macro F1: **0.7568**
   - Out-of-Fold Observation Top-1 Accuracy: **58.25%**
   - Student-Testing Macro F1: **0.7541**
   - Student-Testing Top-1 Accuracy: **58.10%**
   - Log Loss: **0.2086**
   - Single-Observation Latency: **0.216 ms / observation**

8. **Observation-level Top-1 Accuracy, Macro F1-score, and Log Loss Summary**:
   - **Top-1 Accuracy**: $58.25\%$ (an increase of $+6.84\%$ over Logistic Regression).
   - **Macro F1-score**: $0.7568$ (an increase of $+0.1471$ F1 score over linear baseline).
   - **PR-AUC**: $0.6225$.
   - **Log Loss**: $0.2086$ (substantially sharper probability distributions).

9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - **Conclusion**: **Slight Underfitting with Latency Saturation**.
   - **Evidence**:
     - With $k=15$ and distance weighting, the gap between prototype train accuracy ($62.1\%$) and validation accuracy ($58.3\%$) is modest ($\approx 3.8\%$), showing good generalization.
     - However, the model struggles on sparse road segments where candidate density is low, resulting in ambiguous local neighbor sets.
     - Furthermore, inference latency ($0.216\text{ ms/pt}$) is $216\times$ slower than Logistic Regression and $36\times$ slower than LightGBM ($0.006\text{ ms/pt}$), confirming that tree-based ensembles (Random Forest and LightGBM) are superior in both accuracy and speed.
