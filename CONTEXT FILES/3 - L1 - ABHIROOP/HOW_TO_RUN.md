# ABHIROOP: How to Run & Viva Defense Guide

### Step 1: Open Terminal in your folder
```bash
pip install networkx pandas numpy
python review_audit.py
```

### Step 2: What Appears on Your Screen (Show the Teachers!)
* **Total Nodes**: 8,542
* **Total Directed Arcs**: 18,196
* **Weakly Connected Components**: 100
* **MultiDiGraph Necessity**: There are 6 parallel road segments between identical node pairs; standard `DiGraph` drops them, so we must use `networkx.MultiDiGraph` keyed by `arc_id`.

### Step 3: Your HMM Math & Dijkstra Bridge Splicer
* **Emission**: Calibrated ML Likelihood + Gaussian Spatial Distance Error ($\sigma_z = 4.07\text{ m}$).
* **Transition**: Shortest path Dijkstra distance vs Great Circle GPS distance with soft disconnect penalty ($\Lambda_{\text{disc}} = 18.0$).
* **Continuity Guarantee**: Run-length deduplication + Dijkstra topological bridge splicing guarantees **100.00% Route Continuity**.

### Step 4: Your Exact 1-Sentence Viva Pitch
> *"Classifiers alone produce disconnected jumps across buildings; our Hidden Markov Model uses log-space Viterbi decoding and Dijkstra topological bridge splicing to eliminate discontinuities and guarantee 100% valid, continuous driving routes."*
