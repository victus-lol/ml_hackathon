# 02 - ABHIROOP: Tasks Before 11:30 AM & Handoff Prompt

### Exactly What Abhiroop Needs Open on Screen at 11:30 AM:
1. **Screen 1**: Terminal running a NetworkX graph summary on Trip `00000000` showing:
   - Total road junction nodes.
   - Total directed road segments (arcs).
   - Number of weakly connected components.
2. **Screen 2**: HMM Formulation note or code snippet ready to show:
   - Emission: Calibrated ML probability + Gaussian distance measurement.
   - Transition: Shortest path Dijkstra distance vs Great Circle GPS distance.
3. **Abhiroop's 1-Sentence Viva Pitch**:
   > *"Classifiers alone produce disconnected jumps across buildings; our Hidden Markov Model uses log-space Viterbi decoding and Dijkstra topological bridge splicing to eliminate discontinuities and guarantee valid, continuous driving routes."*

---

### COPY-PASTE HANDOFF PROMPT FOR ABHIROOP (Send to Abhiroop now):

```markdown
Hi Abhiroop! Here is your exact task checklist and screen setup for our 11:30 AM Phase 1 review:

1. PULL LATEST REPO:
Run `git pull origin main` in your terminal right now.

2. YOUR 11:30 REVIEW DELIVERABLES (WHAT MUST BE ON YOUR SCREEN):
- Have a Python terminal open showing a loaded NetworkX directed graph for Trip 00000000 printing total nodes, total arcs, and graph connectivity stats.
- Have the HMM math ready (emission: ML score + Gaussian spatial distance; transition: Dijkstra shortest-path distance).

3. QUICK CODE TO RUN & DISPLAY YOUR GRAPH STATS IF NOT DONE:
```python
import networkx as nx, pandas as pd

nodes = pd.read_csv("Student folder/Student folder/student_test/00000000/00000000.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
arcs = pd.read_csv("Student folder/Student folder/student_test/00000000/00000000.arcs", sep=r"\s+", header=None, names=["u", "v"])

G = nx.DiGraph()
for i, row in nodes.iterrows():
    G.add_node(i, pos=(row["lon"], row["lat"]))
for arc_id, row in arcs.iterrows():
    G.add_edge(int(row["u"]), int(row["v"]), arc_id=arc_id)

print(f"=== ROAD NETWORK GRAPH AUDIT (Trip 00000000) ===")
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Directed Arcs: {G.number_of_edges()}")
print(f"Weakly Connected Components: {nx.number_weakly_connected_components(G)}")
```

4. YOUR EXACT 1-SENTENCE VIVA ANSWER WHEN TEACHERS ASK:
"Classifiers alone produce disconnected jumps across buildings; our Hidden Markov Model uses log-space Viterbi decoding and Dijkstra topological bridge splicing to eliminate discontinuities and guarantee valid, continuous driving routes."
```
