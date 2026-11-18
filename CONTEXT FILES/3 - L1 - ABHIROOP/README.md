# 3 - L1 - ABHIROOP: Road Graph, HMM & Viterbi Decoder (Laptop 1 Engine)

### Primary Deliverables
1. **Directed Road Graph Assembly**: Build a directed graph $G = (V, E)$ in NetworkX using `.nodes` and `.arcs`, storing metric physical lengths on edges.
2. **HMM Emission Probability Formulation**:
   $$P(z_t | s_t) \propto P_{\text{ML}}(s_t) \times \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{d(z_t, s_t)^2}{2\sigma^2}\right)$$
   Combining Aditya's calibrated ML confidence with Gaussian spatial projection distance.
3. **HMM Transition Probability Formulation**:
   $$P(s_t | s_{t-1}) \propto \frac{1}{\beta} \exp\left(-\frac{|d_{\text{graph}}(s_{t-1}, s_t) - d_{\text{greatcircle}}(z_{t-1}, z_t)|}{\beta}\right)$$
   Using shortest-path driving distance on the road network (Dijkstra) with smooth exponential penalties for disconnected components.
4. **Log-Space Viterbi Dynamic Programming**: Decode the globally optimal sequence of hidden road segments in log-space to prevent underflow.
5. **Dijkstra Topological Bridge Splicing**: Apply run-length deduplication and bridge splicing (`nx.shortest_path`) to reconnect gaps from GPS dropouts.
6. **Metrics**: Measure Candidate Miss Rate and Route Continuity (targeting $\ge 99\%$). Author Report Section 5.
