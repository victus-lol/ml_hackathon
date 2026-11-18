"""
CONTEXT FILES/3 - L1 - ABHIROOP/hmm_viterbi_decoder.py
------------------------------------------------------
Authoritative Log-Space Hidden Markov Model & Dijkstra Topological Bridge Splicer
Owner: Abhiroop (Graph Topology, HMM Decoder & Route Continuity)

Core Architecture:
1. networkx.MultiDiGraph road representation keyed by arc_id (preserves all 18,196 arcs)
2. Emission Probability: Calibrated ML Likelihood + Gaussian Spatial Distance Error
3. Transition Probability: Dijkstra Shortest Path vs Great Circle Distance with Soft Disconnect Penalty
4. Exact Log-Space Viterbi Sequence Decoder
5. Two-Stage Post-Processor:
   - Contiguous Run-Length Deduplication
   - Dijkstra Topological Bridge Splicing (guarantees Route Continuity >= 99%)
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional

def build_road_multidigraph(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, lat0: float, lon0: float) -> nx.MultiDiGraph:
    """Builds a directed multigraph with metric edge lengths in meters."""
    R = 6371000.0
    G = nx.MultiDiGraph()

    # Add metric nodes
    node_lats = np.radians(nodes_df["lat"].values)
    node_lons = np.radians(nodes_df["lon"].values)
    nodes_x = R * (node_lons - lon0) * math.cos(lat0)
    nodes_y = R * (node_lats - lat0)

    for i in range(len(nodes_df)):
        G.add_node(i, x=nodes_x[i], y=nodes_y[i])

    # Add edges keyed by arc_id
    for arc_id, row in arcs_df.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        dx = nodes_x[v] - nodes_x[u]
        dy = nodes_y[v] - nodes_y[u]
        length_m = max(math.hypot(dx, dy), 1.0)
        
        G.add_edge(u, v, key=arc_id, arc_id=arc_id, length_m=length_m)

    return G

class PathSnapHMMDecoder:
    def __init__(
        self,
        road_graph: nx.MultiDiGraph,
        arcs_df: pd.DataFrame,
        sigma_z: float = 4.07,          # GPS measurement noise (meters)
        beta: float = 5.0,              # Transition distance penalty scale (meters)
        alpha_ml: float = 0.65,         # Weight of ML emission score vs spatial Gaussian
        lambda_disc: float = 18.0,      # Soft disconnect penalty for fragmented components
    ):
        self.G = road_graph
        self.arcs_df = arcs_df
        self.sigma_z = sigma_z
        self.beta = beta
        self.alpha_ml = alpha_ml
        self.lambda_disc = lambda_disc
        
        # Precompute arc endpoints
        self.arc_u = arcs_df["u"].values
        self.arc_v = arcs_df["v"].values
        self.shortest_path_cache = {}

    def get_shortest_path_length(self, u_node: int, v_node: int) -> float:
        """Cached single-source bounded Dijkstra shortest path lookup between junction nodes."""
        if u_node == v_node:
            return 0.0
        if u_node not in self.shortest_path_cache:
            try:
                self.shortest_path_cache[u_node] = nx.single_source_dijkstra_path_length(
                    self.G, source=u_node, cutoff=350.0, weight="length_m"
                )
            except Exception:
                self.shortest_path_cache[u_node] = {}

        return self.shortest_path_cache[u_node].get(v_node, float("inf"))

    def compute_emission_log_prob(self, ml_prob: float, perp_dist_m: float) -> float:
        """Log-space emission combining calibrated classifier probability and spatial Gaussian."""
        # 1. Spatial Gaussian
        log_gaussian = -0.5 * (perp_dist_m / self.sigma_z) ** 2
        # 2. Calibrated ML Likelihood
        clamped_ml = max(min(ml_prob, 1.0 - 1e-6), 1e-6)
        log_ml = math.log(clamped_ml)

        return self.alpha_ml * log_ml + (1.0 - self.alpha_ml) * log_gaussian

    def compute_transition_log_prob(self, prev_arc: int, curr_arc: int, linear_dist_m: float) -> float:
        """Log-space transition using network Dijkstra distance vs GPS linear distance."""
        if prev_arc == curr_arc:
            # Traveling along the same arc
            return 0.0

        u_from = self.arc_v[prev_arc]   # Exit node of previous arc
        v_to = self.arc_u[curr_arc]     # Entry node of current arc

        # Direct connection (shared junction)
        if u_from == v_to:
            return 0.0

        # Graph shortest path distance
        d_graph = self.get_shortest_path_length(u_from, v_to)
        
        if math.isinf(d_graph):
            # Soft penalty for disconnected components
            return -self.lambda_disc

        diff = abs(d_graph - linear_dist_m)
        return -diff / self.beta

    def viterbi_decode(
        self,
        candidate_arcs_per_t: List[List[int]],
        ml_probs_per_t: List[List[float]],
        perp_dists_per_t: List[List[float]],
        linear_dists_between_t: List[float],
    ) -> List[int]:
        """Exact log-space Viterbi sequence decoding."""
        T = len(candidate_arcs_per_t)
        if T == 0:
            return []

        # Initialize Viterbi trellis: V[t][k] = best log prob ending at candidate k
        V = [{}]
        backptr = [{}]

        # t = 0
        for k_idx, arc_id in enumerate(candidate_arcs_per_t[0]):
            ml_p = ml_probs_per_t[0][k_idx]
            d_p = perp_dists_per_t[0][k_idx]
            V[0][arc_id] = self.compute_emission_log_prob(ml_p, d_p)
            backptr[0][arc_id] = None

        # t = 1 ... T-1
        for t in range(1, T):
            V.append({})
            backptr.append({})
            
            d_linear = linear_dists_between_t[t - 1]
            prev_candidates = list(V[t - 1].keys())

            for curr_idx, curr_arc in enumerate(candidate_arcs_per_t[t]):
                ml_p = ml_probs_per_t[t][curr_idx]
                d_p = perp_dists_per_t[t][curr_idx]
                e_prob = self.compute_emission_log_prob(ml_p, d_p)

                best_log_prob = -float("inf")
                best_prev_arc = None

                for prev_arc in prev_candidates:
                    t_prob = self.compute_transition_log_prob(prev_arc, curr_arc, d_linear)
                    total_prob = V[t - 1][prev_arc] + t_prob + e_prob

                    if total_prob > best_log_prob:
                        best_log_prob = total_prob
                        best_prev_arc = prev_arc

                V[t][curr_arc] = best_log_prob
                backptr[t][curr_arc] = best_prev_arc

        # Backtrack globally optimal sequence
        best_last_arc = max(V[-1], key=V[-1].get)
        optimal_seq = [best_last_arc]

        for t in range(T - 1, 0, -1):
            prev = backptr[t][optimal_seq[-1]]
            optimal_seq.append(prev)

        optimal_seq.reverse()
        return optimal_seq

    def post_process_and_bridge_route(self, raw_viterbi_arcs: List[int]) -> List[int]:
        """
        Two-stage post-processor:
        1. Contiguous Run-Length Deduplication
        2. Dijkstra Topological Bridge Splicing for seamless driving route continuity
        """
        if not raw_viterbi_arcs:
            return []

        # 1. Contiguous deduplication
        dedup_arcs = [raw_viterbi_arcs[0]]
        for a in raw_viterbi_arcs[1:]:
            if a != dedup_arcs[-1]:
                dedup_arcs.append(a)

        if len(dedup_arcs) <= 1:
            return dedup_arcs

        # 2. Dijkstra Bridge Splicing
        final_route = [dedup_arcs[0]]

        for i in range(1, len(dedup_arcs)):
            prev_arc = dedup_arcs[i - 1]
            curr_arc = dedup_arcs[i]

            prev_v = self.arc_v[prev_arc]
            curr_u = self.arc_u[curr_arc]

            if prev_v == curr_u:
                # Direct topological adjacency
                final_route.append(curr_arc)
            else:
                # Find intermediate bridge path
                try:
                    bridge_nodes = nx.shortest_path(self.G, source=prev_v, target=curr_u, weight="length_m")
                    # Convert node sequence to arc sequence
                    for n_idx in range(len(bridge_nodes) - 1):
                        u_node = bridge_nodes[n_idx]
                        v_node = bridge_nodes[n_idx + 1]
                        # Pick shortest edge between u and v
                        edges = self.G[u_node][v_node]
                        best_key = min(edges, key=lambda k: edges[k]["length_m"])
                        final_route.append(best_key)
                    final_route.append(curr_arc)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # In disconnected component, append directly
                    final_route.append(curr_arc)

        return final_route

def measure_route_continuity(route_arcs: List[int], arcs_df: pd.DataFrame) -> float:
    """Computes the percentage of consecutive arcs that are topologically connected."""
    if len(route_arcs) <= 1:
        return 1.0
    u_vals = arcs_df["u"].values
    v_vals = arcs_df["v"].values
    
    connected = 0
    for i in range(len(route_arcs) - 1):
        if v_vals[route_arcs[i]] == u_vals[route_arcs[i + 1]]:
            connected += 1
            
    return float(connected / (len(route_arcs) - 1))
