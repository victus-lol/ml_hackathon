"""Candidate-state HMM, log-space Viterbi, and topological bridge splicing."""
from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx
import numpy as np


def make_graph(arcs: np.ndarray, lengths: np.ndarray) -> nx.DiGraph:
    graph = nx.DiGraph()
    for arc_id, ((u, v), length) in enumerate(zip(arcs, lengths)):
        old = graph.get_edge_data(int(u), int(v), {}).get("weight", math.inf)
        if float(length) < old:
            graph.add_edge(int(u), int(v), weight=float(length), arc_id=int(arc_id))
    return graph


def decode(candidates: list[list[dict]], arcs: np.ndarray, lengths: np.ndarray,
           gps_xy: np.ndarray, times: np.ndarray, beta_m: float = 50.0,
           sigma_m: float = 25.0, gap_reset_s: float = 10.0) -> list[int]:
    """Decode one arc per observation. Each candidate has arc_id, p, and distance_m."""
    if not candidates:
        return []
    graph = make_graph(arcs, lengths)

    @lru_cache(maxsize=1024)
    def distances_from(a: int, cutoff: int) -> dict:
        if a not in graph:
            return {}
        return nx.single_source_dijkstra_path_length(graph, a, cutoff=cutoff, weight="weight")

    decoded: list[int] = []
    start = 0
    while start < len(candidates):
        end = start + 1
        while end < len(candidates) and times[end] - times[end - 1] <= gap_reset_s:
            end += 1
        segment = candidates[start:end]
        costs: list[np.ndarray] = []
        back: list[np.ndarray] = []
        first = segment[0]
        costs.append(np.array([
            -math.log(max(float(c["p"]), 1e-9)) + float(c["distance_m"])**2 / (2 * sigma_m**2)
            for c in first
        ], dtype=float))
        for local_t in range(1, len(segment)):
            prev, curr = segment[local_t - 1], segment[local_t]
            observed = float(np.linalg.norm(gps_xy[start + local_t] - gps_xy[start + local_t - 1]))
            cutoff=int(math.ceil((observed+9*beta_m)/50.0)*50)
            prev_arc = [int(x["arc_id"]) for x in prev]
            curr_arc = [int(x["arc_id"]) for x in curr]
            trans = np.full((len(prev), len(curr)), 9.0, dtype=float)
            for i, pa in enumerate(prev_arc):
                pu, pv = arcs[pa]
                for j, ca in enumerate(curr_arc):
                    cu, cv = arcs[ca]
                    d = float(distances_from(int(pv),cutoff).get(int(cu),math.inf))
                    # Include the portions of the two candidate arcs between projections.
                    if math.isfinite(d):
                        d += max(0.0, (1.0 - float(prev[i].get("fraction", .5))) * lengths[pa])
                        d += max(0.0, float(curr[j].get("fraction", .5)) * lengths[ca])
                        trans[i, j] = abs(d - observed) / max(beta_m, 1e-6)
                    elif pv == cu:
                        trans[i, j] = abs(observed) / max(beta_m, 1e-6)
            emissions = np.array([
                -math.log(max(float(c["p"]), 1e-9)) + float(c["distance_m"])**2 / (2 * sigma_m**2)
                for c in curr
            ])
            total = costs[-1][:, None] + trans + emissions[None, :]
            back.append(np.argmin(total, axis=0).astype(np.int32))
            costs.append(total.min(axis=0))
        state = int(np.argmin(costs[-1]))
        chosen = [state]
        for t in range(len(segment) - 1, 0, -1):
            state = int(back[t - 1][state])
            chosen.append(state)
        chosen.reverse()
        decoded.extend(int(segment[t][state]["arc_id"]) for t, state in enumerate(chosen))
        start = end
    return decoded


def bridge_route(route: list[int], arcs: np.ndarray, lengths: np.ndarray) -> list[int]:
    """Deduplicate adjacent states and splice directed shortest paths between them."""
    if not route:
        return []
    unique = [int(route[0])]
    for arc in route[1:]:
        if int(arc) != unique[-1]:
            unique.append(int(arc))
    graph = make_graph(arcs, lengths)
    out = [unique[0]]
    for left, right in zip(unique, unique[1:]):
        _, left_end = arcs[left]
        right_start, _ = arcs[right]
        if left_end != right_start:
            try:
                distance=nx.shortest_path_length(graph,int(left_end),int(right_start),weight="weight")
                # Avoid inventing a very long connector across disconnected-looking
                # sequences or long GPS gaps. Leave those as explicit breaks.
                if distance<=1000.0:
                    nodes = nx.shortest_path(graph, int(left_end), int(right_start), weight="weight")
                    for a, b in zip(nodes, nodes[1:]):
                        edge = graph[a][b]
                        if edge["arc_id"] not in out[-1:]:
                            out.append(int(edge["arc_id"]))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        if right != out[-1]:
            out.append(right)
    return out


def continuity(route: list[int], arcs: np.ndarray) -> float:
    if len(route) < 2:
        return 1.0
    return float(np.mean([arcs[a, 1] == arcs[b, 0] for a, b in zip(route, route[1:])]))
