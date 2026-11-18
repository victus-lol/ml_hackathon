import networkx as nx, pandas as pd

D = "CONTEXT FILES/2 - ADITYA/sample_trip_00000000/00000000"
nodes = pd.read_csv(D + ".nodes", sep=r"\s+", header=None)
arcs = pd.read_csv(D + ".arcs", sep=r"\s+", header=None)

G = nx.MultiDiGraph()
for arc_id, (u, v) in enumerate(zip(arcs[0], arcs[1])):
    G.add_edge(int(u), int(v), key=arc_id, arc_id=arc_id)

print("=== ROAD NETWORK GRAPH AUDIT (Trip 00000000) ===")
print(f"Total Nodes: {len(nodes)} | Directed Arcs: {G.number_of_edges()} | "
      f"Weakly Connected Components: {nx.number_weakly_connected_components(G)}")