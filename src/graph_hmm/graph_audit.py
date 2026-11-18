import networkx as nx
import pandas as pd

TRIP = r"C:\Users\Abhirup Roy\Downloads\Student folder\Student folder\student_test\00000000\00000000"

nodes = pd.read_csv(TRIP + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
arcs  = pd.read_csv(TRIP + ".arcs",  sep=r"\s+", header=None, names=["u", "v"])

G = nx.MultiDiGraph()
for i, row in nodes.iterrows():
    G.add_node(i, pos=(row["lon"], row["lat"]))
for arc_id, row in arcs.iterrows():
    G.add_edge(int(row["u"]), int(row["v"]), key=arc_id, arc_id=arc_id)

comps = list(nx.weakly_connected_components(G))
print("=== ROAD NETWORK GRAPH AUDIT (Trip 00000000) ===")
print("Total Nodes:", G.number_of_nodes())
print("Total Directed Arcs:", G.number_of_edges())
print("Weakly Connected Components:", len(comps))
print("Strongly Connected Components:", nx.number_strongly_connected_components(G))
print("Largest component size:", max(len(c) for c in comps))
print("Isolated nodes:", len(list(nx.isolates(G))))

try:
    route = pd.read_csv(TRIP + ".route", header=None, names=["arc_id"])
    ids = route["arc_id"].tolist()
    ok = sum(arcs.loc[a, "v"] == arcs.loc[b, "u"] for a, b in zip(ids, ids[1:]))
    print("Route length:", len(ids), "| chained transitions:", ok, "of", len(ids) - 1)
except FileNotFoundError:
    print("No .route file for this trip")

print("Arc rows in file:", len(arcs), "| DiGraph edges:", G.number_of_edges())

comp_of = {n: k for k, c in enumerate(comps) for n in c}
route_nodes = {arcs.loc[a, "u"] for a in ids} | {arcs.loc[a, "v"] for a in ids}
print("Route lies in", len({comp_of[n] for n in route_nodes}), "component(s)")