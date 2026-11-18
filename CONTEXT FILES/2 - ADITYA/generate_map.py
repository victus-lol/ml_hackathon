import folium
import pandas as pd

track_path = "CONTEXT FILES/2 - ADITYA/sample_trip_00000000/00000000.track"
nodes_path = "CONTEXT FILES/2 - ADITYA/sample_trip_00000000/00000000.nodes"
arcs_path  = "CONTEXT FILES/2 - ADITYA/sample_trip_00000000/00000000.arcs"

track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"])
nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"])
arcs  = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"])

print(f"Loaded: {len(track)} GPS points, {len(nodes)} nodes, {len(arcs)} arcs")

center_lat = track["lat"].mean()
center_lon = track["lon"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

# Draw road arcs
for _, arc in arcs.iterrows():
    u_idx, v_idx = int(arc["u"]), int(arc["v"])
    if u_idx < len(nodes) and v_idx < len(nodes):
        u_pt = nodes.iloc[u_idx]
        v_pt = nodes.iloc[v_idx]
        folium.PolyLine(
            [(u_pt["lat"], u_pt["lon"]), (v_pt["lat"], v_pt["lon"])],
            color="#333333",
            weight=1.5,
            opacity=0.6,
        ).add_to(m)

# Draw raw GPS points
for _, row in track.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=2.5,
        color="#1E88E5",
        fill=True,
        fill_color="#1E88E5",
        fill_opacity=0.8,
        tooltip=f"t={int(row['t'])}",
    ).add_to(m)

out_file = "CONTEXT FILES/2 - ADITYA/trip_00000000_eda.html"
m.save(out_file)
print(f"Successfully saved interactive map to: {out_file}")
