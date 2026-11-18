import folium
import pandas as pd

TRIP = r"C:\Users\Abhirup Roy\Downloads\Student folder\Student folder\student_test\00000000\00000000"

track = pd.read_csv(TRIP + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
nodes = pd.read_csv(TRIP + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
arcs  = pd.read_csv(TRIP + ".arcs",  sep=r"\s+", header=None, names=["u", "v"])
route = pd.read_csv(TRIP + ".route", header=None, names=["arc_id"])

print("GPS lat range:", track.lat.min(), track.lat.max())
print("GPS lon range:", track.lon.min(), track.lon.max())

def arc_line(a):
    u, v = arcs.loc[a, "u"], arcs.loc[a, "v"]
    return [[nodes.loc[u, "lat"], nodes.loc[u, "lon"]],
            [nodes.loc[v, "lat"], nodes.loc[v, "lon"]]]

m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=16)

# Layer 1: road network near the trip (grey)
pad = 0.003
near = nodes[(nodes.lat.between(track.lat.min() - pad, track.lat.max() + pad)) &
             (nodes.lon.between(track.lon.min() - pad, track.lon.max() + pad))].index
near = set(near)
net = folium.FeatureGroup(name="Road network (grey)")
for a, row in arcs.iterrows():
    if row.u in near and row.v in near:
        folium.PolyLine(arc_line(a), color="gray", weight=2, opacity=0.6).add_to(net)
net.add_to(m)

# Layer 2: reference route (green)
ref = folium.FeatureGroup(name="Reference route (green)")
for a in route.arc_id:
    folium.PolyLine(arc_line(a), color="green", weight=5).add_to(ref)
ref.add_to(m)

# Layer 3: raw GPS points (blue)
gps = folium.FeatureGroup(name="GPS observations (blue)")
for r in track.itertuples():
    folium.CircleMarker([r.lat, r.lon], radius=2, color="blue",
                        fill=True, tooltip=f"t={r.t}").add_to(gps)
gps.add_to(m)

folium.LayerControl().add_to(m)
m.save("outputs/trip_00000000_map.html")
print("Saved outputs/trip_00000000_map.html")