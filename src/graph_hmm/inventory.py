import os, glob
import pandas as pd

DATA_DIR = "data"   # change if your dataset is elsewhere

def read_ws(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)

files = {ext: sorted(glob.glob(f"{DATA_DIR}/**/*.{ext}", recursive=True))
         for ext in ["track", "nodes", "arcs", "route"]}
for ext, fl in files.items():
    print(f"{ext}: {len(fl)} files")

if not files["track"]:
    print("No .track files found. Check DATA_DIR and that the dataset is in the data folder.")
else:
    t = files["track"][0]
    print("\nFirst track file:", t)
    print("Folder contents:", os.listdir(os.path.dirname(t)))

    track = read_ws(t, ["lon", "lat", "t"])
    print(track.head())
    print("rows:", len(track))
    print("lon range:", track.lon.min(), track.lon.max())
    print("lat range:", track.lat.min(), track.lat.max())
    print("time gaps:\n", track.t.diff().value_counts().head())