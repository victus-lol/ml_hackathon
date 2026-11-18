"""Structured EDA for the permitted training and student-test partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pathsnap import load_trip, trip_ids


def inspect_partition(root: Path, partition: str) -> pd.DataFrame:
    base=root/partition
    expected=80 if partition=="train" else 10
    ids=trip_ids(base)
    if len(ids)!=expected: raise ValueError(f"{partition}: expected {expected} trips, found {len(ids)}")
    rows=[]
    for tid in ids:
        folder=base/tid
        data=load_trip(folder,include_route=True)
        track=data["track"]; times=track[:,2]
        gaps=np.diff(times)
        graph=nx.DiGraph(); graph.add_nodes_from(range(len(data["nodes"]))); graph.add_edges_from(data["arcs"].tolist())
        components=list(nx.weakly_connected_components(graph))
        rows.append({
            "partition":partition,"trip_id":tid,"observations":len(track),
            "nodes":len(data["nodes"]),"arcs":len(data["arcs"]),"route_arcs":len(data["route"]),
            "track_bytes":(folder/f"{tid}.track").stat().st_size,
            "nodes_bytes":(folder/f"{tid}.nodes").stat().st_size,
            "arcs_bytes":(folder/f"{tid}.arcs").stat().st_size,
            "route_bytes":(folder/f"{tid}.route").stat().st_size,
            "missing_or_nonfinite_track_values":int((~np.isfinite(track)).sum()),
            "nonpositive_time_gaps":int((gaps<=0).sum()),
            "gaps_over_2s":int((gaps>2).sum()),"max_gap_s":float(gaps.max()) if len(gaps) else 0.,
            "longitude_min":float(track[:,0].min()),"longitude_max":float(track[:,0].max()),
            "latitude_min":float(track[:,1].min()),"latitude_max":float(track[:,1].max()),
            "weak_components":len(components),"isolated_nodes":sum(graph.degree(n)==0 for n in graph.nodes),
            "largest_component_nodes":max(map(len,components)) if components else 0,
        })
        print(f"EDA {partition} trip {tid}: T={len(track)} nodes={len(data['nodes'])} arcs={len(data['arcs'])} route={len(data['route'])}",flush=True)
    return pd.DataFrame(rows)


def run(root:Path, output:Path):
    output.mkdir(parents=True,exist_ok=True)
    frames=[inspect_partition(root,p) for p in ["train","student_test"]]
    all_df=pd.concat(frames,ignore_index=True)
    all_df.to_csv(output/"trip_inventory.csv",index=False)
    summary={"partitions":{},"coordinate_system":"WGS84 input; select a local UTM zone and hemisphere per trip",
        "raw_columns":{"track":["longitude","latitude","timestamp_s"],"nodes":["longitude","latitude"],
                       "arcs":["from_node_id","to_node_id"],"route":["ordered arc_id"]}}
    for partition,df in all_df.groupby("partition"):
        summary["partitions"][partition]={"trip_count":int(len(df)),
          "total_observations":int(df.observations.sum()),
          "observations_per_trip":{"min":int(df.observations.min()),"median":float(df.observations.median()),"max":int(df.observations.max())},
          "total_nodes":int(df.nodes.sum()),"total_arcs":int(df.arcs.sum()),
          "total_raw_bytes":int(df[["track_bytes","nodes_bytes","arcs_bytes","route_bytes"]].to_numpy().sum()),
          "missing_track_values":int(df.missing_or_nonfinite_track_values.sum()),
          "trips_with_nonpositive_time_gaps":int((df.nonpositive_time_gaps>0).sum()),
          "gaps_over_2s":int(df.gaps_over_2s.sum()),"max_gap_s":float(df.max_gap_s.max()),
          "weak_components_median":float(df.weak_components.median()),
          "isolated_nodes_total":int(df.isolated_nodes.sum())}
    (output/"eda_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,4.5))
    for p,df in all_df.groupby("partition"):
        ax.hist(df.observations,bins=12,alpha=.6,label=p)
    ax.set_title("GPS observations per trip");ax.set_xlabel("Observations");ax.set_ylabel("Trips");ax.legend();fig.tight_layout()
    fig.savefig(output/"observations_per_trip.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for p,df in all_df.groupby("partition"):
        ax.scatter(df.arcs,df.nodes,label=p,alpha=.7)
    ax.set_title("Road-network size by trip");ax.set_xlabel("Directed arcs");ax.set_ylabel("Nodes");ax.legend();fig.tight_layout()
    fig.savefig(output/"network_size.png",dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();run(a.dataset_root,a.output_dir)

if __name__=="__main__":main()
