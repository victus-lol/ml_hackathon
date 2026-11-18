"""Offline audit of monotonic route-to-GPS alignment distances."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd,shapely
from shapely.geometry import Point
from pathsnap import load_trip,_route_alignment,trip_ids

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for part in ['train','student_test']:
  for i,tid in enumerate(trip_ids(a.dataset_root/part),1):
   d=load_trip(a.dataset_root/part/tid,include_route=True);aligned=_route_alignment(d)
   geoms=np.asarray(d['geometries'],dtype=object)[aligned]
   points=shapely.points(d['gps_xy'][:,0],d['gps_xy'][:,1])
   distance=np.asarray(shapely.distance(geoms,points),dtype=float)
   rows.append({'partition':part,'trip_id':tid,'observations':len(distance),
      'distance_p50_m':float(np.quantile(distance,.5)),'distance_p90_m':float(np.quantile(distance,.9)),
      'distance_max_m':float(distance.max()),'fraction_within_60m':float(np.mean(distance<=60))})
   print(f"{part} {i}: {tid} p50={rows[-1]['distance_p50_m']:.1f}m within60={rows[-1]['fraction_within_60m']:.3f}",flush=True)
 out=pd.DataFrame(rows);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
 print('\nLargest p50 alignment distances:');print(out.sort_values('distance_p50_m',ascending=False).head(10).to_string(index=False))
if __name__=='__main__':main()
