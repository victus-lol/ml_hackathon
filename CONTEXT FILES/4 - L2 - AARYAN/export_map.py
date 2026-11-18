"""Offline student-test Folium diagnostic map with actual model and decoder outputs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,folium,matplotlib.pyplot as plt
from experiments import positive_probability
from decoder import bridge_route,decode
from pathsnap import FEATURES,_candidate_frame,load_trip

def coords(route,data):
 out=[]
 for aid in route:
  u,v=data['arcs'][int(aid)];a=data['nodes'][u].tolist();b=data['nodes'][v].tolist()
  if out and out[-1]==a:out.append(b)
  else:out.extend([a,b])
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--candidate-dir',type=Path,required=True)
 p.add_argument('--model',type=Path,required=True);p.add_argument('--decoder-config',type=Path,required=True);p.add_argument('--trip-id',default='00000000')
 p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 data=load_trip(a.dataset_root/'student_test'/a.trip_id,include_route=True)
 f=pd.read_csv(a.candidate_dir/'student_test_candidates.csv.gz',dtype={'trip_id':str})
 f=f[f.trip_id==a.trip_id].copy().reset_index(drop=True)
 bundle=joblib.load(a.model);raw=positive_probability(bundle['model'],f[FEATURES].to_numpy(np.float32))
 prob=bundle['calibrator'].predict(raw) if bundle.get('calibrator') is not None else raw
 f['_p']=prob;sets=[];greedy=[]
 for _,g in f.groupby('observation_id',sort=True):
  rows=g.to_dict('records');best=max(rows,key=lambda r:r['_p']);greedy.append(int(best['candidate_arc_id']))
  sets.append([{'arc_id':int(r['candidate_arc_id']),'p':max(float(r['_p']),1e-9),'distance_m':float(r['distance_m']),'fraction':float(r['projection_fraction'])} for r in rows])
 cfg=json.loads(a.decoder_config.read_text(encoding='utf-8'))
 obs=sorted(f.observation_id.unique())
 vit=decode(sets,data['arcs'],data['lengths'],data['gps_xy'][obs],data['track'][obs,2],beta_m=float(cfg['beta_m']),sigma_m=float(cfg['sigma_m']))
 bridged=bridge_route(vit,data['arcs'],data['lengths'])
 gps=[[float(lon),float(lat)] for lon,lat,_ in data['track']]
 classifier=coords(greedy,data);viterbi=coords(vit,data);bridge=coords(bridged);reference=coords(data['route'],data)
 center=[float(np.mean(data['track'][:,1])),float(np.mean(data['track'][:,0]))]
 m=folium.Map(location=center,zoom_start=14,control_scale=True)
 road=folium.FeatureGroup('Road network',show=False)
 for u,v in data['arcs']:
  folium.PolyLine([[data['nodes'][u,1],data['nodes'][u,0]],[data['nodes'][v,1],data['nodes'][v,0]]],color='#a8a8a8',weight=1,opacity=.25).add_to(road)
 road.add_to(m)
 def add_layer(name,points,color,weight,show=True,dash=None):
  group=folium.FeatureGroup(name,show=show)
  if len(points)>1:
   opts={'locations':[[lat,lon] for lon,lat in points],'color':color,'weight':weight,'opacity':.9,'tooltip':name}
   if dash:opts['dash_array']=dash
   folium.PolyLine(**opts).add_to(group)
  group.add_to(m)
 add_layer('Raw GPS',gps,'blue',3,True)
 add_layer('Reference route — offline student-test only',reference,'green',4,False)
 add_layer('Classifier-only route',classifier,'orange',3,False)
 add_layer('Viterbi per-observation route',viterbi,'red',4,True)
 add_layer('Viterbi bridged route — offline diagnostic',bridge,'purple',4,False,'8, 5')
 for lon,lat in gps:folium.CircleMarker([lat,lon],radius=2,color='blue',fill=True,fill_opacity=.7).add_to(m)
 folium.LayerControl(collapsed=False).add_to(m);m.fit_bounds([[lat,lon] for lon,lat in gps])
 html=a.output_dir/f'student_test_{a.trip_id}_diagnostic.html';m.save(str(html))
 fig,ax=plt.subplots(figsize=(9,7))
 ax.plot([x for x,y in gps],[y for x,y in gps],color='royalblue',lw=1.2,label='Raw GPS')
 ax.plot([x for x,y in reference],[y for x,y in reference],color='green',lw=1.3,alpha=.75,label='Reference route (offline)')
 ax.plot([x for x,y in classifier],[y for x,y in classifier],color='darkorange',lw=1.0,alpha=.8,label='Classifier only')
 ax.plot([x for x,y in viterbi],[y for x,y in viterbi],color='red',lw=1.0,alpha=.8,label='Viterbi')
 ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'Student-test trip {a.trip_id}: offline diagnostic');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
 png=a.output_dir/f'student_test_{a.trip_id}_diagnostic.png';fig.savefig(png,dpi=160);plt.close(fig)
 print('Saved',html.resolve());print('Saved',png.resolve())

if __name__=='__main__':main()
