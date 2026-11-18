"""Summarize reference-arc availability in the generated candidate pools."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[]
 for part in ['train','student_test']:
  f=pd.read_csv(a.candidate_dir/f'{part}_candidates.csv.gz',dtype={'trip_id':str})
  groups=f.groupby(['trip_id','observation_id']).label.sum().rename('label_sum').reset_index()
  for tid,g in groups.groupby('trip_id'):
   misses=int((g.label_sum<0).sum());rows.append({'partition':part,'trip_id':tid,'observations':len(g),
       'candidate_misses':misses,'candidate_miss_rate':misses/len(g),
       'observations_with_candidate_truth':int((g.label_sum>0).sum())})
 out=pd.DataFrame(rows).sort_values(['partition','candidate_miss_rate'],ascending=[True,False]);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
 print(out.groupby('partition').agg(trips=('trip_id','count'),observations=('observations','sum'),misses=('candidate_misses','sum'),mean_miss_rate=('candidate_miss_rate','mean')).to_string())
 print('\nHighest miss rates:');print(out.head(8).to_string(index=False))
if __name__=='__main__':main()
