"""Re-run MLP with GroupKFold as its only validation mechanism."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from experiments import make_models,downsample_training,metrics,positive_probability
from pathsnap import FEATURES

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 all_train=pd.read_csv(a.candidate_dir/'train_candidates.csv.gz',dtype={'trip_id':str}); all_test=pd.read_csv(a.candidate_dir/'student_test_candidates.csv.gz',dtype={'trip_id':str})
 train=all_train[all_train.label>=0].copy().reset_index(drop=True); test=all_test[all_test.label>=0].copy().reset_index(drop=True)
 x=train[FEATURES].to_numpy(np.float32);y=train.label.to_numpy(np.int8);groups=train.trip_id.to_numpy()
 tx=all_test[FEATURES].to_numpy(np.float32);ty=test.label.to_numpy(np.int8);cv=GroupKFold(5);oof=np.full(len(train),np.nan,np.float32);folds=[];fit_time=infer_time=0.
 model=make_models(a.seed)['MLP']
 for k,(itr,iva) in enumerate(cv.split(x,y,groups),1):
  fit_idx=downsample_training(train.iloc[itr],a.seed+k).index.to_numpy();m=__import__('sklearn.base',fromlist=['clone']).clone(model)
  t=time.perf_counter();m.fit(x[fit_idx],y[fit_idx]);fit_time+=time.perf_counter()-t
  t=time.perf_counter();pp=positive_probability(m,x[iva]);infer_time+=time.perf_counter()-t;oof[iva]=pp
  val=train.iloc[iva].reset_index(drop=True);ids=set(val.trip_id);full=all_train[all_train.trip_id.isin(ids)]
  z=metrics(val,y[iva],pp,full_frame=full);z['fold']=k;folds.append(z);print(f"fold {k}/5 macro-F1={z['candidate_macro_f1']:.4f} Top1={z['top1_accuracy']:.4f}",flush=True)
 calibrator=IsotonicRegression(out_of_bounds='clip').fit(oof,y)
 final=__import__('sklearn.base',fromlist=['clone']).clone(model);fit_idx=downsample_training(train,a.seed).index.to_numpy()
 t=time.perf_counter();final.fit(x[fit_idx],y[fit_idx]);final_fit=time.perf_counter()-t
 ptrain=calibrator.predict(positive_probability(final,x));train_metrics=metrics(train,y,ptrain,full_frame=all_train)
 t=time.perf_counter();ptest_all=calibrator.predict(positive_probability(final,tx));test_infer=time.perf_counter()-t
 test_metrics=metrics(test,ty,ptest_all[all_test.label.to_numpy()>=0],full_frame=all_test)
 cv_metrics=metrics(train.reset_index(drop=True),y,oof,full_frame=all_train)
 cv_metrics.update({'fold_macro_f1_mean':float(np.mean([f['candidate_macro_f1'] for f in folds])),
    'fold_macro_f1_std':float(np.std([f['candidate_macro_f1'] for f in folds])),
    'folds':folds,'fit_seconds':fit_time,'inference_seconds':infer_time,
    'final_fit_seconds':final_fit,'sampled_training_rows':int(len(fit_idx))})
 test_metrics['inference_seconds']=test_infer
 bundle={'model':final,'calibrator':calibrator,'features':FEATURES,'model_name':'MLP',
    'calibrator_method':'IsotonicRegression fitted on training-only OOF predictions'}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'models').mkdir(exist_ok=True)
 joblib.dump(bundle,a.output_dir/'models'/'mlp.joblib')
 report={'cv':cv_metrics,'train_fit':train_metrics,'student_test':test_metrics}
 (a.output_dir/'mlp_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 est=final.steps[-1][1];curve=pd.DataFrame({'epoch':np.arange(1,len(est.loss_curve_)+1),'training_loss':est.loss_curve_})
 curve.to_csv(a.output_dir/'mlp_training_loss.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,4));ax.plot(curve.epoch,curve.training_loss);ax.set_title('MLP training loss (actual final estimator)');ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.output_dir/'mlp_training_loss.png',dpi=160);plt.close(fig)
 print(json.dumps({'cv_macro_f1':cv_metrics['fold_macro_f1_mean'],'student_test':test_metrics,'n_iter':est.n_iter_,'loss_curve_length':len(est.loss_curve_)},indent=2))
if __name__=='__main__':main()
