#!/usr/bin/env python3
"""Low-cost D7: explain when the NLinear path corrects phase-path errors."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, pytorch_lightning as pl, torch
from torch.utils.data import DataLoader, Subset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_input_candidate_discovery_frozen import load_model
from src.dataset.data_factory import data_provider

MODELS=("weak_residual","rcrf_nlinear_plain")
def features(x):
    # x: B,L,C; all features are channel-averaged magnitudes, fixed before targets.
    a=x.detach().cpu().numpy(); cyc=a.reshape(len(a),30,24,a.shape[-1]); means=cyc.mean(2)
    amp=cyc.std(2); slope=np.abs(a[:,-96:]-a[:,-96:].mean(1,keepdims=True)).mean((1,2))
    return np.stack([np.abs(np.diff(a[:,-96:],axis=1)).mean((1,2)), slope,
      means.std(1).mean(1), amp.std(1).mean(1), np.abs(means[:,-1]-means[:,:-1].mean(1)).mean(1),
      np.abs(a[:,-96:]-a[:,-192:-96]).mean((1,2))],1)
def oof_r2(X,y):
    pred=np.zeros(len(y)); folds=np.array_split(np.arange(len(y)),5)
    for test in folds:
      train=np.setdiff1d(np.arange(len(y)),test); mu=X[train].mean(0); sd=X[train].std(0)+1e-8
      z=(X-mu)/sd; beta=np.linalg.lstsq(np.c_[np.ones(len(train)),z[train]],y[train],rcond=None)[0]
      pred[test]=np.c_[np.ones(len(test)),z[test]]@beta
    return 1-np.square(y-pred).sum()/np.square(y-y.mean()).sum()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--weak-checkpoint',type=Path,required=True);p.add_argument('--rcrf-checkpoint',type=Path,required=True);p.add_argument('--max-samples',type=int,default=512);p.add_argument('--require-cuda',action='store_true');a=p.parse_args()
 if a.output_dir.exists(): p.error('refusing to overwrite output');
 if a.require_cuda and not torch.cuda.is_available():p.error('CUDA required')
 a.output_dir.mkdir(parents=True);pl.seed_everything(2021,workers=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 models={}; exp=None
 for n,c in zip(MODELS,(a.weak_checkpoint,a.rcrf_checkpoint)):
  m,exp=load_model(n,c,192,720,2021);m.to(device).eval();models[n]=m
 ds,_=data_provider(exp.dataset_args,'val');idx=np.linspace(0,len(ds)-1,min(a.max_samples,len(ds)),dtype=int); loader=DataLoader(Subset(ds,idx.tolist()),batch_size=exp.dataset_args.batch_size,shuffle=False)
 rows=[]; arrays={}
 with torch.inference_mode():
  for n,m in models.items():
   collected=[]
   for x,y,xm,ym in loader:
    x,y,xm,ym=[v.to(device) for v in (x,y,xm,ym)]; out,_,_=m(x.float(),xm.float(),m._build_decoder_input(y.float()),ym.float()); truth=y[:,-192:].float(); phase=m.last_phase_forecast[:,-192:]; resid=m.last_residual_forecast[:,-192:]; fused=out[:,-192:]
    pe=(phase-truth).abs().mean((1,2)); fe=(fused-truth).abs().mean((1,2)); re=(resid-truth).abs().mean((1,2)); corr=resid-phase; err=truth-phase
    align=(corr*err).mean((1,2)).cpu().numpy()/(np.sqrt((corr.square().mean((1,2))*err.square().mean((1,2))).cpu().numpy())+1e-8)
    collected.append(np.c_[features(x),pe.cpu(),fe.cpu(),re.cpu(),(pe-fe).cpu(),align])
   z=np.concatenate(collected); X=z[:,:6]; gain=z[:,9]; names=['local_diff','recent_deviation','cycle_level_std','cycle_amplitude_std','last_cycle_shift','daily_lag_change']
   for j,name in enumerate(names): rows.append({'model':n,'stat':'feature_gain_corr','feature':name,'value':float(np.corrcoef(X[:,j],gain)[0,1])})
   rows.append({'model':n,'stat':'gain_oof_r2','feature':'all_six','value':float(oof_r2(X,gain))});rows.append({'model':n,'stat':'mean_phase_to_fused_mae_gain','feature':'none','value':float(gain.mean())});rows.append({'model':n,'stat':'mean_correction_alignment','feature':'none','value':float(z[:,10].mean())});arrays[n]=z
 with (a.output_dir/'internal_path_probe.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=['model','stat','feature','value']);w.writeheader();w.writerows(rows)
 np.savez_compressed(a.output_dir/'sample_path_statistics.npz',**arrays)
 (a.output_dir/'protocol.json').write_text(json.dumps({'dataset':'ETTm1','split':'validation','lookback':720,'horizon':192,'seed':2021,'sample_count':len(idx),'features':['local_diff','recent_deviation','cycle_level_std','cycle_amplitude_std','last_cycle_shift','daily_lag_change'],'target':'phase_mae-fused_mae','evaluation':'5 contiguous time folds'},indent=2)+'\n');print(a.output_dir/'internal_path_probe.csv')
if __name__=='__main__':main()
