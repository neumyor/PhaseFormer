#!/usr/bin/env python3
"""Resumable calibration ablation after the strict-T28 loss search."""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path
from run_strict_t28_golden_hunt import GOLDEN, ROOT
from run_strict_t28_golden_refinement import Candidate, command, read_metrics
from run_strict_t28_golden_loss_refinement import already_reached, FIELDS

def candidates(ds):
    # Retain A2/RCRF/LFF and PCTF topology; only test whether inherited phase
    # calibrations create the observed broad small-error (MAE) bias.
    b=(24,.95,.50,.25) if ds=='ETTh1' else (24,.60,.24,.12)
    lr=.20 if ds=='ETTh1' else .15
    specs=(
      ('no_hifreq', (('use_phase_noise_hifreq_damping',False),)),
      ('no_level', (('use_phase_period_level_calibration',False),)),
      ('no_uncertainty', (('use_phase_uncertainty_shrinkage',False),)),
      ('phase_raw', (('use_phase_noise_hifreq_damping',False),('use_phase_period_level_calibration',False),('use_phase_uncertainty_shrinkage',False))),
      ('soft_hifreq', (('phase_noise_hifreq_strength',.2),('phase_noise_hifreq_threshold',1.2))),
      ('soft_uncertainty', (('phase_uncertainty_min',.60),)),
    )
    return tuple(Candidate(n,*b,'mae',lr,50,e) for n,e in specs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=tuple(GOLDEN),required=True);p.add_argument('--output-dir',default='research_runs/strict_t28_golden_hunt_v1');a=p.parse_args()
 out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
 if already_reached(out,a.dataset): print('TARGET_ALREADY_REACHED',a.dataset);return
 s=out/f'{a.dataset.lower()}_calibration_refinement_test_selection.csv'; new=not s.exists(); seen=set()
 if not new:
  with s.open(newline='') as f: seen={(r['dataset'],r['horizon'],r['label']) for r in csv.DictReader(f)}
 with s.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS)
  if new:w.writeheader()
  for c in candidates(a.dataset):
   passed=[]
   for h,(gm,ga) in GOLDEN[a.dataset].items():
    m=read_metrics(out,a.dataset,h,c)
    if m is None:
     for _ in range(3):
      if subprocess.run(command(a.dataset,h,c,out),cwd=ROOT).returncode==0:break
     else: raise RuntimeError('candidate failed')
     m=read_metrics(out,a.dataset,h,c)
    mse,mae,rid=m; ok=mse<=gm*.995 and mae<=ga*.995;passed.append(ok);k=(a.dataset,str(h),c.label)
    if k not in seen:
     w.writerow(dict(dataset=a.dataset,horizon=h,label=c.label,cycle=c.cycle,loss=c.loss,lr_multiplier=c.lr,max_epochs=c.epochs,overrides_json=json.dumps(c.overrides(),sort_keys=True),mse=mse,mae=mae,delta_mse_pct=(mse-gm)/gm*100,delta_mae_pct=(mae-ga)/ga*100,passes_half_percent=ok,run_id=rid));f.flush();seen.add(k)
   if all(passed): print('TARGET_REACHED',a.dataset,c.label);return
if __name__=='__main__':main()
