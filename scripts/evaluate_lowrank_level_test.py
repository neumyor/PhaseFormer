#!/usr/bin/env python3
"""Evaluate the frozen static-component candidates without further training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lowrank_interaction_stage2 import save_json,digest


def main():
    import yaml
    p=argparse.ArgumentParser()
    p.add_argument('--training-scratch-dir',type=Path,required=True)
    p.add_argument('--scratch-dir',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    training=args.training_scratch_dir.resolve();scratch=args.scratch_dir.resolve();audit=args.audit_dir.resolve()
    original=json.loads((training/'manifest.json').read_text())
    assert json.loads((training/'audit_complete.json').read_text())['status']=='passed'
    assert original['study'] in ('level','parallel') and not original['smoke']
    parallel=original['study']=='parallel'
    frozen=json.loads((training/'frozen_selection.json').read_text()) if parallel else None
    arms=(original['arms'] if parallel else ['i0','tp_pre','components_static','tp_pre_components_static'])
    retained=(frozen['retained_arms'] if parallel else ['components_static','tp_pre_components_static'])
    baselines=(original['baselines'] if parallel else {'i0':'i0','tp_pre':'i0','components_static':'i0','tp_pre_components_static':'components_static'})
    sources=copy.deepcopy(original['sources']);checkpoints=[]
    for source in sources:
        source['training_setting']=source['setting']
        assert source['setting'].endswith('_val')
        source['setting']=source['setting'][:-4]+'_test'
        for arm in arms:
            d=training/'runs'/f"{source['training_setting']}__{arm}"
            result=json.loads((d/'result.json').read_text());cfg=json.loads((d/'config.json').read_text())
            assert not result['pretrained_checkpoint_loaded']
            assert Path(cfg['checkpoint']).is_file()
            assert digest(Path(cfg['checkpoint']))==result['checkpoint_sha256']
            checkpoints.append({'setting':source['setting'],'config_id':arm,
                'path':cfg['checkpoint'],'sha256':result['checkpoint_sha256']})
    manifest={'sources':sources,'arms':arms,'retained_arms':retained,'baselines':baselines,
        'study':original['study'],'evaluation_split':'test','training_scratch_dir':str(training),
        'epochs':original['epochs'],'smoke':False,'training_code_commit':original['code_commit'],
        'comparisons':[['components_static','i0'],['tp_pre_components_static','tp_pre'],
            ['tp_pre_components_static','components_static'],['tp_pre','i0']]}
    if parallel:
        manifest['comparisons']=original['comparisons']
        manifest['validation_frozen_selection']=frozen
    if (scratch/'manifest.json').exists():
        assert json.loads((scratch/'manifest.json').read_text())==manifest,'Refuse changed test selection'
    save_json(scratch/'manifest.json',manifest)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    settings=[{'setting':s['setting'],'dataset':s['config']['dataset'],'horizon':s['config']['horizon'],
        'seed':s['config']['seed'],'lookback':720,'split':'test'} for s in sources]
    run={'experiment_id':audit.name,'code':{'commit':code,'training_commit':original['code_commit']},
        'mechanism':{'description':'Frozen static level/shape fusion and pre-norm tradeoff candidate','arms':arms,'baselines':baselines},
        'experiment':{'settings':settings,'training':{'additional_training':False,'checkpoint':'registered lowest validation Huber'},'metrics':['MSE','MAE']},
        'execution':{'settings':[{'setting':s['setting'],'commands':[sys.argv]} for s in sources],'registered_checkpoints':checkpoints},
        'selection':{'source':'validation','default_candidate':frozen['default_candidate'] if parallel else 'components_static',
            'tradeoff_candidate':None if parallel else 'tp_pre_components_static','test_set_selection':False,
            'prior_test_exposure':True,'selected_configs':[{'setting':s['setting'],'config_id':retained,
                'search_notes':'Frozen before test; failed dynamic level arms excluded; static pre-norm is a secondary MSE tradeoff, not default'} for s in sources]},
        'analysis':{'ranking_metric':'mae','top_k':3,'dedup_rule':'same-channel starts at least H apart per category'},
        'validation':{'status':'incomplete'}}
    if parallel:
        run['mechanism']['description']='Registered parallel post-norm/static-component hypothesis and controls'
        run['selection']['validation_decision']=frozen
        for record in run['selection']['selected_configs']:
            record['search_notes']='Validation decision frozen before test. Registered arms evaluated for confirmation/diagnosis; test cannot reselect.'
    audit.mkdir(parents=True,exist_ok=True)
    if not (audit/'run.yaml').exists():(audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    print(json.dumps({'settings':len(settings),'checkpoint_evaluations':len(checkpoints),'additional_training':False,'default_candidate':run['selection']['default_candidate']}),flush=True)
    if not args.prepare_only:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_lowrank_six.py'),
            '--scratch-dir',str(scratch),'--audit-dir',str(audit),'--gpus',args.gpus],cwd=ROOT,check=True)


if __name__=='__main__':main()
