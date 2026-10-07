#!/usr/bin/env python3
"""Stage 3: matched content-capacity and centering experiments on stage-2 tp_pre."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lowrank_interaction_stage2 import construct, forecast, digest, save_json

ARMS={'tp_pre_continue': ('none','joint'),
      **{f'{mode}_{regime}':(mode,regime) for regime in ('frozen','joint')
         for mode in ('trend','phase','centered')}}
PREFIX='lowrank_interaction_branch.content_correction.'


def train_job(path):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger,build_trainer,restore_best_checkpoint
    from scripts.search_phaseformer import evaluate
    job=json.loads(Path(path).read_text());source=job['source'];out=Path(job['output'])
    out.mkdir(parents=True,exist_ok=True)
    mode,regime=ARMS[job['arm']]
    cfg=copy.deepcopy(source['config'])
    assert digest(cfg['checkpoint'])==source['checkpoint_sha256']
    cfg['hyperparams'].update(lowrank_content_mode=mode,lowrank_content_width=8,
                            train_epochs=job['epochs'])
    cfg.update(max_epochs=job['epochs'],evaluate_test=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg['seed'],workers=True)
    model,args=construct(cfg)
    source_state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    missing=model.load_state_dict(source_state,strict=False)
    assert not missing.unexpected_keys and all(k.startswith(PREFIX) for k in missing.missing_keys)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in source_state.items())
    baseline,_=construct(source['config']);baseline.load_state_dict(source_state,strict=True)
    train_set,train_loader=data_provider(args.dataset_args,'train')
    val_set,val_loader=data_provider(args.dataset_args,'val')
    device=torch.device('cuda:0');model.to(device).eval();baseline.to(device).eval()
    batch=[v[:2] for v in next(iter(val_loader))]
    with torch.inference_mode():
        a,_=forecast(model,batch,device);b,_=forecast(baseline,batch,device)
    assert torch.equal(a,b),f'initial tp_pre identity failed: {(a-b).abs().max()}'
    if regime=='frozen':
        for name,param in model.named_parameters():param.requires_grad_(name.startswith(PREFIX))
    save_json(out/'initialization.json',{'max_abs':float((a-b).abs().max()),
        'missing_keys':missing.missing_keys,'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
        'source_checkpoint_sha256':source['checkpoint_sha256']})
    del baseline
    logger=build_logger(str(out/'lightning'),version='train')
    trainer,checkpoint=build_trainer(max_epochs=job['epochs'],logger=logger,
        patience=job['epochs']+1,checkpoint_dir=str(out/'checkpoints'),progress=False)

    class ModeAudit(pl.Callback):
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            if regime=='frozen':
                # Keep the original model a deterministic frozen function.
                module.eval();module.lowrank_interaction_branch.content_correction.train()
                assert all(not part.training for name,part in module.named_modules()
                           if not name.startswith(PREFIX.rstrip('.')))
                assert all(p.requires_grad==name.startswith(PREFIX) for name,p in module.named_parameters())
            else:
                assert all(part.training for part in module.modules())
    trainer.callbacks.append(ModeAudit())
    if job['smoke']:
        trainer.limit_train_batches=2;trainer.limit_val_batches=2
    pl.seed_everything(cfg['seed'],workers=True)
    model.train();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    elapsed=time.monotonic()-started
    restore_best_checkpoint(model,checkpoint);model.to(device).eval()
    if regime=='frozen':
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in source_state.items()),'frozen weights changed'
    metric=evaluate(model,val_loader,val_set,'val',cfg['horizon'],out)
    addon=model.lowrank_interaction_branch.content_correction
    if job['smoke'] and addon is not None:
        assert addon.scale.item()!=0,'scale did not learn'
    cfg['checkpoint']=checkpoint.best_model_path
    save_json(out/'config.json',cfg)
    save_json(out/'result.json',{'setting':source['setting'],'arm':job['arm'],**metric,
        'elapsed_sec':elapsed,'epochs':trainer.current_epoch,'checkpoint':checkpoint.best_model_path,
        'checkpoint_sha256':digest(checkpoint.best_model_path),'best_val_loss':float(checkpoint.best_model_score),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'added_count':sum(p.numel() for p in addon.parameters()) if addon is not None else 0,
        'content_scale':float(addon.scale) if addon is not None else None,
        'initial_max_abs':0.,'frozen_weights_verified':regime=='frozen',
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'smoke':job['smoke'],
        'environment':{'torch':torch.__version__,'lightning':pl.__version__,'gpu':torch.cuda.get_device_name(0)}})


def launch(path,gpu):
    job=json.loads(path.read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    command=[sys.executable,str(Path(__file__).resolve()),'--job',str(path)]
    save_json(out/'status.json',{'status':'running','command':command,'gpu':gpu})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (out/'process.log').open('a') as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    save_json(out/'status.json',{'status':'completed' if code==0 else 'failed','returncode':code,'command':command,'gpu':gpu})
    if code:raise RuntimeError(f'failed: {path}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-scratch',type=Path);p.add_argument('--scratch-dir',type=Path)
    p.add_argument('--audit-dir',type=Path);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--smoke',action='store_true')
    p.add_argument('--settings',default='');p.add_argument('--job',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.job:train_job(args.job);return
    import yaml
    scratch=args.scratch_dir.resolve();audit=args.audit_dir.resolve()
    sources=[]
    for path in sorted((args.source_scratch/'runs').glob('*__tp_pre/config.json')):
        cfg=json.loads(path.read_text());setting=path.parent.name.removesuffix('__tp_pre')
        if args.settings and setting not in args.settings.split(','):continue
        assert cfg['hyperparams']['lowrank_update_style']=='pre_norm'
        assert cfg['hyperparams']['trend_interaction_mode']=='trend_to_phase'
        assert not cfg['evaluate_test']
        assert cfg['hyperparams']['weak_period_residual_rank']==cfg['horizon']//8
        result=json.loads((path.parent/'result.json').read_text())
        assert not result['smoke']
        sources.append({'setting':setting,'config':cfg,'checkpoint_sha256':digest(cfg['checkpoint']),
                        'source_result':result})
    assert sources
    jobs=[]
    # Complete frozen diagnostics before joint candidates; baseline is first.
    for arm in ARMS:
        for source in sources:
            path=scratch/'jobs'/f"{source['setting']}__{arm}.json"
            job={'source':source,'arm':arm,'epochs':args.epochs,'smoke':args.smoke,
                 'output':str(scratch/'runs'/f"{source['setting']}__{arm}")}
            if path.exists():assert json.loads(path.read_text())==job
            save_json(path,job);jobs.append(path)
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    manifest={'sources':sources,'arms':ARMS,'jobs':[str(x) for x in jobs],'epochs':args.epochs,'smoke':args.smoke,'code_commit':code}
    save_json(scratch/'manifest.json',manifest)
    settings=[{'setting':s['setting'],'dataset':s['config']['dataset'],'horizon':s['config']['horizon'],
               'seed':s['config']['seed'],'lookback':720,'split':'validation'} for s in sources]
    run={'experiment_id':audit.name,'code':{'commit':code,'branch':'codex/trend-token-cross-attention'},
        'mechanism':{'description':'tp_pre broadcast plus equal-parameter trend/phase/centered content correction','width':8,'arms':ARMS},
        'experiment':{'settings':settings,'training':{'epochs':args.epochs,'warm_start':'stage2 valid v2 tp_pre',
            'optimizer':'fresh Adam, source LR/scheduler/batch size/Huber','checkpoint':'lowest validation Huber after training',
            'frozen_mode':'all original parameters and buffers fixed; original modules eval; only new content module trains'},
            'metrics':['MSE','MAE'],'smoke':args.smoke},
        'execution':{'settings':[{'setting':s['setting'],'source_checkpoint_sha256':s['checkpoint_sha256'],
            'source_config':s['config'],'commands':[[sys.executable,str(Path(__file__).resolve()),'--job',str(j)]
            for j in jobs if j.name.startswith(s['setting']+'__')]} for s in sources]},
        'selection':{'source':'validation','selected_configs':[{'setting':s['setting'],'config_id':list(ARMS),
            'search_notes':'all predeclared arms; no sweep; source and current checkpoints selected using same validation'} for s in sources]},
        'analysis':{'ranking_metric':'mae','top_k':5,'dedup_rule':'same channel/category selected starts separated by >=H'},
        'validation':{'status':'incomplete'}}
    audit.mkdir(parents=True,exist_ok=True);(audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    print(json.dumps({'jobs':len(jobs),'settings':len(sources),'code':code}),flush=True)
    if args.prepare_only:return
    gpus=args.gpus.split(',')
    for frozen_stage in (True,False):
        selected=[j for j in jobs if (json.loads(j.read_text())['arm'].endswith('frozen') or
                 json.loads(j.read_text())['arm']=='tp_pre_continue')==frozen_stage]
        def bucket(gpu,paths):
            for path in paths:launch(path,gpu)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(bucket,gpu,selected[i::len(gpus)]) for i,gpu in enumerate(gpus)]
            for future in futures:future.result()
    if not args.smoke:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_lowrank_stage3.py'),'--scratch-dir',str(scratch),
            '--audit-dir',str(audit),'--gpus',args.gpus],cwd=ROOT,check=True)


if __name__=='__main__':main()
