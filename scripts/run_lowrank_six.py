#!/usr/bin/env python3
"""From-scratch controls for three interaction hypotheses and three output gates."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_interaction_stage2 import construct, forecast, save_json, digest, ARMS as OLD_ARMS

ARMS = {'i0': {}, **{k:v for k,v in OLD_ARMS.items() if k != 'i0_continue'},
        'g1_reliability': {'lowrank_fusion_mode':'reliability'},
        'g2_disagreement': {'lowrank_fusion_mode':'disagreement'},
        'g2_without_forecasts': {'lowrank_fusion_mode':'disagreement_without'},
        'g3_components': {'lowrank_fusion_mode':'components'},
        'components_static': {'lowrank_fusion_mode':'components_static'},
        'tp_pre_g2': {'trend_interaction_mode':'trend_to_phase','lowrank_update_style':'pre_norm','lowrank_fusion_mode':'disagreement'},
        'tp_pre_g3': {'trend_interaction_mode':'trend_to_phase','lowrank_update_style':'pre_norm','lowrank_fusion_mode':'components'}}
BASELINES = {arm: 'i0' for arm in ARMS}
BASELINES.update(pt_null='pt_scaled', tp_selective='tp_scaled', g3_components='components_static', tp_pre_g2='g2_disagreement', tp_pre_g3='g3_components')
DEFAULT_SETTINGS = ','.join(f'{d}:{h}:{s}' for d in ('ETTh1','ETTm2','Weather')
                            for h in (96,720) for s in (2021,2022))


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def random_models(cfg, arm):
    import torch
    import pytorch_lightning as pl
    assert not cfg.get('checkpoint') and not cfg.get('init_checkpoint')
    pl.seed_everything(cfg['seed'], workers=True)
    reference, args = construct(cfg)
    common = {k:v.detach().clone() for k,v in reference.state_dict().items()}
    candidate_cfg = copy.deepcopy(cfg)
    candidate_cfg['hyperparams'].update(ARMS[arm])
    # Additional modules cannot shift the RNG used for common branch weights.
    pl.seed_everything(cfg['seed'], workers=True)
    model, _ = construct(candidate_cfg)
    current = model.state_dict()
    assert set(common) <= set(current)
    for k,v in common.items():
        assert current[k].shape == v.shape
        current[k].copy_(v)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in common.items())
    return model, reference, args, candidate_cfg, state_hash(common)


def train_job(path):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate
    job = json.loads(Path(path).read_text()); out = Path(job['output']); out.mkdir(parents=True,exist_ok=True)
    cfg = copy.deepcopy(job['source']['config'])
    cfg.update(checkpoint='', init_checkpoint='', max_epochs=job['epochs'], evaluate_test=False)
    cfg['hyperparams']['train_epochs'] = job['epochs']
    torch.set_num_threads(2); torch.set_float32_matmul_precision('medium')
    assert torch.cuda.is_available()
    original_load = torch.load
    def forbid_load(*a, **kw):
        raise AssertionError('Pretrained checkpoint loading forbidden during initialization')
    torch.load = forbid_load
    try:
        model, reference, args, cfg, initial_hash = random_models(cfg,job['arm'])
    finally:
        torch.load = original_load
    train_set, train_loader = data_provider(args.dataset_args,'train')
    val_set, val_loader = data_provider(args.dataset_args,'val')
    device = torch.device('cuda:0'); model.to(device).eval(); reference.to(device).eval()
    batch = [x[:2] for x in next(iter(val_loader))]
    with torch.inference_mode():
        a,_ = forecast(model,batch,device); b,_ = forecast(reference,batch,device)
        changed = list(batch)
        changed[1] = torch.randn_like(changed[1])
        unrelated,_ = forecast(model,changed,device)
        torch.testing.assert_close(a,unrelated,atol=0,rtol=0)
    max_abs = float((a-b).abs().max())
    identity = job['arm'] == 'i0' or job['arm'].startswith('g') or job['arm'] in {
        'components_static','pt_scaled','tp_scaled','pt_null','tp_selective'}
    if identity: torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert all(p.requires_grad for p in model.parameters())
    save_json(out/'initialization.json',{'common_random_state_sha256':initial_hash,
        'pretrained_checkpoint_loaded':False,'initial_max_abs_vs_random_i0':max_abs,
        'identity_required':identity,'all_parameters_trainable':True})
    del reference
    class Audit(pl.Callback):
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            assert all(m.training for m in module.modules())
            assert all(p.requires_grad for p in module.parameters())
    logger = build_logger(str(out/'lightning'),version='train')
    trainer, checkpoint = build_trainer(max_epochs=job['epochs'],logger=logger,
        patience=8 if job['epochs']>8 else job['epochs']+1,
        checkpoint_dir=str(out/'checkpoints'),progress=False)
    trainer.callbacks.append(Audit())
    if job['smoke']:
        trainer.limit_train_batches=2; trainer.limit_val_batches=2
    pl.seed_everything(cfg['seed'],workers=True)
    model.train(); torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    elapsed=time.monotonic()-started
    restore_best_checkpoint(model,checkpoint); model.to(device).eval()
    metrics=evaluate(model,val_loader,val_set,'val',cfg['horizon'],out)
    cfg['checkpoint']=checkpoint.best_model_path
    save_json(out/'config.json',cfg)
    save_json(out/'result.json',{'setting':job['source']['setting'],'arm':job['arm'],**metrics,
        'elapsed_sec':elapsed,'epochs':trainer.current_epoch,'epoch_budget':job['epochs'],
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'common_random_state_sha256':initial_hash,'pretrained_checkpoint_loaded':False,
        'initial_max_abs':max_abs,'checkpoint':checkpoint.best_model_path,
        'checkpoint_sha256':digest(checkpoint.best_model_path),'best_val_loss':float(checkpoint.best_model_score),
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'smoke':job['smoke'],
        'environment':{'torch':torch.__version__,'lightning':pl.__version__,
                       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}})


def configs(source_dir,settings):
    requested={tuple(s.split(':')) for s in settings.split(',')}; found={}
    for path in sorted((source_dir/'search/runs').glob('*/config.json')):
        cfg=json.loads(path.read_text()); key=tuple(str(cfg[k]) for k in ('dataset','horizon','seed'))
        if key not in requested or cfg['mechanism']!='phaseformer_l_i0_late_fusion':continue
        assert key not in found and not cfg['init_checkpoint'] and not cfg['evaluate_test']
        hp=cfg['hyperparams']; assert hp['weak_period_residual_rank']==cfg['horizon']//8
        assert hp['trend_interaction_mode']=='none' and not hp.get('use_trend_token',False)
        cfg.update(checkpoint='',init_checkpoint='')
        hp.update(lowrank_fusion_mode='static',lowrank_update_style='post_norm',
                  lowrank_null_read=False,lowrank_selective_write=False,lowrank_content_mode='none')
        found[key]={'setting':f'{key[0]}_H{key[1]}_S{key[2]}_val','config':cfg,
                    'config_source':str(path),'config_source_sha256':digest(path)}
    assert set(found)==requested,(set(found),requested)
    return list(found.values())


def launch(path,gpu):
    job=json.loads(path.read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    command=[sys.executable,str(Path(__file__).resolve()),'--job',str(path)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    save_json(out/'status.json',{'status':'running','command':command,'gpu':gpu})
    with (out/'process.log').open('a') as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    save_json(out/'status.json',{'status':'completed' if code==0 else 'failed','returncode':code,'command':command,'gpu':gpu})
    if code:raise RuntimeError(f'failed: {path}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path);p.add_argument('--scratch-dir',type=Path)
    p.add_argument('--audit-dir',type=Path);p.add_argument('--settings',default=DEFAULT_SETTINGS)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--arms',default=','.join(ARMS));p.add_argument('--smoke',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--job',type=Path)
    p.add_argument('--skip-analysis',action='store_true');args=p.parse_args()
    if args.job:train_job(args.job);return
    import yaml
    sources=configs(args.source_dir.resolve(),args.settings);arms=args.arms.split(',')
    assert len(arms)==len(set(arms)) and set(arms)<=set(ARMS) and 'i0' in arms
    assert all(BASELINES[a] in arms for a in arms)
    scratch=args.scratch_dir.resolve();audit=args.audit_dir.resolve();jobs=[]
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for arm in arms:
        for source in sources:
            path=scratch/'jobs'/f"{source['setting']}__{arm}.json"
            job={'source':source,'arm':arm,'epochs':args.epochs,'smoke':args.smoke,
                 'code_commit':code,'output':str(scratch/'runs'/f"{source['setting']}__{arm}")}
            if path.exists():assert json.loads(path.read_text())==job,'Refuse changed resume job'
            save_json(path,job);jobs.append(path)
    manifest={'sources':sources,'arms':arms,'jobs':list(map(str,jobs)),'epochs':args.epochs,
              'smoke':args.smoke,'code_commit':code}
    save_json(scratch/'manifest.json',manifest)
    settings=[{'setting':s['setting'],'dataset':s['config']['dataset'],'horizon':s['config']['horizon'],
               'seed':s['config']['seed'],'lookback':720,'split':'validation'} for s in sources]
    run={'experiment_id':audit.name,'code':{'commit':code,'branch':'codex/trend-token-cross-attention'},
         'mechanism':{'hypotheses':['H1','H2','H3','G1','G2','G3'],'arms':{a:ARMS[a] for a in arms},'baselines':BASELINES},
         'experiment':{'settings':settings,'training':{'epochs':args.epochs,'initialization':'random shared untrained weights; no pretrained checkpoint',
            'all_parameters_trainable':True,'loss':'Huber','checkpoint':'lowest validation Huber','hyperparameters':{s['setting']:s['config']['hyperparams'] for s in sources}},
            'metrics':['MAE','MSE'],'smoke':args.smoke},
         'execution':{'settings':[{'setting':s['setting'],'source_config':s['config'],
            'commands':[[sys.executable,str(Path(__file__).resolve()),'--job',str(j)] for j in jobs if j.name.startswith(s['setting']+'__')]} for s in sources]},
         'selection':{'source':'validation','selected_configs':[{'setting':s['setting'],'config_id':arms,
            'search_notes':'fixed arms; validation used for checkpoints and exploratory hypothesis selection; no test'} for s in sources]},
         'analysis':{'ranking_metric':'mae','top_k':3,'dedup_rule':'same channel/category selected starts at least H apart; maximum nine cases per arm/setting'},
         'validation':{'status':'incomplete'}}
    audit.mkdir(parents=True,exist_ok=True)
    (audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    print(json.dumps({'jobs':len(jobs),'settings':len(sources),'arms':arms,'code':code}),flush=True)
    if args.prepare_only:return
    tasks=queue.Queue()
    for j in jobs:tasks.put(j)
    def bucket(gpu):
        while True:
            try:path=tasks.get_nowait()
            except queue.Empty:return
            launch(path,gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as pool:
        futures=[pool.submit(bucket,g) for g in args.gpus.split(',')]
        for future in futures:future.result()
    if not args.smoke and not args.skip_analysis:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_lowrank_six.py'),'--scratch-dir',str(scratch),
                        '--audit-dir',str(audit),'--gpus',args.gpus],cwd=ROOT,check=True)


if __name__=='__main__':main()
