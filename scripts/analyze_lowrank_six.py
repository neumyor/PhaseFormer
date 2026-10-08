#!/usr/bin/env python3
"""Paired validation, fixed-weight gate interventions and auditable six-hypothesis report."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from scripts.analyze_lowrank_stage2 import load_model, pick_cases, markdown_table
from scripts.run_lowrank_interaction_stage2 import forecast, save_json
from scripts.run_lowrank_six import BASELINES
from src.dataset.data_factory import data_provider

CASE_KEYS=('history','truth','baseline','candidate','off','phase','trend','gate','acceptance')


def pack(cases,path):
    z={'metadata':np.array([json.dumps(c['metadata']) for c in cases]),
       'setting':np.array([c['metadata']['setting'] for c in cases])}
    for k in CASE_KEYS:
        values=[np.asarray(c[k]).reshape(-1) for c in cases]
        z[k+'_length']=np.array([len(v) for v in values])
        z[k]=np.full((len(cases),max(map(len,values))),np.nan,np.float32)
        for i,v in enumerate(values):z[k][i,:len(v)]=v
    np.savez_compressed(path,**z)


def unpack(path):
    with np.load(path,allow_pickle=False) as z:
        return [{'metadata':json.loads(str(m)),**{k:z[k][i,:z[k+'_length'][i]] for k in CASE_KEYS}}
                for i,m in enumerate(z['metadata'])]


def mix(p,t,g,period):
    if g.shape[-1]==1:
        alpha=g.transpose(1,2);return (1-alpha)*p+alpha*t
    b,h,c=p.shape;pc=p.reshape(b,-1,period,c);tc=t.reshape_as(pc)
    level,shape=g[...,0][:,None,None],g[...,1][:,None,None]
    return ((1-shape)*pc+shape*tc+(level-shape)*(tc.mean(2,keepdim=True)-pc.mean(2,keepdim=True))).reshape_as(p)


def replay_fusion(model,batch,device,gate,inputs):
    # Fuse in the model's normalized coordinates before undoing RevIN.
    # Mixing separately denormalized branches amplifies cancellation error.
    prediction=mix(inputs['phase'],inputs['trend'],gate,model.period_len)
    if model.use_revin:
        _,stats=model.revin.normalize(batch[0].float().to(device))
        prediction=model.revin.denormalize(prediction,stats)
    return prediction


def component_errors(prediction,truth,period):
    error=(prediction-truth).double().reshape(len(truth),-1,period,truth.shape[-1])
    level=error.mean(2,keepdim=True)
    level_mse=level.square().mean((1,2))
    shape_mse=(error-level).square().mean((1,2))
    torch.testing.assert_close(level_mse+shape_mse,error.square().mean((1,2)),atol=1e-10,rtol=1e-10)
    return level_mse,shape_mse


def oracle_component_gate(phase,trend,truth,period):
    # One coefficient per sample/channel across all future cycles, as in the model.
    p,t,y=(v.double().reshape(len(v),-1,period,v.shape[-1]) for v in (phase,trend,truth))
    pm,tm,ym=(v.mean(2,keepdim=True) for v in (p,t,y))
    def coefficient(start,end,target):
        delta=end-start
        denominator=delta.square().sum((1,2))
        numerator=((target-start)*delta).sum((1,2))
        return torch.where(denominator>0,numerator/denominator.clamp_min(1e-30),0).clamp(0,1)
    return torch.stack([coefficient(pm,tm,ym),coefficient(p-pm,t-tm,y-ym)],-1).to(phase.dtype)


def separated_shuffle_shifts(count,horizon,seeds=range(5)):
    if count < 2*horizon:
        raise ValueError('Gate shuffle needs at least two nonoverlapping horizons')
    return [int(np.random.default_rng(seed).integers(horizon,count-horizon+1)) for seed in seeds]


def train_mean_gate(model,args,device):
    # deterministic full training loader, including its last partial batch.
    dataset,_=data_provider(args.dataset_args,'train')
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.dataset_args.batch_size if hasattr(args.dataset_args,'batch_size') else 64,
                                        shuffle=False,num_workers=0)
    total=None;count=0
    with torch.inference_mode():
        for batch in loader:
            forecast(model,batch,device);g=model.lowrank_dynamic_fusion.last_gate
            total=g.sum(0) if total is None else total+g.sum(0);count+=len(g)
    return total/count,count


def off_prediction(model,batch,device,mean_gate,fusion_inputs):
    branch=model.lowrank_interaction_branch
    if model.lowrank_dynamic_fusion is not None:
        return replay_fusion(model,batch,device,mean_gate[None].expand(len(batch[0]),-1,-1),fusion_inputs)
    previous=branch.interaction_mode
    try:
        branch.interaction_mode='none';return forecast(model,batch,device)[0]
    finally:branch.interaction_mode=previous


def worker(scratch,setting):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');device=torch.device('cuda:0')
    manifest=json.loads((scratch/'manifest.json').read_text());arms=manifest['arms']
    baselines=manifest.get('baselines',BASELINES);level_study=manifest.get('study')=='level'
    source=next(s for s in manifest['sources'] if s['setting']==setting)
    cases=[];summaries=[];parts=scratch/'analysis_parts';parts.mkdir(exist_ok=True)
    target=parts/f'{setting}.csv';first=True
    for arm in arms:
        model,args=load_model(scratch/'runs'/f'{setting}__{arm}',device)
        args.dataset_args.num_workers=0
        baseline,_=load_model(scratch/'runs'/f'{setting}__{baselines[arm]}',device)
        dataset,loader=data_provider(args.dataset_args,'val')
        branch=model.lowrank_interaction_branch;branch.capture_diagnostics=True
        fusion=model.lowrank_dynamic_fusion
        fusion_inputs={};capture_hook=None
        if fusion is not None:
            def capture_fusion(module,inputs,output):
                fusion_inputs.update(phase=inputs[0].detach(),trend=inputs[1].detach())
            capture_hook=fusion.register_forward_hook(capture_fusion)
        mean_gate=None;gate_count=0
        if fusion is not None:mean_gate,gate_count=train_mean_gate(model,args,device)
        rows=[];gates=[];offset=0
        with torch.inference_mode():
            for batch in loader:
                bp,truth=forecast(baseline,batch,device);cp,_=forecast(model,batch,device)
                phase=model.last_phase_forecast.clone();trend=model.last_lowrank_forecast.clone()
                diag=dict(branch.last_diagnostics)
                if fusion is not None:
                    g=fusion.last_gate.clone();gates.append(g.cpu())
                    torch.testing.assert_close(cp,replay_fusion(model,batch,device,g,fusion_inputs),atol=0,rtol=0)
                else:
                    g=torch.sigmoid(model.lowrank_interaction_gate).transpose(1,2).expand(len(cp),-1,-1)
                off=off_prediction(model,batch,device,mean_gate,fusion_inputs)
                arrays={}
                for label,pred in [('baseline',bp),('candidate',cp),('off',off),('phase',phase),('trend',trend)]:
                    err=pred-truth
                    arrays[label+'_mae']=err.abs().mean(1).cpu().numpy()
                    arrays[label+'_mse']=err.square().mean(1).cpu().numpy()
                    if level_study:
                        level_error,shape_error=component_errors(pred,truth,model.period_len)
                        arrays[label+'_level_mse']=level_error.cpu().numpy()
                        arrays[label+'_shape_mse']=shape_error.cpu().numpy()
                    if label in ('baseline','candidate'):
                        for j,e in enumerate(torch.tensor_split(err,3,dim=1)):
                            arrays[f'{label}_segment{j}_mse']=e.square().mean(1).cpu().numpy()
                        if level_study:
                            arrays[label+'_peak_error']=(pred.amax(1)-truth.amax(1)).abs().cpu().numpy()
                            arrays[label+'_trough_error']=(pred.amin(1)-truth.amin(1)).abs().cpu().numpy()
                if level_study and fusion is not None and fusion.components:
                    normalized_truth=truth
                    if model.use_revin:
                        _,stats=model.revin.normalize(batch[0].float().to(device))
                        normalized_truth=model.revin.normalize_with_stats(truth,stats)
                    oracle_gate=oracle_component_gate(fusion_inputs['phase'],fusion_inputs['trend'],normalized_truth,model.period_len)
                    oracle=replay_fusion(model,batch,device,oracle_gate,fusion_inputs)
                    arrays['oracle_mse']=(oracle-truth).square().mean(1).cpu().numpy()
                    arrays['oracle_gate_level']=oracle_gate[...,0].cpu().numpy()
                    arrays['oracle_gate_shape']=oracle_gate[...,1].cpu().numpy()
                    arrays['gate_regret_mse']=arrays['candidate_mse']-arrays['oracle_mse']
                arrays['gate_level']=g[...,0].cpu().numpy();arrays['gate_shape']=g[...,-1].cpu().numpy()
                arrays['prediction_disagreement_rms']=(phase-trend).square().mean(1).sqrt().cpu().numpy()
                arrays['outside_branch_interval']=((cp<torch.minimum(phase,trend)-1e-5)|(cp>torch.maximum(phase,trend)+1e-5)).float().mean(1).cpu().numpy()
                if diag and 'acceptance' in diag:
                    arrays['acceptance_mean']=diag['acceptance'].mean(-1).cpu().numpy()
                    arrays['acceptance_phase_std']=diag['acceptance'].std(-1,unbiased=False).cpu().numpy()
                    arrays['message_norm']=diag['message_norm'].cpu().numpy()
                if arm in ('pt_null','tp_selective'):
                    flag='null_read' if arm=='pt_null' else 'selective_write'
                    setattr(branch,flag,False)
                    no_null=forecast(model,batch,device)[0];setattr(branch,flag,True)
                    arrays['no_null_mse']=(no_null-truth).square().mean(1).cpu().numpy()
                if arm=='tp_selective':
                    for intervention in ('mean','roll'):
                        branch.receipt_intervention=intervention
                        v=forecast(model,batch,device)[0]
                        arrays[f'receipt_{intervention}_mse']=(v-truth).square().mean(1).cpu().numpy()
                    branch.receipt_intervention='native'
                b,c=arrays['candidate_mse'].shape
                frame=pd.DataFrame({k:v.reshape(-1) for k,v in arrays.items()})
                frame['sample_id']=np.repeat(np.arange(offset,offset+b),c);frame['channel']=np.tile(np.arange(c),b)
                frame['setting']=setting;frame['baseline_config_id']=baselines[arm];frame['candidate_config_id']=arm
                frame['delta_mae']=frame.candidate_mae-frame.baseline_mae;frame['delta_mse']=frame.candidate_mse-frame.baseline_mse
                frame['time_range']=[f'{dataset.timestamps[i+model.seq_len]}/{dataset.timestamps[i+model.seq_len+model.pred_len-1]}' for i in frame.sample_id]
                rows.append(frame);offset+=b
        frame=pd.concat(rows,ignore_index=True)
        if gates:
            bank=torch.cat(gates);n=len(bank);shift=n//2
            assert min(shift,n-shift)>=model.pred_len,'Gate shuffle must separate nonoverlapping forecast windows'
            shifts=separated_shuffle_shifts(n,model.pred_len) if level_study else [shift]
            offset=0;shuffle=[[] for _ in shifts];level_fixed=[];shape_fixed=[]
            with torch.inference_mode():
                for batch in loader:
                    _,truth=forecast(model,batch,device);b=len(truth)
                    for k,shift in enumerate(shifts):
                        indices=(torch.arange(offset,offset+b)+shift)%n
                        shuffled=bank[indices].to(device)
                        if fusion.level_only:
                            shuffled[...,1]=fusion.last_gate[...,1]
                        v=replay_fusion(model,batch,device,shuffled,fusion_inputs)
                        shuffle[k].append((v-truth).square().mean(1).cpu().numpy())
                    if fusion.components:
                        native=fusion.last_gate.clone()
                        for component,values in ((0,level_fixed),(1,shape_fixed)):
                            g=native.clone();g[...,component]=mean_gate[...,component]
                            v=replay_fusion(model,batch,device,g,fusion_inputs);values.append((v-truth).square().mean(1).cpu().numpy())
                    offset+=b
            shuffle_arrays=[np.concatenate(values).reshape(-1) for values in shuffle]
            frame['shuffled_gate_mse']=np.mean(shuffle_arrays,axis=0)
            if level_study:
                for k,values in enumerate(shuffle_arrays):frame[f'shuffled_gate_seed{k}_mse']=values
            if level_fixed:
                frame['fixed_level_mse']=np.concatenate(level_fixed).reshape(-1)
                frame['fixed_shape_mse']=np.concatenate(shape_fixed).reshape(-1)
        result=json.loads((scratch/'runs'/f'{setting}__{arm}'/'result.json').read_text())
        for metric in ('mse','mae'):
            np.testing.assert_allclose(frame['candidate_'+metric].mean(),result['val_'+metric],atol=3e-6,rtol=3e-5)
        record={**result,'config_id':arm,'baseline_config_id':baselines[arm],
                'model':'PhaseFormer-L',
                'key_params':json.dumps(json.loads((scratch/'runs'/f'{setting}__{arm}'/'config.json').read_text())['hyperparams'],sort_keys=True),
                'dataset':source['config']['dataset'],'horizon':model.pred_len,'seed':source['config']['seed'],
                'mse':float(frame.candidate_mse.mean()),'mae':float(frame.candidate_mae.mean()),
                'delta_mse':float(frame.delta_mse.mean()),'delta_mae':float(frame.delta_mae.mean()),
                'gate_train_sample_count':gate_count,'selected':True}
        for col in frame.select_dtypes('number').columns:
            if col not in ('sample_id','channel'):record['mean_'+col]=float(frame[col].mean())
        record['gate_level_std']=float(frame.gate_level.std());record['gate_shape_std']=float(frame.gate_shape.std())
        record['gate_sample_std_within_channel']=float(frame.groupby('channel').gate_level.std().mean())
        record['gate_saturation_fraction']=float(((frame.gate_level<.01)|(frame.gate_level>.99)).mean())
        if level_study:
            for name in ('baseline','candidate'):
                values=frame[name+'_mse']
                record[name+'_mse_p95']=float(values.quantile(.95))
                record[name+'_mse_top1_mean']=float(values.nlargest(max(1,int(np.ceil(len(values)*.01)))).mean())
        summaries.append(record)
        chosen=pick_cases(frame,model.pred_len,top_k=3) if arm!='i0' else []
        cache={}
        with torch.inference_mode():
            for row in chosen:
                sid,ch=int(row['sample_id']),int(row['channel'])
                if (sid,ch) not in cache:
                    start=sid//loader.batch_size*loader.batch_size
                    items=[dataset[i] for i in range(start,min(start+loader.batch_size,len(dataset)))]
                    batch=[torch.as_tensor(np.stack([v[j] for v in items])) for j in range(4)];bi=sid-start
                    bp,truth=forecast(baseline,batch,device);cp,_=forecast(model,batch,device)
                    diag=branch.last_diagnostics
                    g=fusion.last_gate if fusion is not None else torch.sigmoid(model.lowrank_interaction_gate).transpose(1,2).expand(len(cp),-1,-1)
                    data={'history':batch[0][bi,:,ch].numpy(),'truth':truth[bi,:,ch].cpu().numpy(),
                          'baseline':bp[bi,:,ch].cpu().numpy(),'candidate':cp[bi,:,ch].cpu().numpy(),
                          'phase':model.last_phase_forecast[bi,:,ch].cpu().numpy(),
                          'trend':model.last_lowrank_forecast[bi,:,ch].cpu().numpy(),'gate':g[bi,ch].cpu().numpy(),
                          'acceptance':diag['acceptance'][bi,ch].cpu().numpy() if diag else np.array([],dtype=np.float32)}
                    data['off']=off_prediction(model,batch,device,mean_gate,fusion_inputs)[bi,:,ch].cpu().numpy()
                    cache[sid,ch]=data
                cases.append({'metadata':row,**cache[sid,ch]})
        # Fixed column schema across arms, so appended rows remain readable.
        optional=['acceptance_mean','acceptance_phase_std','message_norm','no_null_mse','receipt_mean_mse',
                  'receipt_roll_mse','shuffled_gate_mse','fixed_level_mse','fixed_shape_mse']
        if level_study:
            optional+=['oracle_mse','oracle_gate_level','oracle_gate_shape','gate_regret_mse']
            optional+=[f'shuffled_gate_seed{k}_mse' for k in range(5)]
        for col in optional:
            if col not in frame:frame[col]=np.nan
        frame=frame.reindex(sorted(frame.columns),axis=1)
        frame.to_csv(target,index=False,mode='w' if first else 'a',header=first);first=False
        print(setting,arm,len(frame),'rows',len(chosen),'cases',flush=True)
        if capture_hook is not None:capture_hook.remove()
        del baseline,model,frame;torch.cuda.empty_cache()
    pack(cases,parts/f'{setting}.npz');save_json(parts/f'{setting}.json',summaries)


def merge(scratch,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((scratch/'manifest.json').read_text());run=yaml.safe_load((audit/'run.yaml').read_text())
    level_study=manifest.get('study')=='level'
    settings=[s['setting'] for s in manifest['sources']];results=[];cases=[];selections=[];distributions=[]
    first=True;row_count=0
    for setting in settings:
        frame=pd.read_csv(scratch/'analysis_parts'/f'{setting}.csv');row_count+=len(frame)
        case=unpack(scratch/'analysis_parts'/f'{setting}.npz');cases.extend(case)
        records=json.loads((scratch/'analysis_parts'/f'{setting}.json').read_text());results.extend(records)
        assert len({r['common_random_state_sha256'] for r in records})==1
        assert all(not r['pretrained_checkpoint_loaded'] for r in records)
        horizon=int(records[0]['horizon'])
        for arm in manifest['arms']:
            group=frame[frame.candidate_config_id==arm]
            if arm=='i0':continue
            expected=pick_cases(group,horizon,top_k=3)
            actual=[c['metadata'] for c in case if c['metadata']['candidate_config_id']==arm]
            key=lambda r:(r['category'],r['rank'],r['sample_id'],r['channel'])
            assert {key(r) for r in actual}=={key(r) for r in expected}
            selections.extend({k:r[k] for k in ('setting','candidate_config_id','category','rank','sample_id','channel')} for r in expected)
            distributions.append({'setting':setting,'arm':arm,'sample_mae_win_fraction':float((group.delta_mae<0).mean()),
                'delta_mae_p05':float(group.delta_mae.quantile(.05)),'delta_mae_p95':float(group.delta_mae.quantile(.95)),
                'candidate_mae_p95':float(group.candidate_mae.quantile(.95))})
        frame.to_csv(audit/'sample_errors.csv',index=False,mode='w' if first else 'a',header=first);first=False
    table=pd.DataFrame(results)
    for col in ('mse','mae','delta_mse','delta_mae'):
        table[col]=pd.to_numeric(table[col],errors='raise')
    table.to_csv(audit/'results.csv',index=False);pack(cases,audit/'selected_cases.npz')
    for c in cases:
        for key in ('baseline','candidate','off'):
            e=c[key]-c['truth']
            np.testing.assert_allclose(np.square(e).mean(),c['metadata'][key+'_mse'],atol=5e-5,rtol=5e-4)
            np.testing.assert_allclose(np.abs(e).mean(),c['metadata'][key+'_mae'],atol=3e-5,rtol=3e-4)
    figures=audit/'figures';figures.mkdir(exist_ok=True);refs=[]
    major=(tuple(a for a in manifest['arms'] if a!='i0') if level_study else
           ('pt_scaled','tp_scaled','pt_null','tp_selective','g1_reliability','g2_disagreement','g3_components','tp_pre_g2','tp_pre_g3'))
    for setting in settings:
        for arm in major:
            selected=[c for c in cases if c['metadata']['setting']==setting and c['metadata']['candidate_config_id']==arm and c['metadata']['rank']==1]
            if not selected:continue
            fig,axes=plt.subplots(len(selected),2,figsize=(12,3*len(selected)),squeeze=False)
            for i,c in enumerate(selected):
                m=c['metadata'];hist=c['history'][-192:]
                axes[i,0].plot(np.arange(-len(hist),0),hist,color='.65',label='history')
                for name,col in [('truth','black'),('baseline','#2166ac'),('candidate','#b2182b'),('off','#188e66')]:
                    axes[i,0].plot(c[name],label=name,color=col,lw=1)
                axes[i,0].legend(fontsize=7,ncol=3)
                axes[i,0].set_title(f"{m['category']} {m['sample_id']},c{m['channel']}; dMAE {m['delta_mae']:+.4f}",fontsize=9)
                for name,col in [('truth','black'),('phase','#2166ac'),('trend','#ef8a62')]:
                    axes[i,1].plot(c[name],label=name,color=col,lw=1)
                axes[i,1].legend(fontsize=7)
                gate=', '.join(f'{v:.3f}' for v in c['gate'])
                receipt=f"; receipt mean {c['acceptance'].mean():.3f}" if len(c['acceptance']) else ''
                axes[i,1].set_title(f'output gate {gate}'+receipt,fontsize=9)
            fig.suptitle(f'{setting} / {arm}',fontsize=11);fig.tight_layout()
            rel=f'figures/{setting}__{arm}.png';fig.savefig(audit/rel,dpi=120);plt.close(fig);refs.append(rel)
    # All efficacy comparisons are recomputed from the same evaluation rows.
    comparisons=[('pt_pre','pt_post'),('tp_pre','tp_post'),('pt_scaled','pt_pre'),('tp_scaled','tp_pre'),
                 ('pt_null','pt_scaled'),('tp_selective','tp_scaled'),('g1_reliability','i0'),
                 ('g2_disagreement','i0'),('g2_disagreement','g2_without_forecasts'),
                 ('g3_components','i0'),('g3_components','components_static'),('g3_components','g2_disagreement'),('tp_pre_g2','g2_disagreement'),('tp_pre_g3','g3_components'),('tp_pre_g2','tp_pre'),('tp_pre_g3','tp_pre')]
    if 'comparisons' in manifest:comparisons=manifest['comparisons']
    paired=[]
    for candidate,baseline in comparisons:
        if candidate not in manifest['arms'] or baseline not in manifest['arms']:continue
        a=table[table.config_id==candidate].set_index('setting');b=table[table.config_id==baseline].set_index('setting')
        paired.append({'candidate':candidate,'baseline':baseline,'delta_mse':float((a.mse-b.mse).mean()),
                       'relative_mse_percent':float(((a.mse/b.mse-1)*100).mean()),
                       'median_relative_mse_percent':float(((a.mse/b.mse-1)*100).median()),
                       'relative_mae_percent':float(((a.mae/b.mae-1)*100).mean()),
                       'delta_mae':float((a.mae-b.mae).mean()),'mse_wins':int((a.mse<b.mse).sum()),'settings':len(a)})
    lines=['# Experiment and Objective Error Analysis','', '## 1. Experiment Setup','',
           f"From-scratch {manifest.get('study','six')}-fusion study: {len(table)} runs; budget {manifest['epochs']} epochs. All original parameters trainable.",
           'No pretrained checkpoint loaded. Shared untrained parameters and seed-matched data order. Original rank-r encoder/decoder retained.',
           'Validation-only exploratory selection; validation was previously inspected. No test or Golden superiority claim.','',
           '## 2. Experiment Results','']
    if manifest['smoke']:
        lines.insert(2, '> SMOKE ONLY: two training batches; these values are not efficacy evidence.\n')
    lines+=markdown_table(pd.DataFrame(paired));lines+=['','All settings and arms:','']
    lines+=markdown_table(table[['setting','config_id','mae','mse','delta_mae','delta_mse','parameter_count','elapsed_sec']])
    lines+=['','## 3. Parameter / Configuration Search','',
            'Fixed predeclared arms; no per-arm tuning. G2 without forecast features is an equal-parameter information control. G3 has a static two-component control.',
            'H1 separates pre/post normalization and zero-initialized scaling. H2/H3 compare against their scaled directional controls.',
            'Checkpoint selected by minimum validation Huber; table retains every arm. Gate-feature inputs are detached, but both branches train through the fused loss.','',
            '## 4. Error Distribution','']
    if level_study:
        start=lines.index('Fixed predeclared arms; no per-arm tuning. G2 without forecast features is an equal-parameter information control. G3 has a static two-component control.')
        lines[start:start+2]=[
            'Fixed predeclared matrix: no/pre-norm interaction crossed with static components, dynamic level/static shape, and fully dynamic components; scalar-gate references retained.',
            'Level-only gates share the G3 context and width=16. Shape gate is a learned per-channel constant. Context-zero controls preserve parameter shapes when registered.']
    lines+=markdown_table(pd.DataFrame(distributions));lines+=['','## 5. Horizon-wise Error','']
    segments=['mean_'+s for s in ['baseline_segment0_mse','candidate_segment0_mse','baseline_segment1_mse','candidate_segment1_mse','baseline_segment2_mse','candidate_segment2_mse']]
    lines+=markdown_table(table[['setting','config_id']+segments])
    lines+=['','## 6. High-Error Selection','',
            'Top three baseline MAE, positive MAE regressions and negative MAE improvements; same-channel starts separated by at least H within each category. Maximum nine records per arm/setting. Rankings verified from sample_errors.csv.','',
            '## 7. Case Analysis','']
    for rel in refs:lines.extend([f'![{Path(rel).stem}]({rel})',''])
    lines+=['## 8. Repeated Observable Patterns','',
            'off means the current checkpoint with interaction disabled, or its output gate replaced by its own training-set mean. It is not a retrained model.',
            'Shuffled gates exchange sample identities within the same channel with forecast-window separation >=H. Component gates are intervened separately.',
            'Gate variation, attention mass and correlation are observations, not proof of calibrated reliability.','']
    diagcols=[c for c in ('mean_off_mse','mean_shuffled_gate_mse','mean_no_null_mse','mean_receipt_mean_mse',
                          'mean_receipt_roll_mse','mean_fixed_level_mse','mean_fixed_shape_mse','mean_gate_level',
                          'mean_gate_shape','mean_outside_branch_interval') if c in table]
    lines+=markdown_table(table[['setting','config_id','mse']+diagcols])
    if level_study:
        lines+=['','Level/shape decomposition, branch oracle and tail risk:','']
        columns=['mean_candidate_level_mse','mean_candidate_shape_mse',
                 'mean_phase_level_mse','mean_trend_level_mse','mean_oracle_mse',
                 'mean_gate_regret_mse','candidate_mse_p95','candidate_mse_top1_mean']
        lines+=markdown_table(table[['setting','config_id']+columns])
        interaction=[]
        lookup=table.set_index(['setting','config_id'])
        required={'components_level_dynamic','tp_pre_level_dynamic','components_static','tp_pre_components_static'}
        if required <= set(manifest['arms']):
            for setting in settings:
                mse=lambda arm:float(lookup.loc[(setting,arm),'mse'])
                interaction.append({'setting':setting,'dynamic_level_marginal_without_interaction':
                    mse('components_level_dynamic')-mse('components_static'),
                    'dynamic_level_marginal_with_interaction':mse('tp_pre_level_dynamic')-mse('tp_pre_components_static'),
                    'difference_of_differences':(mse('tp_pre_level_dynamic')-mse('tp_pre_components_static'))-
                    (mse('components_level_dynamic')-mse('components_static'))})
            lines+=['','Interaction changes to dynamic-level marginal benefit (descriptive):','']
            lines+=markdown_table(pd.DataFrame(interaction))
        anchors=[]
        for setting in settings:
            if setting not in {'Weather_H96_S2022_val','Weather_H96_S2021_val'}:continue
            frame=pd.read_csv(scratch/'analysis_parts'/f'{setting}.csv')
            ids=[1702,1956] if 'S2022' in setting else [1980]
            group=frame[(frame.channel==14)&frame.sample_id.isin(ids)]
            columns=['setting','candidate_config_id','sample_id','channel','candidate_mse',
                     'candidate_level_mse','candidate_shape_mse','oracle_mse','gate_level','gate_shape']
            anchors.extend(group[columns].to_dict('records'))
        run['analysis']['historical_anchors']=anchors
        run['analysis']['interaction_marginal_comparisons']=interaction
        if anchors:
            lines+=['','Historical anchors (diagnostics only, excluded from ranked case selection):','']
            lines+=markdown_table(pd.DataFrame(anchors))
    lines+=['','## 9. Objective Defect Summary','',
            'Read the paired comparisons together with the regression cases: average improvement does not imply every sample improves. Component mixing is not pointwise convex mixing of the original forecasts.',
            'A useful fixed-weight intervention does not establish a superior retrained architecture. An eight-epoch loss is a budget-specific result; longer from-scratch confirmation may change ordering.','',
            '## 10. Experiment Scope','',
            f"{len(settings)} validation settings; {row_count} sample/channel/arm records; {len(cases)} selected case records. No dataset-wise stitching of winners.",
            'This report is reproducible from the six audit files and referenced figures. Hypothesis decisions are added after numerical review.','']
    report='\n'.join(lines);(audit/'objective_error_analysis.md').write_text(report)
    listed=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report);assert set(listed)==set(refs)
    assert {p.name for p in figures.iterdir()}=={Path(r).name for r in refs}
    with zipfile.ZipFile(audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in ['objective_error_analysis.md']+refs:z.write(audit/rel,rel)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        for rel in z.namelist():assert z.read(rel)==(audit/rel).read_bytes()
    run['analysis']['selections']=selections;run['analysis']['paired_comparisons']=paired
    run['code']['analysis_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run['execution']['results']=[{k:r[k] for k in ('setting','arm','epochs','elapsed_sec','peak_memory_bytes','environment','common_random_state_sha256')} for r in results]
    run['validation']={'status':'passed','results_checked':True,'ranking_and_cases_checked':True,
                       'report_and_archive_checked':True,'directory_and_settings_checked':True,
                       'no_pretrained_initialization_checked':True,'sample_rows':row_count,'case_records':len(cases)}
    (audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    assert {p.name for p in audit.iterdir()}=={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz','objective_error_analysis.md','objective_error_analysis.zip','figures'}
    assert set(table.setting)==set(settings)=={c['metadata']['setting'] for c in cases}
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    save_json(scratch/'audit_complete.json',run['validation'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch-dir',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--setting');p.add_argument('--merge-only',action='store_true')
    args=p.parse_args();scratch=args.scratch_dir.resolve();audit=args.audit_dir.resolve()
    if args.setting:worker(scratch,args.setting);return
    if not args.merge_only:
        settings=[s['setting'] for s in json.loads((scratch/'manifest.json').read_text())['sources']];gpus=args.gpus.split(',')
        def bucket(gpu,batch):
            for setting in batch:
                command=[sys.executable,str(Path(__file__).resolve()),'--scratch-dir',str(scratch),'--audit-dir',str(audit),'--setting',setting]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with (scratch/f'analysis_{setting}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(bucket,g,settings[i::len(gpus)]) for i,g in enumerate(gpus)]
            for f in futures:f.result()
    merge(scratch,audit)


if __name__=='__main__':main()
