#!/usr/bin/env python3
"""Full-validation stage-three metrics, content ablations and selected activations."""
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
from scripts.analyze_lowrank_stage2 import load_model,pick_cases,markdown_table
from scripts.run_lowrank_interaction_stage2 import construct,forecast,save_json,digest
from scripts.run_lowrank_stage3 import ARMS
from src.dataset.data_factory import data_provider

MODES=('native','off','roll','reverse','half','double')
ANCHORS={'Weather_H720_S2021_val':[(1910,0),(2577,15),(1955,7)],
         'ETTm2_H720_S2022_val':[(7538,4),(8142,3)]}


def source_model(source,device):
    model,args=construct(source['config'],workers=0)
    assert digest(source['config']['checkpoint'])==source['checkpoint_sha256']
    model.load_state_dict(torch.load(source['config']['checkpoint'],map_location='cpu',weights_only=False)['state_dict'])
    return model.to(device).eval(),args


def pack_cases(cases,path):
    packed={'metadata':np.array([json.dumps(c['metadata']) for c in cases]),
            'setting':np.array([c['metadata']['setting'] for c in cases])}
    for key in ('history','truth','baseline','source','native','off','roll','reverse','half','double','raw','update','singular_values'):
        shapes=[list(np.asarray(c[key]).shape) for c in cases]
        flat=[np.asarray(c[key]).reshape(-1) for c in cases]
        packed[key+'_shape']=np.array([json.dumps(s) for s in shapes])
        packed[key+'_length']=np.array([len(a) for a in flat])
        packed[key]=np.full((len(cases),max([len(a) for a in flat]+[0])),np.nan,dtype=np.float32)
        for i,a in enumerate(flat):packed[key][i,:len(a)]=a
    np.savez_compressed(path,**packed)


def unpack_cases(path):
    cases=[]
    with np.load(path,allow_pickle=False) as z:
        for i,metadata in enumerate(z['metadata']):
            c={'metadata':json.loads(str(metadata))}
            for key in ('history','truth','baseline','source','native','off','roll','reverse','half','double','raw','update','singular_values'):
                c[key]=z[key][i,:z[key+'_length'][i]].reshape(json.loads(str(z[key+'_shape'][i])))
            cases.append(c)
    return cases


def worker(scratch,setting):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('medium')
    device=torch.device('cuda:0')
    manifest=json.loads((scratch/'manifest.json').read_text())
    source=next(s for s in manifest['sources'] if s['setting']==setting)
    base,args=source_model(source,device)
    continued,_=load_model(scratch/'runs'/f'{setting}__tp_pre_continue',device)
    dataset,loader=data_provider(args.dataset_args,'val')
    frames,cases=[],[];source_metrics=None
    for arm,(mode,regime) in ARMS.items():
        model,_=load_model(scratch/'runs'/f'{setting}__{arm}',device)
        addon=model.lowrank_interaction_branch.content_correction
        if addon is not None:addon.capture_diagnostics=True
        rows=[];offset=0
        with torch.inference_mode():
            for batch in loader:
                sp,truth=forecast(base,batch,device)
                bp=forecast(continued,batch,device)[0] if regime=='joint' and arm!='tp_pre_continue' else sp
                predictions={};cap={}
                for intervention in MODES:
                    if addon is not None:addon.intervention=intervention
                    predictions[intervention]=forecast(model,batch,device)[0]
                    if intervention=='native' and addon is not None:cap=dict(addon.last_diagnostics)
                    if addon is None:
                        predictions={m:predictions['native'] for m in MODES};break
                if regime=='frozen':
                    torch.testing.assert_close(predictions['off'],sp,atol=0,rtol=0)
                if mode=='trend':
                    torch.testing.assert_close(predictions['native'],predictions['roll'],atol=0,rtol=0)
                    torch.testing.assert_close(predictions['native'],predictions['reverse'],atol=0,rtol=0)
                if mode=='centered':
                    torch.testing.assert_close(cap['update'].mean(1),torch.zeros_like(cap['update'].mean(1)),atol=1e-6,rtol=0)
                arrays={}
                for name,pred in {'source':sp,'baseline':bp,**predictions}.items():
                    err=pred-truth
                    arrays[name+'_mae']=err.abs().mean(1).cpu().numpy()
                    arrays[name+'_mse']=err.square().mean(1).cpu().numpy()
                    if name in ('baseline','native','off'):
                        for j,e in enumerate(torch.tensor_split(err,3,dim=1)):
                            arrays[f'{name}_segment{j}_mse']=e.square().mean(1).cpu().numpy()
                b,c=arrays['native_mse'].shape
                correction=predictions['native']-predictions['off'];needed=truth-predictions['off']
                arrays['output_energy']=correction.square().mean(1).cpu().numpy()
                arrays['output_alignment']=(correction*needed).mean(1).cpu().numpy()
                arrays['output_mean_shift']=correction.mean(1).cpu().numpy()
                np.testing.assert_allclose(arrays['native_mse']-arrays['off_mse'],
                    arrays['output_energy']-2*arrays['output_alignment'],atol=2e-5,rtol=2e-4)
                if cap:
                    raw=cap['raw'];update=cap['update']
                    arrays['raw_norm']=raw.norm(dim=-1).mean(1).reshape(b,c).cpu().numpy()
                    arrays['update_norm']=update.norm(dim=-1).mean(1).reshape(b,c).cpu().numpy()
                    arrays['update_phase_mean_norm']=update.mean(1).norm(dim=-1).reshape(b,c).cpu().numpy()
                    arrays['update_phase_spread']=(update-update.mean(1,keepdim=True)).norm(dim=-1).mean(1).reshape(b,c).cpu().numpy()
                frame=pd.DataFrame({k:v.reshape(-1) for k,v in arrays.items()})
                frame['setting']=setting;frame['candidate_config_id']=arm
                frame['baseline_config_id']='tp_pre_continue' if regime=='joint' and arm!='tp_pre_continue' else 'tp_pre_source'
                frame['sample_id']=np.repeat(np.arange(offset,offset+b),c);frame['channel']=np.tile(np.arange(c),b)
                frame['candidate_mae']=frame.native_mae;frame['candidate_mse']=frame.native_mse
                frame['delta_mae']=frame.native_mae-frame.baseline_mae;frame['delta_mse']=frame.native_mse-frame.baseline_mse
                frame['enabled_minus_disabled_mse']=frame.native_mse-frame.off_mse
                frame['roll_minus_native_mse']=frame.roll_mse-frame.native_mse
                frame['reverse_minus_native_mse']=frame.reverse_mse-frame.native_mse
                frame['time_range']=[f'{dataset.timestamps[i+base.seq_len]}/{dataset.timestamps[i+base.seq_len+base.pred_len-1]}' for i in frame.sample_id]
                rows.append(frame);offset+=b
        frame=pd.concat(rows,ignore_index=True);frames.append(frame)
        selected=pick_cases(frame,base.pred_len)
        for rank,(sid,ch) in enumerate(ANCHORS.get(setting,[]),1):
            match=frame[(frame.sample_id==sid)&(frame.channel==ch)]
            assert len(match)==1
            selected.append({**match.iloc[0].to_dict(),'category':'prespecified_anchor','rank':rank})
        cache={}
        with torch.inference_mode():
            for row in selected:
                sid,ch=int(row['sample_id']),int(row['channel'])
                if (sid,ch) not in cache:
                    # Reuse the original loader batch shape to avoid precision/batching confounds.
                    start=sid//loader.batch_size*loader.batch_size
                    ids=range(start,min(start+loader.batch_size,len(dataset)))
                    items=[dataset[i] for i in ids]
                    batch=[torch.as_tensor(np.stack([item[j] for item in items])) for j in range(4)]
                    bi=sid-start;sp,truth=forecast(base,batch,device)
                    bp=forecast(continued,batch,device)[0] if regime=='joint' and arm!='tp_pre_continue' else sp
                    data={'history':batch[0][bi,:,ch].numpy(),'truth':truth[bi,:,ch].cpu().numpy(),
                          'source':sp[bi,:,ch].cpu().numpy(),'baseline':bp[bi,:,ch].cpu().numpy()}
                    for intervention in MODES:
                        if addon is not None:addon.intervention=intervention
                        pred,_=forecast(model,batch,device);data[intervention]=pred[bi,:,ch].cpu().numpy()
                        if intervention=='native':
                            channels=pred.shape[-1]
                            for name in ('raw','update'):
                                data[name]=addon.last_diagnostics[name][bi*channels+ch].cpu().numpy() if addon is not None else np.zeros((24,1),np.float32)
                            data['singular_values']=np.linalg.svd(data['update'],compute_uv=False)
                    cache[sid,ch]=data
                cases.append({'metadata':row,**cache[sid,ch]})
        del model;torch.cuda.empty_cache()
        print(setting,arm,len(frame),'rows',len(selected),'cases',flush=True)
    parts=scratch/'analysis_parts';parts.mkdir(exist_ok=True)
    pd.concat(frames,ignore_index=True).to_csv(parts/f'{setting}.csv',index=False)
    pack_cases(cases,parts/f'{setting}.npz')


def merge(scratch,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((scratch/'manifest.json').read_text());run=yaml.safe_load((audit/'run.yaml').read_text())
    settings=[s['setting'] for s in manifest['sources']]
    frames=[];cases=[];results=[]
    for setting in settings:
        frame=pd.read_csv(scratch/'analysis_parts'/f'{setting}.csv');frames.append(frame)
        selected=unpack_cases(scratch/'analysis_parts'/f'{setting}.npz');cases.extend(selected)
        for arm,(mode,regime) in ARMS.items():
            result=json.loads((scratch/'runs'/f'{setting}__{arm}'/'result.json').read_text())
            group=frame[frame.candidate_config_id==arm]
            for metric in ('mse','mae'):
                np.testing.assert_allclose(group['native_'+metric].mean(),result['val_'+metric],atol=2e-6,rtol=2e-5)
            expected=pick_cases(group,int(next(s for s in manifest['sources'] if s['setting']==setting)['config']['horizon']))
            actual=[c['metadata'] for c in selected if c['metadata']['candidate_config_id']==arm and c['metadata']['category']!='prespecified_anchor']
            key=lambda r:(r['category'],r['rank'],r['sample_id'],r['channel'])
            assert {key(r) for r in expected}=={key(r) for r in actual}
            results.append({'setting':setting,'config_id':arm,'baseline_config_id':group.baseline_config_id.iloc[0],
                'mode':mode,'regime':regime,'mae':group.native_mae.mean(),'mse':group.native_mse.mean(),
                'delta_mae':group.delta_mae.mean(),'delta_mse':group.delta_mse.mean(),
                'source_mae':group.source_mae.mean(),'source_mse':group.source_mse.mean(),
                'off_mse':group.off_mse.mean(),'roll_mse':group.roll_mse.mean(),'reverse_mse':group.reverse_mse.mean(),
                'half_mse':group.half_mse.mean(),'double_mse':group.double_mse.mean(),
                'selected':True,'parameter_count':result['parameter_count'],'trainable_count':result['trainable_count'],
                'added_count':result['added_count'],'content_scale':result['content_scale'],
                'elapsed_sec':result['elapsed_sec'],'peak_memory_bytes':result['peak_memory_bytes'],
                'checkpoint_sha256':result['checkpoint_sha256']})
        # Frozen source is a fixed reference, not another trained configuration.
        group=frame[frame.candidate_config_id=='tp_pre_continue']
        results.append({'setting':setting,'config_id':'tp_pre_source','regime':'source','mode':'none',
                        'mae':group.source_mae.mean(),'mse':group.source_mse.mean(),'delta_mae':0.,'delta_mse':0.,'selected':True})
    errors=pd.concat(frames,ignore_index=True);results=pd.DataFrame(results)
    errors.to_csv(audit/'sample_errors.csv',index=False);results.to_csv(audit/'results.csv',index=False)
    pack_cases(cases,audit/'selected_cases.npz')
    for c in cases:
        for name,label in [('baseline','baseline'),('native','candidate'),('off','off')]:
            e=c[name]-c['truth']
            np.testing.assert_allclose(np.square(e).mean(),c['metadata'][label+'_mse'],atol=6e-5,rtol=6e-4)
            np.testing.assert_allclose(np.abs(e).mean(),c['metadata'][label+'_mae'],atol=3e-5,rtol=3e-4)
    figures=audit/'figures';figures.mkdir(exist_ok=True);refs=[]
    for setting in settings:
        for arm in ('trend_joint','phase_joint','centered_joint','centered_frozen'):
            pair=[]
            for category in ('candidate_regression','candidate_improvement'):
                eligible=[c for c in cases if c['metadata']['setting']==setting and c['metadata']['candidate_config_id']==arm and c['metadata']['category']==category]
                if eligible:pair.append(min(eligible,key=lambda c:c['metadata']['rank']))
            if not pair:continue
            fig,axes=plt.subplots(len(pair),2,figsize=(12,3.4*len(pair)),squeeze=False)
            for i,c in enumerate(pair):
                meta=c['metadata'];hist=c['history'][-192:]
                axes[i,0].plot(np.arange(-len(hist),0),hist,color='.6',label='history')
                for name,color in [('truth','black'),('baseline','#2166ac'),('native','#b2182b'),('off','#188e66')]:
                    axes[i,0].plot(c[name],label=name,color=color,lw=1)
                axes[i,0].legend(fontsize=7,ncol=3)
                axes[i,0].set_title(f"{meta['category']} {meta['sample_id']},c{meta['channel']}; delta MAE {meta['delta_mae']:+.4f}",fontsize=9)
                im=axes[i,1].imshow(c['update'],aspect='auto',cmap='coolwarm')
                fig.colorbar(im,ax=axes[i,1],fraction=.04)
                axes[i,1].set_title(f"Content update; on/off MSE {meta['native_mse']:.4f}/{meta['off_mse']:.4f}",fontsize=9)
                axes[i,1].set_xlabel('latent coordinate');axes[i,1].set_ylabel('phase slot')
            fig.suptitle(setting+' / '+arm);fig.tight_layout()
            name=f'{setting}__{arm}.png';fig.savefig(figures/name,dpi=130);plt.close(fig);refs.append('figures/'+name)
    pivot=results.pivot(index='setting',columns='config_id',values='mse');comparisons=[]
    for regime in ('frozen','joint'):
        baseline='tp_pre_source' if regime=='frozen' else 'tp_pre_continue'
        for candidate,control in [(f'{m}_{regime}',baseline) for m in ('trend','phase','centered')]+[(f'phase_{regime}',f'trend_{regime}'),(f'centered_{regime}',f'trend_{regime}'),(f'centered_{regime}',f'phase_{regime}')]:
            d=pivot[candidate]-pivot[control]
            comparisons.append({'comparison':candidate+' - '+control,'delta_mse':d.mean(),'wins':f'{(d<0).sum()}/{len(d)}'})
    stats=[]
    for (setting,arm),g in errors.groupby(['setting','candidate_config_id']):
        stats.append({'setting':setting,'arm':arm,'mae_win_fraction':(g.delta_mae<0).mean(),'delta_mae_p95':g.delta_mae.quantile(.95),
            'enabled_minus_disabled_mse':g.enabled_minus_disabled_mse.mean(),'roll_minus_native_mse':g.roll_minus_native_mse.mean(),
            'reverse_minus_native_mse':g.reverse_minus_native_mse.mean(),
            **{k:g[k].mean() for k in ('raw_norm','update_norm','update_phase_mean_norm','update_phase_spread','output_energy','output_alignment','output_mean_shift')}})
    anchors=pd.DataFrame([c['metadata'] for c in cases if c['metadata']['category']=='prespecified_anchor'])
    segments=errors.groupby('candidate_config_id')[[f'{name}_segment{i}_mse' for name in ('baseline','native') for i in range(3)]].mean().reset_index()
    report=['# Stage 3: Phase-conditioned content on tp_pre','', '## 1. Experiment Setup','',
        f"Matched {manifest['epochs']}-epoch continuation from valid stage-2 tp_pre. Six predeclared candidate arms and matched tp_pre continuation; a separate source reference for frozen arms. Width 8, original low-rank E/D and broadcast retained. Frozen backbone stays in eval mode; only content module trains. No test. " + ('SMOKE ONLY; not efficacy evidence.' if manifest['smoke'] else ''),
        '', '## 2. Experiment Results','',*markdown_table(pd.DataFrame(comparisons)), '',
        *markdown_table(results[['setting','config_id','mae','mse','delta_mae','delta_mse','added_count','content_scale']]),
        '', '## 3. Parameter / Configuration Search','',
        'No sweep. All configurations retained. Candidate projection shapes and parameter counts match exactly. Trend-only control uses pooled normalized trend as context. Validation Huber selects checkpoints after training, including baseline; source selection and hypothesis development already used this validation set. Results are exploratory, not a Golden comparison or unbiased generalization estimate.',
        '', '## 4. Error Distribution','',*markdown_table(pd.DataFrame(stats)[['setting','arm','mae_win_fraction','delta_mae_p95']]),
        '', '## 5. Horizon-wise Error','', 'Sample/channel-weighted validation MSE; three equal forecast segments.', '',*markdown_table(segments),
        '', '## 6. High-Error Selection','',
        'Top five baseline-high-error, positive MAE regression and negative MAE improvement per setting/arm; within channel/category separate selected starts by >=H. Five prespecified anchors are additionally retained across arms. All selection indices and activations are in run.yaml and selected_cases.npz.',
        '', '## 7. Case Analysis','',
        'Forecasts and content update heatmaps for top regression/improvement. Values are in train-standardized coordinates. The baseline is source tp_pre for frozen arms and matched continued tp_pre for joint arms. Off disables only the NEW content and retains original broadcast.', '']
    report += [f'![{Path(ref).stem}]({ref})' for ref in refs]
    if len(anchors):report += ['', '### Prespecified anchor cases','',*markdown_table(anchors[['setting','candidate_config_id','sample_id','channel','baseline_mse','native_mse','off_mse','roll_mse','reverse_mse','delta_mse']])]
    report += ['', '## 8. Repeated Observable Patterns','',*markdown_table(pd.DataFrame(stats)),
        '', '## 9. Objective Defect Summary','',
        'Phase content must beat equal-capacity trend expansion and its appropriate baseline before being credited with efficacy. Centering must beat uncentered content. Native-minus-off <0 shows fixed-model benefit, not a retraining result; roll/reverse-minus-native >0 supports alignment dependence but may involve out-of-distribution activations. Frozen off outputs are checked bitwise against source tp_pre. Trend-only permutations are checked bitwise invariant. Centered content has zero phase mean in latent space, not necessarily zero output shift.',
        '', '## 10. Experiment Scope','',
        'Same three dataset/horizon pairs and two seeds as stage 2; four more epochs, no from-scratch training. Initial full-model identity is exact. No gate, new trend token or output-energy regularizer is introduced. Output energy/alignment are measured using labels only for diagnosis; no label-dependent decision enters inference. Runtime, memory, configurations and checkpoint hashes are retained in run.yaml and results.csv.', '']
    (audit/'objective_error_analysis.md').write_text('\n'.join(report))
    with zipfile.ZipFile(audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['objective_error_analysis.md',*refs]:z.write(audit/name,name)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        assert all(z.read(n)==(audit/n).read_bytes() for n in z.namelist())
    assert set(errors.setting)==set(results.setting)=={c['metadata']['setting'] for c in cases}==set(settings)
    assert {p.name for p in audit.iterdir()}=={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz','objective_error_analysis.md','objective_error_analysis.zip','figures'}
    assert {p.name for p in figures.iterdir()}=={Path(r).name for r in refs}
    run['analysis']['selections']=[c['metadata'] for c in cases]
    run['analysis']['comparisons']=comparisons
    for entry in run['execution']['settings']:
        entry['runtime']=[json.loads((scratch/'runs'/f"{entry['setting']}__{arm}"/'result.json').read_text()) for arm in ARMS]
    run['validation']={'results_checked':True,'ranking_and_cases_checked':True,'report_and_archive_checked':True,'directory_and_settings_checked':True,'status':'passed'}
    (audit/'run.yaml').write_text(yaml.safe_dump(json.loads(json.dumps(run,default=lambda x:x.item())),sort_keys=False))
    save_json(scratch/'audit_complete.json',{'status':'passed','rows':len(errors),'cases':len(cases),'figures':len(refs)})
    print(json.dumps({'status':'passed','rows':len(errors),'cases':len(cases),'figures':len(refs)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch-dir',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--worker-setting');p.add_argument('--merge-only',action='store_true')
    args=p.parse_args()
    if args.worker_setting:worker(args.scratch_dir,args.worker_setting);return
    manifest=json.loads((args.scratch_dir/'manifest.json').read_text())
    settings=[s['setting'] for s in manifest['sources']]
    if not args.merge_only:
        gpus=args.gpus.split(',')
        def bucket(gpu,assigned):
            for setting in assigned:
                with (args.scratch_dir/f'analysis_{setting}.log').open('a') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--scratch-dir',str(args.scratch_dir),
                        '--audit-dir',str(args.audit_dir),'--worker-setting',setting],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(bucket,gpu,settings[i::len(gpus)]) for i,gpu in enumerate(gpus)]
            for future in futures:future.result()
    merge(args.scratch_dir,args.audit_dir)


if __name__=='__main__':main()
