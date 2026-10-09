"""Extract fixed-checkpoint features on registered temporal blocks; no optimizer."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.message_switch_diagnostic.core import FEATURES, temporal_roles
from scripts.lowrank_token_interaction.activation_audit import Trace, digest, save_json, mean_cos
from scripts.lowrank_token_interaction.model import load_model, state_hash
from scripts.run_lowrank_interaction_stage2 import forecast
from src.dataset.data_factory import data_provider


def features(model, batch, trace, off_phase):
    native = trace.values
    x = native['x'].transpose(1, 2).double()
    shift = x[..., -24:].mean(-1)-x.mean(-1)
    last = x[..., -1]-x[..., -24:].mean(-1)
    left, right = x[..., :-24], x[..., 24:]
    difference = (left-right).square().mean(-1).sqrt()
    correlation = mean_cos(left-left.mean(-1, keepdim=True), right-right.mean(-1, keepdim=True))
    q = torch.nn.functional.linear(native['m'].double(), model.predictor.decoder.weight.double())
    phase = off_phase.double().reshape(len(x), -1, 24, x.shape[1]).mean(2).transpose(1, 2)
    trend = native['trend'].double().reshape(len(x), -1, 24, x.shape[1]).mean(2).transpose(1, 2)
    level_difference = trend-phase
    gate = native['gate'][..., 0].double()
    d = q*(1-gate[..., None])
    a = level_difference*gate[..., None]
    rms = lambda v: v.square().mean(-1).sqrt()
    cancellation = (rms(a)+rms(d))/rms(a+d).clamp_min(1e-8)
    values = [shift, last, torch.log1p(difference), correlation,
              torch.log1p(rms(d)), torch.log1p(rms(level_difference)), mean_cos(a,d),
              torch.log1p(cancellation.clamp_max(10))]
    return {k:v.cpu().numpy().reshape(-1) for k,v in zip(FEATURES, values)}


def batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        items = [dataset[int(i)] for i in ids]
        yield ids, [torch.as_tensor(np.stack([x[j] for x in items])) for j in range(4)]


def main(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(d,h,s) for d in ('ETTh1','ETTm2','Weather') for h in (96,720) for s in (2021,2022)]
    for j,(name,horizon,seed) in enumerate(jobs):
        if j % args.shards != args.shard:
            continue
        setting = f'{name}_H{horizon}_S{seed}'
        done = args.output/f'{setting}.json'
        if done.exists():
            previous=json.loads(done.read_text())
            assert previous['extract_sha256']==digest(__file__)
            assert previous['core_sha256']==digest(Path(__file__).with_name('core.py'))
            continue
        started=time.monotonic()
        tp_run=args.runs/f'{setting}_val__tp_add'
        model, config = load_model(tp_run, device)
        b0, _ = load_model(args.runs/f'{setting}_val__b0', device)
        initial=state_hash(model.state_dict())
        train,_=data_provider(copy.deepcopy(config.dataset_args),'train')
        val,_=data_provider(copy.deepcopy(config.dataset_args),'val')
        roles,cut=temporal_roles(len(train.data_x),len(val.data_x),720,horizon)
        trace=Trace(model)
        rows=[]; checks={'single_forward_off_replay_max_abs':0.}
        provenance={'setting':setting,'dataset':name,'horizon':horizon,'seed':seed,
                    'training_length':len(train.data_x),'validation_length':len(val.data_x),
                    'cut':cut,'roles':{},'extract_sha256':digest(__file__),
                    'core_sha256':digest(Path(__file__).with_name('core.py')),
                    'checkpoint_sha256':digest(json.loads((tp_run/'config.json').read_text())['checkpoint'])}
        with torch.inference_mode():
            for role,ids in roles.items():
                dataset=train if role in ('fit','cal') else val
                if args.smoke:
                    ids=ids[:8]
                offset=0 if role!='eval' else len(train.data_x)-720
                provenance['roles'][role]={'windows':len(ids),'start':int(ids[0])+offset,
                     'end':int(ids[-1])+720+horizon-1+offset,
                     'history_start':str(dataset.timestamps[int(ids[0])]),
                     'target_end':str(dataset.timestamps[int(ids[-1])+720+horizon-1])}
                for sample_ids,batch in batches(dataset,ids,config.dataset_args.batch_size):
                    trace.enabled=True
                    cp,truth=forecast(model,batch,device)
                    native={k:v.clone() for k,v in trace.values.items()}
                    trace.enabled=False
                    model.lowrank_interaction_branch.intervention='off'
                    off,_=forecast(model,batch,device)
                    model.lowrank_interaction_branch.intervention='native'
                    # Derive the off phase from the native phase and normalized profile.
                    q=torch.nn.functional.linear(native['m'],model.predictor.decoder.weight)
                    profile=q.repeat_interleave(24,-1).transpose(1,2)
                    off_phase=native['phase']-profile
                    sigma=batch[0].float().to(device).var(1,unbiased=False).add(model.revin.eps).sqrt()
                    predicted_off=cp-profile*sigma[:,None]*(1-native['gate'][...,0])[:,None]
                    checks['single_forward_off_replay_max_abs']=max(checks['single_forward_off_replay_max_abs'],float((predicted_off-off).abs().max()))
                    torch.testing.assert_close(predicted_off,off,atol=5e-5,rtol=5e-5)
                    bp,_=forecast(b0,batch,device)
                    values=features(model,batch,trace,off_phase)
                    for label,prediction in [('candidate',cp),('off',off),('baseline',bp)]:
                        err=prediction.double()-truth.double()
                        values[label+'_mse']=err.square().mean(1).cpu().numpy().reshape(-1)
                        values[label+'_mae']=err.abs().mean(1).cpu().numpy().reshape(-1)
                    channels=cp.shape[-1]
                    frame=pd.DataFrame(values)
                    frame['sample_id']=np.repeat(sample_ids,channels)
                    frame['channel']=np.tile(np.arange(channels),len(sample_ids))
                    frame['setting']=setting;frame['role']=role
                    frame['dataset']=name;frame['horizon']=horizon;frame['seed']=seed
                    frame['time_range']=[f'{dataset.timestamps[int(i)+720]}/{dataset.timestamps[int(i)+720+horizon-1]}' for i in frame.sample_id]
                    rows.append(frame)
        trace.close()
        assert initial==state_hash(model.state_dict())
        frame=pd.concat(rows,ignore_index=True)
        frame.to_csv(args.output/f'{setting}.csv',index=False)
        provenance.update(checks=checks,state_unchanged=True,seconds=time.monotonic()-started,
                          csv_sha256=digest(args.output/f'{setting}.csv'),smoke=args.smoke)
        save_json(done,provenance)
        print(setting,len(frame),flush=True)
        del model,b0,train,val,rows,frame
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args())
