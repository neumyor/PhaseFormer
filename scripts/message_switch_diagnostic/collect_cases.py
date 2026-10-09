"""Replay program-ranked held-out cases with fixed weights and decisions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.analyze_lowrank_six import pack
from scripts.lowrank_token_interaction.model import load_model, state_hash
from scripts.run_lowrank_interaction_stage2 import forecast
from src.dataset.data_factory import data_provider
from scripts.message_switch_diagnostic.core import temporal_roles
from scripts.lowrank_token_interaction.activation_audit import save_json, digest


def main(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0')
    requests=json.loads((args.scratch/'case_requests.json').read_text())
    all_cases=[]
    for setting in sorted({r['setting'] for r in requests}):
        source=setting.removesuffix('_eval')
        model,config=load_model(args.runs/f'{source}_val__tp_add',device)
        base,_=load_model(args.runs/f'{source}_val__b0',device)
        before=state_hash(model.state_dict())
        train,_=data_provider(config.dataset_args,'train')
        data,_=data_provider(config.dataset_args,'val')
        roles,_=temporal_roles(len(train.data_x),len(data.data_x),720,model.pred_len)
        ids=roles['eval'];batchsize=config.dataset_args.batch_size
        cache={}
        with torch.inference_mode():
            for request in [r for r in requests if r['setting']==setting]:
                sample,ch=int(request['sample_id']),int(request['channel'])
                position=int(np.searchsorted(ids,sample));assert ids[position]==sample
                start=position//batchsize*batchsize
                if start not in cache:
                    samples=ids[start:start+batchsize]
                    items=[data[int(i)] for i in samples]
                    batch=[torch.as_tensor(np.stack([x[j] for x in items])) for j in range(4)]
                    native,truth=forecast(model,batch,device)
                    phase=model.last_phase_forecast.clone();trend=model.last_lowrank_forecast.clone()
                    gate=model.lowrank_dynamic_fusion.last_gate.clone()
                    model.lowrank_interaction_branch.intervention='off'
                    off,_=forecast(model,batch,device)
                    model.lowrank_interaction_branch.intervention='native'
                    bp,_=forecast(base,batch,device)
                    cache[start]=(batch[0].numpy(),truth.cpu().numpy(),native.cpu().numpy(),
                                   off.cpu().numpy(),bp.cpu().numpy(),phase.cpu().numpy(),
                                   trend.cpu().numpy(),gate.cpu().numpy())
                history,truth,native,off,bp,phase,trend,gate=cache[start]
                i=position-start
                candidate=off[i,:,ch] if request['disabled'] else native[i,:,ch]
                case={'metadata':request,'history':history[i,:,ch],'truth':truth[i,:,ch],
                      'candidate':candidate,'baseline':bp[i,:,ch],'off':off[i,:,ch],
                      'phase':phase[i,:,ch],'trend':trend[i,:,ch],'gate':gate[i,ch],
                      'acceptance':np.array([float(not request['disabled'])],dtype=np.float32)}
                for label,array in [('candidate',candidate),('baseline',bp[i,:,ch]),('off',off[i,:,ch]),('native',native[i,:,ch])]:
                    error=array.astype(np.float64)-truth[i,:,ch].astype(np.float64)
                    for metric,value in [('mse',np.square(error).mean()),('mae',np.abs(error).mean())]:
                        np.testing.assert_allclose(value,request[label+'_'+metric],atol=5e-5,rtol=5e-4)
                all_cases.append(case)
        assert before==state_hash(model.state_dict())
        print(setting,'case replay passed',flush=True)
        del model,base,train,data,cache
        torch.cuda.empty_cache()
    pack(all_cases,args.audit/'selected_cases.npz')
    save_json(args.scratch/'case_replay_complete.json',{'cases':len(all_cases),'state_unchanged':True,
              'script_sha256':digest(__file__),'case_npz_sha256':digest(args.audit/'selected_cases.npz')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    main(parser.parse_args())
