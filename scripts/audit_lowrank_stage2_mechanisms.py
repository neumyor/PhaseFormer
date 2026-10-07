#!/usr/bin/env python3
"""Frozen-checkpoint activation interventions; never trains or reads test data.

Run on the stage-2 host with --source-scratch, --source-audit, --output and
--work-dir. Six independent setting workers can use six GPUs. The diagnostic
cohort is 128 equally spaced validation windows (all channels), selected without
looking at errors. Ten detailed cases use the prior audit's rank-one cases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from scripts.analyze_lowrank_stage2 import load_model, markdown_table
from scripts.run_lowrank_interaction_stage2 import ARMS, construct, forecast, load_i0
from src.dataset.data_factory import data_provider


def select_cases(source_audit):
    groups = {
        ('ETTh1_H96_S2021_val', 'pt_post'): ['candidate_regression'],
        ('ETTh1_H96_S2022_val', 'pt_post'): ['candidate_regression'],
        ('Weather_H720_S2021_val', 'pt_null'): ['candidate_regression', 'candidate_improvement'],
        ('Weather_H720_S2021_val', 'tp_selective'): ['candidate_regression', 'candidate_improvement'],
        ('ETTm2_H720_S2022_val', 'tp_selective'): ['candidate_regression', 'candidate_improvement'],
        ('Weather_H720_S2022_val', 'tp_scaled'): ['candidate_regression', 'candidate_improvement'],
    }
    with np.load(source_audit / 'selected_cases.npz', allow_pickle=False) as z:
        metadata = [json.loads(str(x)) for x in z['metadata']]
    cases = [m for m in metadata if m['rank'] == 1 and
             m['category'] in groups.get((m['setting'], m['candidate_config_id']), [])]
    assert len(cases) == 10
    return cases


class Intervention:
    """Analysis-only replacement, independently checked against native forward."""
    def __init__(self, branch):
        self.branch = branch
        self.mode = 'native'
        self.capture = {}
        self.original_pt = branch._phase_to_trend
        self.original_tp = branch._trend_to_phase
        branch._phase_to_trend = types.MethodType(
            lambda _branch, latent, phase: self.update(latent, phase, True), branch)
        branch._trend_to_phase = types.MethodType(
            lambda _branch, latent, phase: self.update(latent, phase, False), branch)

    def restore(self):
        self.branch._phase_to_trend = self.original_pt
        self.branch._trend_to_phase = self.original_tp

    def update(self, latent, phase_latent, to_trend):
        br = self.branch
        b, c, slots, d = phase_latent.shape
        trend = latent.reshape(b*c, 1, br.rank)
        phase = phase_latent.reshape(b*c, slots, d)
        pre = br.update_style != 'post_norm'
        t, p = (br.trend_norm(trend), br.phase_norm(phase)) if pre else (trend, phase)
        if to_trend:
            attn, output = br.phase_to_trend_attention, br.trend_out
            q, k, v = br.trend_query(t), br.phase_key(p), br.phase_value(p)
        else:
            attn, output = br.trend_to_phase_attention, br.phase_out
            q, k, v = br.phase_query(p), br.trend_key(t), br.trend_value(t)
        if pre:
            wq, wk, wv = attn.in_proj_weight.chunk(3)
            bq, bk, bv = attn.in_proj_bias.chunk(3)
            q, k, v = F.linear(q, wq, bq), F.linear(k, wk, bk), F.linear(v, wv, bv)
            scores = q @ k.transpose(-2, -1) / math.sqrt(d)
            conditional = scores.softmax(-1)
            if br.null_read or br.selective_write:
                extended = torch.cat([scores, br.null_score.expand_as(scores[..., :1])], -1)
                weights = extended.softmax(-1)[..., :-1]
            else:
                weights = conditional
            original_weights = weights.clone()
            if self.mode == 'no_null':
                weights = conditional
            elif self.mode == 'uniform_phase':
                weights = weights.sum(-1, keepdim=True).expand_as(weights) / weights.shape[-1]
            elif self.mode == 'mean_gate':
                weights = weights.mean(-2, keepdim=True).expand_as(weights)
            elif self.mode == 'roll_gate':
                weights = weights.roll(slots//2, dims=-2)
            mass = weights.sum(-1, keepdim=True)
            attended = F.linear(weights @ v, attn.out_proj.weight) + mass * attn.out_proj.bias
            message = F.linear(attended, output.weight) + mass * output.bias
        else:
            attended, weights = attn(q, k, v, need_weights=True)
            original_weights = weights
            message = output(attended)
            mass = weights.sum(-1, keepdim=True)
        original = trend if to_trend else phase
        scale = br.interaction_scale if br.update_style == 'scaled_pre_norm' else 1.
        beta = {'off': 0., 'quarter': .25, 'half': .5, 'double': 2.}.get(self.mode, 1.)
        if pre:
            updated = original + beta * scale * message
        else:
            norm = br.trend_norm if to_trend else br.phase_norm
            if self.mode == 'off':
                updated = original
            elif self.mode == 'norm_only':
                updated = norm(original)
            elif self.mode == 'message_only':
                updated = original + message
            else:
                updated = norm(original + beta * message)
        self.capture = {
            'trend': trend.reshape(b,c,1,br.rank).detach(),
            'phase': phase.reshape(b,c,slots,d).detach(),
            'original': original.reshape(b,c,*original.shape[1:]).detach(),
            'updated': updated.reshape(b,c,*updated.shape[1:]).detach(),
            'message': message.reshape(b,c,*message.shape[1:]).detach(),
            'acceptance': mass.reshape(b,c,-1).detach(),
            'weights': weights.reshape(b,c,*weights.shape[1:]).detach(),
            'original_weights': original_weights.reshape(b,c,*original_weights.shape[1:]).detach(),
        }
        shape = (b,c,br.rank) if to_trend else (b,c,slots,d)
        return updated.reshape(shape), weights.reshape(b,c,slots)


def modes_for(arm):
    modes = ['native', 'off', 'quarter', 'half', 'double']
    if arm.endswith('post'):
        modes += ['norm_only', 'message_only']
    if arm == 'pt_null':
        modes += ['no_null', 'uniform_phase']
    if arm == 'tp_selective':
        modes += ['mean_gate', 'roll_gate', 'no_null']
    return modes


def batch_at(dataset, ids):
    items = [dataset[int(i)] for i in ids]
    return [torch.as_tensor(np.stack([x[j] for x in items])) for j in range(4)]


def summarize_activation(cap):
    original, updated, message = cap['original'], cap['updated'], cap['message']
    change = updated-original
    weights = cap['weights']
    conditional = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    return {
        'original_norm': original.norm(dim=-1).mean(-1),
        'message_norm': message.norm(dim=-1).mean(-1),
        'update_norm': change.norm(dim=-1).mean(-1),
        'update_ratio': change.norm(dim=-1).mean(-1) / original.norm(dim=-1).mean(-1).clamp_min(1e-12),
        'token_cosine': F.cosine_similarity(original, updated, dim=-1).mean(-1),
        'acceptance_mean': cap['acceptance'].mean(-1),
        'acceptance_std': cap['acceptance'].std(-1, unbiased=False),
        'conditional_entropy': -(conditional * conditional.clamp_min(1e-20).log()).sum(-1).mean(-1),
        'conditional_max': conditional.max(-1).values.mean(-1),
    }


def worker(args):
    torch.set_num_threads(2)
    # Highest precision supports output-space decomposition without TF32 noise.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    setting = args.setting
    cases = [c for c in select_cases(args.source_audit) if c['setting'] == setting]
    base, data_args = load_model(args.source_scratch/'runs'/f'{setting}__i0_continue', device)
    data_args.dataset_args.num_workers = 0
    dataset, _ = data_provider(data_args.dataset_args, 'val')
    uniform = set(np.linspace(0, len(dataset)-1, args.windows, dtype=int).tolist())
    ids = sorted(uniform | {c['sample_id'] for c in cases})
    rows, packed, checks = [], {}, []
    for arm in list(ARMS)[1:]:
        run_dir = args.source_scratch/'runs'/f'{setting}__{arm}'
        model, _ = load_model(run_dir, device)
        br = model.lowrank_interaction_branch
        probe = Intervention(br)
        for start in range(0, len(ids), 16):
            current = ids[start:start+16]
            batch = batch_at(dataset, current)
            with torch.inference_mode():
                bp, truth = forecast(base, batch, device)
                preds, activations = {}, {}
                for mode in modes_for(arm):
                    probe.mode = mode
                    preds[mode], _ = forecast(model, batch, device)
                    if mode == 'native':
                        activations = dict(probe.capture)
                    elif mode in {'mean_gate', 'roll_gate'}:
                        torch.testing.assert_close(
                            probe.capture['acceptance'].mean(-1),
                            activations['acceptance'].mean(-1), atol=2e-6, rtol=2e-6)
                        torch.testing.assert_close(
                            (probe.capture['updated']-probe.capture['original']).mean(-2),
                            (activations['updated']-activations['original']).mean(-2),
                            atol=2e-6, rtol=2e-6)
                    elif mode == 'uniform_phase':
                        torch.testing.assert_close(probe.capture['acceptance'],
                            activations['acceptance'], atol=2e-6, rtol=2e-6)
                if arm.startswith('pt_') and not arm.endswith('post'):
                    # The unchanged rank decoder and output fusion are linear in
                    # a fixed P->T message; this is a structural intervention check.
                    torch.testing.assert_close(preds['half'],
                        (preds['native']+preds['off'])/2, atol=3e-6, rtol=3e-6)
                probe.restore()
                native, _ = forecast(model, batch, device)
                diff = (native-preds['native']).abs().max().item()
                torch.testing.assert_close(native, preds['native'], atol=2e-6, rtol=2e-6)
                # Off must exactly match the original branch's no-interaction path.
                old_mode = br.interaction_mode
                br.interaction_mode = 'none'
                direct_off, _ = forecast(model, batch, device)
                br.interaction_mode = old_mode
                torch.testing.assert_close(direct_off, preds['off'], atol=2e-6, rtol=2e-6)
                probe = Intervention(br)
                checks.append(diff)
                act = {k:v.cpu().numpy() for k,v in summarize_activation(activations).items()}
                delta = preds['native']-preds['off']
                required = truth-preds['off']
                energy = delta.square().mean(1)
                alignment = (delta*required).mean(1)
                residual_energy = required.square().mean(1)
                cosine = alignment / (energy*residual_energy).sqrt().clamp_min(1e-12)
                # Exact identity in output space, regardless of network nonlinearities.
                change_mse = (preds['native']-truth).square().mean(1) - residual_energy
                torch.testing.assert_close(change_mse, energy-2*alignment, atol=2e-5, rtol=2e-5)
                for bi, sid in enumerate(current):
                    for channel in range(truth.shape[-1]):
                        role = 'uniform' if sid in uniform else 'case_only'
                        row = {'setting':setting,'candidate_config_id':arm,
                               'sample_id':sid,'channel':channel,'cohort':role,
                               'time_range':f'{dataset.timestamps[sid+base.seq_len]}/{dataset.timestamps[sid+base.seq_len+base.pred_len-1]}',
                               **{k:float(v[bi,channel]) for k,v in act.items()},
                               'output_energy':float(energy[bi,channel]),
                               'output_alignment':float(alignment[bi,channel]),
                               'output_cosine':float(cosine[bi,channel]),
                               'oracle_output_scale':float(alignment[bi,channel]/energy[bi,channel].clamp_min(1e-16)),
                               'interaction_scale':float(br.interaction_scale) if hasattr(br,'interaction_scale') else 1.,
                               'fusion_gate':float(model.lowrank_interaction_gate.sigmoid().reshape(-1)[channel])}
                        for name, pred in {'baseline':bp, **preds}.items():
                            err = pred[bi,:,channel]-truth[bi,:,channel]
                            row[name+'_mse'] = float(err.square().mean())
                            row[name+'_mae'] = float(err.abs().mean())
                        rows.append(row)
                        for index, case in enumerate(cases):
                            if case['sample_id'] != sid or case['channel'] != channel:
                                continue
                            key = f'case{index}__{arm}__'
                            packed[key+'metadata'] = np.array(json.dumps({**row,'selection':case}))
                            for name, pred in {'truth':truth,'baseline':bp, **preds}.items():
                                packed[key+name] = pred[bi,:,channel].cpu().numpy()
                            packed[key+'history'] = batch[0][bi,:,channel].numpy()
                            for name, value in activations.items():
                                packed[key+name+'_activation'] = value[bi,channel].cpu().numpy()
            print(setting, arm, start+len(current), '/', len(ids), flush=True)
        probe.restore()
        del model
    # At source-i0 weights: separate insertion-time LN damage from trained adaptation.
    initial_rows = []
    manifest = json.loads((args.source_scratch/'manifest.json').read_text())
    source = next(s for s in manifest['sources'] if s['setting'] == setting)
    if setting.startswith('ETTh1'):
        from scripts.run_lowrank_interaction_stage2 import prepare_config
        cfg = source['config']
        state = torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
        for arm in ('pt_post','pt_pre','pt_scaled'):
            torch.manual_seed(cfg['seed'])
            model, _ = construct(prepare_config(cfg,arm,4))
            load_i0(model,state)
            model.to(device).eval()
            probe = Intervention(model.lowrank_interaction_branch)
            for case in cases:
                batch = batch_at(dataset,[case['sample_id']])
                channel=case['channel']
                predictions={}
                with torch.inference_mode():
                    for mode in modes_for(arm):
                        probe.mode=mode
                        pred,truth=forecast(model,batch,device)
                        predictions[mode]=pred[0,:,channel].cpu().numpy()
                row={'setting':setting,'arm':arm,'sample_id':case['sample_id'],'channel':channel}
                for mode,pred in predictions.items():
                    row[mode+'_mse']=float(np.mean((pred-truth[0,:,channel].cpu().numpy())**2))
                initial_rows.append(row)
            probe.restore()
            del model
    out=args.work_dir/setting
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv',index=False)
    np.savez_compressed(out/'cases.npz',**packed)
    (out/'checks.json').write_text(json.dumps({'native_max_abs':max(checks),'initial_cases':initial_rows,'uniform_windows':len(uniform),'validation_windows':len(dataset)},indent=2))


def merge(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((args.source_scratch/'manifest.json').read_text())
    settings=sorted(s['setting'] for s in manifest['sources'])
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    frame=pd.concat([pd.read_csv(args.work_dir/s/'rows.csv') for s in settings],ignore_index=True)
    frame.to_csv(out/'sample_errors.csv',index=False)
    uniform=frame[frame.cohort=='uniform']
    metric_columns=[c for c in uniform if c.endswith('_mse') or c.endswith('_mae')]
    result=uniform.groupby(['setting','candidate_config_id'])[metric_columns].mean().reset_index()
    result.to_csv(out/'results.csv',index=False)
    packed={}
    for setting in settings:
        with np.load(args.work_dir/setting/'cases.npz',allow_pickle=False) as z:
            packed.update({setting+'__'+k:z[k] for k in z.files})
    packed['setting']=np.array(settings)
    np.savez_compressed(out/'selected_cases.npz',**packed)
    cases=select_cases(args.source_audit)
    case_rows,refs=[],[]
    for setting in settings:
        local_cases=[c for c in cases if c['setting']==setting]
        for index,case in enumerate(local_cases):
            arm=case['candidate_config_id']
            key=f'{setting}__case{index}__{arm}__'
            meta=json.loads(str(packed[key+'metadata']))
            case_rows.append(meta)
            fig,axes=plt.subplots(3,2,figsize=(13,10))
            history=packed[key+'history'][-192:]
            axes[0,0].plot(np.arange(-len(history),0),history,color='0.65',label='history')
            for name,color in [('truth','black'),('baseline','#24679c'),('native','#c04445'),('off','#27906b')]:
                axes[0,0].plot(packed[key+name],label=name,color=color,lw=1.1)
            axes[0,0].legend(fontsize=8,ncol=3)
            axes[0,0].set_title('Forecasts (train-standardized units)')
            acceptance=packed[key+'acceptance_activation'].reshape(-1)
            weights=packed[key+'weights_activation'].reshape(-1)
            if len(acceptance)==1:
                axes[0,1].bar(np.arange(len(weights)),weights,label='real phase mass')
                axes[0,1].axhline((1-acceptance[0]),color='#c04445',ls='--',label=f'null mass {1-acceptance[0]:.3f}')
            else:
                axes[0,1].bar(np.arange(len(acceptance)),acceptance,label='trend receipt')
                axes[0,1].axhline(acceptance.mean(),color='#c04445',ls='--',label='slot mean')
            axes[0,1].set_ylim(0,1.02)
            axes[0,1].set_title('Attention receipt by phase slot')
            axes[0,1].legend(fontsize=8)
            original=packed[key+'original_activation']
            change=packed[key+'updated_activation']-original
            for ax,values,title in [(axes[1,0],original,'Original receiving token(s)'),(axes[1,1],change,'Actual token update')]:
                lim=max(float(np.abs(values).max()),1e-12)
                im=ax.imshow(values,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim)
                fig.colorbar(im,ax=ax,fraction=.04)
                ax.set_title(title)
                ax.set_xlabel('Latent coordinate');ax.set_ylabel('Receiving slot')
            modes=modes_for(arm)
            axes[2,0].bar(np.arange(len(modes)),[meta[m+'_mse'] for m in modes],color='#567c9d')
            axes[2,0].set_xticks(np.arange(len(modes)),modes,rotation=35,ha='right',fontsize=8)
            axes[2,0].set_title('Frozen-weight intervention MSE')
            required=packed[key+'truth']-packed[key+'off']
            delta=packed[key+'native']-packed[key+'off']
            axes[2,1].plot(required,label='needed correction',color='black',lw=1)
            axes[2,1].plot(delta,label='actual output correction',color='#c04445',lw=1)
            axes[2,1].legend(fontsize=8)
            axes[2,1].set_title(f"Correction cosine {meta['output_cosine']:.3f}; energy {meta['output_energy']:.4g}")
            fig.suptitle(f"{setting} / {arm} / {case['category']} / sample {case['sample_id']}, channel {case['channel']}",fontsize=12)
            fig.tight_layout()
            name=f"{setting}__{arm}__{case['category']}.png"
            fig.savefig(figures/name,dpi=130);plt.close(fig)
            refs.append('figures/'+name)
    avg=result.groupby('candidate_config_id')[metric_columns].mean()
    # All-validation observational check from the previous audit, without new inference.
    cols=['setting','candidate_config_id','acceptance_mean','acceptance_phase_std','update_norm','prediction_change_rms','enabled_minus_disabled_mse','delta_mse']
    full_parts=[]
    for chunk in pd.read_csv(args.source_audit/'sample_errors.csv',usecols=cols,chunksize=250000):
        full_parts.append(chunk[chunk.candidate_config_id.isin(['pt_null','tp_selective'])])
    full=pd.concat(full_parts,ignore_index=True)
    observations=[]
    for (setting,arm),g in full.groupby(['setting','candidate_config_id']):
        meaningful=g[g.prediction_change_rms>1e-4]
        corr=g[['acceptance_mean','enabled_minus_disabled_mse']].corr(method='spearman').iloc[0,1]
        for label,sub in [('all',g),('accept<0.1',g[g.acceptance_mean<.1]),('accept>0.9',g[g.acceptance_mean>.9])]:
            observations.append({'setting':setting,'arm':arm,'bin':label,'n':len(sub),
                'mean_enabled_minus_disabled_mse':sub.enabled_minus_disabled_mse.mean(),
                'harm_fraction':(sub.enabled_minus_disabled_mse>1e-6).mean(),
                'acceptance_error_spearman':corr,
                'meaningful_output_change_n':len(meaningful)})
    diagnostics=[]
    for (setting,arm),g in uniform.groupby(['setting','candidate_config_id']):
        active=g[g.output_energy>1e-8]
        diagnostics.append({'setting':setting,'arm':arm,'active_n':len(active),'n':len(g),
            'mean_output_energy':g.output_energy.mean(),'mean_output_alignment':g.output_alignment.mean(),
            'wrong_direction_fraction':(active.output_alignment<=0).mean(),
            'overstep_fraction':((active.output_alignment>0)&(active.oracle_output_scale<.5)).mean(),
            'mean_update_ratio':g.update_ratio.mean(),'mean_gate_std':g.acceptance_std.mean()})
    initial=[]
    checks={}
    for setting in settings:
        checks[setting]=json.loads((args.work_dir/setting/'checks.json').read_text())
        initial.extend(checks[setting]['initial_cases'])
    lines=['# Frozen-checkpoint mechanism and activation audit','',
           '## 1. Scope and selection','',
           f'No training or test evaluation. Six settings; {args.windows} equally spaced validation windows per setting, all channels. Windows overlap and are not independent statistical replicates. Ten detailed cases use pre-existing rank-one regression/improvement cases, chosen by the declared setting/arm groups. They are explanatory cases, not a representative performance sample.',
           '', 'Float32 highest matmul precision; original stage-2 efficacy metrics used medium precision. Native reimplementation and off-path outputs are checked against the unmodified model on every batch. New metrics describe this diagnostic cohort, not the full validation efficacy results.',
           '', '## 2. Intervention definitions','',
           '`off`: original token bypass; `quarter/half/double`: message coefficient times 0.25/0.5/2 (post-norm still applies LN). `norm_only`: LN(original), message zero. `message_only`: original+message, bypass LN. `no_null`: remove null and renormalize real attention. `uniform_phase`: uniform real-phase weights preserving total real mass. `mean_gate`: same sample/channel mean receipt at every phase. `roll_gate`: cyclic shift by 12 slots, preserving gate histogram. All weights remain frozen; altered states may be out of the training distribution.',
           '', '## 3. Diagnostic cohort means','']
    lines += markdown_table(avg.reset_index()[['candidate_config_id','baseline_mse','native_mse','off_mse','half_mse','double_mse']])
    lines += ['', '### Routing interventions by setting','']
    lines += markdown_table(result[result.candidate_config_id.isin(['pt_null','tp_selective'])][['setting','candidate_config_id','native_mse','off_mse','no_null_mse','uniform_phase_mse','mean_gate_mse','roll_gate_mse']])
    lines += ['', '### Normalization interventions','']
    lines += markdown_table(result[result.candidate_config_id.str.endswith('post')][['setting','candidate_config_id','native_mse','off_mse','norm_only_mse','message_only_mse']])
    lines += ['', '## 4. Output correction decomposition','',
              'Let d=y_on-y_off, e=y_true-y_off. Exactly: MSE_on-MSE_off = mean(d²)-2 mean(d e). The first term is correction energy; the second measures alignment. The oracle output scale mean(d e)/mean(d²) uses future labels and is diagnostic only, not an available inference signal or a selected model hyperparameter. With positive alignment but scale below 0.5 the correction overshoots enough to hurt; nonpositive alignment is a wrong direction. Fractions below exclude output energy <=1e-8.', '']
    lines += markdown_table(pd.DataFrame(diagnostics))
    lines += ['', '## 5. Full-validation observational checks','',
              'These reuse the original audit at its original precision. Harm fraction uses enabled-minus-disabled MSE >1e-6; correlations are descriptive and overlapping windows are not independent. Empty bins are NaN. Gate magnitude alone does not establish reliability.','']
    lines += markdown_table(pd.DataFrame(observations))
    lines += ['', '## 6. Source-i0 insertion-time checks','',
              'Reconstructed with the original seed before model construction and common source weights. These are initial networks, not separately trained models.','']
    lines += markdown_table(pd.DataFrame(initial))
    lines += ['', '## 7. Detailed selected cases','']
    lines += markdown_table(pd.DataFrame(case_rows)[['setting','candidate_config_id','sample_id','channel','acceptance_mean','acceptance_std','update_norm','update_ratio','token_cosine','output_energy','output_alignment','output_cosine','oracle_output_scale','baseline_mse','native_mse','off_mse','half_mse']])
    for ref in refs:
        lines += ['',f'![{Path(ref).stem}]({ref})']
    lines += ['', '## 8. Interpretation limits','',
              'No claim of generalization, significance or superiority to Golden. Native-vs-off isolates a frozen-model intervention, not a retraining effect. Null/gate ablations can shift the activation distribution. Initial-vs-trained observations do not establish a universal optimization law. T->P selective updates are scalar multiples of one shared value direction within each sample/channel; selectivity changes magnitude, not message direction. The rank-r trend token is a learned compression, not a guaranteed low-frequency trend. No model parameter is modified or checkpoint overwritten.', '']
    report=out/'objective_error_analysis.md'
    report.write_text('\n'.join(lines))
    with zipfile.ZipFile(out/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['objective_error_analysis.md',*refs]:z.write(out/name,name)
    with zipfile.ZipFile(out/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        assert all(z.read(n)==(out/n).read_bytes() for n in z.namelist())
    assert len(case_rows)==10 and set(frame.setting)==set(settings)
    assert set(result.setting)==set(settings) and len(result)==48
    for case in case_rows:
        key=(case['setting'],case['candidate_config_id'],case['sample_id'],case['channel'])
        match=frame[(frame.setting==key[0])&(frame.candidate_config_id==key[1])&(frame.sample_id==key[2])&(frame.channel==key[3])]
        assert len(match)==1
        np.testing.assert_allclose(float(match.iloc[0].native_mse),case['native_mse'],rtol=1e-6)
    run={'experiment_id':out.name,'code':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
         'source_experiment':args.source_audit.name,'experiment':{'settings':settings,'training':False,'split':'validation','windows_per_setting':args.windows},
         'selection':{'source':'previous validation audit','cases':cases,'cohort':'equally spaced indices; independent of errors'},
         'checks':checks,'full_validation_observations':observations,'output_diagnostics':diagnostics,
         'validation':{'results_checked':True,'ranking_and_cases_checked':True,'report_and_archive_checked':True,'directory_and_settings_checked':True,'status':'passed'}}
    (out/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    assert {p.name for p in out.iterdir()}=={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz','objective_error_analysis.md','objective_error_analysis.zip','figures'}
    print(json.dumps({'status':'passed','rows':len(frame),'cases':len(cases),'figures':len(refs)}),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('source-scratch','source-audit','output','work-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--setting');p.add_argument('--windows',type=int,default=128)
    p.add_argument('--gpus',default='0,1,2,3,4,5');p.add_argument('--merge-only',action='store_true')
    args=p.parse_args()
    if args.setting:
        worker(args);return
    if not args.merge_only:
        manifest=json.loads((args.source_scratch/'manifest.json').read_text())
        settings=sorted(s['setting'] for s in manifest['sources'])
        gpus=args.gpus.split(',')
        args.work_dir.mkdir(parents=True,exist_ok=True)
        def launch(pair):
            i,setting=pair
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[i%len(gpus)],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,__file__,*sys.argv[1:],'--setting',setting]
            with (args.work_dir/(setting+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(launch,enumerate(settings)))
    merge(args)


if __name__=='__main__':
    main()
