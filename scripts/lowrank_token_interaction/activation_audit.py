#!/usr/bin/env python3
"""Read-only TP-Add activations and fixed-checkpoint diagnostic interventions.

Execute against the frozen training checkout with --repo, --runs, --scratch,
--audit. No optimizer, training loader, model source or checkpoint is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def mean_cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-20)


ACTIVATIONS = ('trend_token', 'phase_tokens', 'message', 'compressed_token', 'cycle_profile')


def pack_cases(cases, path):
    import numpy as np
    from scripts.analyze_lowrank_six import pack
    pack(cases, path)
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in z.files}
    for key in ACTIVATIONS:
        arrays = [np.asarray(c[key]) for c in cases]
        flat = [v.reshape(-1) for v in arrays]
        values[key+'_length'] = np.array([len(v) for v in flat])
        values[key+'_shape'] = np.array([json.dumps(v.shape) for v in arrays])
        values[key] = np.full((len(cases), max(map(len, flat))), np.nan, np.float32)
        for i, v in enumerate(flat):
            values[key][i, :len(v)] = v
    np.savez_compressed(path, **values)


def unpack_cases(path):
    import numpy as np
    from scripts.analyze_lowrank_six import unpack
    cases = unpack(path)
    with np.load(path, allow_pickle=False) as z:
        for key in ACTIVATIONS:
            for i, case in enumerate(cases):
                case[key] = z[key][i, :z[key+'_length'][i]].reshape(json.loads(str(z[key+'_shape'][i])))
    return cases


def message_maps(model):
    """Maps in normalized input coordinates, invariant to latent basis changes."""
    import torch
    branch = model.lowrank_interaction_branch
    if model.predictor.use_mlp or not isinstance(model.predictor.decoder, torch.nn.Linear):
        raise ValueError('The linear-predictor derivation does not apply')
    head = branch.lowrank_head
    if head.pool_factor != 1 or head.smooth_ratio or model.revin.affine:
        raise ValueError('Unexpected input or RevIN transform')
    a = branch.communication.source.weight.double()
    b = branch.communication.output.weight.double()
    w = model.predictor.decoder.weight.double()
    k = w @ b @ a
    f = k @ head.encoder.weight.double()
    # E reads x-x_last; include centering in the full normalized-input map.
    f = f.clone()
    f[:, -1] -= f.sum(-1)
    bias = k @ head.encoder.bias.double()
    u, s, vh = torch.linalg.svd(f, full_matrices=False)
    _, sw, vhw = torch.linalg.svd(w, full_matrices=False)
    active = sw > sw.max() * 1e-10
    projector = vhw[active].T @ vhw[active]
    return {'map': f, 'bias': bias, 'u': u, 's': s, 'vh': vh,
            'projector': projector, 'latent_map': k}


class Trace:
    def __init__(self, model):
        self.enabled = True
        self.values = {}
        self.hooks = [
            model.lowrank_interaction_branch.register_forward_pre_hook(self.branch),
            model.lowrank_interaction_branch.communication.register_forward_hook(self.communication),
            model.lowrank_dynamic_fusion.register_forward_hook(self.fusion),
        ]

    def branch(self, module, inputs):
        if self.enabled:
            self.values['x'] = inputs[0].detach().clone()

    def communication(self, module, inputs, output):
        if self.enabled:
            self.values.update(t=inputs[0].detach().clone(), z=inputs[1].detach().clone(),
                               m=output[2][:, :, 0].detach().clone())

    def fusion(self, module, inputs, output):
        if self.enabled:
            self.values.update(phase=inputs[0].detach().clone(), trend=inputs[1].detach().clone(),
                               gate=module.last_gate.detach().clone())

    def close(self):
        for hook in self.hooks:
            hook.remove()


def overridden_prediction(model, batch, device, message):
    from scripts.run_lowrank_interaction_stage2 import forecast
    def override(module, inputs, output):
        t, z = inputs
        delta = message.unsqueeze(-2).expand_as(z)
        return t, z + delta, delta
    hook = model.lowrank_interaction_branch.communication.register_forward_hook(override)
    try:
        return forecast(model, batch, device)[0]
    finally:
        hook.remove()


def worker(cli):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.lowrank_token_interaction.model import load_model, state_hash
    from scripts.run_lowrank_interaction_stage2 import forecast
    from scripts.analyze_lowrank_six import component_errors
    from src.dataset.data_factory import data_provider
    torch.set_num_threads(2)
    # Medium/TF32 rounds Z and Z+m differently and spoils linear attribution.
    # Record original-precision metrics separately; diagnose in full float32.
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    jobs = [(d, h, s, split) for d in ('ETTh1', 'ETTm2', 'Weather')
            for h in (96, 720) for s in (2021, 2022) for split in ('val', 'test')]
    parts = cli.scratch / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    for index, (dataset_name, horizon, seed, split) in enumerate(jobs):
        if index % cli.shards != cli.shard:
            continue
        setting = f'{dataset_name}_H{horizon}_S{seed}_{split}'
        stem = f'{dataset_name}_H{horizon}_S{seed}_val'
        run = cli.runs / f'{stem}__tp_add'
        config = json.loads((run / 'config.json').read_text())
        authority = json.loads((run / 'result.json').read_text())
        checkpoint_hash = digest(config['checkpoint'])
        done = parts / f'{setting}.json'
        if done.exists():
            previous = json.loads(done.read_text())
            if previous['checkpoint_sha256'] != checkpoint_hash or previous['script_sha256'] != digest(__file__):
                raise ValueError('Completed analysis provenance changed')
            continue
        started = time.monotonic()
        model, args = load_model(run, device)
        baseline, _ = load_model(cli.runs / f'{stem}__b0', device)
        original_state = state_hash(model.state_dict())
        trace = Trace(model)
        maps = message_maps(model)
        data, loader = data_provider(args.dataset_args, split)
        frames, offset = [], 0
        checks = {'hook_identity_max_abs': 0., 'linear_reconstruction_max_abs': 0.,
                  'half_scale_forward_max_abs': 0., 'null_projection_forward_max_abs': 0.,
                  'rank_two_reconstruction_max_abs': 0.}
        with torch.inference_mode():
            for batch in loader:
                bp, truth = forecast(baseline, batch, device)
                trace.enabled = True
                cp, _ = forecast(model, batch, device)
                native = dict(trace.values)
                phase_native = model.last_phase_forecast.clone()
                trend_native = model.last_lowrank_forecast.clone()
                trace.enabled = False
                branch = model.lowrank_interaction_branch
                branch.intervention = 'off'
                try:
                    off, _ = forecast(model, batch, device)
                    phase_off = model.last_phase_forecast.clone()
                    torch.testing.assert_close(model.last_lowrank_forecast, trend_native, atol=0, rtol=0)
                finally:
                    branch.intervention = 'native'
                bcount, _, channels = cp.shape
                t, z, m = (native[k].double() for k in ('t', 'z', 'm'))
                gate = native['gate'].double()
                sigma = batch[0].float().to(device).var(1, unbiased=False).add(model.revin.eps).sqrt().double()
                q = torch.nn.functional.linear(m, model.predictor.decoder.weight.double())
                def to_output(profile):
                    return profile.repeat_interleave(model.period_len, -1).transpose(1, 2) * sigma[:, None] * (1-gate[..., 0])[:, None]
                direct = to_output(q)
                delta = cp.double() - off.double()
                error = float((direct-delta).abs().max())
                checks['linear_reconstruction_max_abs'] = max(checks['linear_reconstruction_max_abs'], error)
                if error > 5e-5 * (1 + float(delta.abs().max())):
                    raise AssertionError(f'Output is not explained by the linear level message: {error}')
                x = native['x'].transpose(1, 2).double()
                coeff = x @ maps['vh'][:2].T
                modes = [coeff[..., j, None] * maps['s'][j] * maps['u'][:, j] for j in range(2)]
                qbias = maps['bias'].expand_as(q)
                reconstructed = to_output(modes[0] + modes[1] + qbias)
                error = float((reconstructed-delta).abs().max())
                checks['rank_two_reconstruction_max_abs'] = max(checks['rank_two_reconstruction_max_abs'], error)
                if error > 5e-5 * (1 + float(delta.abs().max())):
                    raise AssertionError('Input-space rank decomposition failed')
                rank_one = off.double() + to_output(modes[0] + qbias)
                input_only = off.double() + to_output(modes[0] + modes[1])
                bias_only = off.double() + to_output(qbias)
                if offset == 0:
                    repeated, _ = forecast(model, batch, device)
                    checks['hook_identity_max_abs'] = float((repeated-cp).abs().max())
                    torch.testing.assert_close(repeated, cp, atol=0, rtol=0)
                    half = overridden_prediction(model, batch, device, native['m'] * .5)
                    projected = overridden_prediction(model, batch, device, (m @ maps['projector']).to(native['m']))
                    checks['half_scale_forward_max_abs'] = float((half.double()-(off.double()+.5*delta)).abs().max())
                    checks['null_projection_forward_max_abs'] = float((projected-cp).abs().max())
                    torch.testing.assert_close(half.double(), off.double()+.5*delta, atol=3e-5, rtol=3e-5)
                    torch.testing.assert_close(projected, cp, atol=3e-5, rtol=3e-5)
                predictions = {'baseline': bp, 'candidate': cp, 'off': off,
                               'half': off.double()+.5*delta, 'strong': off.double()+1.5*delta,
                               'rank_one': rank_one, 'input_only': input_only, 'bias_only': bias_only}
                arrays = {}
                for name, prediction in predictions.items():
                    err = prediction.double()-truth.double()
                    arrays[name+'_mse'] = err.square().mean(1)
                    arrays[name+'_mae'] = err.abs().mean(1)
                    if name in ('baseline', 'candidate', 'off'):
                        level, shape = component_errors(prediction, truth, model.period_len)
                        arrays[name+'_level_mse'], arrays[name+'_shape_mse'] = level, shape
                        for j, segment in enumerate(torch.tensor_split(err, 3, dim=1)):
                            arrays[f'{name}_segment{j}_mse'] = segment.square().mean(1)
                torch.set_float32_matmul_precision('medium')
                try:
                    medium_bp, _ = forecast(baseline, batch, device)
                    medium_cp, _ = forecast(model, batch, device)
                finally:
                    torch.set_float32_matmul_precision('highest')
                for name, prediction in [('medium_baseline', medium_bp), ('medium_candidate', medium_cp)]:
                    err = prediction.double()-truth.double()
                    arrays[name+'_mse'], arrays[name+'_mae'] = err.square().mean(1), err.abs().mean(1)
                receiving_norm = z.square().mean(-2).sum(-1).sqrt()
                message_sq = m.square().sum(-1)
                null_sq = (m-m @ maps['projector']).square().sum(-1)
                zmean = z.mean(-2)
                output = delta.transpose(1, 2)
                required = (truth.double()-off.double()).transpose(1, 2)
                denergy = output.square().mean(-1)
                dot = (output*required).mean(-1)
                level_delta = delta.reshape(bcount, -1, model.period_len, channels).mean(2)
                phase_level_off = phase_off.double().reshape(bcount, -1, model.period_len, channels).mean(2)
                trend_level = trend_native.double().reshape(bcount, -1, model.period_len, channels).mean(2)
                lowrank_level_write = gate[..., 0][:, None] * (trend_level-phase_level_off)
                cancellation = (level_delta.square().mean(1).sqrt()+lowrank_level_write.square().mean(1).sqrt()) / (level_delta+lowrank_level_write).square().mean(1).sqrt().clamp_min(1e-12)
                shaped = delta.reshape(bcount, -1, model.period_len, channels)
                shapefraction = (shaped-shaped.mean(2, keepdim=True)).square().mean((1, 2)) / denergy.clamp_min(1e-20)
                q0, q1 = modes
                compressed = torch.nn.functional.linear(t, branch.communication.source.weight.double())
                arrays.update(trend_norm=t.norm(dim=-1), compressed0=compressed[..., 0], compressed1=compressed[..., 1],
                    phase_norm=receiving_norm, phase_common_norm=zmean.norm(dim=-1),
                    phase_centered_norm=(z-zmean.unsqueeze(-2)).square().mean(-2).sum(-1).sqrt(),
                    message_norm=message_sq.sqrt(), relative_message_norm=message_sq.sqrt()/receiving_norm.clamp_min(1e-12),
                    message_phase_cosine=mean_cos(m, zmean), null_energy=null_sq, message_energy=message_sq,
                    null_energy_fraction=null_sq/message_sq.clamp_min(1e-20),
                    predictor_message_rms=q.square().mean(-1).sqrt(), output_message_mse=denergy,
                    output_message_rms=denergy.sqrt(), output_message_shape_fraction=shapefraction,
                    output_message_shape_energy=(shaped-shaped.mean(2, keepdim=True)).square().mean((1, 2)),
                    residual_alignment=mean_cos(output, required), message_residual_dot=dot,
                    alpha_oracle=dot/denergy.clamp_min(1e-20), native_minus_off_mse=arrays['candidate_mse']-arrays['off_mse'],
                    direct_gain_linear_term=2*dot, direct_cost_quadratic_term=denergy,
                    cycle_message_cosine=mean_cos(q, torch.ones_like(q)),
                    mode0_output_energy=q0.square().mean(-1), mode1_output_energy=q1.square().mean(-1),
                    bias_output_energy=qbias.square().mean(-1),
                    mode_cross_term=(q0*q1).mean(-1),
                    lowrank_write_cosine=mean_cos(level_delta.transpose(1, 2), lowrank_level_write.transpose(1, 2)),
                    cancellation_ratio=cancellation, gate_level=gate[..., 0], gate_shape=gate[..., 1],
                    history_recent_shift=(x[..., -model.period_len:].mean(-1)-x.mean(-1)),
                    phase_shape_change=arrays['candidate_shape_mse']-arrays['off_shape_mse'])
                frame = pd.DataFrame({k: v.cpu().numpy().reshape(-1) for k, v in arrays.items()})
                frame['sample_id'] = np.repeat(np.arange(offset, offset+bcount), channels)
                frame['channel'] = np.tile(np.arange(channels), bcount)
                frame['setting'], frame['split'] = setting, split
                frame['baseline_config_id'], frame['candidate_config_id'] = 'b0', 'tp_add'
                frame['delta_mse'] = frame.candidate_mse-frame.baseline_mse
                frame['delta_mae'] = frame.candidate_mae-frame.baseline_mae
                frame['time_range'] = [f'{data.timestamps[i+model.seq_len]}/{data.timestamps[i+model.seq_len+horizon-1]}' for i in frame.sample_id]
                np.testing.assert_allclose(frame.native_minus_off_mse, frame.direct_cost_quadratic_term-frame.direct_gain_linear_term, atol=2e-10, rtol=2e-10)
                frames.append(frame)
                offset += bcount
        frame = pd.concat(frames, ignore_index=True)
        if split == 'val':
            for name in ('mse', 'mae'):
                np.testing.assert_allclose(frame['medium_candidate_'+name].mean(), authority['val_'+name], atol=3e-6, rtol=3e-5)
        if state_hash(model.state_dict()) != original_state:
            raise AssertionError('Diagnostic changed model weights')
        # One program-ranked regression example per setting, or maximal baseline MAE.
        ranked = frame[frame.delta_mae>0].sort_values(['delta_mae', 'sample_id', 'channel'], ascending=[False, True, True], kind='stable')
        category = 'candidate_regression'
        if ranked.empty:
            ranked = frame.sort_values(['baseline_mae', 'sample_id', 'channel'], ascending=[False, True, True], kind='stable')
            category = 'baseline_high_error'
        row = ranked.iloc[0].to_dict()
        sid, ch = int(row['sample_id']), int(row['channel'])
        start = sid//loader.batch_size*loader.batch_size
        items = [data[i] for i in range(start, min(start+loader.batch_size, len(data)))]
        batch = [torch.as_tensor(np.stack([v[j] for v in items])) for j in range(4)]
        with torch.inference_mode():
            bp, truth = forecast(baseline, batch, device)
            trace.enabled = True
            cp, _ = forecast(model, batch, device)
            captured = dict(trace.values)
            bi = sid-start
            case = {'metadata': {**row, 'category': category, 'rank': 1},
                    'history': batch[0][bi, :, ch].numpy(), 'truth': truth[bi, :, ch].cpu().numpy(),
                    'baseline': bp[bi, :, ch].cpu().numpy(), 'candidate': cp[bi, :, ch].cpu().numpy(),
                    'phase': model.last_phase_forecast[bi, :, ch].cpu().numpy(),
                    'trend': model.last_lowrank_forecast[bi, :, ch].cpu().numpy(),
                    'gate': captured['gate'][bi, ch].cpu().numpy(), 'acceptance': np.array([], dtype=np.float32),
                    'trend_token': captured['t'][bi, ch].cpu().numpy(),
                    'phase_tokens': captured['z'][bi, ch].cpu().numpy(),
                    'message': captured['m'][bi, ch].cpu().numpy(),
                    'compressed_token': branch.communication.source(captured['t'])[bi, ch].cpu().numpy(),
                    'cycle_profile': model.predictor.decoder.weight.matmul(captured['m'][bi, ch]).cpu().numpy()}
            trace.enabled = False
            branch.intervention = 'off'
            try:
                case['off'] = forecast(model, batch, device)[0][bi, :, ch].cpu().numpy()
            finally:
                branch.intervention = 'native'
        np.testing.assert_allclose(np.mean((case['candidate']-case['truth'])**2), row['candidate_mse'], rtol=3e-5, atol=3e-6)
        frame.to_csv(parts/f'{setting}.csv', index=False)
        pack_cases([case], parts/f'{setting}.npz')
        records = []
        for name in predictions:
            record = {'setting': setting, 'config_id': 'tp_add' if name=='candidate' else 'b0' if name=='baseline' else 'diagnostic_'+name,
                      'dataset': dataset_name, 'horizon': horizon, 'seed': seed, 'split': split,
                      'baseline_config_id': 'b0', 'model': 'PhaseFormer-L TP-Add',
                      'mse': float(frame[name+'_mse'].mean()), 'mae': float(frame[name+'_mae'].mean()),
                      'delta_mse': float((frame[name+'_mse']-frame.baseline_mse).mean()),
                      'delta_mae': float((frame[name+'_mae']-frame.baseline_mae).mean()),
                      'selected': name=='candidate', 'key_params': 'Existing weights; fixed diagnostic; no training'}
            if name=='candidate':
                for col in frame.select_dtypes('number'):
                    if col not in ('sample_id', 'channel'):
                        record['mean_'+col] = float(frame[col].mean())
                record.update(message_relative_p95=float(frame.relative_message_norm.quantile(.95)),
                    null_energy_weighted_fraction=float(frame.null_energy.sum()/max(frame.message_energy.sum(), 1e-20)),
                    output_shape_energy_fraction=float(frame.output_message_shape_energy.sum()/max(frame.output_message_mse.sum(), 1e-20)),
                    direct_mse_win_fraction=float((frame.native_minus_off_mse<0).mean()),
                    native_mse_p95=float(frame.candidate_mse.quantile(.95)),
                    oracle_alpha_global=float(frame.message_residual_dot.sum()/max(frame.output_message_mse.sum(), 1e-20)))
            records.append(record)
        summary = {'setting': setting, 'records': records, 'checkpoint_sha256': checkpoint_hash,
                   'baseline_checkpoint_sha256': digest(json.loads((cli.runs/f'{stem}__b0/config.json').read_text())['checkpoint']),
                   'config_sha256': digest(run/'config.json'), 'script_sha256': digest(__file__),
                   'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'data_sha256': digest(Path(args.dataset_args.root_path)/args.dataset_args.data_path),
                   'test_or_val_windows': len(data), 'checks': checks,
                   'input_map_singular_values': maps['s'].cpu().tolist(),
                   'encoder_bias_profile': maps['bias'].cpu().tolist(),
                   'input_map': maps['map'].cpu().tolist(), 'input_mode0': maps['vh'][0].cpu().tolist(),
                   'output_mode0': maps['u'][:, 0].cpu().tolist(),
                   'mean_cycle_input_kernel': maps['map'].mean(0).cpu().tolist(),
                   'input_last_step_energy_fraction': float(maps['map'][:, -1].square().sum()/maps['map'].square().sum().clamp_min(1e-20)),
                   'input_recent24_energy_fraction': float(maps['map'][:, -24:].square().sum()/maps['map'].square().sum().clamp_min(1e-20)),
                   'gate_level_channel_range': float(native['gate'][0, :, 0].max()-native['gate'][0, :, 0].min()),
                   'environment': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'matmul': 'highest for attribution; original medium also evaluated'},
                   'elapsed_seconds': time.monotonic()-started, 'selection': case['metadata']}
        save_json(done, summary)
        trace.close()
        print(setting, 'COMPLETE', 'native', records[1]['mse'], 'off', records[2]['mse'], flush=True)
        del model, baseline, trace, frame, frames, maps
        torch.cuda.empty_cache()


def markdown(frame):
    return frame.to_markdown(index=False, floatfmt='.6f').splitlines()


def merge(cli):
    import numpy as np
    import pandas as pd
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parts = cli.scratch/'parts'
    metadata = [json.loads(p.read_text()) for p in sorted(parts.glob('*.json'))]
    expected = {f'{d}_H{h}_S{s}_{split}' for d in ('ETTh1', 'ETTm2', 'Weather') for h in (96, 720) for s in (2021, 2022) for split in ('val', 'test')}
    if {m['setting'] for m in metadata} != expected:
        raise ValueError('Incomplete activation matrix')
    cli.audit.mkdir(parents=True, exist_ok=True)
    figures = cli.audit/'figures'
    figures.mkdir(exist_ok=True)
    results = pd.DataFrame([r for m in metadata for r in m['records']])
    results.to_csv(cli.audit/'results.csv', index=False)
    cases, first = [], True
    grouped_stats = []
    channel_statistics = []
    for m in metadata:
        setting = m['setting']
        frame = pd.read_csv(parts/f'{setting}.csv')
        columns = ['baseline_mse','candidate_mse','off_mse','delta_mse','baseline_mae',
                   'candidate_mae','delta_mae','output_message_rms','relative_message_norm',
                   'residual_alignment','cancellation_ratio','lowrank_write_cosine']
        channel_frame = frame.groupby('channel')[columns].mean().reset_index()
        channel_frame['setting'] = setting
        channel_frame['n'] = frame.groupby('channel').size().values
        channel_statistics.extend(channel_frame.to_dict('records'))
        frame.to_csv(cli.audit/'sample_errors.csv', index=False, mode='w' if first else 'a', header=first, float_format='%.9g')
        first = False
        cases.extend(unpack_cases(parts/f'{setting}.npz'))
        for column in ('relative_message_norm', 'output_message_rms', 'residual_alignment'):
            bins = pd.qcut(frame[column].rank(method='first'), 5, labels=False)
            for q, group in frame.groupby(bins):
                grouped_stats.append({'setting': setting, 'split': frame['split'].iloc[0], 'group_variable': column, 'quintile': int(q)+1,
                    'n': len(group), 'native_minus_off_mse': float(group.native_minus_off_mse.mean()),
                    'native_minus_b0_mse': float(group.delta_mse.mean()), 'alpha_oracle_median': float(group.alpha_oracle.median())})
        for name in ('candidate', 'baseline', 'off'):
            for metric in ('mse', 'mae'):
                config_id = 'tp_add' if name=='candidate' else 'b0' if name=='baseline' else 'diagnostic_off'
                value = results.loc[(results.setting==setting)&(results.config_id==config_id), metric].iloc[0]
                np.testing.assert_allclose(frame[name+'_'+metric].mean(), value, atol=1e-10, rtol=1e-10)
        ranked = frame[frame.delta_mae>0].sort_values(['delta_mae','sample_id','channel'], ascending=[False, True, True], kind='stable')
        if not ranked.empty:
            assert (int(ranked.iloc[0].sample_id), int(ranked.iloc[0].channel)) == (int(m['selection']['sample_id']), int(m['selection']['channel']))
    pack_cases(cases, cli.audit/'selected_cases.npz')
    native = results[results.config_id=='tp_add'].copy()
    rows = []
    for split in ('val', 'test'):
        group = results[results.split==split]
        base = group[group.config_id=='b0'].set_index('setting')
        off = group[group.config_id=='diagnostic_off'].set_index('setting')
        for config_id, values in group.groupby('config_id', sort=False):
            values = values.set_index('setting').sort_index()
            rows.append({'split': split, 'config_id': config_id, 'mse': values.mse.mean(), 'mae': values.mae.mean(),
                'vs_b0_mse_pct': ((values.mse/base.mse-1)*100).mean(),
                'vs_off_mse_pct': ((values.mse/off.mse-1)*100).mean(),
                'mse_wins_vs_b0': int((values.mse<base.mse).sum())})
    report = ['# Experiment and Objective Error Analysis: TP-Add activations', '',
        '## 1. Experiment Setup', '',
        'Existing validation-best b0/TP-Add weights; 3 datasets × H96/720 × seeds2021/2022 × validation/test =24 settings. '
        'No training, optimizer, checkpoint or original model source changes. Focus on TP-Add was chosen after viewing prior test results; '
        'this is test-informed diagnostic exploration, not an independent blind test. Fixed interventions below do not select a new deployed configuration.', '',
        'Activation attribution uses highest float32 precision: the initial medium/TF32 precision check failed at0.00068–0.00088 maximum forward-versus-linear discrepancy. '
        'Original medium-precision b0/TP-Add metrics are also evaluated over every full split and retained in sample_errors.csv and native summary mean_medium_* columns. '
        'These original-precision validation metrics must reproduce registered training results; high-precision diagnostics are reported explicitly,not substituted silently.', '',
        '## 2. Experiment Results', '', *markdown(pd.DataFrame(rows)), '',
        'Raw MSE means give each setting equal weight; relative changes first divide within each setting and then average. '
        'Same-checkpoint off measures reliance, not the outcome of training without messages. Independent b0 provides the architecture comparison.', '',
        '## 3. Parameter / Configuration Search', '',
        'Predeclared diagnostic readings: native, off, half message,1.5× message,input-map rank-one with encoder bias retained,input-only,bias-only. '
        'All eight configurations including independent b0 are reported at every setting. Oracle alpha is label-based attribution, never inference input. '
        'No hyperparameter tuning or new mechanism training. Test-informed focus is disclosed; no configuration is chosen from these new diagnostic test values.', '',
        '## 4. Error Distribution', '',
        'Quintiles are calculated separately within each setting; activation norms alone are not evidence of reliability.', '',
        *markdown(pd.DataFrame(grouped_stats).groupby(['split','group_variable','quintile'], as_index=False)[['native_minus_off_mse','native_minus_b0_mse','alpha_oracle_median']].mean()), '',
        '## 5. Horizon-wise Error and Information Path', '',
        't=E(x−x_last), m=BAt, Z_slot′=Z_slot+m. All restored predictors are linear. '
        'q=Wm is identical at every phase offset of a cycle. Reassembly repeats q once per period; '
        'static-component fusion multiplies this level-only write by1−g_level; RevIN multiplies by the input-window sigma. '
        'Thus Δforecast=(1−g_level)·sigma·repeat_period(WBA t). Direct cycle-shape change must vanish up to numerical rounding. '
        'Changes in shape error versus independent b0 can arise through learned backbone coadaptation, not directly through this broadcast write.', '',
        *markdown(native[['setting','mean_output_message_rms','mean_output_message_shape_fraction','mean_phase_shape_change',
                          'mean_lowrank_write_cosine','mean_cancellation_ratio']]), '',
        '## 6. High-Error Selection', '',
        'One maximum positive candidate-minus-b0 MAE regression per setting, ordered by delta/sample/channel; '
        'fallback is maximum baseline MAE. All24 examples are saved for setting coverage; '
        'detailed plots are limited to the six largest regressions (three validation,three test). No manual case choice.', '',
        '## 7. Case Analysis', '']
    refs = []
    chosen = []
    for split in ('val', 'test'):
        chosen.extend(sorted((c for c in cases if c['metadata']['split']==split), key=lambda c: (-c['metadata']['delta_mae'], c['metadata']['setting']))[:3])
    for case in chosen:
        meta = case['metadata']
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
        hist, truth = case['history'], case['truth']
        axes[0].plot(np.arange(-len(hist), 0), hist, color='gray', alpha=.6, label='history')
        for key in ('truth', 'baseline', 'candidate', 'off'):
            axes[0].plot(case[key], label=key)
        axes[0].set_xlim(-min(96,len(hist)), len(truth)); axes[0].legend(ncol=5, fontsize=8)
        axes[0].set_title(f"{meta['setting']}, sample{int(meta['sample_id'])}, channel{int(meta['channel'])}; ΔMAE={meta['delta_mae']:.4f}")
        axes[1].plot(case['candidate']-case['off'], label='message write')
        axes[1].plot(truth-case['off'], label='required correction', alpha=.65)
        axes[1].axhline(0, color='gray', linewidth=.7); axes[1].legend(); axes[1].set_xlabel('Forecast step')
        common=case['phase_tokens'].mean(0)
        activation=np.stack([common, case['message'], common+case['message']])
        limit=max(float(np.abs(activation).max()),1e-9)
        axes[2].imshow(activation,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        axes[2].set_yticks([0,1,2],['phase mean','broadcast message','updated phase mean'])
        axes[2].set_xlabel('Latent coordinate'); axes[2].set_title(f"Compressed token: {case['compressed_token'].round(4)}")
        name=f"{meta['setting']}__sample{int(meta['sample_id'])}_c{int(meta['channel'])}.png"
        fig.savefig(figures/name, dpi=150); plt.close(fig)
        ref='figures/'+name; refs.append(ref)
        report.extend([f"### {meta['setting']} sample{int(meta['sample_id'])}, channel{int(meta['channel'])}", '',
            f"Baseline/candidate MAE={meta['baseline_mae']:.6f}/{meta['candidate_mae']:.6f}; "
            f"message RMS={meta['output_message_rms']:.6f}; oracle α={meta['alpha_oracle']:.6f}; "
            f"message/residual cosine={meta['residual_alignment']:.6f}. These are measured quantities,not a causal training explanation.", '', f'![{name}]({ref})', ''])
    kernel_rows=[]
    for m in metadata:
        if m['setting'].endswith('_val'):
            s=np.asarray(m['input_map_singular_values'])
            kernel_rows.append({'setting':m['setting'], 'map_rank1_energy_fraction':float(s[0]**2/max(np.sum(s**2),1e-20)),
                'last_step_input_energy_fraction':m['input_last_step_energy_fraction'],
                'recent24_input_energy_fraction':m['input_recent24_energy_fraction'],
                'gate_level_channel_range':m['gate_level_channel_range']})
    report.extend(['## 8. Repeated Observable Patterns', '',
        *markdown(native[['setting','mean_relative_message_norm','message_relative_p95','null_energy_weighted_fraction',
                          'mean_mode0_output_energy','mean_mode1_output_energy','mean_bias_output_energy',
                          'direct_mse_win_fraction','oracle_alpha_global']]), '',
        *markdown(pd.DataFrame(kernel_rows)), '',
        'Singular modes use the complete normalized-input→cycle-message map WBAE(I−1e_lastᵀ), not the arbitrary rank-latent basis. '
        'Rank-one intervention retains the original encoder-bias write. Null-space removal changes receiver activations but must preserve predictions; '
        'validated on the first original-size batch of every setting. Norm/cosine statistics stay descriptive.', '',
        '## 9. Objective Defect Summary', '',
        'Separate under/over-sized writes (fixed scale interventions),redundant output directions (rank-one), '
        'constant encoder-bias writes (input-only/bias-only),and redundant level contributions (write cosine/cancellation). '
        'Optimization suggestions must follow cross-seed validation evidence,not a single extreme sample. '
        'The output map has rank at most2 and never reads the receiving phase state; TP-Add here is a low-rank level write rather than state-dependent interaction. '
        'It could be moved to the forecast-level path algebraically; absorbing it into the original shared trend decoder additionally requires channel-independent gate ratios. '
        'No numerical fold or new architecture is claimed without checking that condition.', '',
        '## 10. Experiment Scope', '',
        '24 existing-checkpoint settings; all192 comparison/diagnostic records retained. Full sample/channel activation/error statistics in sample_errors.csv (nine significant CSV digits; aggregate records computed before rounding); '
        '24 program-selected examples in selected_cases.npz. Six detailed case plots. No new training or default-architecture decision. '
        'Previous validation reuse and test exposure remain; post-hoc oracle quantities are not deployable. Results are diagnostic evidence,not final architecture superiority.', ''])
    (cli.audit/'objective_error_analysis.md').write_text('\n'.join(report))
    with zipfile.ZipFile(cli.audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for ref in ['objective_error_analysis.md', *refs]:
            z.write(cli.audit/ref, ref)
    settings = [{k: next(r for r in m['records'] if r['config_id']=='tp_add')[k] for k in ('setting','dataset','horizon','seed','split')} for m in metadata]
    run = {'experiment_id': cli.audit.name,
           'code': {'repository':'PhaseFormer', 'commit':cli.analysis_commit, 'model_source_commit':metadata[0]['source_commit'],
                    'modified_files':['scripts/lowrank_token_interaction/activation_audit.py']},
           'mechanism': {'description':'Read-only TP-Add activation decomposition and fixed-weight diagnostics','feature_flag':'tp_add'},
           'experiment': {'baseline':'b0','candidate':'tp_add','settings':settings,'training':'none; existing from-scratch checkpoints',
                          'metrics':['MSE','MAE','activations','level/shape','same-checkpoint counterfactuals']},
           'execution': {'environment':metadata[0]['environment'],'settings':[
               {'setting':m['setting'],'commands':['activation_audit.py --repo <frozen-checkout> --runs <existing-runs> --scratch <independent-scratch> --audit <canonical-audit> --shards 4 --shard 0|1|2|3'],
                'runtime':m['elapsed_seconds'],'checkpoint_sha256':m['checkpoint_sha256'],'baseline_checkpoint_sha256':m['baseline_checkpoint_sha256'],
                'config_sha256':m['config_sha256'],'data_sha256':m['data_sha256'],'script_sha256':m['script_sha256'], 'checks':m['checks'],
                'input_map_singular_values':m['input_map_singular_values'],'gate_level_channel_range':m['gate_level_channel_range'],
                'input_mode0':m['input_mode0'],'output_mode0':m['output_mode0'],
                'mean_cycle_input_kernel':m['mean_cycle_input_kernel'],
                'input_last_step_energy_fraction':m['input_last_step_energy_fraction'],
                'input_recent24_energy_fraction':m['input_recent24_energy_fraction']} for m in metadata]},
           'selection': {'source':'test','selected_configs':[
               {'setting':s['setting'],'config_id':'tp_add','search_notes':'Focus chosen after prior nine-method test comparison; no new configuration selected from activation interventions.'} for s in settings]},
           'analysis': {'ranking_metric':'mae','top_k':1,'dedup_rule':'One program-ranked case per setting; six detailed plots total',
                        'channel_statistics':channel_statistics,
                        'worker_commit':'2f0a14d','initial_diagnostic_failures':'Four medium/TF32 attribution checks failed at max discrepancies0.0006807–0.0008816; no successful outputs overwritten. Repeated with highest precision and retained original medium metrics.',
                        'selections':[m['selection'] for m in metadata], 'quintile_statistics':grouped_stats},
           'validation': {'results_checked':True,'ranking_and_cases_checked':True,'report_and_archive_checked':False,'directory_and_settings_checked':False,'status':'incomplete'}}
    (cli.audit/'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    verify(cli)
    print(pd.DataFrame(rows).to_string(index=False))


def verify(cli):
    import numpy as np
    import pandas as pd
    import yaml
    required={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz','objective_error_analysis.md','objective_error_analysis.zip','figures'}
    assert {p.name for p in cli.audit.iterdir()}==required
    run = yaml.safe_load((cli.audit/'run.yaml').read_text())
    expected={s['setting'] for s in run['experiment']['settings']}
    results=pd.read_csv(cli.audit/'results.csv')
    assert set(results.setting)==expected and len(results)==192
    assert set(s['setting'] for s in run['execution']['settings'])==expected
    assert set(s['setting'] for s in run['selection']['selected_configs'])==expected
    sample_settings=set()
    for frame in pd.read_csv(cli.audit/'sample_errors.csv', chunksize=100000):
        sample_settings.update(frame.setting)
    assert sample_settings==expected
    cases=unpack_cases(cli.audit/'selected_cases.npz')
    assert {c['metadata']['setting'] for c in cases}==expected
    for c in cases:
        for label in ('baseline','candidate','off'):
            for metric in ('mse','mae'):
                err=c[label].astype(np.float64)-c['truth'].astype(np.float64)
                value=np.mean(err**2 if metric=='mse' else np.abs(err))
                np.testing.assert_allclose(value,c['metadata'][label+'_'+metric],atol=3e-6,rtol=3e-5)
    report=(cli.audit/'objective_error_analysis.md').read_text()
    refs=re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)',report)
    assert {p.name for p in (cli.audit/'figures').iterdir()}=={Path(r).name for r in refs}
    with zipfile.ZipFile(cli.audit/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        for ref in z.namelist():
            assert z.read(ref)==(cli.audit/ref).read_bytes()
    run['validation']={k:True for k in ('results_checked','ranking_and_cases_checked','report_and_archive_checked','directory_and_settings_checked')}
    run['validation']['status']='passed'
    (cli.audit/'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    print('AUDIT PASSED',len(expected),'settings',len(cases),'cases',len(refs),'figures',flush=True)


def self_check():
    import torch
    torch.manual_seed(81)
    x=torch.randn(5,3,20,dtype=torch.double)
    e=torch.randn(4,20,dtype=torch.double)
    a=torch.randn(2,4,dtype=torch.double)
    b=torch.randn(7,2,dtype=torch.double)
    w=torch.randn(3,7,dtype=torch.double)
    z=torch.randn(5,3,6,7,dtype=torch.double)
    t=(x-x[...,-1:])@e.T
    m=t@a.T@b.T
    observed=(z+m.unsqueeze(-2))@w.T-z@w.T
    q=m@w.T
    torch.testing.assert_close(observed,q.unsqueeze(-2).expand_as(observed),atol=1e-12,rtol=1e-12)
    f=w@b@a@e
    f[:,-1]-=f.sum(-1)
    torch.testing.assert_close(x@f.T,q,atol=1e-12,rtol=1e-12)
    _,s,vh=torch.linalg.svd(w,full_matrices=False)
    projection=vh.T@vh
    torch.testing.assert_close(m@projection@w.T,q,atol=1e-12,rtol=1e-12)
    err=torch.randn_like(q)
    torch.testing.assert_close((err+q).square().mean(-1)-err.square().mean(-1),
                              2*(err*q).mean(-1)+q.square().mean(-1),atol=1e-12,rtol=1e-12)
    print('SELF CHECK PASSED: broadcast, input map, predictor null space, MSE attribution')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--runs',type=Path)
    p.add_argument('--scratch',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--shards',type=int,default=4)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--analysis-commit',default='uncommitted')
    p.add_argument('--merge',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--self-check',action='store_true')
    cli=p.parse_args()
    os.chdir(cli.repo)
    sys.path.insert(0,str(cli.repo))
    if cli.self_check:
        self_check()
    elif cli.verify:
        verify(cli)
    elif cli.merge:
        merge(cli)
    else:
        if not 0<=cli.shard<cli.shards:
            p.error('Invalid shard')
        worker(cli)


if __name__=='__main__':
    main()
