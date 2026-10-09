#!/usr/bin/env python3
"""Validation-only sample audit and sequential GPU latency measurement."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.phase_context_interaction.model import ARMS, IsolatedBranch, load_model
from scripts.phase_context_interaction.run import commit, provenance, run_child, STOP, request_stop, stop_children
from scripts.run_lowrank_interaction_stage2 import forecast, save_json, digest
from scripts.analyze_lowrank_stage2 import pick_cases, markdown_table
from scripts.analyze_lowrank_six import (
    component_errors, oracle_component_gate, replay_fusion, pack, unpack)
from src.dataset.data_factory import data_provider


def off_prediction(model, batch, device):
    branch = model.lowrank_interaction_branch
    if not isinstance(branch, IsolatedBranch):
        return forecast(model, batch, device)[0]
    previous = branch.intervention
    try:
        branch.intervention = 'off'
        return forecast(model, batch, device)[0]
    finally:
        branch.intervention = previous


def read_manifest(scratch):
    manifest = json.loads((scratch / 'manifest.json').read_text())
    if manifest['code_commit'] != commit() or manifest['source_hashes'] != provenance():
        raise ValueError('Analysis source differs from frozen manifest')
    if manifest['evaluation_split'] != 'val' or manifest['arms'] != list(ARMS):
        raise ValueError('Unregistered analysis scope')
    return manifest


def release_diagnostic_caches(model):
    """Module.to does not move ordinary cached output tensors."""
    for module in model.modules():
        for name in list(vars(module)):
            if name.startswith('last_'):
                value = getattr(module,name)
                setattr(module,name,{} if isinstance(value,dict) else None)


def benchmark(scratch, gpu):
    """Serial randomized timings; one resident model; batch1 and native batch."""
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    device = torch.device(f'cuda:{gpu}')
    manifest = read_manifest(scratch)
    records = []
    for source in manifest['sources']:
        setting = source['setting']
        models = {}
        for arm in ARMS:
            models[arm], args = load_model(scratch / 'runs' / f'{setting}__{arm}', 'cpu')
        dataset, _ = data_provider(args.dataset_args, 'val')
        sizes = sorted({1, min(source['config']['batch_size'], len(dataset))})
        summaries = {arm: {'setting': setting, 'config_id': arm} for arm in ARMS}
        for size in sizes:
            items = [dataset[i] for i in range(size)]
            batch = [torch.as_tensor(np.stack([x[j] for x in items])).float().to(device) for j in range(4)]
            timings = {arm: [] for arm in ARMS}
            memories = {arm: [] for arm in ARMS}
            with torch.inference_mode():
                base_model = models['b0'].to(device)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                    forecast(base_model, batch, device)
                    torch.cuda.synchronize(device)
                lower_flops = sum(e.flops for e in prof.key_averages())
                release_diagnostic_caches(base_model)
                base_model.to('cpu')
                torch.cuda.empty_cache()
            with torch.inference_mode():
                for sweep in range(5):
                    order = list(ARMS)
                    random.Random(8100+sweep).shuffle(order)
                    for arm in order:
                        model = models[arm].to(device)
                        model.lowrank_interaction_branch.capture_diagnostics = False
                        for _ in range(10):
                            forecast(model, batch, device)
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        for _ in range(30):
                            start.record()
                            forecast(model, batch, device)
                            end.record()
                            end.synchronize()
                            timings[arm].append(start.elapsed_time(end))
                        memories[arm].append(torch.cuda.max_memory_allocated(device))
                        release_diagnostic_caches(model)
                        model.to('cpu')
                        torch.cuda.empty_cache()
            baseline = float(np.median(timings['b0']))
            base95 = float(np.quantile(timings['b0'], .95))
            base_mem = max(memories['b0'])
            for arm in ARMS:
                tag = 'batch1' if size == 1 else 'native_batch'
                summary = summaries[arm]
                summary.update({f'{tag}_size': size,
                    f'{tag}_latency_median_ms': float(np.median(timings[arm])),
                    f'{tag}_latency_p95_ms': float(np.quantile(timings[arm], .95)),
                    f'{tag}_latency_vs_b0': float(np.median(timings[arm])/baseline),
                    f'{tag}_p95_vs_b0': float(np.quantile(timings[arm], .95)/base95),
                    f'{tag}_latency_std_ms': float(np.std(timings[arm])),
                    f'{tag}_peak_bytes': max(memories[arm]),
                    f'{tag}_peak_vs_b0': max(memories[arm])/base_mem})
                if size == 1:
                    model = models[arm]
                    d, r, slots = model.latent_dim, model.lowrank_interaction_branch.rank, model.period_len
                    k = min(2,r,d)
                    channels = batch[0].shape[-1]
                    read = (slots-1)*d + d + (2*d-1)*k
                    write = (2*k-1)*d + slots*d
                    cost = {'b0':0, 'tp_add':(2*r-1)*k+write,
                            'context_full':(2*r-1)*k+read+k+write,
                            'phase_only':read+d+write}[arm]*channels
                    if lower_flops is None or lower_flops <= 0:
                        raise ValueError('No positive whole-forward profiler lower bound')
                    summary.update(communication_arithmetic_ops=cost,
                        total_forward_counted_flops_lower_bound=lower_flops,
                        added_compute_ratio_upper_bound=cost/lower_flops,
                        compute_count_note='Profiler counted mm/bmm/mul FLOPs are a lower bound; explicit new-path scalar arithmetic includes projections/reduction/division/subtraction/broadcast. Ratio upper-bounds added arithmetic versus unchanged backbone.')
            del batch, items
        for arm in ARMS:
            summary = summaries[arm]
            summary.update(latency_median_ms=summary['native_batch_latency_median_ms'],
                latency_vs_b0=summary['native_batch_latency_vs_b0'],
                benchmark_gpu=torch.cuda.get_device_name(device),
                benchmark_protocol='medium float32; batch1/native; one resident model; 5 randomized sweeps; warmup10 +150 single-forward CUDA-event timings')
            records.append(summary)
        del models, model
        torch.cuda.empty_cache()
    save_json(scratch / 'benchmark.json', records)


def worker(scratch, setting):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda:0')
    manifest = read_manifest(scratch)
    source = next(s for s in manifest['sources'] if s['setting'] == setting)
    parts = scratch / 'analysis_parts'
    parts.mkdir(exist_ok=True)
    summaries, cases = [], []
    first = True
    for arm in ARMS:
        run_dir = scratch / 'runs' / f'{setting}__{arm}'
        baseline, _ = load_model(scratch / 'runs' / f'{setting}__b0', device)
        model, args = load_model(run_dir, device)
        dataset, loader = data_provider(args.dataset_args, 'val')
        branch = model.lowrank_interaction_branch
        branch.capture_diagnostics = True
        inputs = {}
        def capture(module, values, output):
            inputs.update(phase=values[0].detach(), trend=values[1].detach())
        hook = model.lowrank_dynamic_fusion.register_forward_hook(capture)
        frames, offset = [], 0
        with torch.inference_mode():
            for batch in loader:
                bp, truth = forecast(baseline, batch, device)
                cp, _ = forecast(model, batch, device)
                phase = model.last_phase_forecast.clone()
                trend = model.last_lowrank_forecast.clone()
                native = dict(inputs)
                diag = dict(branch.last_diagnostics) if isinstance(branch, IsolatedBranch) else {}
                gate = model.lowrank_dynamic_fusion.last_gate.clone()
                torch.testing.assert_close(gate, gate[:1].expand_as(gate), atol=0, rtol=0)
                torch.testing.assert_close(cp, replay_fusion(model, batch, device, gate, native), atol=0, rtol=0)
                off = off_prediction(model, batch, device)
                torch.testing.assert_close(model.lowrank_dynamic_fusion.last_gate, gate, atol=0, rtol=0)
                arrays = {}
                for label, prediction in [('baseline', bp), ('candidate', cp), ('off', off), ('phase', phase), ('trend', trend)]:
                    error = prediction - truth
                    arrays[label + '_mae'] = error.abs().mean(1).cpu().numpy()
                    arrays[label + '_mse'] = error.square().mean(1).cpu().numpy()
                    level, shape = component_errors(prediction, truth, model.period_len)
                    arrays[label + '_level_mse'] = level.cpu().numpy()
                    arrays[label + '_shape_mse'] = shape.cpu().numpy()
                    if label in ('baseline', 'candidate'):
                        for j, segment in enumerate(torch.tensor_split(error, 3, dim=1)):
                            arrays[f'{label}_segment{j}_mse'] = segment.square().mean(1).cpu().numpy()
                        arrays[label + '_peak_error'] = (prediction.amax(1) - truth.amax(1)).abs().cpu().numpy()
                        arrays[label + '_trough_error'] = (prediction.amin(1) - truth.amin(1)).abs().cpu().numpy()
                _, stats = model.revin.normalize(batch[0].float().to(device))
                normalized_truth = model.revin.normalize_with_stats(truth, stats)
                norm_off = model.revin.normalize_with_stats(off, stats).double()
                norm_cp = model.revin.normalize_with_stats(cp, stats).double()
                direct = norm_cp - norm_off
                residual = normalized_truth.double() - norm_off
                regret = direct.square().mean(1) - 2*(direct*residual).mean(1)
                arrays['direct_error_increase'] = regret.cpu().numpy()
                arrays['direct_harm_positive'] = regret.clamp_min(0).cpu().numpy()
                arrays['direct_harm_indicator'] = (regret > 1e-10).double().cpu().numpy()
                dc = direct.reshape(len(direct),-1,model.period_len,direct.shape[-1]).mean(2)
                tc = native['trend'].double().reshape_as(direct).reshape(len(direct),-1,model.period_len,direct.shape[-1]).mean(2)
                phase_off = model.revin.normalize_with_stats(model.last_phase_forecast, stats).double()
                pc = phase_off.reshape(len(direct),-1,model.period_len,direct.shape[-1]).mean(2)
                # Trend's level correction relative to the phase-only off forecast.
                tc = (tc-pc)*gate[...,0].double().unsqueeze(1)
                rms = lambda q: q.square().mean(1).sqrt()
                valid = (rms(dc)>.01)&(rms(tc)>.01)
                ratio = (rms(dc)+rms(tc))/(rms(dc+tc).clamp_min(1e-6))
                arrays['cancellation_valid'] = valid.double().cpu().numpy()
                arrays['severe_cancellation'] = (valid&(ratio>5)&((dc*tc).mean(1)<0)).double().cpu().numpy()
                arrays['cancellation_ratio'] = ratio.cpu().numpy()

                oracle_gate = oracle_component_gate(native['phase'], native['trend'], normalized_truth, model.period_len)
                oracle = replay_fusion(model, batch, device, oracle_gate, native)
                arrays['oracle_mse'] = (oracle - truth).square().mean(1).cpu().numpy()
                arrays['gate_regret_mse'] = arrays['candidate_mse'] - arrays['oracle_mse']
                arrays['gate_level'] = gate[..., 0].cpu().numpy()
                arrays['gate_shape'] = gate[..., 1].cpu().numpy()
                for key in ('update_norm', 'original_norm', 'relative_update_norm'):
                    arrays[key] = diag[key].cpu().numpy() if diag else np.zeros_like(arrays['candidate_mse'])
                b, c = arrays['candidate_mse'].shape
                frame = pd.DataFrame({k: v.reshape(-1) for k, v in arrays.items()})
                frame['sample_id'] = np.repeat(np.arange(offset, offset + b), c)
                frame['channel'] = np.tile(np.arange(c), b)
                frame['setting'] = setting
                frame['baseline_config_id'] = 'b0'
                frame['candidate_config_id'] = arm
                frame['split'] = 'val'
                frame['delta_mae'] = frame.candidate_mae - frame.baseline_mae
                frame['delta_mse'] = frame.candidate_mse - frame.baseline_mse
                frame['time_range'] = [f'{dataset.timestamps[i + model.seq_len]}/{dataset.timestamps[i + model.seq_len + model.pred_len - 1]}' for i in frame.sample_id]
                frames.append(frame)
                offset += b
        frame = pd.concat(frames, ignore_index=True)
        result = json.loads((run_dir / 'result.json').read_text())
        if result['code_commit'] != manifest['code_commit'] or result['pretrained_checkpoint_loaded']:
            raise ValueError('Invalid training provenance')
        for metric in ('mse', 'mae'):
            np.testing.assert_allclose(frame['candidate_' + metric].mean(), result['val_' + metric], atol=3e-6, rtol=3e-5)
        record = {**result, 'config_id': arm, 'baseline_config_id': 'b0',
            'dataset': source['config']['dataset'], 'horizon': model.pred_len, 'seed': source['config']['seed'],
            'model': 'PhaseFormer-L', 'key_params': json.dumps(json.loads((run_dir / 'config.json').read_text()), sort_keys=True),
            'mse': float(frame.candidate_mse.mean()), 'mae': float(frame.candidate_mae.mean()),
            'delta_mse': float(frame.delta_mse.mean()), 'delta_mae': float(frame.delta_mae.mean()),
            'selected': arm == 'b0', 'sample_mae_win_fraction': float((frame.delta_mae < 0).mean())}
        for col in frame.select_dtypes('number'):
            if col not in ('sample_id', 'channel'):
                record['mean_' + col] = float(frame[col].mean())
        for label in ('baseline', 'candidate'):
            mae_values = frame[label + '_mae']
            record[label + '_mae_p95'] = float(mae_values.quantile(.95))
            record[label + '_mae_top1_mean'] = float(mae_values.nlargest(max(1,int(np.ceil(len(mae_values)*.01)))).mean())
            values = frame[label + '_mse']
            record[label + '_mse_p95'] = float(values.quantile(.95))
            record[label + '_mse_top1_mean'] = float(values.nlargest(max(1, int(np.ceil(len(values) * .01)))).mean())
        summaries.append(record)
        chosen = pick_cases(frame, model.pred_len, top_k=3) if arm != 'b0' else []
        cache = {}
        with torch.inference_mode():
            for row in chosen:
                sid, ch = int(row['sample_id']), int(row['channel'])
                if (sid, ch) not in cache:
                    start = sid // loader.batch_size * loader.batch_size
                    items = [dataset[i] for i in range(start, min(start + loader.batch_size, len(dataset)))]
                    batch = [torch.as_tensor(np.stack([v[j] for v in items])) for j in range(4)]
                    bi = sid - start
                    bp, truth = forecast(baseline, batch, device)
                    cp, _ = forecast(model, batch, device)
                    data = {'history': batch[0][bi, :, ch].numpy(), 'truth': truth[bi, :, ch].cpu().numpy(),
                        'baseline': bp[bi, :, ch].cpu().numpy(), 'candidate': cp[bi, :, ch].cpu().numpy(),
                        'phase': model.last_phase_forecast[bi, :, ch].cpu().numpy(),
                        'trend': model.last_lowrank_forecast[bi, :, ch].cpu().numpy(),
                        'gate': model.lowrank_dynamic_fusion.last_gate[bi, ch].cpu().numpy(),
                        'acceptance': np.array([], dtype=np.float32)}
                    data['off'] = off_prediction(model, batch, device)[bi, :, ch].cpu().numpy()
                    cache[sid, ch] = data
                cases.append({'metadata': row, **cache[sid, ch]})
        frame.reindex(sorted(frame.columns), axis=1).to_csv(parts / f'{setting}.csv', index=False,
            mode='w' if first else 'a', header=first)
        first = False
        print(setting, arm, len(frame), 'rows', len(chosen), 'cases', flush=True)
        hook.remove()
        del baseline, model, frame, frames
        torch.cuda.empty_cache()
    pack(cases, parts / f'{setting}.npz')
    save_json(parts / f'{setting}.json', summaries)


def paired(table, candidate, baseline):
    a = table[table.config_id == candidate].set_index('setting').sort_index()
    b = table[table.config_id == baseline].set_index('setting').sort_index()
    assert a.index.equals(b.index)
    delta = (a.mse / b.mse - 1) * 100
    tasks = pd.DataFrame({'dataset': a.dataset, 'horizon': a.horizon,
                          'delta': a.mse - b.mse}).groupby(['dataset', 'horizon']).delta.mean()
    return {'candidate': candidate, 'baseline': baseline,
        'relative_mse_percent': float(delta.mean()), 'median_relative_mse_percent': float(delta.median()),
        'relative_mae_percent': float(((a.mae / b.mae - 1) * 100).mean()),
        'mse_wins': int((delta < 0).sum()), 'settings': len(a), 'task_wins': int((tasks < 0).sum())}


def pilot_decision(table):
    """Apply fixed practical gates; never auto-select a per-task winner."""
    checks = {}
    details = {}
    full = table[table.config_id == 'context_full'].set_index('setting').sort_index()
    for baseline in ('b0', 'tp_add'):
        base = table[table.config_id == baseline].set_index('setting').sort_index()
        delta = (full.mse/base.mse-1)*100
        mae = (full.mae/base.mae-1)*100
        checks[baseline+'_effect'] = bool(delta.mean() <= -1 and (delta<0).sum()>=3
                                          and mae.mean()<=0 and delta.max()<=2)
        checks[baseline+'_tail'] = bool(all(
            ((full[column]/base[column]-1)*100).max() <= 2
            for column in ('candidate_mae_p95','candidate_mae_top1_mean')))
        details[baseline] = {'relative_mse_mean_percent':float(delta.mean()),
            'relative_mae_mean_percent':float(mae.mean()), 'mse_wins':int((delta<0).sum()),
            'max_setting_mse_regression_percent':float(delta.max()),
            'raw_mse_mean_delta':float(full.mse.mean()-base.mse.mean()),
            'raw_mae_mean_delta':float(full.mae.mean()-base.mae.mean())}
    base=table[table.config_id=='b0'].set_index('setting').sort_index()
    checks['parameters'] = bool((full.parameter_count/base.parameter_count<=1.01).all())
    checks['compute'] = bool((full.added_compute_ratio_upper_bound<=.05).all())
    for tag in ('batch1','native_batch'):
        checks[tag+'_latency'] = bool((full[tag+'_latency_vs_b0']<=1.05).all()
                                      and (full[tag+'_p95_vs_b0']<=1.1).all())
        checks[tag+'_memory'] = bool((full[tag+'_peak_vs_b0']<=1.05).all())
    checks['epoch_training_cost'] = bool((full.epoch_train_median_seconds/base.epoch_train_median_seconds<=1.1).all())
    return {'advance_to_controls':bool(all(checks.values())), 'checks':checks, 'paired_details':details,
            'scope':'Pilot only; no test read; no significance claim; H1-H3 need subsequent controls.',
            'timing_note':'Threshold crossings need serial same-GPU remeasurement before interpreting boundary failures.'}


def merge(scratch, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = read_manifest(scratch)
    run = yaml.safe_load((audit / 'run.yaml').read_text())
    if run['code']['commit'] != manifest['code_commit']:
        raise ValueError('Canonical audit metadata differs from manifest')
    settings = {s['setting'] for s in manifest['sources']}
    results, cases, selections, patterns = [], [], [], []
    count, first = 0, True
    for setting in sorted(settings):
        frame = pd.read_csv(scratch / 'analysis_parts' / f'{setting}.csv')
        records = json.loads((scratch / 'analysis_parts' / f'{setting}.json').read_text())
        if {r['config_id'] for r in records} != set(ARMS) or len(records) != len(ARMS):
            raise ValueError('Missing arms')
        assert len({r['common_random_state_sha256'] for r in records}) == 1
        assert len({r['dataset_sha256'] for r in records}) == 1
        startup = [json.loads((scratch/'runs'/f'{setting}__{arm}'/'gradient_startup.json').read_text()) for arm in ARMS]
        fingerprints = [[(r['step'],r['batch_x_sha256']) for r in rows] for rows in startup]
        if not all(h==fingerprints[0] for h in fingerprints):
            raise ValueError('Paired training data orders differ')
        current = unpack(scratch / 'analysis_parts' / f'{setting}.npz')
        for arm in ARMS:
            group = frame[frame.candidate_config_id == arm]
            record = next(r for r in records if r['config_id'] == arm)
            for metric in ('mse', 'mae'):
                np.testing.assert_allclose(group['candidate_' + metric].mean(), record[metric], atol=1e-7)
            if arm == 'b0':
                continue
            expected = pick_cases(group, record['horizon'], top_k=3)
            actual = [c['metadata'] for c in current if c['metadata']['candidate_config_id'] == arm]
            key = lambda r: (r['category'], r['rank'], r['sample_id'], r['channel'])
            assert {key(r) for r in expected} == {key(r) for r in actual}
            selections.extend({k: r[k] for k in ('setting', 'candidate_config_id', 'category', 'rank', 'sample_id', 'channel')} for r in expected)
        count += len(frame)
        frame.to_csv(audit / 'sample_errors.csv', index=False, mode='w' if first else 'a', header=first)
        first = False
        results.extend(records)
        cases.extend(current)
    table = pd.DataFrame(results)
    latency = pd.DataFrame(json.loads((scratch / 'benchmark.json').read_text()))
    assert len(latency) == len(table)
    table = table.merge(latency, on=['setting', 'config_id'], validate='one_to_one')
    for entry in run['execution']['settings']:
        group = table[table.setting == entry['setting']]
        entry['runtime'] = {'summed_training_seconds': float(group.elapsed_sec.sum()),
            'completed_runs': len(group), 'environments': group.environment.tolist()}
    comparisons = [paired(table, arm, 'b0') for arm in ARMS[1:]]
    comparisons += [paired(table, 'context_full', 'tp_add'), paired(table, 'context_full', 'phase_only')]
    decision = pilot_decision(table)
    if manifest['smoke']:
        decision['advance_to_controls'] = False
        decision['scope'] = 'SMOKE: pipeline check only; rubric values are not efficacy evidence.'
    table.to_csv(audit / 'results.csv', index=False)
    pack(cases, audit / 'selected_cases.npz')
    for case in cases:
        for key in ('baseline', 'candidate', 'off'):
            error = case[key] - case['truth']
            np.testing.assert_allclose(np.square(error).mean(), case['metadata'][key + '_mse'], rtol=5e-4, atol=5e-5)
            np.testing.assert_allclose(np.abs(error).mean(), case['metadata'][key + '_mae'], rtol=3e-4, atol=3e-5)
    figures = audit / 'figures'
    figures.mkdir(exist_ok=True)
    refs = []
    # Coverage first, then globally ranked main-candidate examples; <=10 plotted cases.
    main_cases = [c for c in cases if c['metadata']['candidate_config_id']=='context_full']
    selected = []
    for setting in sorted(settings):
        group = [c for c in main_cases if c['metadata']['setting']==setting]
        selected.append(max(group, key=lambda c:(c['metadata']['delta_mae'], -c['metadata']['sample_id'])))
    identity = lambda c:(c['metadata']['setting'],c['metadata']['sample_id'],c['metadata']['channel'])
    for case in sorted(main_cases,key=lambda c:(-abs(c['metadata']['delta_mae']),identity(c))):
        if len(selected)>=10:break
        if identity(case) not in {identity(c) for c in selected}:selected.append(case)
    for case in selected:
        m=case['metadata']
        fig,ax=plt.subplots(figsize=(10,3))
        history=case['history'][-192:]
        ax.plot(np.arange(-len(history),0),history,color='.65',label='history')
        for key,color in [('truth','black'),('baseline','#2166ac'),('candidate','#b2182b'),('off','#188e66')]:
            ax.plot(case[key],color=color,label=key,lw=1)
        ax.legend(fontsize=8)
        ax.set_title(f"{m['setting']} sample{m['sample_id']} c{m['channel']}; delta MAE {m['delta_mae']:+.5f}")
        fig.tight_layout()
        rel=f"figures/{m['setting']}__sample{m['sample_id']}_c{m['channel']}.png"
        fig.savefig(audit/rel,dpi=120);plt.close(fig)
        refs.append(rel)
    identities = set()
    for case in cases:
        m = case['metadata']
        identity = (m['setting'], m['candidate_config_id'], m['sample_id'], m['channel'])
        if identity in identities:
            continue
        identities.add(identity)
        patterns.append({'config_id': m['candidate_config_id'], 'unique_cases': 1,
            'level_dominant': int(m['candidate_level_mse'] > m['candidate_shape_mse']),
            'std_above_truth': int(case['candidate'].std() > case['truth'].std()),
            'absolute_mean_bias_above_baseline': int(abs((case['candidate']-case['truth']).mean()) > abs((case['baseline']-case['truth']).mean()))})
    columns = ['setting', 'config_id', 'mse', 'mae', 'parameter_count', 'communication_parameters',
               'elapsed_sec', 'latency_median_ms', 'latency_vs_b0', 'peak_memory_bytes']
    lines = ['# Experiment and Objective Error Analysis', '', '## 1. Experiment Setup', '',
        'PhaseFormer-L non-attention communication; static learned per-variable level/shape output gates in every arm. Full model trains from scratch. Original rank, E/D, RevIN and phase backbone retained.',
        f"{len(settings)} validation settings, {len(table)} checkpoints; epoch ceiling {manifest['epochs']}, patience8, minimum validation Huber checkpoint. Communication width min(2,r,d). Zero-initialized message exit, no new norm/bias/activation. No test read.",
        '> SMOKE ONLY: four training batches; not efficacy evidence.' if manifest['smoke'] else 'Validation has prior exposure. No Golden-superiority, significance or fresh blind-generalization claim.', '',
        '## 2. Experiment Results', '']
    lines += markdown_table(pd.DataFrame(comparisons)) + [''] + markdown_table(table[columns])
    lines += ['', '## 3. Parameter / Configuration Search', '',
        'Four fixed arms: b0; TP-Add BAt; context_full B(At-C mean(Z)); phase_only -BC mean(Z). No per-arm tuning. All source paths retain gradients.',
        'The minus sign is absorbable into C. Only the joint rank-two shared write is constrained; no identifiable deduplication, Kalman, orthogonality or contraction claim. Direct writes affect cycle level only. Research direction was informed by prior test inspection; no new test read.',
        'Zero decoder and zero message exit cause staggered gradient startup. Per-worker first-eight-step gradient records verify eventual message learning; they do not imply equal learning speed.',
        'Latency uses one GPU with randomized serial arm sweeps after all training. Batch1 and the registered training batch size; full forward, warmup and CUDA synchronization; diagnostic capture disabled. Ratios are environment-specific measurements.', '',
        '## 4. Error Distribution', '']
    lines += markdown_table(table[['setting', 'config_id', 'sample_mae_win_fraction', 'candidate_mse_p95', 'candidate_mse_top1_mean']])
    lines += ['', '## 5. Horizon-wise Error', '']
    lines += markdown_table(table[['setting', 'config_id'] + [f'mean_candidate_segment{i}_mse' for i in range(3)]])
    lines += ['', '## 6. High-Error Selection', '',
        'Top3 baseline MAE, positive MAE regression and negative MAE improvement, within-category/channel start separation >=H. At most9 per candidate/setting. All rankings verified from the complete sample table.', '',
        '## 7. Case Analysis', '']
    for rel in refs:
        lines += [f'![{Path(rel).stem}]({rel})', '']
    lines += ['## 8. Repeated Observable Patterns', '',
        'Counts below deduplicate sample/channel identities across selection categories; selected cases do not estimate population prevalence.', '']
    lines += markdown_table(pd.DataFrame(patterns).groupby('config_id').sum().reset_index())
    lines += ['', '## 9. Objective Defect Summary', '',
        'off disables token interaction in the same trained checkpoint and preserves its static output gates; it is not a retrained baseline. Oracle reads future truth only during post-training diagnostics and is not deployable.', '']
    lines += markdown_table(table[['setting', 'config_id', 'mean_candidate_level_mse', 'mean_candidate_shape_mse',
        'mean_phase_mse', 'mean_trend_mse', 'mean_off_mse', 'mean_oracle_mse', 'mean_relative_update_norm']])
    lines += ['', '## 10. Experiment Scope', '',
        f'{count} sample/channel/arm rows, {len(cases)} selected case records. All configurations retained; decisions require numerical review against preregistered gates. No dataset-wise stitching of winners.',
        'Conditional context controls, bidirectional combinations, communication-width search and independent-seed/cross-task confirmation are not automatically triggered by this runner.', '']
    lines += ['## 11. Preregistered Pilot Rubric', '',
              'Research focus was informed by earlier test inspection. These new results use validation only.',
              'All gates must pass; timing boundary failures require isolated confirmation. No summed score overrides an effect/efficiency failure.',
              '```json', json.dumps(decision, indent=2), '```', '']
    report = '\n'.join(lines)
    (audit / 'objective_error_analysis.md').write_text(report)
    listed = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    assert set(listed) == set(refs)
    assert {p.name for p in figures.iterdir()} == {Path(r).name for r in refs}
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for rel in ['objective_error_analysis.md', *refs]:
            archive.write(audit / rel, rel)
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip') as archive:
        assert set(archive.namelist()) == {'objective_error_analysis.md', *refs}
        for rel in archive.namelist():
            assert archive.read(rel) == (audit / rel).read_bytes()
    assert {p.name for p in audit.iterdir()} == {'run.yaml', 'results.csv', 'sample_errors.csv',
        'selected_cases.npz', 'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}
    assert set(table.setting) == settings == {c['metadata']['setting'] for c in cases}
    run['analysis'].update(selections=selections, paired_comparisons=comparisons)
    run['analysis']['pilot_rubric'] = decision
    run['analysis']['fixed_cancellation_thresholds'] = {
        'normalized_write_rms_min':.01, 'denominator_epsilon':1e-6,
        'severe_ratio_min':5, 'requires_opposite_direction':True,
        'reference':'RevIN normalized units, frozen before smoke/formal training'}
    channel_summaries = []
    for setting in sorted(settings):
        frame=pd.read_csv(scratch/'analysis_parts'/f'{setting}.csv')
        for (arm,channel),group in frame.groupby(['candidate_config_id','channel']):
            channel_summaries.append({'setting':setting,'config_id':arm,'channel':int(channel),
                'baseline_mse':float(group.baseline_mse.mean()),'candidate_mse':float(group.candidate_mse.mean()),
                'baseline_mae':float(group.baseline_mae.mean()),'candidate_mae':float(group.candidate_mae.mean())})
    run['analysis']['channel_summaries'] = channel_summaries
    run['validation'] = {'status': 'passed', 'results_checked': True, 'ranking_and_cases_checked': True,
        'paired_initialization_dataset_and_startup_order_checked': True,
        'report_and_archive_checked': True, 'directory_and_settings_checked': True,
        'sample_rows': count, 'case_records': len(cases)}
    (audit / 'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    save_json(scratch / 'audit_complete.json', {**run['validation'],
        'artifact_hashes': {str(p.relative_to(audit)): digest(p)
                           for p in audit.rglob('*') if p.is_file()}})
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--setting')
    args = parser.parse_args()
    scratch, audit = args.scratch_dir.resolve(), args.audit_dir.resolve()
    if args.setting:
        with (scratch / f'analysis_{args.setting}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(scratch, args.setting)
        return
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (scratch / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_manifest(scratch)
        complete = scratch / 'audit_complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            actual = {str(p.relative_to(audit)): digest(p) for p in audit.rglob('*') if p.is_file()}
            if saved.get('status') != 'passed' or saved.get('artifact_hashes') != actual:
                raise ValueError('Completed audit artifacts changed')
            return
        benchmark(scratch, int(args.gpus.split(',')[0]))
        settings = [s['setting'] for s in manifest['sources']]
        gpus = args.gpus.split(',')
        def bucket(gpu, batch):
            for setting in batch:
                if STOP.is_set():
                    return
                command = [sys.executable, str(Path(__file__).resolve()), '--scratch-dir', str(scratch),
                           '--audit-dir', str(audit), '--setting', setting]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                with (scratch / f'analysis_{setting}.log').open('a') as log:
                    code = run_child(command, env, log)
                if code:
                    stop_children()
                    raise RuntimeError(f'Analysis worker failed: {setting}')
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(bucket, gpu, settings[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
        if STOP.is_set():
            raise RuntimeError('Analysis stopped before merge')
        merge(scratch, audit)


if __name__ == '__main__':
    main()
