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

from scripts.lowrank_token_interaction.model import ARMS, IsolatedBranch, load_model
from scripts.lowrank_token_interaction.run import commit, provenance, run_child, STOP, request_stop, stop_children
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


def benchmark(scratch, gpu):
    """One process/GPU, after all training; never overlap arm timings."""
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    device = torch.device(f'cuda:{gpu}')
    manifest = read_manifest(scratch)
    records = []
    for source in manifest['sources']:
        setting = source['setting']
        models = {}
        for arm in ARMS:
            models[arm], args = load_model(scratch / 'runs' / f'{setting}__{arm}', device)
        args.dataset_args.num_workers = 0
        dataset, _ = data_provider(args.dataset_args, 'val')
        size = min(32, len(dataset))
        items = [dataset[i] for i in range(size)]
        batch = [torch.as_tensor(np.stack([x[j] for x in items])).float().to(device) for j in range(4)]
        timings = {arm: [] for arm in ARMS}
        memories = {}
        with torch.inference_mode():
            for arm, model in models.items():
                model.lowrank_interaction_branch.capture_diagnostics = False
                for _ in range(10):
                    forecast(model, batch, device)
                torch.cuda.synchronize(device)
            for sweep in range(5):
                order = list(ARMS)
                random.Random(8100 + sweep).shuffle(order)
                for arm in order:
                    model = models[arm]
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    before = torch.cuda.memory_allocated(device)
                    start = time.perf_counter()
                    for _ in range(30):
                        forecast(model, batch, device)
                    torch.cuda.synchronize(device)
                    timings[arm].append((time.perf_counter() - start) / 30 * 1000)
                    memories[arm] = max(memories.get(arm, 0), torch.cuda.max_memory_allocated(device) - before)
        baseline = float(np.median(timings['b0']))
        for arm in ARMS:
            records.append({'setting': setting, 'config_id': arm, 'benchmark_batch_size': size,
                'latency_median_ms': float(np.median(timings[arm])),
                'latency_min_ms': min(timings[arm]), 'latency_max_ms': max(timings[arm]),
                'latency_vs_b0': float(np.median(timings[arm]) / baseline),
                'incremental_peak_memory_bytes': memories[arm],
                'benchmark_gpu': torch.cuda.get_device_name(device),
                'benchmark_protocol': 'float32/medium; warmup10;5 randomized serial sweeps x30; CUDA synchronized; diagnostics off'})
        del models, model, batch, items
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
            'selected': False, 'sample_mae_win_fraction': float((frame.delta_mae < 0).mean())}
        for col in frame.select_dtypes('number'):
            if col not in ('sample_id', 'channel'):
                record['mean_' + col] = float(frame[col].mean())
        for label in ('baseline', 'candidate'):
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
    comparisons = [paired(table, arm, 'b0') for arm in ARMS[1:]]
    comparisons += [paired(table, 'pt_correct', 'pt_add'), paired(table, 'tp_mod', 'tp_add')]
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
    for setting in sorted(settings):
        for arm in ARMS[1:]:
            selected = [c for c in cases if c['metadata']['setting'] == setting
                        and c['metadata']['candidate_config_id'] == arm and c['metadata']['rank'] == 1]
            if not selected:
                raise ValueError(f'No selected audit cases: {setting}/{arm}')
            fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3 * len(selected)), squeeze=False)
            for i, case in enumerate(selected):
                m = case['metadata']
                history = case['history'][-192:]
                axes[i, 0].plot(np.arange(-len(history), 0), history, color='.65', label='history')
                for key, color in [('truth', 'black'), ('baseline', '#2166ac'), ('candidate', '#b2182b'), ('off', '#188e66')]:
                    axes[i, 0].plot(case[key], color=color, label=key, lw=1)
                axes[i, 0].legend(fontsize=7, ncol=3)
                axes[i, 0].set_title(f"{m['category']} {m['sample_id']}/c{m['channel']}; deltaMAE {m['delta_mae']:+.4f}", fontsize=9)
                for key, color in [('truth', 'black'), ('phase', '#2166ac'), ('trend', '#ef8a62')]:
                    axes[i, 1].plot(case[key], color=color, label=key, lw=1)
                axes[i, 1].legend(fontsize=7)
                axes[i, 1].set_title(f"static output gates {case['gate'][0]:.3f}, {case['gate'][1]:.3f}", fontsize=9)
            fig.suptitle(f'{setting} / {arm}', fontsize=11)
            fig.tight_layout()
            rel = f'figures/{setting}__{arm}.png'
            fig.savefig(audit / rel, dpi=120)
            plt.close(fig)
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
        'Five fixed arms, no per-arm tuning. PT-add: t+B V mean(Z); PT-correct: t+B(V mean(Z)-Ct); TP-add: Z+BAt; TP-mod: Z+Z*(BAt). Source and message paths retain gradients.',
        'PT-correct self term can be absorbed into the trainable encoder; this comparison concerns parameterization/optimization and phase information, not a uniquely identifiable error-correction or Kalman mechanism. General B,V,C have no orthogonal/non-expansion guarantee.',
        'Zero decoder and zero message exit cause staggered gradient startup. Per-worker first-eight-step gradient records verify eventual message learning; they do not imply equal learning speed.',
        'Latency uses one GPU with randomized serial arm sweeps after all training. Same batch32, full forward, warmup and CUDA synchronization; diagnostic capture disabled. Ratios are environment-specific measurements.', '',
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
    run['validation'] = {'status': 'passed', 'results_checked': True, 'ranking_and_cases_checked': True,
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
