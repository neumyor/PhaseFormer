#!/usr/bin/env python3
"""Validation-only paired predictions, activations and portable stage-2 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_lowrank_interaction_stage2 import ARMS, construct, forecast, save_json
from src.dataset.data_factory import data_provider


def load_model(run, device):
    cfg = json.loads((run / 'config.json').read_text())
    model, args = construct(cfg)
    state = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), args


def pick_cases(frame, horizon, top_k=5):
    selections = []
    for category, column, ascending in (
        ('baseline_high_error', 'baseline_mae', False),
        ('candidate_regression', 'delta_mae', False),
        ('candidate_improvement', 'delta_mae', True),
    ):
        ranked = frame.sort_values([column, 'sample_id', 'channel'],
                                   ascending=[ascending, True, True], kind='stable')
        if category == 'candidate_regression':
            ranked = ranked[ranked.delta_mae > 0]
        elif category == 'candidate_improvement':
            ranked = ranked[ranked.delta_mae < 0]
        chosen = []
        for row in ranked.to_dict('records'):
            if any(row['channel'] == r['channel'] and abs(row['sample_id'] - r['sample_id']) < horizon for r in chosen):
                continue
            chosen.append(row)
            selections.append({**row, 'category': category, 'rank': len(chosen)})
            if len(chosen) == top_k:
                break
    return selections


def worker(scratch, setting):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda:0')
    baseline, args = load_model(scratch / 'runs' / f'{setting}__i0_continue', device)
    dataset, loader = data_provider(args.dataset_args, 'val')
    parts = scratch / 'analysis_parts'
    parts.mkdir(exist_ok=True)
    frames, all_cases = [], []
    horizon, lookback = baseline.pred_len, baseline.seq_len
    for arm in list(ARMS)[1:]:
        model, _ = load_model(scratch / 'runs' / f'{setting}__{arm}', device)
        branch = model.lowrank_interaction_branch
        branch.capture_diagnostics = True
        rows = []
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                bp, truth = forecast(baseline, batch, device)
                cp, _ = forecast(model, batch, device)
                diagnostics = branch.last_diagnostics
                original_mode = branch.interaction_mode
                try:
                    branch.interaction_mode = 'none'
                    disabled, _ = forecast(model, batch, device)
                finally:
                    branch.interaction_mode = original_mode
                arrays = {}
                for name, pred in [('baseline', bp), ('candidate', cp), ('disabled', disabled)]:
                    error = pred - truth
                    arrays[name + '_mae'] = error.abs().mean(1).cpu().numpy()
                    arrays[name + '_mse'] = error.square().mean(1).cpu().numpy()
                    for segment, e in enumerate(torch.tensor_split(error, 3, dim=1)):
                        arrays[f'{name}_segment{segment}_mse'] = e.square().mean(1).cpu().numpy()
                arrays['prediction_change_rms'] = (cp - disabled).square().mean(1).sqrt().cpu().numpy()
                for key in ('message_norm', 'update_norm', 'original_norm'):
                    if key in diagnostics:
                        arrays[key] = diagnostics[key].cpu().numpy()
                if diagnostics:
                    a = diagnostics['acceptance']
                    arrays['acceptance_mean'] = a.mean(-1).cpu().numpy()
                    arrays['acceptance_phase_std'] = a.std(-1, unbiased=False).cpu().numpy()
                bsz, channels = arrays['candidate_mae'].shape
                frame = pd.DataFrame({key: value.reshape(-1) for key, value in arrays.items()})
                frame['sample_id'] = np.repeat(np.arange(offset, offset + bsz), channels)
                frame['channel'] = np.tile(np.arange(channels), bsz)
                frame['setting'] = setting
                frame['baseline_config_id'] = 'i0_continue'
                frame['candidate_config_id'] = arm
                frame['delta_mae'] = frame.candidate_mae - frame.baseline_mae
                frame['delta_mse'] = frame.candidate_mse - frame.baseline_mse
                frame['enabled_minus_disabled_mse'] = frame.candidate_mse - frame.disabled_mse
                ids = frame.sample_id.to_numpy()
                frame['time_range'] = [f'{dataset.timestamps[i + lookback]}/{dataset.timestamps[i + lookback + horizon - 1]}' for i in ids]
                rows.append(frame)
                offset += bsz
        frame = pd.concat(rows, ignore_index=True)
        selected = pick_cases(frame, horizon)
        # Second pass only over selected windows; full predictions never persist.
        cache = {}
        with torch.inference_mode():
            for row in selected:
                key = (int(row['sample_id']), int(row['channel']))
                if key not in cache:
                    sample, channel = key
                    batch = [torch.as_tensor(v).unsqueeze(0) for v in dataset[sample]]
                    bp, truth = forecast(baseline, batch, device)
                    cp, _ = forecast(model, batch, device)
                    diagnostics = branch.last_diagnostics
                    cache[key] = {
                        'history': batch[0][0, :, channel].numpy(),
                        'truth': truth[0, :, channel].cpu().numpy(),
                        'baseline': bp[0, :, channel].cpu().numpy(),
                        'candidate': cp[0, :, channel].cpu().numpy(),
                        'acceptance': diagnostics['acceptance'][0, channel].cpu().numpy()
                            if diagnostics else np.array([], dtype=np.float32),
                    }
                all_cases.append({'metadata': row, **cache[key]})
        frames.append(frame)
        del model
        torch.cuda.empty_cache()
        print(f'{setting} {arm}: {len(frame)} rows, {len(selected)} cases', flush=True)
    pd.concat(frames, ignore_index=True).to_csv(parts / f'{setting}.csv', index=False)
    packed = {'metadata': np.array([json.dumps(c['metadata']) for c in all_cases])}
    for key in ('history', 'truth', 'baseline', 'candidate', 'acceptance'):
        length = max(len(c[key]) for c in all_cases)
        packed[key] = np.full((len(all_cases), length), np.nan, dtype=np.float32)
        for i, c in enumerate(all_cases):
            packed[key][i, :len(c[key])] = c[key]
    np.savez_compressed(parts / f'{setting}.npz', **packed)


def markdown_table(frame):
    # Avoid an optional tabulate dependency in the experiment environment.
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return lines


def merge(scratch, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = json.loads((scratch / 'manifest.json').read_text())
    run = yaml.safe_load((audit / 'run.yaml').read_text())
    sources = {s['setting']: s for s in manifest['sources']}
    settings = sorted(sources)
    frames, cases = [], []
    for setting in settings:
        frame = pd.read_csv(scratch / 'analysis_parts' / f'{setting}.csv')
        frames.append(frame)
        with np.load(scratch / 'analysis_parts' / f'{setting}.npz', allow_pickle=False) as packed:
            for i, metadata in enumerate(packed['metadata']):
                cases.append({'metadata': json.loads(str(metadata)),
                              **{k: packed[k][i][np.isfinite(packed[k][i])] for k in ('history', 'truth', 'baseline', 'candidate', 'acceptance')}})
    errors = pd.concat(frames, ignore_index=True)
    errors.to_csv(audit / 'sample_errors.csv', index=False)
    results = []
    for setting in settings:
        source = sources[setting]
        base_result = json.loads((scratch / 'runs' / f'{setting}__i0_continue' / 'result.json').read_text())
        for arm in ARMS:
            result = json.loads((scratch / 'runs' / f'{setting}__{arm}' / 'result.json').read_text())
            cfg = json.loads((scratch / 'runs' / f'{setting}__{arm}' / 'config.json').read_text())
            init = json.loads((scratch / 'runs' / f'{setting}__{arm}' / 'initialization.json').read_text())
            assert not cfg['evaluate_test'] and not result['smoke']
            results.append({'setting': setting, 'config_id': arm, 'dataset': cfg['dataset'],
                'horizon': cfg['horizon'], 'seed': cfg['seed'], 'model': 'PhaseFormer-L',
                'key_params': json.dumps(ARMS[arm], sort_keys=True),
                'mse': result['val_mse'], 'mae': result['val_mae'],
                'delta_mse': result['val_mse'] - base_result['val_mse'],
                'delta_mae': result['val_mae'] - base_result['val_mae'], 'selected': True,
                'elapsed_sec': result['elapsed_sec'], 'parameter_count': result['parameter_count'],
                'interaction_scale': result['interaction_scale'], 'null_score': result['null_score'],
                'initial_max_abs': result['initial_max_abs'],
                'source_checkpoint_sha256': source['source_checkpoint_sha256']})
            if init['identity_required']:
                assert init['max_abs_vs_i0'] == 0
            if arm != 'i0_continue':
                subset = errors[(errors.setting == setting) & (errors.candidate_config_id == arm)]
                for metric in ('mae', 'mse'):
                    np.testing.assert_allclose(subset['candidate_' + metric].mean(), result['val_' + metric], rtol=2e-5, atol=2e-6)
                    np.testing.assert_allclose(subset['baseline_' + metric].mean(), base_result['val_' + metric], rtol=2e-5, atol=2e-6)
                expected = pick_cases(subset, int(cfg['horizon']))
                expected_keys = {(r['category'], r['rank'], r['sample_id'], r['channel']) for r in expected}
                actual = [c for c in cases if c['metadata']['setting'] == setting and c['metadata']['candidate_config_id'] == arm]
                actual_keys = {(c['metadata']['category'], c['metadata']['rank'], c['metadata']['sample_id'], c['metadata']['channel']) for c in actual}
                assert expected_keys == actual_keys
    results = pd.DataFrame(results)
    results.to_csv(audit / 'results.csv', index=False)
    # Pad numeric arrays, with explicit lengths, to avoid pickle-based NPZ data.
    arrays = {'metadata': np.array([json.dumps(c['metadata']) for c in cases]),
              'setting': np.array([c['metadata']['setting'] for c in cases]),
              'sample_id': np.array([c['metadata']['sample_id'] for c in cases]),
              'channel': np.array([c['metadata']['channel'] for c in cases])}
    for key in ('history', 'truth', 'baseline', 'candidate', 'acceptance'):
        arrays[key + '_length'] = np.array([len(c[key]) for c in cases])
        arrays[key] = np.full((len(cases), max(arrays[key + '_length'])), np.nan, dtype=np.float32)
        for i, case in enumerate(cases):
            arrays[key][i, :len(case[key])] = case[key]
    np.savez_compressed(audit / 'selected_cases.npz', **arrays)
    for case in cases:
        for name in ('baseline', 'candidate'):
            error = case[name] - case['truth']
            # Different inference batch sizes can change TF32 rounding slightly.
            np.testing.assert_allclose(np.abs(error).mean(), case['metadata'][name + '_mae'], rtol=3e-3, atol=3e-4)
            np.testing.assert_allclose(np.square(error).mean(), case['metadata'][name + '_mse'], rtol=6e-3, atol=6e-4)
    figures = audit / 'figures'
    figures.mkdir(exist_ok=True)
    refs = []
    for setting in settings:
        for arm in ('pt_scaled', 'tp_scaled', 'pt_null', 'tp_selective'):
            pair = []
            for category in ('candidate_regression', 'candidate_improvement'):
                available = [c for c in cases if c['metadata']['setting'] == setting and c['metadata']['candidate_config_id'] == arm and c['metadata']['category'] == category]
                if available:
                    pair.append(min(available, key=lambda c: c['metadata']['rank']))
            if not pair:
                continue
            fig, axes = plt.subplots(len(pair), 1, figsize=(11, 3.1 * len(pair)), squeeze=False)
            for ax, case in zip(axes[:, 0], pair):
                meta = case['metadata']
                hist = case['history'][-192:]
                ax.plot(np.arange(-len(hist), 0), hist, color='0.55', label='History')
                for key, color in [('truth', 'black'), ('baseline', '#176b9c'), ('candidate', '#b43b4c')]:
                    ax.plot(case[key], label=key, color=color, linewidth=1.2)
                ax.axvline(0, color='0.6', linewidth=0.7)
                ax.set_title(f"{meta['category']}: sample {meta['sample_id']}, channel {meta['channel']}; delta MAE {meta['delta_mae']:+.4f}", fontsize=10)
                ax.set_ylabel('Train-scaled value')
                ax.legend(ncol=4, fontsize=8)
            axes[-1, 0].set_xlabel('Steps from forecast start')
            fig.suptitle(f'{setting}: {arm}', fontsize=12)
            fig.tight_layout()
            name = f'{setting}__{arm}.png'
            fig.savefig(figures / name, dpi=130)
            plt.close(fig)
            refs.append(f'figures/{name}')
    avg = results.groupby('config_id', sort=False)[['delta_mae', 'delta_mse']].mean().reset_index()
    incremental = []
    pivot = results.pivot(index='setting', columns='config_id', values='mse')
    for candidate, control in [('pt_pre', 'pt_post'), ('tp_pre', 'tp_post'), ('pt_scaled', 'pt_pre'), ('tp_scaled', 'tp_pre'), ('pt_null', 'pt_scaled'), ('tp_selective', 'tp_scaled')]:
        diff = pivot[candidate] - pivot[control]
        incremental.append({'comparison': f'{candidate} - {control}', 'mean_delta_mse': diff.mean(), 'wins': f'{(diff < 0).sum()}/{len(diff)}'})
    stats = []
    for (setting, arm), group in errors.groupby(['setting', 'candidate_config_id']):
        record = {'setting': setting, 'arm': arm, 'mae_win_fraction': (group.delta_mae < 0).mean(),
                  'delta_mae_p95': group.delta_mae.quantile(.95),
                  'enabled_minus_disabled_mse': group.enabled_minus_disabled_mse.mean()}
        for key in ('message_norm', 'update_norm', 'acceptance_mean', 'acceptance_phase_std'):
            record[key] = group[key].mean() if key in group else float('nan')
        stats.append(record)
    segments = errors.groupby('candidate_config_id')[[f'candidate_segment{i}_mse' for i in range(3)]].mean().reset_index()
    report = ['# Experiment and Objective Error Analysis', '', '## 1. Experiment Setup', '',
        'Stage 2: matched four-epoch fine-tuning from paired stage-1 low-rank i0 checkpoints. '
        'All arms retain L=720, rank=H/8, the legacy decoder and static fusion. '
        'Validation only; no test evaluation. The nine fixed arms share each source config and optimizer schedule. '
        'These exploratory results do not replace Golden/test results.', '',
        '## 2. Experiment Results', '', 'Setting-equal means relative to i0_continue:', '',
        *markdown_table(avg), '', 'Controlled incremental comparisons:', '',
        *markdown_table(pd.DataFrame(incremental)), '', 'All setting results:', '',
        *markdown_table(results[['setting', 'config_id', 'mae', 'mse', 'delta_mae', 'delta_mse']]), '',
        '## 3. Parameter / Configuration Search', '',
        'No hyperparameter sweep. All nine predeclared arms are retained; selected=true means included in the comparison, not chosen as a winner. '
        'Checkpoint selection uses validation Huber loss. Source config, checkpoint SHA256, commands and runtime are in run.yaml. '
        'Stage-1 i0 selection also used this validation set; this is not an unbiased generalization estimate.', '',
        '## 4. Error Distribution', '', *markdown_table(pd.DataFrame(stats)[['setting', 'arm', 'mae_win_fraction', 'delta_mae_p95']]), '',
        '## 5. Horizon-wise Error', '', 'Validation sample/channel-weighted MSE over three equal forecast segments:', '',
        *markdown_table(segments), '', '## 6. High-Error Selection', '',
        'Top five per setting/arm/category by MAE; per channel, selected starts must differ by at least H. '
        'Regression requires positive delta and improvement negative delta. All cases, timestamps and ranks are in selected_cases.npz and run.yaml.', '',
        '## 7. Case Analysis', '',
        'Each figure shows the top deduplicated regression and improvement, when present. '
        'Curves are in the train-fitted standardized space used by evaluation; predictions are re-evaluated on selected windows. '
        'Their metrics are checked against full-loader metrics with a small tolerance for batch-dependent TF32 rounding.', '']
    report.extend(f'![{Path(ref).stem}]({ref})' for ref in refs)
    report += ['', '## 8. Repeated Observable Patterns', '',
        'Raw message and effective update norms are per-token norms averaged over receiving slots. '
        'Acceptance is real-key mass: 1-null_mass for P->T, or per-phase trend receipt for T->P. '
        'NaN identifies legacy arms without these diagnostics. Zero phase-gate spread indicates broadcast-like receipt.', '',
        *markdown_table(pd.DataFrame(stats)[['setting', 'arm', 'message_norm', 'update_norm', 'acceptance_mean', 'acceptance_phase_std', 'enabled_minus_disabled_mse']]), '',
        '## 9. Objective Defect Summary', '',
        'Assess proposal 1 against its normalization and extra-training controls; assess null read against pt_scaled and selective write against tp_scaled. '
        'A candidate that does not improve these controls has no demonstrated incremental benefit in this pilot. '
        'Enabled-minus-disabled errors hold trained weights fixed but disabling interactions can be out of distribution; '
        'the legacy post-norm controls bypass both the message and its post-LayerNorm in this diagnostic. '
        'they measure reliance and do not establish the outcome of training without interactions. '
        'Attention concentration, null mass and latent norms alone do not establish reliability or causation.', '',
        '## 10. Experiment Scope', '',
        'Three dataset/horizon pairs, two seeds each, all validation. No automatic expansion or test-set selection. '
        'The base latent remains a single rank-r token. New interactions make the overall input-output map nonlinear; '
        'retaining a rank-r bottleneck does not imply that the whole forecast map remains a linear rank-r operator.', '']
    report_path = audit / 'objective_error_analysis.md'
    report_path.write_text('\n'.join(report))
    actual_refs = re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)', report_path.read_text())
    assert actual_refs == refs
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['objective_error_analysis.md', *refs]:
            assert not (audit / name).is_symlink()
            z.write(audit / name, name)
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip') as z:
        assert set(z.namelist()) == {'objective_error_analysis.md', *refs}
        for name in z.namelist():
            assert z.read(name) == (audit / name).read_bytes()
    assert set(errors.setting) == set(results.setting) == set(arrays['setting']) == set(settings)
    assert set(p.name for p in audit.iterdir()) == {'run.yaml', 'results.csv', 'sample_errors.csv', 'selected_cases.npz', 'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}
    assert set(p.name for p in figures.iterdir()) == {Path(r).name for r in refs}
    run['analysis']['selections'] = [c['metadata'] for c in cases]
    for entry in run['execution']['settings']:
        setting = entry['setting']
        entry['runtime'] = [json.loads((scratch / 'runs' / f'{setting}__{arm}' / 'result.json').read_text()) for arm in ARMS]
        entry['source_checkpoint'] = sources[setting]['config']['checkpoint']
        entry['source_checkpoint_sha256'] = sources[setting]['source_checkpoint_sha256']
    run['validation'] = {'results_checked': True, 'ranking_and_cases_checked': True,
                         'report_and_archive_checked': True, 'directory_and_settings_checked': True, 'status': 'passed'}
    (audit / 'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    save_json(scratch / 'audit_complete.json', {'status': 'passed', 'rows': len(errors), 'cases': len(cases), 'figures': len(refs)})
    print(json.dumps({'status': 'passed', 'rows': len(errors), 'cases': len(cases), 'audit': str(audit)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--worker-setting')
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if args.worker_setting:
        worker(args.scratch_dir, args.worker_setting)
        return
    manifest = json.loads((args.scratch_dir / 'manifest.json').read_text())
    settings = [s['setting'] for s in manifest['sources']]
    if not args.merge_only:
        gpus = args.gpus.split(',')
        def bucket(gpu, assigned):
            for setting in assigned:
                with (args.scratch_dir / f'analysis_{setting}.log').open('a') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--scratch-dir', str(args.scratch_dir),
                        '--audit-dir', str(args.audit_dir), '--worker-setting', setting],
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(bucket, gpu, settings[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
    merge(args.scratch_dir, args.audit_dir)


if __name__ == '__main__':
    main()
