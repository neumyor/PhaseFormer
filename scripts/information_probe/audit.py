"""Freeze calibration decisions, then merge and verify the portable probe audit."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scripts.information_probe.core import CROSS, CONTROLS, FAMILIES, rubric, summarize
from scripts.information_probe.run import JOBS
from scripts.lowrank_token_interaction.activation_audit import save_json, digest
from scripts.analyze_lowrank_six import pack, unpack
from scripts.analyze_lowrank_stage2 import markdown_table, pick_cases


def frozen_parts(scratch):
    return [(scratch/f'{d}_H{h}_S{s}', json.loads((scratch/f'{d}_H{h}_S{s}'/'frozen.json').read_text()))
            for d, h, s in JOBS]


def freeze(args):
    parts = frozen_parts(args.scratch)
    means = {name: float(np.mean([(f['families'][name]['mse']/f['baseline_calibration']['mse']-1)*100
                                  for _, f in parts])) for name in CROSS}
    selected = min(('b0', *CROSS), key=lambda n: (means.get(n, 0.), n != 'b0', n))
    path = args.scratch/'selection_frozen.json'
    if path.exists():
        assert json.loads(path.read_text())['selected_family'] == selected
        return
    save_json(path, {'selected_family': selected, 'calibration_mean_relative_mse': means,
                    'evaluation_opened': False, 'frozen_at': time.time(),
                    'source_hashes': {str(p.name): digest(p/'frozen.json') for p, _ in parts},
                    'policy': 'Lowest calibration setting-equal relative MSE; no-op eligible. No evaluation tuning.'})
    print(json.dumps({'selected': selected, 'calibration': means}), flush=True)


def finalize(args):
    args.audit.mkdir(parents=True, exist_ok=True)
    parts = frozen_parts(args.scratch)
    selection = json.loads((args.scratch/'selection_frozen.json').read_text())
    selected = selection['selected_family']
    eval_results = pd.concat([pd.read_csv(p/'results.csv') for p, _ in parts], ignore_index=True)
    errors = pd.concat([pd.read_csv(p/'sample_errors.csv') for p, _ in parts], ignore_index=True)
    cases = [c for p, _ in parts for c in unpack(p/'selected_cases.npz')]
    calibration = []
    for (path, frozen), (dataset, horizon, seed) in zip(parts, JOBS):
        for trial in frozen['attempts']:
            family = trial['family']
            calibration.append({'setting': frozen['setting']+'_eval', 'dataset': dataset,
                'horizon': horizon, 'seed': seed, 'split': 'calibration',
                'config_id': family+f"_df{trial['degrees']:g}", 'family': family,
                'degrees': trial['degrees'], 'mse': trial['mse'], 'mae': trial['mae'],
                'mae_p95': trial['mae_p95'], 'mae_top1': trial['mae_top1'],
                'selected': trial['degrees'] == frozen['families'][family]['degrees'],
                'relative_mse': (trial['mse']/frozen['baseline_calibration']['mse']-1)*100,
                'relative_mae': (trial['mae']/frozen['baseline_calibration']['mae']-1)*100})
    eval_results['split'] = 'evaluation'
    eval_results['family'] = eval_results.config_id
    results = pd.concat([pd.DataFrame(calibration), eval_results], ignore_index=True)
    results.to_csv(args.audit/'results.csv', index=False)
    errors.to_csv(args.audit/'sample_errors.csv', index=False)
    pack(cases, args.audit/'selected_cases.npz')
    decision = rubric(eval_results, selected)
    avg = eval_results.groupby('config_id')[['relative_mse', 'relative_mae', 'relative_p95',
                                           'relative_top1', 'degrees']].mean().reset_index()
    settings, executions, configs = [], [], []
    for path, frozen in parts:
        provenance = json.loads((path/'evaluate_complete.json').read_text())
        assert provenance['completed_at'] > selection['frozen_at']
        assert digest(path/'frozen.json') == selection['source_hashes'][path.name]
        assert provenance['state_unchanged']
        assert provenance['roles']['fit']['end'] < provenance['roles']['cal']['start']
        assert provenance['roles']['cal']['end'] < provenance['roles']['eval']['start']
        setting = path.name+'_eval'
        settings.append({'setting': setting, 'dataset': provenance['dataset'], 'horizon': provenance['horizon'],
                         'seed': provenance['seed'], 'lookback': 720, 'split': 'support-disjoint validation tail'})
        executions.append({'setting': setting, 'commands': ['run.py --stage calibrate', 'audit.py --stage freeze',
                                                           'run.py --stage evaluate', 'audit.py --stage finalize'],
                           'runtime': provenance})
        configs.append({'setting': setting, 'config_id': selected,
                        'search_notes': frozen['families'][selected]['degrees'] if selected != 'b0' else 'no-op'})
    run = {'experiment_id': args.audit.name,
        'code': {'branch': 'codex/trend-token-cross-attention', 'commit': args.commit,
                 'modified_files': ['scripts/information_probe/', 'tests/test_information_probe.py'],
                 'frozen_probe_hashes': executions[0]['runtime']['source_hashes']},
        'mechanism': {'description': 'Frozen b0 four-mode residual ridge probes; both directional summaries and disjoint mismatches',
                      'feature_flag': 'Offline diagnostic only; original model untouched'},
        'experiment': {'baseline': 'b0', 'candidate': selected, 'settings': settings,
                      'training': 'No backbone training. Per-checkpoint fit-only summaries and ridge; calibration selects effective degrees.',
                      'metrics': ['MSE', 'MAE', 'p95 MAE', 'top1% MAE', 'orthogonal oracle', 'contiguous start blocks']},
        'execution': {'environment': {'inference': 'Torch 2.6.0+cu124 / A800 / highest float32; diagnostics float64'},
                      'settings': executions},
        'selection': {'source': 'validation', 'selected_configs': configs,
                      'calibration_freeze': selection,
                      'search': {p.name: f for p, f in parts}},
        'analysis': {'ranking_metric': 'mae', 'top_k': 3,
                     'dedup_rule': 'Same-channel sample starts at least horizon apart within category',
                     'selections': [c['metadata'] for c in cases], 'decision': decision,
                     'predictor_training_runs': 0, 'ridge_fits': 12*10*4,
                     'decision_status': 'advance_confirmation' if decision['pass'] else 'stop_tested_linear_information_hypotheses'}}
    # Full original residual metrics are checked against every retained evaluation candidate.
    for setting, group in errors.groupby('setting'):
        for config in eval_results[eval_results.setting == setting].config_id:
            prefix = 'baseline' if config == 'b0' else config
            expected = summarize({m: group[prefix+'_'+m].to_numpy() for m in ('mse', 'mae')})
            record = eval_results[(eval_results.setting == setting)&(eval_results.config_id == config)].iloc[0]
            for metric, value in expected.items():
                np.testing.assert_allclose(value, record[metric], rtol=1e-10, atol=1e-10)
        chosen = pick_cases(group, int(group.horizon.iloc[0]), top_k=3)
        key = lambda c: (c['category'], c['rank'], c['sample_id'], c['channel'])
        assert {key(c) for c in chosen} == {key(c['metadata']) for c in cases if c['metadata']['setting'] == setting}
    for case in cases:
        for name in ('baseline', 'candidate'):
            error = case[name].astype(np.float64)-case['truth'].astype(np.float64)
            for metric, value in [('mse', np.square(error).mean()), ('mae', np.abs(error).mean())]:
                np.testing.assert_allclose(value, case['metadata'][name+'_'+metric], rtol=5e-4, atol=5e-5)
    blocks = errors.groupby(['setting', 'support_block'])[['baseline_mse', 'candidate_mse']].mean().reset_index()
    blocks['relative_mse'] = (blocks.candidate_mse/blocks.baseline_mse-1)*100
    oracle = errors.groupby('setting')[['normalized_mode_energy', 'normalized_residual_energy',
                                      'baseline_mse', 'orthogonal_oracle_mse']].mean().reset_index()
    oracle['mode_energy_fraction'] = oracle.normalized_mode_energy/oracle.normalized_residual_energy
    oracle['oracle_relative_mse'] = (oracle.orthogonal_oracle_mse/oracle.baseline_mse-1)*100
    run['analysis']['block_descriptions'] = blocks.to_dict('records')
    run['analysis']['mode_oracle'] = oracle.to_dict('records')
    figures = args.audit/'figures'; figures.mkdir(exist_ok=True)
    chosen = []
    for setting in sorted(errors.setting.unique()):
        group = [c for c in cases if c['metadata']['setting'] == setting]
        chosen.append(max(group, key=lambda c: c['metadata']['delta_mae']))
    chosen = sorted(chosen, key=lambda c: -c['metadata']['delta_mae'])[:10]
    refs = []
    stats = []
    for case in cases:
        m = case['metadata']; truth = case['truth'].astype(np.float64)
        record = {k: m[k] for k in ('setting', 'category', 'rank', 'sample_id', 'channel')}
        record.update(truth_std=float(truth.std()), baseline_std=float(case['baseline'].std()),
                      candidate_std=float(case['candidate'].std()), truth_range=float(np.ptp(truth)))
        for name in ('baseline', 'candidate'):
            for i, e in enumerate(np.array_split(case[name].astype(np.float64)-truth, 3)):
                record[f'{name}_third{i+1}_mse'] = float(np.square(e).mean())
        stats.append(record)
    run['analysis']['case_descriptive_statistics'] = stats
    for case in chosen:
        m = case['metadata']; fig, ax = plt.subplots(figsize=(10, 3.5))
        history = case['history'][-192:]
        ax.plot(np.arange(-len(history), 0), history, color='.65', label='history')
        for name, color in [('truth', 'black'), ('baseline', '#2166ac'), ('candidate', '#b2182b')]:
            ax.plot(case[name], color=color, label=name, lw=1)
        ax.set_title(f"{m['setting']} sample{m['sample_id']} c{m['channel']}; MAE delta {m['delta_mae']:+.5f}")
        ax.set_xlabel('Forecast step'); ax.legend(fontsize=8); fig.tight_layout()
        rel = f"figures/{m['setting']}__s{m['sample_id']}_c{m['channel']}.png"
        fig.savefig(args.audit/rel, dpi=120); plt.close(fig); refs.append(rel)
    lines = ['# Experiment and Objective Error Analysis: information complementarity', '',
        '## 1. Experiment Setup', '',
        'Three datasets, H96/H720, seeds2021/2022; independently trained b0 with static level/shape fusion. Frozen weights, no new test inference. Train prefix60% fits probes, train suffix40% calibrates, validation tail evaluates with complete history-plus-target support isolation. Fit/cal starts sampled every24; evaluation uses every legal start.',
        '**Exposure:** existing backbone trained on all original training data and selected checkpoints using full validation. Prior test results influenced research direction. This is a diagnostic screen, not blind or unbiased generalization and not a Golden improvement claim.', '',
        'Four orthonormal modes: constant, cycle-index slope, period24 first sine/cosine. Supervision is normalized final b0 residual. Features only use historical activations, anchor and receiver original forecast coefficients. No future values enter predictor invocation or features. These names do not constrain tokens to pure trend/period information.', '',
        '## 2. Experiment Results', '',
        f"Calibration-selected cross family: **{selected}**. Advancement rubric: **{decision['pass']}**. Complete fused forecast determines advancement, not branch-only improvement.", '']
    lines += markdown_table(avg)
    lines += ['', 'Positive relative percentages denote degradation. Settings weighted equally.', '']
    lines += markdown_table(eval_results[eval_results.config_id == selected][['setting', 'mse', 'mae', 'relative_mse', 'relative_mae']])
    lines += ['', '## 3. Parameter / Configuration Search', '',
        'Each checkpoint separately fits PCA and ridge, pooling variables without channel/task labels. Trend uses original t; phase-mean uses mean slots; phase-slot uses concatenated mean, fixed sine and cosine slot contrasts. Fit-only centering/scaling/PCA; four components per summary. Receiver base contains anchor and its four original output coefficients.',
        'own2 uses two receiver axes; own4 adds two receiver axes; cross adds two source axes to own2. Both own4 and cross have9 inputs and40 scalar readout coefficients. At each trial ridge slope effective degrees are exactly1/2/3/4, plus unpenalized intercept. Search includes no-op. Each family chooses minimum calibration MSE; primary cross family selected on setting-equal calibration relative MSE before opening evaluation.',
        'receiver_matched and mismatch_matched use the primary family\'s calibration-selected effective degrees exactly; independently calibrated controls are also retained. All480 ridge fits and600 calibration attempts are recorded in run.yaml and results.csv. This matches readout degrees and search budget, not arbitrary representation quality.',
        'Mismatched source uses the same variable at support-disjoint fit times. Fit uses deterministic half rotation; calibration/evaluation use fixed fit-bank indices from sample starts. This is a historical mismatch control with possible source distribution drift, not proof of pure information causality.', '']
    cal_avg = pd.DataFrame(calibration).groupby(['family', 'degrees'])[['relative_mse', 'relative_mae']].mean().reset_index()
    lines += markdown_table(cal_avg)
    lines += ['', '## 4. Error Distribution', '',
        'p95/top1% MAE are computed from every evaluated sample-variable forecast. The first result table preserves each candidate\'s mean relative tails.', '',
        'Orthogonal label oracle describes recoverable error in the four tested modes; it is not a deployable predictor:', '']
    lines += markdown_table(oracle[['setting', 'mode_energy_fraction', 'oracle_relative_mse']])
    lines += ['', 'Contiguous start-index block ranges below show temporal variation. Adjacent block windows can overlap; these are descriptive ranges, not confidence intervals or independent samples. Seeds reuse labels. Small-block uncertainty cannot establish significance.', '']
    lines += markdown_table(blocks.groupby('setting').relative_mse.agg(['min', 'max', 'mean']).reset_index())
    lines += ['', '## 5. Horizon-wise Error', '',
        'H96/H720 population results are retained separately. Per-third MSE below describes selected snapshots only.', '',
        '## 6. High-Error Selection', '',
        'Programmed Top3 baseline MAE, positive MAE regression, negative MAE improvement per setting; same-variable starts separated by horizon within category. All ranked requests retained, including duplicates across categories. Ten largest regression setting representatives plotted.', '',
        '## 7. Case Analysis', '']
    for case, rel in zip(chosen, refs):
        m = case['metadata']
        stat = next(r for r in stats if all(r[k] == m[k] for k in ('setting', 'category', 'rank', 'sample_id', 'channel')))
        lines += [f"### {m['setting']} sample{m['sample_id']} channel{m['channel']}", '',
            f"Program rank {m['rank']} in {m['category']}; {m['time_range']}. Baseline/candidate MAE {m['baseline_mae']:.6f}/{m['candidate_mae']:.6f}. Added output is restricted to the four modes; this observation alone does not identify why a source summary failed.", '']
        lines += markdown_table(pd.DataFrame([{'prediction': name, **{f'third{i}_mse': stat[f'{name}_third{i}_mse'] for i in (1, 2, 3)}} for name in ('baseline', 'candidate')]))
        lines += ['', f'![{Path(rel).stem}]({rel})', '']
    lines += ['## 8. Repeated Observable Patterns', '',
        f"Across {len(cases)} ranked snapshots, candidate standard deviation exceeds truth in {sum(r['candidate_std'] > r['truth_std'] for r in stats)}, and exceeds baseline in {sum(r['candidate_std'] > r['baseline_std'] for r in stats)}. Selected-case descriptions do not estimate population prevalence.", '',
        '## 9. Objective Defect Summary', '',
        'The rubric below separates final residual prediction from same-capacity receiver readout and historical mismatches. A correction beating b0 is insufficient if receiver readout explains it. Frozen probe gains are diagnostic, not jointly trained architecture gains. Negative results concern these four output modes, linear summaries and temporal protocol only.', '',
        '```json', json.dumps(decision, indent=2), '```', '',
        '## 10. Experiment Scope', '',
        f"12 settings; {len(errors)} evaluated sample-variable rows; {len(cases)} ranked snapshots; 480 ridge fits; zero backbone training and zero new test inference. No product probes, new Gate, or attention search is launched unless this preregistered screen passes. Current timing/parameter budgets apply to future deployable architectures; probe overhead is not claimed to meet them.", '']
    report = '\n'.join(lines)
    (args.audit/'objective_error_analysis.md').write_text(report)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for rel in ('objective_error_analysis.md', *refs):
            archive.write(args.audit/rel, rel)
    expected_settings = {s['setting'] for s in settings}
    assert set(results.setting) == set(errors.setting) == {c['metadata']['setting'] for c in cases} == expected_settings
    assert {s['setting'] for s in executions} == {s['setting'] for s in configs} == expected_settings
    assert re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report) == refs
    assert {p.relative_to(args.audit).as_posix() for p in figures.iterdir()} == set(refs)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip') as archive:
        assert set(archive.namelist()) == {'objective_error_analysis.md', *refs}
        for rel in archive.namelist():
            assert not (args.audit/rel).is_symlink()
            assert archive.read(rel) == (args.audit/rel).read_bytes()
    run['validation'] = {'status': 'passed', 'results_checked': True, 'ranking_and_cases_checked': True,
                         'report_and_archive_checked': True, 'directory_and_settings_checked': True}
    (args.audit/'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    assert {p.name for p in args.audit.iterdir()} == {'run.yaml', 'results.csv', 'sample_errors.csv',
        'selected_cases.npz', 'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}
    save_json(args.scratch/'audit_complete.json', {'decision': decision, 'rows': len(errors),
        'artifact_hashes': {str(p.relative_to(args.audit)): digest(p) for p in args.audit.rglob('*') if p.is_file()}})
    print(json.dumps({'decision': decision, 'rows': len(errors), 'cases': len(cases)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--commit', default='unrecorded')
    parser.add_argument('--stage', choices=('freeze', 'finalize'), required=True)
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else finalize(args)
