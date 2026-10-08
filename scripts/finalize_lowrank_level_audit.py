#!/usr/bin/env python3
"""Close the registered level-fusion study without test-based reselection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import yaml

RETAINED = ['components_static', 'tp_pre_components_static']
PAIRS = [('components_static', 'i0'), ('tp_pre_components_static', 'tp_pre'),
         ('tp_pre_components_static', 'components_static'),
         ('components_level_dynamic', 'components_static'),
         ('tp_pre_level_dynamic', 'tp_pre_components_static'),
         ('components_level_dynamic', 'g3_components'),
         ('tp_pre_level_dynamic', 'tp_pre_g3')]
MARKER = '\n## 11. Final Research Decision\n'


def paired(table, candidate, baseline):
    a = table[table.config_id == candidate].set_index('setting').sort_index()
    b = table[table.config_id == baseline].set_index('setting').sort_index()
    assert a.index.equals(b.index)
    relative = (a.mse / b.mse - 1) * 100
    tasks = pd.DataFrame({'dataset': a.dataset, 'horizon': a.horizon,
                          'delta_mse': a.mse - b.mse}).groupby(['dataset', 'horizon']).delta_mse.mean()
    return {'candidate': candidate, 'baseline': baseline,
            'relative_mse_percent': float(relative.mean()),
            'median_relative_mse_percent': float(relative.median()),
            'relative_mae_percent': float(((a.mae / b.mae - 1) * 100).mean()),
            'mse_wins': int((relative < 0).sum()), 'settings': len(a),
            'task_mse_wins': int((tasks < 0).sum()), 'tasks': len(tasks),
            **{k + '_relative_percent': float(((a[k] / b[k] - 1) * 100).mean())
               for k in ['mean_candidate_level_mse', 'mean_candidate_shape_mse',
                         'candidate_mse_p95', 'candidate_mse_top1_mean']}}


def md_table(records, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in records:
        values = [f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def close(audit, validation, history):
    run = yaml.safe_load((audit / 'run.yaml').read_text())
    assert run['validation']['status'] == 'passed'
    table = pd.read_csv(audit / 'results.csv')
    split = str(table.setting.iloc[0]).rsplit('_', 1)[1]
    comparisons = [paired(table, a, b) for a, b in PAIRS
                   if {a, b} <= set(table.config_id)]
    dynamic = [paired(validation, a, b) for a, b in PAIRS[3:5]]
    for row in dynamic:
        row['performance_gate_passed'] = (row['relative_mse_percent'] < 0
            and row['median_relative_mse_percent'] < 0 and row['mse_wins'] >= 8
            and row['task_mse_wins'] >= 4 and row['relative_mae_percent'] <= 0)
    assert not any(r['performance_gate_passed'] for r in dynamic)
    diagnostics = []
    for arm, baseline in PAIRS[3:5]:
        a = validation[validation.config_id == arm].set_index('setting')
        b = validation[validation.config_id == baseline].set_index('setting')
        diagnostics.append({'config_id': arm,
            'fixed_level_relative_mse_percent': float(((a.mean_fixed_level_mse / a.mse - 1) * 100).mean()),
            'shuffled_level_relative_mse_percent': float(((a.mean_shuffled_gate_mse / a.mse - 1) * 100).mean()),
            'oracle_vs_static_relative_mse_percent': float(((a.mean_oracle_mse / b.mse - 1) * 100).mean()),
            'oracle_worse_than_static_settings': int((a.mean_oracle_mse > b.mse).sum())})
    matched = 0
    if history:
        previous = pd.read_csv(history / 'results.csv').set_index(['setting', 'config_id'])
        current = validation.set_index(['setting', 'config_id'])
        common = current.index.intersection(previous.index)
        np.testing.assert_array_equal(current.loc[common, ['mse', 'mae']], previous.loc[common, ['mse', 'mae']])
        matched = len(common)
    selection = run.setdefault('selection', {})
    selection.update(source='validation', default_candidate=RETAINED[0],
                     tradeoff_candidate=RETAINED[1], test_set_selection=False,
                     prior_test_exposure=True)
    for record in selection['selected_configs']:
        record['config_id'] = RETAINED
        record['search_notes'] = 'Validation-frozen B0 default and B1 MSE tradeoff; no selection changes from test.'
    table['selected'] = table.config_id.isin(RETAINED)
    table.to_csv(audit / 'results.csv', index=False)
    run['analysis']['final_decision'] = {
        'default_candidate': RETAINED[0], 'tradeoff_candidate': RETAINED[1],
        'rejected_default_upgrades': ['components_level_dynamic', 'tp_pre_level_dynamic'],
        'validation_advancement_checks': dynamic, 'split_comparisons': comparisons,
        'validation_gate_diagnostics': diagnostics, 'historical_exact_matches': matched,
        'additional_training': 0, 'core_training_runs': 96,
        'context_zero_and_cross_task_extension': 'Not triggered: both dynamic-level candidates fail performance gates.',
        'golden_comparability': 'Numerical reference only; full original training protocol is not established as matched.',
        'causal_limit': 'Oracle uses future truth and is not deployable. Intervention and oracle do not prove causes of joint-training changes.'}
    report = (audit / 'objective_error_analysis.md').read_text().split(MARKER)[0]
    report = report.replace('An eight-epoch loss is a budget-specific result; longer from-scratch confirmation may change ordering.',
                            'Results use a 30-epoch ceiling and validation checkpoint selection; another training budget may change ordering.')
    lines = [MARKER, '',
        'Selection was frozen using validation before this stage\'s test evaluation: B0 (static level/static shape, no interaction) is the default research candidate. B1 (T->P pre-norm/static components) is only an MSE tradeoff. Test does not change these roles.', '',
        f'Final {split} paired comparisons; negative percentages indicate lower error, each setting receives equal weight:', '',
        md_table(comparisons, ['candidate', 'baseline', 'relative_mse_percent', 'median_relative_mse_percent',
                              'relative_mae_percent', 'mse_wins', 'settings', 'task_mse_wins', 'tasks']), '',
        'Validation L0 and L1 both fail the preregistered advancement conditions. No context-zero control, third-seed training or cross-task expansion is warranted by this matrix. Total core training remains 96 runs; final test is 48 checkpoint evaluations with zero training.', '',
        'Static-component validation shape MSE improves in all 12 settings versus the original scalar gate. Its average benefit is small, and its p95/top-1% tail metrics are not uniformly better. Retention is a parsimonious research choice, not a statistical-significance claim.', '',
        'Validation component/tail diagnostics (setting-equal relative changes):', '',
        md_table([paired(validation, a, b) for a, b in PAIRS],
                 ['candidate', 'baseline', 'mean_candidate_level_mse_relative_percent',
                  'mean_candidate_shape_mse_relative_percent', 'candidate_mse_p95_relative_percent',
                  'candidate_mse_top1_mean_relative_percent']), '',
        'Fixed-weight dynamic-level interventions and label-informed oracle diagnostics:', '',
        md_table(diagnostics, ['config_id', 'fixed_level_relative_mse_percent',
                  'shuffled_level_relative_mse_percent', 'oracle_vs_static_relative_mse_percent',
                  'oracle_worse_than_static_settings']), '',
        'Gate matching has measurable value inside the trained dynamic models, but this does not make those models better than independently trained static components. Oracle error is below the static baseline in every setting, so a pure absence of within-branch reachable forecasts does not explain the aggregate loss. The oracle reads future truth; this does not establish learnable reliability or rule out branch-quality problems. Identifying a training cause would require a separately registered experiment.', '',
        'Removing dynamic shape is not supported: L0 loses to D0 in 11/12 settings and L1 loses to D1 in 9/12. Weather-H96-S2022 validation sample1702/channel14 illustrates this limitation: L0 MSE 33.272280 versus D0 15.393433, including shape MSE 22.539290 versus 10.425045. This historical anchor is diagnostic, not a selection criterion. Other Weather anchors regress, so no universal extreme-sample protection is claimed.', '',
        'Pre-norm is not established as complementary to dynamic level: the combination fails against B1 and lacks repeatable uniform gains. B1 lowers mean MSE versus B0 on validation while worsening mean relative MAE and winning only 6/12 settings. Its level and shape errors move in different directions; prefer B0 as the simpler default.', '',
        f'{matched} historical same-seed/settings checkpoint metrics exactly reproduce the prior 30-epoch study. This confirms reproducibility, not extra independent seeds. Two scalar/static ETTh1-H96-S2022 runs reached the 30-epoch ceiling; conclusions are budget-specific, and convergence is not guaranteed.', '',
        'The tasks and validation/test splits have prior exposure. This stage did not use test for selection; it is not a fresh blind test. Golden values remain the authoritative original PhaseFormer reference, but matching lookback/horizon does not establish matching full training protocol. No Golden-superiority or unbiased-generalization claim is made.', '']
    report += '\n'.join(lines)
    (audit / 'objective_error_analysis.md').write_text(report)
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    assert len(refs) == len(set(refs))
    assert all(Path(r).parts[0] == 'figures' and len(Path(r).parts) == 2 for r in refs)
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in ['objective_error_analysis.md', *refs]:
            assert not (audit / rel).is_symlink()
            z.write(audit / rel, rel)
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip') as z:
        assert set(z.namelist()) == {'objective_error_analysis.md', *refs}
        for rel in z.namelist():
            assert z.read(rel) == (audit / rel).read_bytes()
    assert {p.name for p in audit.iterdir()} == {'run.yaml', 'results.csv', 'sample_errors.csv',
        'selected_cases.npz', 'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}
    assert {p.name for p in (audit / 'figures').iterdir()} == {Path(r).name for r in refs}
    run['validation']['final_decision_and_archive_checked'] = True
    (audit / 'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    print(json.dumps({'audit': str(audit), 'comparisons': comparisons,
        'report_sha256': hashlib.sha256((audit / 'objective_error_analysis.md').read_bytes()).hexdigest()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-audit', type=Path, required=True)
    parser.add_argument('--test-audit', type=Path, required=True)
    parser.add_argument('--history-audit', type=Path)
    args = parser.parse_args()
    validation = pd.read_csv(args.validation_audit / 'results.csv')
    for audit in [args.validation_audit, args.test_audit]:
        close(audit, validation, args.history_audit)


if __name__ == '__main__':
    main()
