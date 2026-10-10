"""Verify full coverage and append test-only interaction tables to prior results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.extend_interaction_test import FAMILIES, SETTINGS

NAMES = {
    'i0': 'PhaseFormer-L single gate (8 epochs)',
    'pt_post': 'P->T Post-Norm', 'pt_pre': 'P->T Pre-Norm',
    'pt_scaled': 'P->T zero-initialized scaled residual', 'pt_null': 'P->T Null read',
    'tp_post': 'T->P Post-Norm', 'tp_pre': 'T->P Pre-Norm (8 epochs)',
    'tp_scaled': 'T->P zero-initialized scaled residual', 'tp_selective': 'T->P selective phase write',
    'b0': 'PhaseFormer-L static level/shape gate', 'context_full': 'Phase-conditioned T->P message',
    'phase_only': 'Phase-only context correction', 'tp_mod_centered': 'T->P centered modulation',
}


def golden_reference(path):
    requested = {(dataset, horizon) for dataset, horizon, _ in SETTINGS}
    values = {}
    for row in csv.reader(path.read_text().splitlines(), delimiter='|'):
        if len(row) != 6:
            continue
        dataset, horizon, mse, mae = (v.strip() for v in row[1:5])
        if not horizon.isdigit() or (dataset, int(horizon)) not in requested:
            continue
        key = (dataset, int(horizon))
        if key in values:
            raise ValueError(f'Duplicate Golden task: {key}')
        values[key] = (float(mse), float(mae))
    assert set(values) == requested
    return pd.Series(np.mean(list(values.values()), axis=0), index=['test_mse', 'test_mae'])


def table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join('---' for _ in columns)+' |']
    for row in frame.to_dict('records'):
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                value = '-'
            elif column in ('MSE', 'MAE'):
                value = f'{value:.4f}'
            elif column.startswith('Delta'):
                value = f'{value:+.3f}%'
            values.append(str(value))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main(args):
    manifest = json.loads((args.scratch/'manifest.json').read_text())
    complete = json.loads((args.scratch/'complete.json').read_text())
    freeze = json.loads((args.scratch/'training_complete_before_test.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((args.scratch/'evaluations').glob('*.json'))]
    results = pd.DataFrame(rows)
    expected = {(f, d, h, s, a) for f, (_, _, arms) in FAMILIES.items()
                for d, h, s in SETTINGS for a in arms}
    identity = list(zip(results.family, results.dataset, results.horizon, results.seed, results.arm))
    assert len(identity) == len(set(identity)) == 168 and set(identity) == expected
    assert complete == {'training_jobs': 22, 'test_inference_runs': 168}
    assert freeze['jobs'] == 22 and int(results.new_training.sum()) == 22
    assert results.state_unchanged.all()
    assert set(results.precision) == {'medium'}
    assert set(results.lookback) == {720}
    assert np.isfinite(results[['test_mse', 'test_mae']].to_numpy()).all()
    assert (results.test_samples > 0).all()
    for _, group in results.groupby(['dataset', 'horizon']):
        assert group.dataset_sha256.nunique() == group.test_samples.nunique() == 1
    for entry in manifest['entries']:
        row = results[results.key == entry['key']].iloc[0]
        if entry['existing']:
            assert row.checkpoint_sha256 == entry['source_checkpoint_sha256']
            assert row.config_sha256 == entry['source_config_sha256']
        else:
            path = args.scratch/'new_runs'/entry['key']
            result = json.loads((path/'result.json').read_text())
            initialization = json.loads((path/'initialization.json').read_text())
            assert result['epoch_budget'] == 30 and result['effective_patience'] == 8
            assert not result['pretrained_checkpoint_loaded'] and not result['smoke']
            assert result['checkpoint_sha256'] == row.checkpoint_sha256
            assert initialization['initial_max_abs_vs_b0'] == 0
            assert initialization['all_parameters_trainable'] and initialization['label_and_time_marks_unreachable']
    # Identical source b0 weights should produce the same full-test numbers in both envelopes.
    base = results[results.arm == 'b0'].pivot(index=['dataset', 'horizon', 'seed'], columns='family', values=['test_mse', 'test_mae'])
    for metric in ('test_mse', 'test_mae'):
        np.testing.assert_allclose(base[metric]['context'], base[metric]['centered'], atol=1e-12, rtol=1e-12)
    old = pd.read_csv(args.previous/'results.csv')
    old_base = old[(old.family == 'lightweight')&(old.arm == 'b0')]
    joined = results[(results.family == 'context')&(results.arm == 'b0')].merge(
        old_base, on=['dataset', 'horizon', 'seed'], suffixes=('_new', '_old'))
    assert len(joined) == 12
    for metric in ('test_mse', 'test_mae'):
        np.testing.assert_allclose(joined[metric+'_new'], joined[metric+'_old'], atol=1e-12, rtol=1e-12)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output/'results.csv', index=False)
    task = results.groupby(['family', 'arm', 'dataset', 'horizon'])[['test_mse', 'test_mae']].agg(['mean', 'std']).reset_index()
    task.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in task.columns]
    task.to_csv(args.output/'setting_means.csv', index=False)
    paper = pd.read_csv(args.previous/'dataset_averages.csv')
    anchor = golden_reference(args.golden)
    stored_golden = paper[paper.arm == '论文 Golden'][['test_mse', 'test_mae']].mean()
    np.testing.assert_allclose(anchor, stored_golden, atol=1e-15, rtol=1e-15)
    average = results.groupby(['family', 'arm'])[['test_mse', 'test_mae']].mean()
    new_rows = []
    for (family, arm), values in average.iterrows():
        baseline = average.loc[(family, 'i0' if family == 'normalization' else 'b0')]
        new_rows.append({'Family': family, 'Method': NAMES[arm], 'MSE': values.test_mse, 'MAE': values.test_mae,
            'Delta MSE vs Golden': (values.test_mse/anchor.test_mse-1)*100,
            'Delta MAE vs Golden': (values.test_mae/anchor.test_mae-1)*100,
            'Delta MSE paired': (values.test_mse/baseline.test_mse-1)*100,
            'Delta MAE paired': (values.test_mae/baseline.test_mae-1)*100})
    references = [{'Family': 'reference', 'Method': label, 'MSE': values.test_mse,
                   'MAE': values.test_mae, 'Delta MSE vs Golden': (values.test_mse/anchor.test_mse-1)*100,
                   'Delta MAE vs Golden': (values.test_mae/anchor.test_mae-1)*100}
                  for label, values in [('Golden (original PhaseFormer)', anchor),
                      ('Original PhaseFormer-L q=1/8', paper[paper.arm == '论文 PhaseFormer-L q=1/8'][['test_mse', 'test_mae']].mean())]]
    new = pd.DataFrame(references+new_rows)
    new.to_csv(args.output/'new_method_averages.csv', index=False)
    prior = paper.groupby(['arm', 'method'])[['test_mse', 'test_mae']].mean().reset_index()
    master = []
    for row in prior.to_dict('records'):
        master.append({'Family': 'previous', 'Method': row['method'], 'MSE': row['test_mse'], 'MAE': row['test_mae'],
            'Delta MSE vs Golden': (row['test_mse']/anchor.test_mse-1)*100,
            'Delta MAE vs Golden': (row['test_mae']/anchor.test_mae-1)*100})
    # Preserve the two previously evaluated 30-epoch Pre-Norm variants as separate rows.
    long = pd.read_csv(args.long_test/'results.csv')
    for arm in ('tp_pre', 'tp_pre_components_static'):
        group = long[long.arm == arm]
        assert len(group) == 12
        mse, mae = group.test_mse.mean(), group.test_mae.mean()
        master.append({'Family': 'previous_long', 'Method': arm+' (30 epochs)', 'MSE': mse, 'MAE': mae,
                       'Delta MSE vs Golden': (mse/anchor.test_mse-1)*100,
                       'Delta MAE vs Golden': (mae/anchor.test_mae-1)*100})
    master = pd.concat([pd.DataFrame(master), new[new.Family != 'reference'].drop(columns=['Delta MSE paired', 'Delta MAE paired'])], ignore_index=True)
    master = pd.concat([master[master.Method == '论文 Golden'], master[master.Method != '论文 Golden']], ignore_index=True)
    master.to_csv(args.output/'all_method_averages.csv', index=False)
    lines = ['# Extended Interaction Comparison: Full Test Matrix', '',
        'ETTh1 / ETTm2 / Weather x H96/H720; lookback720. Existing seeds2021/2022 for every interaction. '
        'Each task first averages two seeds, then six tasks receive equal weight. Golden anchor uses the original published PhaseFormer values in docs/PhaseFormer_gold_standard.md on these exact six tasks (MSE0.2935, MAE0.3323333333). Original PhaseFormer-L q1/8 uses the existing three-seed E14 means.', '',
        'Delta = (candidate aggregate mean / reference aggregate mean - 1) * 100%; negative means lower error. '
        'This is a ratio of aggregate means, not a mean of per-setting percentages. Metrics are calculated from all test forecast elements, with no dropped batch.', '',
        '## Newly Completed Test Table', '',
        table(new, ['Family', 'Method', 'MSE', 'MAE', 'Delta MSE vs Golden', 'Delta MAE vs Golden']), '',
        '## Combined Previous and New Table', '',
        table(master, ['Family', 'Method', 'MSE', 'MAE', 'Delta MSE vs Golden', 'Delta MAE vs Golden']), '',
        '## Six Task Details', '']
    for dataset, horizon in sorted({(d, h) for d, h, _ in SETTINGS}):
        block = task[(task.dataset == dataset)&(task.horizon == horizon)]
        records = pd.DataFrame([{'Family': row.family, 'Method': NAMES[row.arm],
                                'MSE': row.test_mse_mean, 'MAE': row.test_mae_mean}
                               for row in block.itertuples()])
        lines += [f'### {dataset} H{horizon}', '', table(records, ['Family', 'Method', 'MSE', 'MAE']), '']
    lines += ['## Provenance and Limits', '',
        'Normalization/Null/selective arms retain their original8-epoch training budget; all108 corresponding checkpoints reused. '
        'Context and centered arms retain30-epoch maximum / patience8. '
        'Reused context candidates:8; reused centered candidates:6; newly trained context:16 and centered:6. '
        'The missing static b0 controls reuse the previously reproduced identical architecture/protocol. '
        'New runs use from-scratch matched common initialization, trainable original backbone, rankH/8, period24, original Huber/LR/batch, and minimum-validation-Huber checkpoint.', '',
        '22 formal training jobs completed before168 full-test evaluations. No hyperparameter search or winner-dependent follow-up. '
        'Prior pilots had negative validation results; this extension was explicitly requested for full coverage. '
        'Smoke runs exercise two missing H96 family paths and do not read test or contribute metrics. '
        'Previous test exposure remains; this is not a blind or unbiased new benchmark. '
        'Paper and interaction models differ in seed coverage and some training/fusion protocols; numerical comparisons cannot isolate interaction causality or establish statistical significance. '
        'Golden PhaseFormer entries remain fixed references; no Golden improvement claim is made.', '',
        'Independent coverage, finite metrics, data hash/window counts, unchanged source checkpoint/config hashes, new-run initialization, static-b0 reproduction and aggregation checks passed. '
        'Per-task seed standard deviations are retained in setting_means.csv; two shared-seed runs do not establish significance.', '']
    (args.output/'comparison.md').write_text('\n'.join(lines))
    metadata = {'status': 'passed', 'settings': 12, 'test_records': len(results), 'new_training_runs': 22,
                'manifest': manifest, 'training_frozen_before_test': freeze, 'complete': complete,
                'validation': {'coverage': True, 'source_hashes': True, 'data_alignment': True,
                               'initialization': True, 'baseline_reproduction': True, 'aggregation': True},
                'summary_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'delta_reference': {'model': 'original PhaseFormer Golden', 'file': str(args.golden),
                                    'source_sha256': hashlib.sha256(args.golden.read_bytes()).hexdigest(),
                                    'mse': float(anchor.test_mse), 'mae': float(anchor.test_mae),
                                    'formula': '(candidate aggregate mean / Golden aggregate mean - 1) * 100'}}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(table(new, ['Family', 'Method', 'MSE', 'MAE', 'Delta MSE vs Golden', 'Delta MAE vs Golden']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('research_runs/interaction_existing_test_20261009'))
    parser.add_argument('--long-test', type=Path, default=Path('research_runs/lowrank_level_fusion_next_test'))
    parser.add_argument('--golden', type=Path, default=Path('docs/PhaseFormer_gold_standard.md'))
    main(parser.parse_args())
