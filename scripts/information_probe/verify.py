"""Independent CSV/NPZ/archive verification without reopening model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import yaml
from scripts.analyze_lowrank_stage2 import pick_cases
from scripts.analyze_lowrank_six import unpack
from scripts.information_probe.core import rubric


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    run = yaml.safe_load((args.audit/'run.yaml').read_text())
    results = pd.read_csv(args.audit/'results.csv')
    evals = results[results.split == 'evaluation']
    selected = run['selection']['calibration_freeze']['selected_family']
    expected_settings = {s['setting'] for s in run['experiment']['settings']}
    assert {s['setting'] for s in run['execution']['settings']} == expected_settings
    assert {s['setting'] for s in run['selection']['selected_configs']} == expected_settings
    assert set(results.setting) == expected_settings
    searches = run['selection']['search']
    cal_means = {}
    for family in ('t_pmean', 't_pslot', 'p_trend'):
        values = []
        for name, frozen in searches.items():
            attempts = [a for a in frozen['attempts'] if a['family'] == family]
            trial = min(attempts, key=lambda a: (a['mse'], a['degrees']))
            assert trial == frozen['families'][family]
            values.append((trial['mse']/frozen['baseline_calibration']['mse']-1)*100)
        cal_means[family] = float(np.mean(values))
    choice = min(('b0', *cal_means), key=lambda n: (cal_means.get(n, 0.), n != 'b0', n))
    assert choice == selected
    for name, frozen in searches.items():
        for trial in frozen['attempts']:
            row = results[(results.setting == name+'_eval')&(results.split == 'calibration')&
                          (results.config_id == trial['family']+f"_df{trial['degrees']:g}")].iloc[0]
            for metric in ('mse', 'mae', 'mae_p95', 'mae_top1'):
                np.testing.assert_allclose(row[metric], trial[metric], atol=1e-12, rtol=1e-12)
            fitted = trial['fitted']
            if fitted is not None:
                assert fitted['features'] == (7 if trial['family'].endswith('own2') else 9)
                np.testing.assert_allclose(fitted['slope_edf'], trial['degrees'], atol=1e-10)
    cases = unpack(args.audit/'selected_cases.npz')
    assert {c['metadata']['setting'] for c in cases} == expected_settings
    tables = []
    for chunk in pd.read_csv(args.audit/'sample_errors.csv', chunksize=50000):
        tables.append(chunk)
    errors = pd.concat(tables, ignore_index=True)
    assert set(errors.setting) == expected_settings
    assert not errors.duplicated(['setting', 'sample_id', 'channel']).any()
    largest_difference = 0.
    for setting, frame in errors.groupby('setting'):
        for config in evals[evals.setting == setting].config_id:
            prefix = 'baseline' if config == 'b0' else config
            mae = frame[prefix+'_mae'].to_numpy()
            actual = {'mse': frame[prefix+'_mse'].mean(), 'mae': mae.mean(),
                      'mae_p95': np.quantile(mae, .95),
                      'mae_top1': np.sort(mae)[-max(1, int(np.ceil(len(mae)*.01))):].mean()}
            row = evals[(evals.setting == setting)&(evals.config_id == config)].iloc[0]
            for metric, value in actual.items():
                largest_difference = max(largest_difference, abs(value-row[metric]))
                np.testing.assert_allclose(value, row[metric], rtol=1e-10, atol=1e-10)
        for metric in ('mse', 'mae'):
            np.testing.assert_allclose(frame['delta_'+metric], frame['candidate_'+metric]-frame['baseline_'+metric], atol=1e-12)
        requests = pick_cases(frame, int(frame.horizon.iloc[0]), top_k=3)
        key = lambda m: (m['category'], m['rank'], m['sample_id'], m['channel'])
        assert {key(m) for m in requests} == {key(c['metadata']) for c in cases if c['metadata']['setting'] == setting}
        if selected != 'b0':
            primary = evals[(evals.setting == setting)&(evals.config_id == selected)].iloc[0]
            for control in ('receiver_matched', 'mismatch_matched'):
                other = evals[(evals.setting == setting)&(evals.config_id == control)].iloc[0]
                np.testing.assert_allclose(primary.degrees, other.degrees, atol=1e-10)
                assert primary.parameters == other.parameters
    for case in cases:
        for name in ('baseline', 'candidate'):
            e = case[name].astype(np.float64)-case['truth'].astype(np.float64)
            for metric, value in [('mse', np.square(e).mean()), ('mae', np.abs(e).mean())]:
                np.testing.assert_allclose(value, case['metadata'][name+'_'+metric], atol=5e-5, rtol=5e-4)
    measured = rubric(evals, selected)
    recorded = run['analysis']['decision']
    assert measured['pass'] == recorded['pass']
    assert measured.get('checks') == recorded.get('checks')
    for metric in ('relative_mse', 'relative_mae'):
        if metric in measured:
            np.testing.assert_allclose(measured[metric], recorded[metric], atol=1e-12, rtol=1e-12)
    report = (args.audit/'objective_error_analysis.md').read_text()
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    assert len(refs) == len(set(refs))
    assert all(r.startswith('figures/') and '..' not in Path(r).parts for r in refs)
    assert {p.relative_to(args.audit).as_posix() for p in (args.audit/'figures').iterdir()} == set(refs)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip') as archive:
        assert set(archive.namelist()) == {'objective_error_analysis.md', *refs}
        for rel in archive.namelist():
            assert archive.read(rel) == (args.audit/rel).read_bytes()
    assert {p.name for p in args.audit.iterdir()} == {'run.yaml', 'results.csv', 'sample_errors.csv',
        'selected_cases.npz', 'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}
    output = {'status': 'passed', 'settings': len(expected_settings), 'sample_rows': len(errors),
              'cases': len(cases), 'summary_max_abs_difference': largest_difference,
              'decision': run['analysis']['decision'], 'artifact_hashes': {
                  str(p.relative_to(args.audit)): sha(p) for p in args.audit.rglob('*') if p.is_file()}}
    print(json.dumps(output, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    main(parser.parse_args())
