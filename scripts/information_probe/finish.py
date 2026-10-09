"""Add measured closure and reproduction metadata without changing frozen results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import yaml


def main(args):
    run = yaml.safe_load((args.audit/'run.yaml').read_text())
    if 'closure' in run['analysis']:
        print(json.dumps(run['analysis']['closure']))
        return
    args.scratch.mkdir(parents=True, exist_ok=True)
    previous = args.scratch/'closure_previous_artifacts'
    previous.mkdir(exist_ok=False)
    for name in ('run.yaml', 'objective_error_analysis.md', 'objective_error_analysis.zip'):
        shutil.copy2(args.audit/name, previous/name)
    data = pd.read_csv(args.audit/'results.csv')
    data = data[data.split == 'evaluation']
    selected = run['selection']['calibration_freeze']['selected_family']
    assert selected == 't_pmean', 'This closure is specific to the completed information_probe_20261009 screen'
    main_rows = data[data.config_id == selected]
    summary = {'selected': selected, 'setting_wins': int((main_rows.relative_mse < 0).sum()),
               'task_wins': int((main_rows.groupby(['dataset', 'horizon']).relative_mse.mean() < 0).sum()),
               'non_noop_settings': int((main_rows.degrees > 0).sum()),
               'relative_mse': float(main_rows.relative_mse.mean()),
               'relative_mae': float(main_rows.relative_mae.mean()),
               'candidate_means': data.groupby('config_id')[['relative_mse', 'relative_mae']].mean().to_dict('index'),
               'followup': 'H1 screen failed; no product probes, prefix backbone confirmation or new architecture training.'}
    if run['analysis']['decision']['pass']:
        raise ValueError('A positive screen requires a separate confirmation plan')
    run['analysis']['closure'] = summary
    run['analysis']['closure_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run['reproduction'] = {'frozen_code_commit': run['code']['commit'], 'source_runs': args.runs,
        'launch': f'python scripts/information_probe/launch.py --runs {args.runs} --scratch research_runs/information_probe_20261009_scratch --audit research_runs/information_probe_20261009 --commit {run["code"]["commit"]}',
        'verify': 'python scripts/information_probe/verify.py --audit research_runs/information_probe_20261009',
        'output_policy': 'Use a new scratch/audit directory when reproducing; preserve original completed outputs.'}
    for i, record in enumerate(run['execution']['settings']):
        record['commands'] = [f'python scripts/information_probe/run.py --runs {args.runs} --scratch research_runs/information_probe_20261009_scratch --stage {stage} --shards 4 --shard {i % 4}' for stage in ('calibrate', 'evaluate')]
        record['commands'].insert(1, 'python scripts/information_probe/audit.py --scratch research_runs/information_probe_20261009_scratch --stage freeze')
        record['commands'].append(f'python scripts/information_probe/audit.py --scratch research_runs/information_probe_20261009_scratch --audit research_runs/information_probe_20261009 --commit {run["code"]["commit"]} --stage finalize')
    report_path = args.audit/'objective_error_analysis.md'
    report = report_path.read_text()
    insertion = ('The selected cross-branch readout changes only %d/12 settings after calibration no-op fallback. '
        'It wins MSE in %d/12 settings and %d/6 dataset-horizon tasks; setting-equal MSE/MAE changes are %+.6f%%/%+.6f%%. '
        'These small values do not establish significance. Independently calibrated trend-own4 achieves %+.6f%% mean MSE, '
        'while the exactly matched-degree receiver and historical mismatch controls achieve %+.6f%%/%+.6f%%. '
        'The primary cross readout does not establish benefit beyond receiver readout or the mismatch control. '
        'Even the other two fixed cross families do not reach the 1%% practical gate. '
        'ETTm2-H720-S2021/S2022 primary MSE changes are %+.6f%%/%+.6f%%, which do not demonstrate cross-seed consistency.\n\n'
        'Ridge fitting uses unweighted normalized residual targets; calibration/evaluation MSE uses original dataset-standardized units. '
        'These objectives weight sample-variable scales differently. Weather has much larger normalized residual energy than ETT. '
        'Dividing by historical scale gives low-variance histories greater weight, but population variance attribution was not separately measured; '
        'normalized mode-energy fractions must not be confused with raw-MSE oracle fractions. '
        'Fit weighting and richer summaries are untested. Therefore this negative result only concerns the registered linear four-mode probes, '
        'not all predictive information or all possible interactions. No prefix-trained reference, width2 product probe or joint architecture training is launched.\n\n'
        'Smoke provenance: v1 failed on an unsupported six-axis short-horizon phase summary; v2 case replay detected an error-sign mismatch. '
        'The final v3 smoke passed all12 settings and snapshot/archive checks. All formal inference uses the corrected frozen code; '
        'earlier smoke measurements are excluded from evidence.\n\n') % (
            summary['non_noop_settings'], summary['setting_wins'], summary['task_wins'], summary['relative_mse'], summary['relative_mae'],
            summary['candidate_means']['t_own4']['relative_mse'], summary['candidate_means']['receiver_matched']['relative_mse'],
            summary['candidate_means']['mismatch_matched']['relative_mse'],
            main_rows[main_rows.setting == 'ETTm2_H720_S2021_eval'].relative_mse.iloc[0],
            main_rows[main_rows.setting == 'ETTm2_H720_S2022_eval'].relative_mse.iloc[0])
    anchor = '## 9. Objective Defect Summary\n\n'
    assert report.count(anchor) == 1
    report = report.replace(anchor, anchor+insertion)
    report = report.replace('Ten largest regression setting representatives plotted.',
        'Ten per-setting ranked representatives with largest signed MAE delta plotted. Settings with no correction contribute baseline-high-error snapshots.')
    report_path.write_text(report)
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for rel in ('objective_error_analysis.md', *refs):
            archive.write(args.audit/rel, rel)
    (args.audit/'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--runs', required=True)
    main(parser.parse_args())
