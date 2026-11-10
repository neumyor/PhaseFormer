#!/usr/bin/env python3
"""B0H+ in the component-MSE experiment: train the missing seeds, export predictions, redo the report.

The component-MSE dump (``~/niuyiming/component_mse_20261023/predictions``) already holds
truth and predictions for PhaseFormer, PhaseFormer-LA, DLinear, SparseTSF, FITS and
CycleNet on 6 datasets x {96, 192} x 3 seeds.  This script adds B0H+ at each setting's
best known configuration (``docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv``):

* seed 2021 reuses the run that produced the recorded number -- its job file is located by
  matching the run directory, so the exported prediction is the exact configured run;
* seeds 2022/2023 retrain the same configuration with the same frozen worker, so the
  comparison is a seed study at a fixed configuration.

Every export re-checks the element-mean test MSE against the run's recorded value before
writing, so a silently different model cannot enter the dump.

    python3 scripts/component_mse/b0hp.py --prepare --scratch-dir research_runs/component_b0hp_v1
    python3 scripts/component_mse/b0hp.py --run --scratch-dir research_runs/component_b0hp_v1 --gpus 0,1
    python3 scripts/component_mse/b0hp.py --export
    python3 scripts/component_mse/b0hp.py --report --out docs/PhaseFormer_VLDBJ_trend_b0hp.md

Kept outside ``scripts/vldbj_m1`` so the frozen-protocol hash stays valid.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SETTINGS = tuple((d, h) for d in ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
                 for h in (96, 192))
SEEDS = (2021, 2022, 2023)
MODEL_KEY = 'B0H+'
BASE_MODELS = ('PhaseFormer', 'PhaseFormerLA', 'DLinear', 'SparseTSF', 'FITS', 'CycleNet')
DEFAULT_DUMP = Path.home() / 'niuyiming/component_mse_20261023/predictions'


def read_rows(csv_path):
    with open(csv_path) as handle:
        rows = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader(handle)}
    missing = [k for k in SETTINGS if k not in rows]
    if missing:
        raise SystemExit(f'missing rows in {csv_path}: {missing}')
    return {k: rows[k] for k in SETTINGS}


def job_name(dataset, horizon, seed):
    return f'{MODEL_KEY}__{dataset}_H{horizon}_S{seed}'


def find_existing_job(run_path):
    """The job file whose output is the run that produced a recorded number."""
    target = Path(run_path).name
    for path in sorted(ROOT.glob('research_runs/*/*/jobs/*.json')):
        job = json.loads(path.read_text())
        if Path(job['output']).name == target:
            return path, job
    raise SystemExit(f'no job file for run {run_path}')


def build_seed_job(row, seed, sources, base):
    from scripts.vldbj_m1.run import write_immutable
    dataset, horizon = row['dataset'], int(row['horizon'])
    period = int(row['period'])
    source = copy.deepcopy(sources[(dataset, horizon, seed)])
    source['config']['period'] = period
    source['config']['hyperparams']['period_len'] = period
    gate_mult = float(row['gate_mult'] or row['ext_lr'])
    readout_mult = float(row['readout_mult'] or row['ext_lr'])
    return dict(
        base, dataset=dataset, horizon=horizon, seed=seed, arm='b0', protocol='e14',
        init='scratch', head=row['head'], shared_gate=row['shared'] == 'True',
        variant_id=None,
        variant={'gate_init': float(row['gate_init']), 'extension_lr_mult': readout_mult,
                 'communication_width': 2},
        extension_lr_mult=readout_mult, backbone_lr_mult=float(row['bb_lr']),
        warm_source=None, epoch_test=False, source=source,
        lr_split={'gate_mult': gate_mult, 'readout_mult': readout_mult},
        level_input='history+token', token_source=row['token_source'],
        token_zero_init=row['token_zero_init'] == 'True',
        b0hp_source={'csv_run': row['run'], 'csv_config_hash': row['config_hash']})


def prepare(scratch, csv_path):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    rows = read_rows(csv_path)
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': 'v1', 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    jobs = []
    for key, row in rows.items():
        dataset, horizon = key
        for seed in SEEDS:
            name = job_name(dataset, horizon, seed)
            if seed == 2021:
                path, existing = find_existing_job(row['run'])
                jobs.append({'setting': name, 'seed': seed, 'reuse_job': str(path),
                             'run': existing['output']})
                continue
            job = build_seed_job(row, seed, sources, base)
            job['output'] = str(scratch / 'v1' / 'runs' / name)
            path = scratch / 'v1' / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            jobs.append({'setting': name, 'seed': seed, 'job': str(path), 'run': job['output']})
    (scratch / 'v1' / 'manifest.json').parent.mkdir(parents=True, exist_ok=True)
    (scratch / 'v1' / 'manifest.json').write_text(json.dumps(jobs, indent=1))
    print(json.dumps(jobs, indent=1))
    return jobs


def load_manifest(scratch):
    return json.loads((scratch / 'v1' / 'manifest.json').read_text())


def run(scratch, gpus, slots):
    from scripts.lr_split_run import run_jobs
    paths = [item['job'] for item in load_manifest(scratch) if 'job' in item]
    print(f'{len(paths)} training jobs', flush=True)
    gpus = [int(g) for g in gpus.split(',') for _ in range(slots)]
    print('launching on', gpus, flush=True)
    run_jobs(paths, gpus)


def job_of(item):
    if 'job' in item:
        return json.loads(Path(item['job']).read_text())
    return json.loads(Path(item['reuse_job']).read_text())


def export_one(item, dump_root):
    import numpy as np
    import torch

    from scripts.component_mse import dump
    from scripts.vldbj_m1.proto import build_model
    from src.dataset.data_factory import data_provider

    job = job_of(item)
    run_dir = Path(job['output'])
    record = json.loads((run_dir / 'result.json').read_text())
    checkpoint = record['checkpoint']
    model, args, _, _ = build_model(job)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state)
    device = torch.device('cuda:0')
    model.to(device).eval()
    test_set, test_loader = data_provider(args.dataset_args, 'test')
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in test_loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
            predictions.append(pred.cpu().numpy())
            truths.append(by[:, -model.pred_len:].cpu().numpy())
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    mse, mae = dump.test_arrays(prediction, truth)
    recorded = record['best']['test_mse']
    if abs(mse - recorded) > max(1e-5, 1e-4 * recorded):
        raise AssertionError(f'{item["setting"]}: exported MSE {mse} != recorded {recorded}')
    dump.save_truth(dump_root, job['dataset'], job['horizon'], truth, test_set.scaler)
    dump.save_prediction(dump_root, MODEL_KEY, job['dataset'], job['horizon'], job['seed'], prediction)
    dump.save_result(dump_root, MODEL_KEY, job['dataset'], job['horizon'], job['seed'], {
        'model': MODEL_KEY, 'dataset': job['dataset'], 'horizon': job['horizon'],
        'seed': job['seed'], 'lookback': 720, 'arm': job['arm'], 'head': job['head'],
        'shared_gate': job['shared_gate'], 'level_input': job.get('level_input'),
        'token_source': job.get('token_source', 'norm'),
        'token_zero_init': bool(job.get('token_zero_init', False)),
        'gate_init': job['variant']['gate_init'],
        'gate_mult': job.get('lr_split', {}).get('gate_mult', job['extension_lr_mult']),
        'readout_mult': job.get('lr_split', {}).get('readout_mult', job['extension_lr_mult']),
        'backbone_lr_mult': job['backbone_lr_mult'], 'period': job['source']['config'].get('period', 24),
        'test_mse': mse, 'test_mae': mae, 'test_size': int(prediction.shape[0]),
        'channels': int(prediction.shape[2]), 'checkpoint': checkpoint,
        'run_dir': str(run_dir), 'space': 'standardized (train-fitted StandardScaler)',
        'selected_on': 'test at seed 2021 (project protocol)',
    })
    print(json.dumps({'setting': item['setting'], 'seed': job['seed'], 'test_mse': mse,
                      'test_mae': mae}), flush=True)


def export(scratch, dump_root, seed=None, skip_existing=False):
    from scripts.component_mse import dump
    for item in load_manifest(scratch):
        if seed is not None and item['seed'] != seed:
            continue
        job = job_of(item)
        existing = dump.result_path(dump_root, MODEL_KEY, job['dataset'], job['horizon'], job['seed'])
        if skip_existing and existing.is_file():
            continue
        record = Path(item['run']) / 'result.json'
        if not record.is_file():
            raise SystemExit(f'run not finished: {item["run"]}')
        export_one(item, dump_root)


# ---------------------------------------------------------------- report
def collect(dump_root, period, definitions):
    """Scores for every model/seed; cached per setting so a rerun only scores new runs."""
    from scripts.component_mse import dump
    from scripts.component_mse import trend_definitions as td
    models = BASE_MODELS + (MODEL_KEY,)
    cache_path = Path(dump_root) / 'results' / f'trend_scores_p{period}.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    rows = []
    updated = False
    for dataset, horizon in SETTINGS:
        truth = dump.load_truth(dump_root, dataset, horizon)
        for model in models:
            for seed in SEEDS:
                missing = [d for d in definitions
                           if f'{model}|{dataset}|{horizon}|{seed}|{d}' not in cache]
                if missing:
                    try:
                        prediction = dump.load_prediction(dump_root, model, dataset, horizon, seed)
                    except FileNotFoundError:
                        continue
                    scores = td.score_file(prediction, truth, period, missing)
                    for definition, values in scores.items():
                        cache[f'{model}|{dataset}|{horizon}|{seed}|{definition}'] = values
                        updated = True
                for definition in definitions:
                    key = f'{model}|{dataset}|{horizon}|{seed}|{definition}'
                    if key not in cache:
                        continue
                    rows.append({'definition': definition, 'dataset': dataset, 'horizon': horizon,
                                 'model': model, 'seed': seed, **cache[key]})
    if updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return rows, models


def summarise(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['definition'], row['dataset'], row['horizon'], row['model']),
                           []).append(row)
    return {key: {m: sum(r[m] for r in items) / len(items) for m in ('trend', 'periodic', 'total')
                  if all(m in r for r in items)} for key, items in grouped.items()}


def report(dump_root, out, period=24, models=None):
    # cycle-level is the definition used in the previous component report; folded is
    # its orthogonal-projection sibling. The remaining definitions were compared once
    # in PhaseFormer_component_mse_report.md and are not repeated here.
    rows, available = collect(dump_root, period, ('cycle-level', 'folded'))
    models = tuple(m for m in (models or available) if m in available)
    summary = summarise(rows)
    settings = list(SETTINGS)
    lines = ['# B0H+ 的趋势 / 周期成分对比（6 数据集 × 96/192，3 seed 均值）', '',
             f'分解周期 P={period}；误差在训练用的标准化空间；趋势成分为每周期取一个水平值'
             '（cycle-level），另附 folded、线性、lowpass 等定义。', '']
    for definition in ('cycle-level', 'folded'):
        if not any(k[0] == definition for k in summary):
            continue
        lines += [f'## {definition}', '', '| Model | ' +
                  ' | '.join(f'{d}-{h}' for d, h in settings) + ' |',
                  '|' + '---|' * (len(settings) + 1)]
        for model in models:
            cells = []
            for dataset, horizon in settings:
                key = (definition, dataset, horizon, model)
                cells.append(f'{summary[key]["trend"]:.4f}' if key in summary else '-')
            lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    def mean_of(definition, metric, model):
        vals = [summary[(definition, d, h, model)][metric] for d, h in settings
                if (definition, d, h, model) in summary and metric in summary[(definition, d, h, model)]]
        return sum(vals) / len(vals) if vals else float('nan')

    base_models = [m for m in models if m not in ('PhaseFormer', MODEL_KEY)]
    header = '| 定义 | 成分 | ' + ' | '.join(models) + ' | B0H+ vs PF | B0H+ vs 最好基线 |'
    lines += ['## 12 setting 平均与相对变化', '', header,
              '|' + '---|' * (len(models) + 3)]
    for definition in ('cycle-level', 'folded'):
        for metric in ('trend', 'periodic', 'total'):
            values = {m: mean_of(definition, metric, m) for m in models}
            pf = values['PhaseFormer']
            bh = values[MODEL_KEY]
            base = min(values[m] for m in base_models) if base_models else float('nan')
            cells = ' | '.join(f'{values[m]:.4f}' for m in models)
            lines.append(f'| {definition} | {metric} | {cells} | '
                         f'{(bh - pf) / pf * 100:+.2f}% | {(bh - base) / base * 100:+.2f}% |')
    text = '\n'.join(lines) + '\n'
    print(text)
    if out:
        Path(out).write_text(text)
        Path(out).with_suffix('.json').write_text(json.dumps(
            {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v for k, v in summary.items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-dir', type=Path, default=ROOT / 'research_runs/component_b0hp_v1')
    parser.add_argument('--csv', type=Path,
                        default=ROOT / 'docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv')
    parser.add_argument('--dump-root', type=Path, default=DEFAULT_DUMP)
    parser.add_argument('--gpus', default='0,1,2,3,4,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--export-only-seed', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--models', default=None,
                        help='comma-separated subset, e.g. PhaseFormer,B0H+,DLinear,SparseTSF,FITS,CycleNet')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.scratch_dir, args.csv)
    if args.run:
        run(args.scratch_dir, args.gpus, args.slots_per_gpu)
    if args.export:
        export(args.scratch_dir, args.dump_root, args.export_only_seed, args.skip_existing)
    if args.report:
        selected = [m.strip() for m in args.models.split(',')] if args.models else None
        report(args.dump_root, args.out, models=selected)


if __name__ == '__main__':
    main()
