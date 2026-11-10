#!/usr/bin/env python3
"""Add an N-BEATS / ETSFormer / BasisFormer style trend branch to original PhaseFormer.

Four arms per setting, all under the same E14 protocol and the same seed:

  pf      original PhaseFormer, no branch (the control)
  nb      + N-BEATS interpretable trend stack
  ets     + ETSFormer level and growth with damping
  basis   + BasisFormer learnable smooth basis with attention coefficients

Every branch is zero at initialization, so each arm starts exactly at PhaseFormer.
The comparison of interest is two-sided: against ``pf`` it measures what the branch
adds on top of PhaseFormer; against B0H+ (the tuned phase + level model from
``research_runs/b0hp_v1`` / ``traffic_b0hp_v1``) it measures whether copying these
designs closes the gap to our level branch without any tuning.

    python3 scripts/trend_branch/run.py --check
    python3 scripts/trend_branch/run.py --run --gpus 0,1,2,3,4,7 --slots-per-gpu 2
    python3 scripts/trend_branch/run.py --export
    python3 scripts/trend_branch/run.py --report --out docs/PhaseFormer_VLDBJ_trend_branches.md

Kept outside ``scripts/vldbj_m1`` so the frozen-protocol hash stays valid.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/trend_branch_v1'
SETTINGS = tuple((d, h) for d in ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
                 for h in (96, 192))
SEED = 2021
KINDS = ('pf', 'nb', 'ets', 'basis')
MODEL_KEYS = {'pf': 'PF-e14', 'nb': 'PF+NB', 'ets': 'PF+ETS', 'basis': 'PF+Basis'}
DEFAULT_DUMP = Path.home() / 'niuyiming/component_mse_20261023/predictions'


# ---------------------------------------------------------------- jobs
def job_name(dataset, horizon, kind):
    return f'{dataset}_H{horizon}_S{SEED}__{kind}'


def build_jobs(scratch):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': 'v1', 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for dataset, horizon in SETTINGS:
        for kind in KINDS:
            job = dict(base, dataset=dataset, horizon=horizon, seed=SEED, arm='pf',
                       protocol='e14', init='scratch', head='S2', shared_gate=False,
                       variant_id=None, variant=None, extension_lr_mult=1.0,
                       backbone_lr_mult=1.0, warm_source=None, epoch_test=False,
                       source=copy.deepcopy(sources[(dataset, horizon, SEED)]),
                       trend_branch=kind)
            name = job_name(dataset, horizon, kind)
            job['output'] = str(scratch / 'v1' / 'runs' / name)
            path = scratch / 'v1' / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def run_jobs(paths, gpus):
    """One subprocess per job, each in its own process so CUDA_VISIBLE_DEVICES applies."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import queue
    import subprocess
    tasks = queue.Queue()
    for path in paths:
        tasks.put(str(path))
    failures = []

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
            if (out / 'result.json').is_file():
                continue
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', path],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(path)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


# ---------------------------------------------------------------- worker
def patched_build_model():
    """proto.build_model with the requested trend branch attached."""
    from scripts.trend_branch import branches
    from scripts.vldbj_m1 import proto
    original = proto.build_model

    def build(job):
        model, args, cfg, common = original(job)
        kind = job.get('trend_branch', 'pf')
        if kind != 'pf':
            branches.attach(model, kind, job.get('branch_options'),
                            float(job.get('branch_lr_mult', 1.0)))
        return model, args, cfg, common
    return build


def worker(job_path):
    from scripts.vldbj_m1 import proto
    proto.build_model = patched_build_model()
    proto.run_job(job_path)


def run(scratch, gpus, slots, shard=None):
    paths = sorted((scratch / 'v1' / 'jobs').glob('*.json'))
    if shard:
        index, count = (int(v) for v in shard.split('/'))
        paths = paths[index::count]
    print(f'{len(paths)} jobs', flush=True)
    gpus = [int(g) for g in gpus.split(',') for _ in range(slots)]
    print('launching on', gpus, flush=True)
    run_jobs(paths, gpus)


# ---------------------------------------------------------------- export
def export(scratch, dump_root, skip_unfinished=False):
    from scripts.component_mse import dump
    from scripts.vldbj_m1 import proto
    from src.dataset.data_factory import data_provider
    import numpy as np
    import torch

    proto.build_model = patched_build_model()
    for path in sorted((scratch / 'v1' / 'jobs').glob('*.json')):
        job = json.loads(path.read_text())
        run_dir = Path(job['output'])
        if skip_unfinished and not (run_dir / 'result.json').is_file():
            print(json.dumps({'setting': path.stem, 'skipped': 'not finished'}), flush=True)
            continue
        record = json.loads((run_dir / 'result.json').read_text())
        model, args, _, _ = proto.build_model(job)
        state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
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
            raise AssertionError(f'{path.name}: exported MSE {mse} != recorded {recorded}')
        key = MODEL_KEYS[job['trend_branch']]
        dump.save_truth(dump_root, job['dataset'], job['horizon'], truth, test_set.scaler)
        dump.save_prediction(dump_root, key, job['dataset'], job['horizon'], job['seed'], prediction)
        dump.save_result(dump_root, key, job['dataset'], job['horizon'], job['seed'], {
            'model': key, 'dataset': job['dataset'], 'horizon': job['horizon'],
            'seed': job['seed'], 'lookback': 720, 'arm': 'pf', 'branch': job['trend_branch'],
            'test_mse': mse, 'test_mae': mae, 'test_size': int(prediction.shape[0]),
            'channels': int(prediction.shape[2]), 'checkpoint': record['checkpoint'],
            'run_dir': str(run_dir), 'space': 'standardized (train-fitted StandardScaler)',
        })
        print(json.dumps({'setting': path.stem, 'test_mse': mse, 'test_mae': mae}), flush=True)


# ---------------------------------------------------------------- check
def check(scratch):
    """Zero at init, and every branch parameter receives a gradient after one step."""
    import torch
    from scripts.vldbj_m1 import proto

    proto.build_model = patched_build_model()
    paths = sorted((scratch / 'v1' / 'jobs').glob('*.json'))
    picks = [p for p in paths if p.stem in ('Weather_H96_S2021__nb', 'Weather_H96_S2021__ets',
                                            'Weather_H96_S2021__basis', 'Weather_H96_S2021__pf',
                                            'Traffic_H192_S2021__basis')]
    if len(picks) != 5:
        raise SystemExit(f'expected five check jobs, found {[p.stem for p in picks]}')
    for path in picks:
        job = json.loads(path.read_text())
        job['smoke'] = True
        model, _, _, _ = proto.build_model(job)
        model.train()
        x = torch.randn(2, model.seq_len, model.enc_in)
        mark = torch.zeros(2, model.seq_len, 4)
        y = torch.randn(2, model.seq_len + model.pred_len, model.enc_in)
        ymark = torch.zeros(2, model.seq_len + model.pred_len, 4)
        out = model(x, mark, model._build_decoder_input(y), ymark)[0]
        if job['trend_branch'] == 'pf':
            print(f'{path.stem}: pf forward {tuple(out.shape)}')
            continue
        branch = model.trend_branch
        delta = branch(x.float())
        zero_at_init = float(delta.abs().max()) < 1e-6
        if hasattr(branch, 'gate'):
            with torch.no_grad():
                branch.gate.fill_(0.1)
            out = model(x, mark, model._build_decoder_input(y), ymark)[0]
        elif job['trend_branch'] == 'nb':
            # The zero-initialized output layers block gradients to the MLPs on the
            # first step; give them a small non-zero value to test the whole branch.
            with torch.no_grad():
                for block in branch.blocks:
                    block.to_forecast.weight.normal_(0.0, 0.01)
                    block.to_backcast.weight.normal_(0.0, 0.01)
            out = model(x, mark, model._build_decoder_input(y), ymark)[0]
        out[:, -model.pred_len:].sum().backward()
        missing = [n for n, p in branch.named_parameters()
                   if p.grad is None or float(p.grad.abs().sum()) == 0]
        if job['trend_branch'] == 'nb':
            # The last block's backcast is not consumed by a later block (by design).
            last = len(branch.blocks) - 1
            missing = [n for n in missing
                       if n not in (f'blocks.{last}.to_backcast.weight',
                                    f'blocks.{last}.to_backcast.bias')]
        print(f'{path.stem}: delta_zero={zero_at_init} '
              f'params={sum(p.numel() for p in branch.parameters())} no_grad={missing}')
        if not zero_at_init or missing:
            raise SystemExit('branch check failed')
    print('check passed')


# ---------------------------------------------------------------- report
def b0hp_numbers(csv_path):
    import csv
    out = {}
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            key = (row['dataset'], int(row['horizon']))
            if key in SETTINGS:
                out[key] = (float(row['test_mse']), float(row['test_mae']))
    return out


def report(scratch, dump_root, out, csv_path):
    from scripts.component_mse import dump, trend_definitions as td
    rows = []
    models = ('PF-e14', 'PF+NB', 'PF+ETS', 'PF+Basis')
    for dataset, horizon in SETTINGS:
        truth = dump.load_truth(dump_root, dataset, horizon)
        for model in models:
            for seed in (SEED,):
                try:
                    prediction = dump.load_prediction(dump_root, model, dataset, horizon, seed)
                except FileNotFoundError:
                    continue
                scores = td.score_file(prediction, truth, 24, ('cycle-level', 'folded'))
                for definition, values in scores.items():
                    rows.append({'definition': definition, 'dataset': dataset, 'horizon': horizon,
                                 'model': model, 'seed': seed, **values})
    summary = {}
    for row in rows:
        summary.setdefault((row['definition'], row['dataset'], row['horizon'], row['model']),
                           []).append(row)
    summary = {k: {m: sum(r[m] for r in v) / len(v) for m in ('trend', 'periodic', 'total')}
               for k, v in summary.items()}
    b0hp = b0hp_numbers(csv_path)
    settings = list(SETTINGS)
    lines = ['# PhaseFormer + 三种趋势分支（N-BEATS / ETSFormer / BasisFormer）的对比', '',
             '12 个 setting（6 数据集 × 96/192），seed 2021，E14 协议，分支零初始化，'
             '因此训练起点与 PhaseFormer 完全相同。趋势成分用 cycle-level 折叠加法分解（P=24）。', '',
             '## 1. 总测试 MSE / MAE', '',
             '| Setting | PF | PF+NB | PF+ETS | PF+Basis | B0H+ |', '|---|---:|---:|---:|---:|---:|']
    def total(model, dataset, horizon, metric='total'):
        key = ('cycle-level', dataset, horizon, model)
        return summary[key][metric] if key in summary else float('nan')
    for dataset, horizon in settings:
        cells = [f'{total(m, dataset, horizon):.4f}' for m in models]
        bh = b0hp.get((dataset, horizon), (float('nan'),) * 2)[0]
        lines.append(f'| {dataset}-{horizon} | ' + ' | '.join(cells) + f' | {bh:.4f} |')
    means = {m: sum(total(m, d, h) for d, h in settings) / len(settings) for m in models}
    bh_mean = sum(b0hp[k][0] for k in settings) / len(settings)
    lines += ['| **平均** | ' + ' | '.join(f'**{means[m]:.4f}**' for m in models) +
              f' | **{bh_mean:.4f}** |', '']
    lines += ['## 2. 趋势成分 MSE（cycle-level）', '',
              '| Setting | PF | PF+NB | PF+ETS | PF+Basis | B0H+ |', '|---|---:|---:|---:|---:|---:|']
    b0hp_trend = {}
    for dataset, horizon in settings:
        try:
            prediction = dump.load_prediction(dump_root, 'B0H+', dataset, horizon, SEED)
            truth = dump.load_truth(dump_root, dataset, horizon)
            b0hp_trend[(dataset, horizon)] = td.score_file(
                prediction, truth, 24, ('cycle-level',))['cycle-level']['trend']
        except FileNotFoundError:
            b0hp_trend[(dataset, horizon)] = float('nan')
    for dataset, horizon in settings:
        cells = [f'{total(m, dataset, horizon, "trend"):.4f}' for m in models]
        lines.append(f'| {dataset}-{horizon} | ' + ' | '.join(cells) +
                     f' | {b0hp_trend[(dataset, horizon)]:.4f} |')
    trend_means = {m: sum(total(m, d, h, 'trend') for d, h in settings) / len(settings) for m in models}
    bh_trend_mean = sum(b0hp_trend.values()) / len(settings)
    lines += ['| **平均** | ' + ' | '.join(f'**{trend_means[m]:.4f}**' for m in models) +
              f' | **{bh_trend_mean:.4f}** |', '']
    lines += ['## 3. 相对变化', '',
              '| 模型 | 总 MSE vs PF | 趋势 MSE vs PF | 总 MSE vs B0H+ | 趋势 MSE vs B0H+ |',
              '|---|---:|---:|---:|---:|']
    for m in models:
        lines.append(f'| {m} | {(means[m]-means["PF-e14"])/means["PF-e14"]*100:+.2f}% | '
                     f'{(trend_means[m]-trend_means["PF-e14"])/trend_means["PF-e14"]*100:+.2f}% | '
                     f'{(means[m]-bh_mean)/bh_mean*100:+.2f}% | '
                     f'{(trend_means[m]-bh_trend_mean)/bh_trend_mean*100:+.2f}% |')
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(out).write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-dir', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--dump-root', type=Path, default=DEFAULT_DUMP)
    parser.add_argument('--csv', type=Path,
                        default=ROOT / 'docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv')
    parser.add_argument('--gpus', default='0,1,2,3,4,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--shard', default=None, help='i/n: run only jobs i, i+n, ...')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--skip-unfinished', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/PhaseFormer_VLDBJ_trend_branches.md')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    if args.prepare:
        paths = build_jobs(args.scratch_dir)
        print(f'{len(paths)} jobs under {args.scratch_dir / "v1"}')
    if args.check:
        check(args.scratch_dir)
    if args.run:
        run(args.scratch_dir, args.gpus, args.slots_per_gpu, args.shard)
    if args.export:
        export(args.scratch_dir, args.dump_root, args.skip_unfinished)
    if args.report:
        report(args.scratch_dir, args.dump_root, args.out, args.csv)


if __name__ == '__main__':
    main()
