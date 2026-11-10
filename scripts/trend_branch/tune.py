#!/usr/bin/env python3
"""Tune the three trend branches so each is compared to PhaseFormer at its best reasonable setting.

The first round gave every branch the backbone's learning rate and no architectural choice.
N-BEATS' polynomial trend then ran away (ETTh1-192 +17%), ETS was neutral and the basis branch
was slightly positive.  This round gives the branch its own learning-rate multiplier and, for
N-BEATS, a lower polynomial degree, then selects per setting by **validation** MSE among the
candidates (the default run from ``trend_branch_v1`` plus the new cells).  Test numbers are
reported only for the selected run, so the selection is blind to the test split.

Cells (12 settings each):

  nb     degree 2, branch lr x0.1      degree 1 (linear trend), lr x0.3
  ets    branch lr x0.3
  basis  branch lr x0.3

The default runs (branch lr x1.0, default architecture) are candidates as well, so a cell only
wins where it improves the validation loss.

    python3 scripts/trend_branch/tune.py --prepare
    python3 scripts/trend_branch/tune.py --run --gpus 0,1,2,3,4,7 --slots-per-gpu 2
    python3 scripts/trend_branch/tune.py --select --export
    python3 scripts/trend_branch/tune.py --report --out docs/PhaseFormer_VLDBJ_trend_branches_tuned.md

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

DEFAULT_SCRATCH = ROOT / 'research_runs/trend_branch_tune_v1'
DEFAULT_SCRATCH_UNTUNED = ROOT / 'research_runs/trend_branch_v1'
SETTINGS = tuple((d, h) for d in ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
                 for h in (96, 192))
SEED = 2021
BRANCHES = ('nb', 'ets', 'basis')
CELLS = (
    ('nb_d2_lr01', 'nb', {'degree': 2}, 0.1),
    ('nb_d1_lr03', 'nb', {'degree': 1}, 0.3),
    ('ets_lr03', 'ets', {}, 0.3),
    ('basis_lr03', 'basis', {}, 0.3),
    # Second round for N-BEATS only: its first-round cells still trail PhaseFormer on the
    # trend component, so two gentler variants are added before the selection is redone.
    ('nb_d2_lr003', 'nb', {'degree': 2}, 0.03),
    ('nb_d1_lr01', 'nb', {'degree': 1}, 0.1),
)
MODEL_KEYS = {'nb': 'T-NB', 'ets': 'T-ETS', 'basis': 'T-Basis'}
DEFAULT_DUMP = Path.home() / 'niuyiming/component_mse_20261023/predictions'


def build_jobs(scratch):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': 'v1', 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for dataset, horizon in SETTINGS:
        for tag, kind, options, lr_mult in CELLS:
            job = dict(base, dataset=dataset, horizon=horizon, seed=SEED, arm='pf',
                       protocol='e14', init='scratch', head='S2', shared_gate=False,
                       variant_id=None, variant=None, extension_lr_mult=1.0,
                       backbone_lr_mult=1.0, warm_source=None, epoch_test=False,
                       source=copy.deepcopy(sources[(dataset, horizon, SEED)]),
                       trend_branch=kind, branch_options=options,
                       branch_lr_mult=lr_mult, tune_tag=tag)
            name = f'{dataset}_H{horizon}_S{SEED}__{tag}'
            job['output'] = str(scratch / 'v1' / 'runs' / name)
            path = scratch / 'v1' / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def run(scratch, gpus, slots, cells=None):
    from scripts.trend_branch.run import run_jobs
    paths = sorted((scratch / 'v1' / 'jobs').glob('*.json'))
    if cells:
        keep = {c.strip() for c in cells.split(',')}
        paths = [p for p in paths if any(p.stem.endswith('__' + c) for c in keep)]
    gpus = [int(g) for g in gpus.split(',') for _ in range(slots)]
    print(f'{len(paths)} jobs on {gpus}', flush=True)
    run_jobs(paths, gpus)


def candidates(scratch, untuned):
    """Per (setting, branch): every candidate with its job path and validation MSE."""
    out = {}
    for path in sorted((untuned / 'v1' / 'jobs').glob('*.json')):
        job = json.loads(path.read_text())
        if job['trend_branch'] not in BRANCHES:
            continue
        record = Path(job['output']) / 'result.json'
        if not record.is_file():
            continue
        out.setdefault((job['dataset'], job['horizon'], job['trend_branch']), []).append({
            'job': str(path), 'tag': 'default', 'lr_mult': 1.0,
            'val_mse': json.loads(record.read_text())['best']['val_mse']})
    for path in sorted((scratch / 'v1' / 'jobs').glob('*.json')):
        job = json.loads(path.read_text())
        record = Path(job['output']) / 'result.json'
        if not record.is_file():
            continue
        out.setdefault((job['dataset'], job['horizon'], job['trend_branch']), []).append({
            'job': str(path), 'tag': job['tune_tag'], 'lr_mult': job['branch_lr_mult'],
            'val_mse': json.loads(record.read_text())['best']['val_mse']})
    return out


def score_val_trend(scratch, untuned):
    """Cycle-level trend MSE of every candidate on the validation split (cached)."""
    import numpy as np
    import torch

    from scripts.component_mse import trend_definitions as td
    from scripts.trend_branch.run import patched_build_model
    from scripts.vldbj_m1 import proto
    from src.dataset.data_factory import data_provider

    cache_path = scratch / 'v1' / 'val_trend.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    proto.build_model = patched_build_model()
    device = torch.device('cuda:0')
    for key, items in candidates(scratch, untuned).items():
        for item in items:
            if item['job'] in cache:
                continue
            job = json.loads(Path(item['job']).read_text())
            record = json.loads((Path(job['output']) / 'result.json').read_text())
            model, args, _, _ = proto.build_model(job)
            state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
            model.load_state_dict(state)
            model.to(device).eval()
            val_set, val_loader = data_provider(args.dataset_args, 'val')
            predictions, truths = [], []
            with torch.inference_mode():
                for batch in val_loader:
                    bx, by, bxm, bym = [v.float().to(device) for v in batch]
                    pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
                    predictions.append(pred.cpu().numpy())
                    truths.append(by[:, -model.pred_len:].cpu().numpy())
            prediction = np.concatenate(predictions)
            truth = np.concatenate(truths)
            cache[item['job']] = td.score_file(prediction, truth, 24, ('cycle-level',))['cycle-level']['trend']
            cache_path.write_text(json.dumps(cache))
            print(json.dumps({'job': Path(item['job']).stem, 'val_trend_mse': cache[item['job']]}), flush=True)
    return cache


def select(scratch, untuned, metric='val_mse'):
    picks = {}
    val_trend = {}
    if metric == 'val_trend':
        val_trend = json.loads((scratch / 'v1' / 'val_trend.json').read_text())
    for key, items in candidates(scratch, untuned).items():
        for item in items:
            item['score'] = val_trend.get(item['job'], float('inf')) if metric == 'val_trend' else item['val_mse']
        best = min(items, key=lambda item: item['score'])
        picks['|'.join(map(str, key))] = best
        print(f'{key[0]:>10}-{key[1]:<4} {key[2]:>5}: ' +
              ' '.join(f"{i['tag']}({i['score']:.4f})" for i in items) +
              f" -> {best['tag']}")
    (scratch / 'v1' / 'selection.json').write_text(json.dumps(picks, indent=1))
    return picks


def export(scratch, untuned, dump_root):
    import numpy as np
    import torch

    from scripts.component_mse import dump
    from scripts.trend_branch.run import patched_build_model
    from scripts.vldbj_m1 import proto
    from src.dataset.data_factory import data_provider

    picks = json.loads((scratch / 'v1' / 'selection.json').read_text())
    proto.build_model = patched_build_model()
    for key, pick in picks.items():
        job = json.loads(Path(pick['job']).read_text())
        run_dir = Path(job['output'])
        existing = dump.result_path(dump_root, MODEL_KEYS[job['trend_branch']],
                                    job['dataset'], job['horizon'], job['seed'])
        if existing.is_file() and json.loads(existing.read_text()).get('run_dir') == str(run_dir):
            print(json.dumps({'key': key, 'skipped': 'already exported'}), flush=True)
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
            raise AssertionError(f'{job["dataset"]}-{job["horizon"]}: {mse} != {recorded}')
        model_key = MODEL_KEYS[job['trend_branch']]
        dump.save_truth(dump_root, job['dataset'], job['horizon'], truth, test_set.scaler)
        dump.save_prediction(dump_root, model_key, job['dataset'], job['horizon'], job['seed'], prediction)
        dump.save_result(dump_root, model_key, job['dataset'], job['horizon'], job['seed'], {
            'model': model_key, 'dataset': job['dataset'], 'horizon': job['horizon'],
            'seed': job['seed'], 'branch': job['trend_branch'], 'tag': pick['tag'],
            'branch_lr_mult': job.get('branch_lr_mult', 1.0),
            'branch_options': job.get('branch_options', {}),
            'val_mse': record['best']['val_mse'], 'test_mse': mse, 'test_mae': mae,
            'checkpoint': record['checkpoint'], 'run_dir': str(run_dir),
        })
        print(json.dumps({'key': key, 'tag': pick['tag'], 'test_mse': mse}), flush=True)


def report(scratch, dump_root, out):
    from scripts.component_mse import dump, trend_definitions as td

    picks = json.loads((scratch / 'v1' / 'selection.json').read_text())
    models = ('PF-e14', 'T-NB', 'T-ETS', 'T-Basis', 'B0H+')
    summary = {}
    for dataset, horizon in SETTINGS:
        truth = dump.load_truth(dump_root, dataset, horizon)
        for model in models:
            try:
                prediction = dump.load_prediction(dump_root, model, dataset, horizon, SEED)
            except FileNotFoundError:
                continue
            scores = td.score_file(prediction, truth, 24, ('cycle-level',))
            summary[(dataset, horizon, model)] = scores['cycle-level']

    def mean(model, metric):
        vals = [summary[(d, h, model)][metric] for d, h in SETTINGS if (d, h, model) in summary]
        return sum(vals) / len(vals)

    lines = ['# 调参后的三种趋势分支 vs 原版 PhaseFormer vs B0H+', '',
             '12 个 setting（6 数据集 × 96/192），seed 2021，E14 协议；趋势成分为 cycle-level（P=24）。',
             '每个 setting 的分支配置在候选 cells 中按**验证集 MSE** 选出（候选包含第一轮默认配置和本轮新 cell），',
             '测试集只用于报告选中配置的结果。', '', '## 1. 总测试 MSE', '',
             '| Setting | PhaseFormer | PF+NB | PF+ETS | PF+Basis | B0H+ | 选中 cell（NB / ETS / Basis） |',
             '|---|---:|---:|---:|---:|---:|---|']
    for dataset, horizon in SETTINGS:
        cells = [f'{summary[(dataset, horizon, m)]["total"]:.4f}' if (dataset, horizon, m) in summary else '-'
                 for m in models]
        tags = []
        for branch in BRANCHES:
            pick = picks.get(f'{dataset}|{horizon}|{branch}')
            tags.append(pick['tag'] if pick else '-')
        lines.append(f'| {dataset}-{horizon} | ' + ' | '.join(cells) + ' | ' + ' / '.join(tags) + ' |')
    lines += ['| **平均** | ' + ' | '.join(f'**{mean(m, "total"):.4f}**' for m in models) + ' | |', '']
    lines += ['## 2. 趋势成分 MSE（cycle-level）', '',
              '| Setting | PhaseFormer | PF+NB | PF+ETS | PF+Basis | B0H+ |',
              '|---|---:|---:|---:|---:|---:|']
    for dataset, horizon in SETTINGS:
        cells = [f'{summary[(dataset, horizon, m)]["trend"]:.4f}' if (dataset, horizon, m) in summary else '-'
                 for m in models]
        lines.append(f'| {dataset}-{horizon} | ' + ' | '.join(cells) + ' |')
    lines += ['| **平均** | ' + ' | '.join(f'**{mean(m, "trend"):.4f}**' for m in models) + ' | |', '']
    lines += ['## 3. 相对变化（12 setting 平均）', '',
              '| 模型 | 总 MSE | 趋势 MSE | 总 MSE vs PF | 趋势 MSE vs PF | 总 MSE vs B0H+ | 趋势 MSE vs B0H+ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    pf_total, pf_trend = mean('PF-e14', 'total'), mean('PF-e14', 'trend')
    bh_total, bh_trend = mean('B0H+', 'total'), mean('B0H+', 'trend')
    for m in models:
        total, trend = mean(m, 'total'), mean(m, 'trend')
        lines.append(f'| {m} | {total:.4f} | {trend:.4f} | {(total - pf_total) / pf_total * 100:+.2f}% | '
                     f'{(trend - pf_trend) / pf_trend * 100:+.2f}% | '
                     f'{(total - bh_total) / bh_total * 100:+.2f}% | {(trend - bh_trend) / bh_trend * 100:+.2f}% |')
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(out).write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-dir', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--untuned-dir', type=Path, default=DEFAULT_SCRATCH_UNTUNED)
    parser.add_argument('--dump-root', type=Path, default=DEFAULT_DUMP)
    parser.add_argument('--gpus', default='0,1,2,3,4,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--select-metric', default='val_mse', choices=('val_mse', 'val_trend'))
    parser.add_argument('--score-val-trend', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--cells', default=None, help='comma-separated tune tags to prepare/run')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/PhaseFormer_VLDBJ_trend_branches_tuned.md')
    args = parser.parse_args()
    if args.prepare:
        paths = build_jobs(args.scratch_dir)
        print(f'{len(paths)} jobs under {args.scratch_dir / "v1"}')
    if args.run:
        run(args.scratch_dir, args.gpus, args.slots_per_gpu, args.cells)
    if args.score_val_trend:
        score_val_trend(args.scratch_dir, args.untuned_dir)
    if args.select:
        select(args.scratch_dir, args.untuned_dir, args.select_metric)
    if args.export:
        export(args.scratch_dir, args.untuned_dir, args.dump_root)
    if args.report:
        report(args.scratch_dir, args.dump_root, args.out)


if __name__ == '__main__':
    main()
