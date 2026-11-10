#!/usr/bin/env python3
"""Ablate the amplitude pathway of PhaseFormer-LA at its best known settings.

Every setting keeps the configuration of its best known PhaseFormer-LA run
(cycle length, gate init, extension learning-rate multiplier, communication
width, backbone learning-rate multiplier, level head and gate sharing).  The
only change is the arm: ``m1`` -> ``b0``.  ``b0`` keeps the phase pathway and
the level pathway (the cycle-mean readout written onto the horizon profiles) but
drops ``IsolatedBranch.communication``, the centered multiplicative modulation
that scales the phase representation by a level-driven factor while leaving
every future cycle mean unchanged.  That modulation is the amplitude pathway of
the model, so the b0 run is LA minus amplitude.

Everything else is held fixed by the frozen worker: same E14 phase-only source
config, same seed (2021), same 30-epoch Huber protocol, same common random
state.  ``paired_models`` builds b0 first and copies its state into m1, so the
two models start from bit-identical shared parameters; ``--check-init`` asserts
this and that the only extra parameters are the modulation module.

Kept outside ``scripts/vldbj_m1`` so the frozen-protocol hash is untouched.

    python3 scripts/ablation_no_amplitude.py --prepare
    python3 scripts/ablation_no_amplitude.py --check-init
    python3 scripts/ablation_no_amplitude.py --run --gpus 1,6 --slots-per-gpu 2
    python3 scripts/ablation_no_amplitude.py --report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402
from scripts.vldbj_m1.wide import name as wide_name  # noqa: E402

SEED = 2021
DEFAULT_SCRATCH = ROOT / 'research_runs/ablation_no_amplitude_v1'

# dataset, horizon, period, gate_init, extension_lr_mult, width, backbone_lr_mult,
# level head, shared gate, m1 test MSE, m1 test MAE -- read from the seed-2021 run
# of exactly this configuration (same run, same read as the b0 arm).  Six settings
# are a little worse here than in the best-known table, which takes the best of
# three seeds; using the seed-2021 value keeps the ablation a matched pair.
SETTINGS = [
    ('ETTh1', 96, 24, 0.1, 0.1, 3, 1.0, 'S2', False, 0.360207, 0.387184),
    ('ETTh1', 192, 24, 0.1, 100.0, 2, 1.0, 'S2', False, 0.389171, 0.408935),
    ('ETTh1', 336, 24, 0.99, 1.0, 2, 1.0, 'S2', False, 0.419581, 0.426534),
    ('ETTh1', 720, 24, 0.1667, 3.0, 3, 1.0, 'S2', False, 0.417196, 0.437077),
    ('ETTh2', 96, 24, 0.999, 3.3333, 2, 1.0, 'S2', True, 0.273685, 0.334761),
    ('ETTh2', 192, 24, 0.1, 0.1, 4, 0.3, 'S2', True, 0.334671, 0.377117),
    ('ETTh2', 336, 24, 0.5, 1.0, 2, 1.0, 'S2', True, 0.362946, 0.403491),
    ('ETTh2', 720, 24, 0.999, 3.0, 2, 1.0, 'S2', True, 0.388919, 0.426404),
    ('ETTm1', 96, 48, 0.1, 3.0, 2, 1.0, 'S2', True, 0.285722, 0.338722),
    ('ETTm1', 192, 48, 0.1, 10.0, 4, 2.0, 'S2', True, 0.322985, 0.364021),
    ('ETTm1', 336, 48, 0.1, 10.0, 2, 1.0, 'S2', True, 0.350841, 0.380277),
    ('ETTm1', 720, 24, 0.0333, 10.0, 2, 1.0, 'S1', False, 0.407414, 0.412912),
    ('ETTm2', 96, 24, 0.99, 30.0, 2, 1.0, 'S3', False, 0.161133, 0.252045),
    ('ETTm2', 192, 24, 0.2, 1.0, 2, 1.0, 'S3', False, 0.217313, 0.292964),
    ('ETTm2', 336, 24, 0.2, 1.0, 2, 1.0, 'S1', True, 0.265274, 0.325022),
    ('ETTm2', 720, 24, 0.99, 30.0, 2, 1.0, 'S3', True, 0.343351, 0.378977),
    ('Weather', 96, 24, 0.999, 30.0, 2, 1.0, 'S2', False, 0.143889, 0.191407),
    ('Weather', 192, 24, 0.6, 3.3333, 2, 1.0, 'S2', False, 0.184392, 0.230291),
    ('Weather', 336, 24, 0.9, 10.0, 2, 1.0, 'S2', False, 0.234461, 0.272527),
    ('Weather', 720, 24, 0.2, 10.0, 3, 1.0, 'S2', False, 0.309591, 0.326370),
    ('Electricity', 96, 24, 0.001, 10.0, 2, 10.0, 'S2', False, 0.127767, 0.219876),
    ('Electricity', 192, 24, 0.001, 0.3, 2, 3.0, 'S2', False, 0.143955, 0.234892),
    ('Electricity', 336, 24, 0.1, 3.3333, 4, 1.0, 'S2', False, 0.164346, 0.256778),
    ('Electricity', 720, 24, 0.1667, 0.1, 2, 1.0, 'S2', False, 0.197208, 0.283181),
    ('Traffic', 96, 24, 0.5, 30.0, 2, 1.0, 'S2', False, 0.358660, 0.232938),
    ('Traffic', 192, 24, 0.1, 0.1, 4, 0.3, 'S2', False, 0.378097, 0.241848),
    ('Traffic', 336, 12, 0.3, 0.2, 2, 3.0, 'S2', False, 0.381990, 0.250858),
    ('Traffic', 720, 12, 0.99, 0.1, 2, 1.0, 'S2', False, 0.422935, 0.272938),
]
FIELDS = ('dataset', 'horizon', 'period', 'gate_init', 'extension_lr_mult', 'width',
          'backbone_lr_mult', 'head', 'shared', 'm1_mse', 'm1_mae')
ROWS = [dict(zip(FIELDS, row)) for row in SETTINGS]


def variant(row):
    return {'gate_init': row['gate_init'], 'extension_lr_mult': row['extension_lr_mult'],
            'communication_width': row['width']}


def m1_name(row):
    base = wide_name(row['dataset'], row['horizon'], SEED, variant(row),
                     row['backbone_lr_mult'], row['head'], row['shared'])
    return base + (f"__p{row['period']}" if row['period'] != 24 else '')


def b0_name(row):
    return m1_name(row).replace('__m1__', '__b0__') + '__noamp'


def build_jobs(scratch, stage, rows):
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for row in rows:
        source = sources[(row['dataset'], row['horizon'], SEED)]
        if row['period'] != 24:
            import copy
            source = copy.deepcopy(source)
            source['config']['period'] = row['period']
            source['config']['hyperparams']['period_len'] = row['period']
        job = dict(base, dataset=row['dataset'], horizon=row['horizon'], seed=SEED,
                   arm='b0', protocol='e14', init='scratch', head=row['head'],
                   shared_gate=row['shared'], variant_id=None, variant=variant(row),
                   extension_lr_mult=float(row['extension_lr_mult']),
                   backbone_lr_mult=float(row['backbone_lr_mult']),
                   warm_source=None, epoch_test=False, source=source,
                   ablation='no_amplitude', m1_reference={'name': m1_name(row),
                                                          'test_mse': row['m1_mse'],
                                                          'test_mae': row['m1_mae']})
        job['output'] = str(scratch / stage / 'runs' / b0_name(row))
        path = scratch / stage / 'jobs' / f"{Path(job['output']).name}.json"
        write_immutable(path, job)
        paths.append(path)
    order = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 4}
    return sorted(paths, key=lambda p: (order.get(p.name.split('_')[0], 9), p.name))


def check_init(rows):
    """b0 and m1 must share every parameter except the modulation module."""
    import torch
    from scripts.vldbj_m1.model import construct, prepare_config
    from scripts.vldbj_m1.proto import build_model
    sources = e14_sources()
    ok = True
    for row in rows:
        template = {'dataset': row['dataset'], 'horizon': row['horizon'], 'seed': SEED,
                    'head': row['head'], 'shared_gate': row['shared'], 'variant': variant(row),
                    'init': 'scratch', 'protocol': 'e14', 'static_q': False,
                    'source': sources[(row['dataset'], row['horizon'], SEED)]}
        if row['period'] != 24:
            import copy
            source = copy.deepcopy(template['source'])
            source['config']['period'] = row['period']
            source['config']['hyperparams']['period_len'] = row['period']
            template['source'] = source
        b0, _, _, _ = build_model(dict(template, arm='b0'))
        m1, _, _, _ = build_model(dict(template, arm='m1'))
        small, big = b0.state_dict(), m1.state_dict()
        missing = set(small) - set(big)
        shared_diff = [k for k in small if not torch.equal(small[k].float(), big[k].float())]
        extra = sorted(set(big) - set(small))
        only_modulation = all(k.startswith('lowrank_interaction_branch.communication.') for k in extra)
        shapes = all(big[k].shape == small[k].shape for k in small)
        good = not missing and not shared_diff and only_modulation and shapes
        ok &= good
        print(f"{row['dataset']:>11}-{row['horizon']:>3}: shared={len(small)} "
              f"identical={not shared_diff} extra={len(extra)} modulation_only={only_modulation} "
              f"{'' if good else 'FAIL'}")
        del b0, m1
    if not ok:
        raise SystemExit('initialization check failed')
    print('init check: b0 == m1 except the amplitude (modulation) parameters')


def collect(scratch, stage, rows):
    out = []
    for row in rows:
        run = scratch / stage / 'runs' / b0_name(row)
        record = run / 'result.json'
        item = dict(row, run=str(run), done=record.is_file(),
                    b0_mse=None, b0_mae=None, b0_done=record.is_file())
        if item['done']:
            best = json.loads(record.read_text())['best']
            item['b0_mse'], item['b0_mae'] = best['test_mse'], best['test_mae']
        out.append(item)
    return out


def report(scratch, stage, rows):
    items = collect(scratch, stage, rows)
    done = [i for i in items if i['done']]
    print(f"{'Dataset':>11} {'H':>4} | {'LA MSE':>8} {'no-Amp MSE':>10} {'d%':>7} | "
          f"{'LA MAE':>8} {'no-Amp MAE':>10} {'d%':>7}")
    for i in items:
        if not i['done']:
            print(f"{i['dataset']:>11} {i['horizon']:>4} | (not finished)")
            continue
        dm = (i['b0_mse'] - i['m1_mse']) / i['m1_mse'] * 100
        da = (i['b0_mae'] - i['m1_mae']) / i['m1_mae'] * 100
        print(f"{i['dataset']:>11} {i['horizon']:>4} | {i['m1_mse']:>8.5f} {i['b0_mse']:>10.5f} "
              f"{dm:>+6.2f}% | {i['m1_mae']:>8.5f} {i['b0_mae']:>10.5f} {da:>+6.2f}%")
    print()
    print(f"{'Dataset':>11} | {'LA MSE':>8} {'no-Amp MSE':>10} {'d%':>7} | "
          f"{'LA MAE':>8} {'no-Amp MAE':>10} {'d%':>7}")
    lines = []
    for dataset in dict.fromkeys(i['dataset'] for i in items):
        group = [i for i in items if i['dataset'] == dataset]
        if not all(i['done'] for i in group):
            continue
        a_mse = sum(i['m1_mse'] for i in group) / len(group)
        b_mse = sum(i['b0_mse'] for i in group) / len(group)
        a_mae = sum(i['m1_mae'] for i in group) / len(group)
        b_mae = sum(i['b0_mae'] for i in group) / len(group)
        row = (f"{dataset:>11} | {a_mse:>8.4f} {b_mse:>10.4f} "
               f"{(b_mse - a_mse) / a_mse * 100:>+6.2f}% | {a_mae:>8.4f} {b_mae:>10.4f} "
               f"{(b_mae - a_mae) / a_mae * 100:>+6.2f}%")
        lines.append((a_mse, b_mse, a_mae, b_mae, row))
        print(row)
    if lines:
        n = len(lines)
        a_mse = sum(x[0] for x in lines) / n
        b_mse = sum(x[1] for x in lines) / n
        a_mae = sum(x[2] for x in lines) / n
        b_mae = sum(x[3] for x in lines) / n
        print(f"{'7-dataset mean':>11} | {a_mse:>8.4f} {b_mse:>10.4f} "
              f"{(b_mse - a_mse) / a_mse * 100:>+6.2f}% | {a_mae:>8.4f} {b_mae:>10.4f} "
              f"{(b_mae - a_mae) / a_mae * 100:>+6.2f}%")
    if len(done) == len(items):
        wins = sum(i['b0_mse'] < i['m1_mse'] for i in items)
        print(f'\nremoving the amplitude pathway improves MSE in {wins}/{len(items)} settings')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--stage', default='ablation')
    parser.add_argument('--gpus', default='1,6')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-init', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--datasets', default=None,
                        help='comma-separated dataset filter, e.g. ETTh1,ETTh2')
    args = parser.parse_args()

    rows = ROWS
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(',')}
        rows = [r for r in rows if r['dataset'] in keep]

    if args.check_init:
        check_init(rows)
        return
    if args.prepare or args.run:
        args.scratch.mkdir(parents=True, exist_ok=True)
        paths = build_jobs(args.scratch, args.stage, rows)
        print(f'{len(paths)} jobs under {args.scratch / args.stage}')
        if args.run:
            gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
            print('launching on', gpus, flush=True)
            for attempt in (1, 2):
                try:
                    run_jobs([str(p) for p in paths], gpus)
                    break
                except RuntimeError as exc:
                    print(f'pass {attempt} ended with failures: {exc}', flush=True)
                    if attempt == 2:
                        raise
    if args.report or args.run:
        report(args.scratch, args.stage, rows)


if __name__ == '__main__':
    main()
