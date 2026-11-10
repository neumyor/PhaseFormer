#!/usr/bin/env python3
"""Ablate the level pathway of PhaseFormer-LA at its best known settings.

Same 28 settings and the same paired seed-2021 comparison as
``ablation_no_amplitude.py``.  The arm stays ``m1``: the phase pathway, the
cycle-mean encoder and the amplitude modulation all remain.  The level readout
-- ``lowrank_head.decode_latent``, the cycle means written onto the three
horizon profiles and fused with the phase forecast -- is forced to zero, so the
forecast can no longer take its level from that pathway.  The encoder still
feeds the amplitude modulation, which is what separates this from the
amplitude ablation.

Initialization is unchanged by construction: the ablation adds no parameters and
removes none, so every parameter of the two models is bit-identical at start.

    python3 scripts/ablation_no_level.py --check-init
    python3 scripts/ablation_no_level.py --prepare
    python3 scripts/ablation_no_level.py --run --gpus 1,6 --slots-per-gpu 3
    python3 scripts/ablation_no_level.py --report
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation_no_amplitude import ROWS, SEED, m1_name, variant  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402

DEFAULT_SCRATCH = ROOT / 'research_runs/ablation_no_level_v1'
AMPLITUDE_SCRATCH = ROOT / 'research_runs/ablation_no_amplitude_v1'


def nl_name(row):
    return m1_name(row) + '__nolvl'


def build_jobs(scratch, stage, rows):
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for row in rows:
        source = sources[(row['dataset'], row['horizon'], SEED)]
        if row['period'] != 24:
            source = copy.deepcopy(source)
            source['config']['period'] = row['period']
            source['config']['hyperparams']['period_len'] = row['period']
        job = dict(base, dataset=row['dataset'], horizon=row['horizon'], seed=SEED,
                   arm='m1', protocol='e14', init='scratch', head=row['head'],
                   shared_gate=row['shared'], variant_id=None, variant=variant(row),
                   extension_lr_mult=float(row['extension_lr_mult']),
                   backbone_lr_mult=float(row['backbone_lr_mult']),
                   warm_source=None, epoch_test=False, source=source,
                   level_zero=True, ablation='no_level',
                   m1_reference={'name': m1_name(row), 'test_mse': row['m1_mse'],
                                 'test_mae': row['m1_mae']})
        job['output'] = str(scratch / stage / 'runs' / nl_name(row))
        path = scratch / stage / 'jobs' / f"{Path(job['output']).name}.json"
        write_immutable(path, job)
        paths.append(path)
    order = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 4}
    return sorted(paths, key=lambda p: (order.get(p.name.split('_')[0], 9), p.name))


def check_init(rows):
    """The ablation adds and removes nothing; only the forward changes."""
    import torch
    from scripts.vldbj_m1.proto import build_model
    sources = e14_sources()
    for row in rows:
        source = sources[(row['dataset'], row['horizon'], SEED)]
        if row['period'] != 24:
            source = copy.deepcopy(source)
            source['config']['period'] = row['period']
            source['config']['hyperparams']['period_len'] = row['period']
        template = {'dataset': row['dataset'], 'horizon': row['horizon'], 'seed': SEED,
                    'head': row['head'], 'shared_gate': row['shared'],
                    'variant': variant(row), 'init': 'scratch', 'protocol': 'e14',
                    'source': source}
        full, args, _, _ = build_model(dict(template, arm='m1'))
        cut, _, _, _ = build_model(dict(template, arm='m1', level_zero=True))
        a, b = full.state_dict(), cut.state_dict()
        same_keys = set(a) == set(b)
        identical = same_keys and all(torch.equal(a[k], b[k]) for k in a)
        torch.manual_seed(0)
        bx = torch.randn(2, full.seq_len, full.enc_in)
        bxm = torch.zeros(2, full.seq_len, 4)
        by = torch.randn(2, full.seq_len + full.pred_len, full.enc_in)
        bym = torch.zeros(2, full.seq_len + full.pred_len, 4)
        full.eval(); cut.eval()
        with torch.inference_mode():
            y_full = full(bx, bxm, full._build_decoder_input(by), bym)[0][:, -full.pred_len:]
            y_cut = cut(bx, bxm, cut._build_decoder_input(by), bym)[0][:, -cut.pred_len:]
        moved = not torch.equal(y_full, y_cut)
        flag = '' if (same_keys and identical and moved) else 'FAIL'
        print(f"{row['dataset']:>11}-{row['horizon']:>3}: params={sum(p.numel() for p in full.parameters())} "
              f"same_keys={same_keys} identical={identical} forecast_changes={moved} {flag}")
        if flag:
            raise SystemExit('level ablation check failed')
        del full, cut
    print('init check: identical parameters, zeroed level readout changes the forecast')


def collect(scratch, stage, rows):
    items = []
    for row in rows:
        run = scratch / stage / 'runs' / nl_name(row)
        record = run / 'result.json'
        item = dict(row, done=record.is_file())
        if item['done']:
            best = json.loads(record.read_text())['best']
            item['mse'], item['mae'] = best['test_mse'], best['test_mae']
        items.append(item)
    return items


def amplitude_lookup(rows):
    out = {}
    for row in rows:
        name = m1_name(row).replace('__m1__', '__b0__') + '__noamp'
        record = AMPLITUDE_SCRATCH / 'ablation' / 'runs' / name / 'result.json'
        if record.is_file():
            best = json.loads(record.read_text())['best']
            out[(row['dataset'], row['horizon'])] = (best['test_mse'], best['test_mae'])
    return out


def report(scratch, stage, rows):
    items = collect(scratch, stage, rows)
    amp = amplitude_lookup(rows)
    print(f"{'Dataset':>11} {'H':>4} | {'LA':>8} {'-amplitude':>11} {'d%':>7} | "
          f"{'-level':>8} {'d%':>7}")
    for i in items:
        key = (i['dataset'], i['horizon'])
        a = amp.get(key)
        if not i['done']:
            print(f"{i['dataset']:>11} {i['horizon']:>4} | (not finished)")
            continue
        left = f"{a[0]:>11.5f} {(a[0] - i['m1_mse']) / i['m1_mse'] * 100:>+6.2f}%" if a else f"{'—':>11} {'—':>7}"
        print(f"{i['dataset']:>11} {i['horizon']:>4} | {i['m1_mse']:>8.5f} {left} | "
              f"{i['mse']:>8.5f} {(i['mse'] - i['m1_mse']) / i['m1_mse'] * 100:>+6.2f}%")
    print()
    print(f"{'Dataset':>11} | {'LA':>8} {'-amplitude':>11} {'d%':>7} | {'-level':>8} {'d%':>7}")
    rows_out = []
    for dataset in dict.fromkeys(i['dataset'] for i in items):
        group = [i for i in items if i['dataset'] == dataset]
        if not all(i['done'] for i in group):
            continue
        la = sum(i['m1_mse'] for i in group) / len(group)
        nl = sum(i['mse'] for i in group) / len(group)
        am = [amp[(i['dataset'], i['horizon'])][0] for i in group if (i['dataset'], i['horizon']) in amp]
        am = sum(am) / len(am) if len(am) == len(group) else None
        left = f"{am:>11.4f} {(am - la) / la * 100:>+6.2f}%" if am else f"{'—':>11} {'—':>7}"
        print(f"{dataset:>11} | {la:>8.4f} {left} | {nl:>8.4f} {(nl - la) / la * 100:>+6.2f}%")
        rows_out.append((la, am, nl))
    if rows_out and all(r[1] for r in rows_out):
        n = len(rows_out)
        la, am, nl = (sum(r[k] for r in rows_out) / n for k in range(3))
        print(f"{'7-dataset mean':>11} | {la:>8.4f} {am:>11.4f} {(am - la) / la * 100:>+6.2f}% | "
              f"{nl:>8.4f} {(nl - la) / la * 100:>+6.2f}%")


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
    parser.add_argument('--datasets', default=None)
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
