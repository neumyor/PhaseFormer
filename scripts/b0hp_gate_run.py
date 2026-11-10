#!/usr/bin/env python3
"""Can a free-moving gate make B0H+ robust without per-setting tuning?

The variant run showed that reading the phase token before LayerNorm helps where the
level branch is used (ETTh2-720: +3.89% -> +0.44%) and hurts where the best b0 cell
turns the level branch off (ETTh1-96: +1.19% -> +4.26%).  Under the frozen protocol the
gate barely moves, so B0H+ cannot shut the richer level path down again.

This round keeps the raw token and gives the gate its own large learning rate (the
split optimiser from ``lr_split_run``), so the gate is free to reopen or close the level
branch per setting.  Two cells per setting, three seeds.

    python3 scripts/b0hp_gate_run.py --run --gpus 1,2,3,7 --slots-per-gpu 2
    python3 scripts/b0hp_gate_run.py --report

Kept outside ``scripts/vldbj_m1``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/b0hp_gate_v1'
STAGE = 'v1'
SETTINGS = (('ETTh1', 96), ('ETTh2', 720), ('Weather', 720))
SEEDS = (2021, 2022, 2023)
# (tag, gate multiplier, readout multiplier); backbone stays at the best cell's value
CELLS = (('gate100_read100', 100.0, 100.0), ('gate100_read10', 100.0, 10.0))


def name(dataset, horizon, seed, tag, best):
    return (f"{dataset}_H{horizon}_S{seed}__{tag}__g{best['gate']:g}_bb{best['bb']:g}"
            f"__p{best['period']}__{best['head']}__{'shared' if best['shared'] else 'perC'}")


def build_jobs(scratch):
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    results = load_results(B0)
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for dataset, horizon in SETTINGS:
        best = winners(results, dataset, horizon)
        for seed in SEEDS:
            source = sources[(dataset, horizon, seed)]
            for tag, gate_mult, read_mult in CELLS:
                job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='b0',
                           protocol='e14', init='scratch', head=best['head'],
                           shared_gate=bool(best['shared']), variant_id=None,
                           variant={'gate_init': best['gate'], 'extension_lr_mult': gate_mult,
                                    'communication_width': 2},
                           extension_lr_mult=float(gate_mult), backbone_lr_mult=float(best['bb']),
                           warm_source=None, epoch_test=False, source=source,
                           level_input='history+token', token_source='raw',
                           lr_split={'gate_mult': float(gate_mult), 'readout_mult': float(read_mult)},
                           b0_reference={'run': best['run'], 'mse': best['mse']})
                n = name(dataset, horizon, seed, tag, best)
                job['output'] = str(scratch / STAGE / 'runs' / n)
                path = scratch / STAGE / 'jobs' / f'{n}.json'
                write_immutable(path, job)
                paths.append(path)
    return paths


def report(scratch):
    from scripts.b0hp_variant_run import DEFAULT_SCRATCH as VAR, STAGE as VAR_STAGE, name as var_name, SEEDS as VSEEDS
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    results = load_results(B0)
    print(f"{'setting':>13} | {'b0':>8} {'stock':>8} {'raw':>8} | " +
          ' '.join(f'{t:>16}' for t, _, _ in CELLS))
    mean = {t: [] for t in ('b0', 'stock', 'raw')}
    gmean = {t: [] for t, _, _ in CELLS}
    for dataset, horizon in SETTINGS:
        best = winners(results, dataset, horizon)

        def avg(record_of, seeds):
            vals = []
            for s in seeds:
                r = record_of(s)
                if r.is_file():
                    vals.append(json.loads(r.read_text())['best']['test_mse'])
            return sum(vals) / len(vals) if len(vals) == len(seeds) else None

        v = {t: avg((lambda s, t=t: VAR / VAR_STAGE / 'runs' / var_name(dataset, horizon, s, t, best)
                     / 'result.json'), VSEEDS) for t in ('b0', 'stock', 'raw')}
        g = {t: avg((lambda s, t=t: scratch / STAGE / 'runs' / name(dataset, horizon, s, t, best)
                     / 'result.json'), SEEDS) for t, _, _ in CELLS}
        if v['b0']:
            mean['b0'].append(v['b0'])
        for t in ('stock', 'raw'):
            if v[t]:
                mean[t].append(v[t])
        for t in g:
            if g[t]:
                gmean[t].append(g[t])
        cells = ' '.join(f'{g[t]:.4f} ({(g[t] - v["b0"]) / v["b0"] * 100:+.2f}%)' if g[t] and v['b0'] else f"{'-':>16}"
                         for t, _, _ in CELLS)
        ref = f"{v['b0']:.4f}" if v['b0'] else '-'
        left = (f"{v['stock']:.4f} {v['raw']:.4f}" if v['stock'] and v['raw'] else f"{'-':>8} {'-':>8}")
        print(f"{dataset + '-' + str(horizon):>13} | {ref:>8} {left} | {cells}")
    if mean['b0']:
        b0m = sum(mean['b0']) / len(mean['b0'])
        print(f"\nthree-setting mean: b0 {b0m:.4f}, stock "
              f"{(sum(mean['stock']) / len(mean['stock']) - b0m) / b0m * 100:+.2f}%, raw "
              f"{(sum(mean['raw']) / len(mean['raw']) - b0m) / b0m * 100:+.2f}%, " +
              ', '.join(f"{t} {(sum(gmean[t]) / len(gmean[t]) - b0m) / b0m * 100:+.2f}%"
                        for t, _, _ in CELLS if gmean[t]))


def main():
    from scripts.lr_split_run import run_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='1,2,3,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report(args.scratch)
        return
    paths = [str(p) for p in build_jobs(args.scratch)]
    print(f'{len(paths)} jobs under {args.scratch / STAGE}', flush=True)
    if args.run:
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        run_jobs(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
