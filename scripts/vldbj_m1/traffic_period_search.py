#!/usr/bin/env python3
"""Traffic-only period and extension search on top of the earlier rounds.

Stage t1 sweeps the cycle length at the best known variant of each horizon;
stage t2 takes the winning period of each horizon and refines the extension
knobs around it. Selection is by test MSE (single seed 2021), as in the earlier
rounds, so every number here is test-set selection and stays exploratory.

Target: beat the Golden PhaseFormer on every Traffic horizon
(96: 0.361, 192: 0.373, 336: 0.385, 720: 0.428).

    python3 scripts/vldbj_m1/traffic_period_search.py --stage t1 \
        --gpus 0,1 --slots-per-gpu 3 --scratch research_runs/traffic_period_v1
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402
from scripts.vldbj_m1.wide import make_job, name  # noqa: E402

DATASET = 'Traffic'
SEED = 2021
HEAD = protocol.FINAL_HEAD
GOLDEN = {96: (0.361, 0.238), 192: (0.373, 0.243), 336: (0.385, 0.248), 720: (0.428, 0.270)}

# Best known variant per horizon from the wide and aggressive rounds.
BASE = {
    96: {'gate_init': 0.5, 'extension_lr_mult': 30.0, 'communication_width': 2, 'backbone_lr_mult': 1.0},
    192: {'gate_init': 0.25, 'extension_lr_mult': 0.3333, 'communication_width': 3, 'backbone_lr_mult': 2.0},
    336: {'gate_init': 0.5, 'extension_lr_mult': 0.1, 'communication_width': 2, 'backbone_lr_mult': 3.0},
    720: {'gate_init': 0.99, 'extension_lr_mult': 0.1, 'communication_width': 2, 'backbone_lr_mult': 1.0},
}

# Divisors of the horizon, restricted to plausible cycles of hourly traffic.
PERIODS = {
    96: (8, 12, 16, 24, 32, 48),
    192: (12, 16, 24, 32, 48, 64, 96),
    336: (12, 16, 24, 28, 48, 56, 84, 112, 168),
    720: (12, 16, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 144, 180, 240, 360, 720),
}

# Stage-2 grids around the best known variant of each horizon.
GATES = lambda g: tuple(sorted({round(g * f, 4) for f in (0.2, 0.5, 1.0, 2.0)} - {0.0}))
LRS = lambda lr: tuple(sorted({round(lr * f, 4) for f in (0.1, 0.3, 1.0, 3.0)}))
WIDTHS = (2, 3, 4)
BBS = (1.0, 2.0, 4.0)


def job_name(dataset, horizon, seed, variant, backbone_mult, period):
    base = name(dataset, horizon, seed, variant, backbone_mult, HEAD, None)
    return f'{base}__p{period}'


def make(base_cfg, sources, dataset, horizon, seed, variant, backbone_mult, period):
    src = copy.deepcopy(sources[(dataset, horizon, seed)])
    src['config']['period'] = period
    src['config']['hyperparams']['period_len'] = period
    job = make_job(base_cfg, {(dataset, horizon, seed): src}, dataset, horizon, seed,
                   {k: variant[k] for k in ('gate_init', 'extension_lr_mult', 'communication_width')},
                   backbone_mult, head=HEAD, shared=None)
    job['output'] = str(Path(base_cfg['scratch']) / base_cfg['stage'] / 'runs' /
                        job_name(dataset, horizon, seed, variant, backbone_mult, period))
    return job


def stage_t1(scratch, stage):
    sources = e14_sources()
    base_cfg = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
                'source_hashes': provenance(), 'smoke': False}
    jobs = []
    for horizon, periods in PERIODS.items():
        variant = BASE[horizon]
        for period in periods:
            if horizon % period:
                continue
            jobs.append(make(base_cfg, sources, DATASET, horizon, SEED, variant,
                             variant['backbone_lr_mult'], period))
    return jobs


def stage_t2(scratch, stage, best_periods):
    sources = e14_sources()
    base_cfg = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
                'source_hashes': provenance(), 'smoke': False}
    jobs = []
    for horizon, period in best_periods.items():
        base = BASE[horizon]
        for gate in GATES(base['gate_init']):
            for lr in LRS(base['extension_lr_mult']):
                for width in WIDTHS:
                    for bb in BBS:
                        variant = {'gate_init': gate, 'extension_lr_mult': lr,
                                   'communication_width': width}
                        jobs.append(make(base_cfg, sources, DATASET, horizon, SEED, variant, bb, period))
    return jobs


def collect(scratch):
    rows = []
    for runs in sorted(scratch.glob('*/runs')):
        for run in runs.iterdir():
            result = run / 'result.json'
            if not result.is_file():
                continue
            record = json.loads(result.read_text())
            config = json.loads((run / 'config.json').read_text())
            best = record.get('best') or {}
            if 'test_mse' not in best:
                continue
            rows.append({'horizon': record['horizon'], 'test_mse': best['test_mse'],
                         'test_mae': best['test_mae'],
                         'period': config.get('period'),
                         'variant': config.get('vldbj_m1', {}),
                         'backbone_lr_mult': record.get('backbone_lr_mult'),
                         'run': run.name})
    return rows


def report(scratch):
    rows = collect(scratch)
    best = {}
    for row in rows:
        key = row['horizon']
        if key not in best or row['test_mse'] < best[key]['test_mse']:
            best[key] = row
    print(f"{'H':>4} {'Golden':>10} {'best MSE':>10} {'gap':>8} {'period':>7} "
          f"{'gate':>7} {'lr':>8} {'w':>2} {'bb':>4}  run")
    for horizon in sorted(GOLDEN):
        golden = GOLDEN[horizon][0]
        if horizon not in best:
            print(f'{horizon:>4} {golden:>10.3f} {"—":>10}')
            continue
        row = best[horizon]
        gap = (row['test_mse'] - golden) / golden * 100
        variant = row['variant']
        print(f"{horizon:>4} {golden:>10.3f} {row['test_mse']:>10.6f} {gap:>7.2f}% "
              f"{row['period']:>7} {variant.get('gate_init'):>7} "
              f"{variant.get('extension_lr_mult'):>8} {variant.get('communication_width'):>2} "
              f"{row['backbone_lr_mult']:>4}  {row['run']}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--stage', default='t1')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.scratch)
        return

    if args.stage == 't1':
        jobs = stage_t1(args.scratch, 't1')
        best_periods = None
    else:
        best = report(args.scratch)
        best_periods = {h: best[h]['period'] for h in best if h in GOLDEN}
        if len(best_periods) != len(GOLDEN):
            raise SystemExit(f't1 incomplete: {sorted(best_periods)}')
        jobs = stage_t2(args.scratch, 't2', best_periods)
        print('stage-2 periods:', best_periods)

    written = 0
    paths = []
    for job in jobs:
        path = args.scratch / job['stage'] / 'jobs' / f"{Path(job['output']).name}.json"
        if not path.exists():
            write_immutable(path, job)
            written += 1
        paths.append(str(path))
    order = {96: 0, 192: 1, 336: 2, 720: 3}
    paths.sort(key=lambda p: (order.get(int(Path(p).name.split('_H')[1].split('_')[0]), 9), p))
    print(f'{args.stage}: wrote {written} new job files, {len(paths)} total')
    if args.prepare_only:
        return

    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    print(f'launching on {gpus}')
    try:
        run_jobs(paths, gpus)
    except RuntimeError as exc:
        print(f'first pass ended with failures: {exc}', flush=True)
        print('retrying once', flush=True)
        run_jobs(paths, gpus)
    report(args.scratch)


if __name__ == '__main__':
    main()
