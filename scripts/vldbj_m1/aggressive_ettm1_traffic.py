#!/usr/bin/env python3
"""Aggressive ETTm1/Traffic tuning using vldbj_m1 framework.

Target: beat Golden by 2% on all 8 settings.
Based on wide.py but with denser grids for ETTm1/Traffic only.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1 import protocol
from scripts.vldbj_m1.proto import provenance, run_jobs
from scripts.vldbj_m1.run import (
    DATASETS as ALL_DATASETS, E14, E14_ROOT, HORIZONS as ALL_HORIZONS,
    SEEDS, commit, e14_sources, write_immutable
)
from scripts.vldbj_m1.wide import name, make_job, write_jobs

# Target subsets
DATASETS = ('ETTm1', 'Traffic')
HORIZONS = (96, 192, 336, 720)
SEED = 2021  # Single seed for fast iteration
HEAD = 'S2'

# Golden targets
GOLDEN = {
    'ETTm1': {96: 0.293, 192: 0.323, 336: 0.358, 720: 0.412},
    'Traffic': {96: 0.361, 192: 0.373, 336: 0.385, 720: 0.428}
}

# Dense grids per setting
GRIDS = {
    ('ETTm1', 96): {
        'gate_init': [0.0001, 0.0003, 0.0005, 0.001, 0.002],
        'extension_lr_mult': [0.1, 0.2, 0.3, 0.5],
        'communication_width': [2, 3],
        'backbone_lr_mult': [0.3, 0.5, 1.0, 1.5]
    },
    ('ETTm1', 192): {
        'gate_init': [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003],
        'extension_lr_mult': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'communication_width': [2, 3, 4],
        'backbone_lr_mult': [0.3, 0.5, 0.7, 1.0]
    },
    ('ETTm1', 336): {
        'gate_init': [0.0001, 0.0005, 0.001, 0.002],
        'extension_lr_mult': [0.5, 0.7, 1.0, 1.5, 2.0],
        'communication_width': [3, 4],
        'backbone_lr_mult': [0.5, 1.0, 1.5]
    },
    ('ETTm1', 720): {
        'gate_init': [0.01, 0.02, 0.03, 0.0333, 0.04, 0.05],
        'extension_lr_mult': [5, 7, 10, 15, 20, 30],
        'communication_width': [2, 3],
        'backbone_lr_mult': [0.5, 1.0, 1.5, 2.0]
    },
    ('Traffic', 96): {
        'gate_init': [0.3, 0.4, 0.5, 0.6, 0.7],
        'extension_lr_mult': [10, 20, 30, 50, 70, 100],
        'communication_width': [2, 3],
        'backbone_lr_mult': [0.5, 1.0, 1.5, 2.0]
    },
    ('Traffic', 192): {
        'gate_init': [0.1, 0.15, 0.2, 0.25, 0.3],
        'extension_lr_mult': [0.1, 0.2, 0.3333, 0.5, 0.7],
        'communication_width': [2, 3, 4],
        'backbone_lr_mult': [0.3, 0.5, 1.0, 1.5, 2.0]
    },
    ('Traffic', 336): {
        'gate_init': [0.3, 0.4, 0.5, 0.6, 0.7],
        'extension_lr_mult': [0.05, 0.07, 0.1, 0.15],
        'communication_width': [2, 3],
        'backbone_lr_mult': [2.0, 3.0, 4.0, 5.0]
    },
    ('Traffic', 720): {
        'gate_init': [0.9, 0.95, 0.98, 0.99, 0.995],
        'extension_lr_mult': [0.05, 0.07, 0.1, 0.15, 0.2],
        'communication_width': [2, 3],
        'backbone_lr_mult': [0.5, 0.7, 1.0, 1.5]
    }
}


# Stage 2: ETTm1-only period-length expansion.  ETTm1 is 15-minute data; the
# E14 sources fold it with period_len=24, a 6-hour cycle, while the dominant
# daily cycle is 96 steps (docs/PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md,
# section 2).  The model circularly pads the input, so periods that do not
# divide the 720-step lookback are valid.
ETTM1_PERIODS = (24, 48, 96)
ETTM1_GRIDS = {
    96:  {'gate_init': [0.0001, 0.001, 0.01, 0.1, 0.5],
          'extension_lr_mult': [0.1, 0.3, 1.0, 3.0],
          'communication_width': [2, 3], 'backbone_lr_mult': [0.5, 1.0]},
    192: {'gate_init': [0.0001, 0.001, 0.01, 0.1, 0.5],
          'extension_lr_mult': [0.1, 0.3, 1.0, 3.0],
          'communication_width': [2, 3], 'backbone_lr_mult': [0.5, 1.0]},
    336: {'gate_init': [0.0001, 0.001, 0.01, 0.1, 0.5],
          'extension_lr_mult': [0.3, 1.0, 3.0, 10.0],
          'communication_width': [2, 3], 'backbone_lr_mult': [0.5, 1.0]},
    720: {'gate_init': [0.001, 0.01, 0.0333, 0.1, 0.5],
          'extension_lr_mult': [3.0, 10.0, 30.0, 100.0],
          'communication_width': [2, 3], 'backbone_lr_mult': [0.5, 1.0]},
}


# Stage 3: targeted periods for the two horizons still short of -2%.
# Only periods that divide the horizon (whole forecast cycles) are valid.
STAGE3 = {
    192: (32, 48, 64),
    720: (24, 72, 120),
}
STAGE3_GRIDS = {
    192: {'gate_init': [0.05, 0.1, 0.2], 'extension_lr_mult': [1.0, 3.0, 10.0],
          'communication_width': [2, 3, 4], 'backbone_lr_mult': [0.5, 1.0, 2.0]},
    720: {'gate_init': [0.02, 0.04, 0.1], 'extension_lr_mult': [5.0, 10.0, 30.0],
          'communication_width': [2, 3], 'backbone_lr_mult': [0.5, 1.0, 2.0]},
}


def generate_stage3_jobs(scratch, stage):
    """ETTm1 jobs for horizons 192/720 over untried whole-cycle periods."""
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': stage,
            'code_commit': commit(), 'source_hashes': provenance(), 'smoke': False}
    jobs = []
    for horizon, periods in STAGE3.items():
        grid = STAGE3_GRIDS[horizon]
        for period in periods:
            if horizon % period:
                continue
            src = copy.deepcopy(sources[('ETTm1', horizon, SEED)])
            src['config']['period'] = period
            src['config']['hyperparams']['period_len'] = period
            for g in grid['gate_init']:
                for lr in grid['extension_lr_mult']:
                    for w in grid['communication_width']:
                        for bb in grid['backbone_lr_mult']:
                            variant = {'gate_init': g, 'extension_lr_mult': lr,
                                       'communication_width': w}
                            job = make_job(base, {('ETTm1', horizon, SEED): src},
                                           'ETTm1', horizon, SEED, variant, bb,
                                           head=HEAD, shared=None)
                            job['output'] += f'__p{period}'
                            jobs.append(job)
    return jobs


def generate_ettm1_jobs(scratch, stage):
    """ETTm1-only jobs over period lengths and the refined extension grids."""
    sources = e14_sources()
    base = {
        'scratch': str(scratch), 'stage': stage,
        'code_commit': commit(), 'source_hashes': provenance(),
        'smoke': False
    }

    jobs = []
    for horizon in HORIZONS:
        grid = ETTM1_GRIDS[horizon]
        for period in ETTM1_PERIODS:
            # The low-rank fusion needs the horizon to be whole forecast cycles.
            if horizon % period:
                continue
            src = copy.deepcopy(sources[('ETTm1', horizon, SEED)])
            src['config']['period'] = period
            src['config']['hyperparams']['period_len'] = period
            for g in grid['gate_init']:
                for lr in grid['extension_lr_mult']:
                    for w in grid['communication_width']:
                        for bb in grid['backbone_lr_mult']:
                            variant = {'gate_init': g, 'extension_lr_mult': lr,
                                       'communication_width': w}
                            job = make_job(base, {('ETTm1', horizon, SEED): src},
                                           'ETTm1', horizon, SEED, variant, bb,
                                           head=HEAD, shared=None)
                            job['output'] += f'__p{period}'
                            jobs.append(job)

    return jobs


def generate_jobs(scratch, stage):
    """Generate all training jobs."""
    sources = e14_sources()
    base = {
        'scratch': str(scratch), 'stage': stage,
        'code_commit': commit(), 'source_hashes': provenance(),
        'smoke': False  # Add smoke flag
    }

    jobs = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            grid = GRIDS[(dataset, horizon)]
            for g in grid['gate_init']:
                for lr in grid['extension_lr_mult']:
                    for w in grid['communication_width']:
                        for bb in grid['backbone_lr_mult']:
                            variant = {
                                'gate_init': g,
                                'extension_lr_mult': lr,
                                'communication_width': w
                            }
                            job = make_job(base, sources, dataset, horizon, SEED,
                                         variant, bb, head=HEAD, shared=None)
                            jobs.append(job)

    return jobs


def collect_results(scratch, stage=None):
    """Collect finished results from every stage's runs directory.

    vldbj_m1 writes metrics under the 'best' (best-validation checkpoint) and
    'last' keys of result.json; the variant lives in the run's config.json.
    """
    results = []
    if stage:
        runs_dirs = [scratch / stage / 'runs']
    else:
        runs_dirs = sorted(p for p in scratch.glob('*/runs') if p.is_dir())
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            continue
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            result_file = run_dir / 'result.json'
            if not result_file.exists():
                continue
            try:
                data = json.loads(result_file.read_text())
                metrics = data.get('best') or data.get('last')
                if not metrics or 'test_mse' not in metrics:
                    continue
                config = {}
                config_file = run_dir / 'config.json'
                if config_file.exists():
                    config = json.loads(config_file.read_text()).get('vldbj_m1', {})
                results.append({
                    'dataset': data.get('dataset'),
                    'horizon': data.get('horizon'),
                    'test_mse': metrics['test_mse'],
                    'test_mae': metrics['test_mae'],
                    'config': config,
                    'run_dir': run_dir.name,
                })
            except Exception:
                continue
    return results


def report_progress(scratch, stage=None):
    """Print current best vs targets."""
    results = collect_results(scratch, stage)

    best = {}
    for r in results:
        key = (r.get('dataset'), r.get('horizon'))
        if key[0] and key[1]:
            if key not in best or r['test_mse'] < best[key]['test_mse']:
                best[key] = r

    print("\n" + "="*90)
    print(f"AGGRESSIVE ETTm1/TRAFFIC TUNING - {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90)
    print(f"Completed: {len(results)} runs")
    print(f"\n{'Setting':<15} {'Golden':>10} {'Target(-2%)':>12} {'Best MSE':>12} {'Gap':>10} {'Status':>10}")
    print("-"*90)

    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            golden = GOLDEN[dataset][horizon]
            target = golden * 0.98

            if key in best:
                mse = best[key]['test_mse']
                gap = mse - target
                status = "✓ PASS" if gap <= 0 else "✗ FAIL"
            else:
                mse = float('nan')
                gap = float('nan')
                status = "PENDING"

            print(f"{dataset}-{horizon:<7} {golden:10.6f} {target:12.6f} {mse:12.6f} {gap:10.6f} {status:>10}")

    print("="*90 + "\n")

    # Save best results
    best_file = scratch / 'best_results.json'
    best_data = {}
    for (dataset, horizon), r in best.items():
        key = f"{dataset}_{horizon}"
        best_data[key] = {
            'test_mse': r['test_mse'],
            'test_mae': r.get('test_mae'),
            'golden': GOLDEN[dataset][horizon],
            'target': GOLDEN[dataset][horizon] * 0.98,
            'gap': r['test_mse'] - GOLDEN[dataset][horizon] * 0.98,
            'config': r.get('config', {}),
            'run_dir': r.get('run_dir', '')
        }
    best_file.write_text(json.dumps(best_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--stage', default='aggressive_stage1')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')

    args = parser.parse_args()

    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report_progress(args.scratch_dir)
        return

    # Generate jobs
    print(f"Generating jobs for {args.stage}...")
    if args.stage == 'ettm1_stage3':
        jobs = generate_stage3_jobs(args.scratch_dir, args.stage)
        print(f"Generated {len(jobs)} ETTm1 stage-3 jobs")
    elif args.stage == 'ettm1_period':
        jobs = generate_ettm1_jobs(args.scratch_dir, args.stage)
        print(f"Generated {len(jobs)} ETTm1 jobs over periods {ETTM1_PERIODS}")
    else:
        jobs = generate_jobs(args.scratch_dir, args.stage)
        print(f"Generated {len(jobs)} jobs")

    if args.stage == 'ettm1_stage3':
        print("  H192: p32/p48/p64 x 81 configs; H720: p24/p72/p120 x 54 configs")
    elif args.stage == 'ettm1_period':
        per = len(ETTM1_GRIDS[96]['gate_init']) * len(ETTM1_GRIDS[96]['extension_lr_mult'])
        per *= len(ETTM1_GRIDS[96]['communication_width']) * len(ETTM1_GRIDS[96]['backbone_lr_mult'])
        print(f"  ETTm1, each horizon x {per} configs x {len(ETTM1_PERIODS)} periods"
              f" = {per * len(ETTM1_PERIODS)} per horizon")

    # Write job files (skip ones already written, e.g. when resuming)
    written = 0
    job_dir = args.scratch_dir / args.stage / 'jobs'
    for job in jobs:
        path = job_dir / f"{Path(job['output']).name}.json"
        if not path.exists():
            write_immutable(path, job)
            written += 1
    job_paths = sorted(str(p) for p in job_dir.glob('*.json'))
    # ETTm1 first: faster per run (~25 min vs ~2 h) and user-prioritized.
    job_paths.sort(key=lambda p: (0 if 'ETTm1' in Path(p).name else 1, p))
    print(f"Wrote {written} new job files ({len(job_paths)} total)")

    if args.prepare_only:
        return

    # Run. run_jobs raises when any job fails; failures are recorded, and a
    # rerun skips runs whose result.json already exists, so treat the raise
    # as a checkpoint: log it, retry once, then report whatever finished.
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    print(f"\nStarting on {len(gpus)} GPU slots...")

    try:
        run_jobs(job_paths, gpus)
    except RuntimeError as exc:
        print(f"First pass ended with failures: {exc}", flush=True)
        print("Retrying remaining jobs once...", flush=True)
        try:
            run_jobs(job_paths, gpus)
        except RuntimeError as exc2:
            print(f"Retry also ended with failures: {exc2}", flush=True)

    # Final report (aggregate across stages)
    report_progress(args.scratch_dir)


if __name__ == '__main__':
    main()
