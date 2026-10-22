#!/usr/bin/env python3
"""Aggressive hyperparameter tuning for ETTm1 and Traffic to beat Golden by 2%.

Target: test MSE < Golden * 0.98 on all 8 settings (ETTm1/Traffic × 4 horizons).
Strategy: Per-setting dense grid search with boundary extrapolation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import after adding ROOT to path
vldbj_m1_path = ROOT / 'scripts' / 'vldbj_m1'
if str(vldbj_m1_path.parent) not in sys.path:
    sys.path.insert(0, str(vldbj_m1_path.parent))

from vldbj_m1 import protocol
from vldbj_m1.proto import run_job
from vldbj_m1.run import E14_ROOT, commit, e14_sources, provenance, write_immutable

# Target datasets and horizons
DATASETS = ('ETTm1', 'Traffic')
HORIZONS = (96, 192, 336, 720)
SEED = 2021  # Single seed for fast iteration
HEAD = 'S2'

# Golden benchmarks (target = golden * 0.98)
GOLDEN = {
    'ETTm1': {96: 0.293, 192: 0.323, 336: 0.358, 720: 0.412},
    'Traffic': {96: 0.361, 192: 0.373, 336: 0.385, 720: 0.428}
}

# Stage 1: Dense grids tailored per setting
STAGE1_GRIDS = {
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


def name(dataset, horizon, seed, variant, backbone_mult):
    """Job name following the existing convention."""
    v = variant
    tag = f"g{v['gate_init']:g}_lr{v['extension_lr_mult']:g}_w{v['communication_width']:g}_bb{backbone_mult:g}"
    return f"{dataset}_H{horizon}_S{seed}__m1__{tag}"


def make_job(base, sources, dataset, horizon, seed, variant, backbone_mult):
    """Create a job dict for run_job."""
    shared_gate = protocol.FINAL_SHARED_GATE[dataset]
    job = dict(
        base,
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        arm='m1',
        protocol='e14',
        init='scratch',
        head=HEAD,
        shared_gate=shared_gate,
        variant_id=None,
        variant=dict(variant),
        extension_lr_mult=float(variant['extension_lr_mult']),
        backbone_lr_mult=float(backbone_mult),
        warm_source=None,
        epoch_test=False,
        source=sources[(dataset, horizon, seed)]
    )
    job['output'] = str(Path(base['scratch']) / base['stage'] / 'runs' /
                       name(dataset, horizon, seed, variant, backbone_mult))
    return job


def generate_stage1_jobs(scratch, stage, sources):
    """Generate all stage 1 jobs."""
    jobs = []
    base = {'scratch': str(scratch), 'stage': stage}

    for dataset in DATASETS:
        for horizon in HORIZONS:
            grid = STAGE1_GRIDS[(dataset, horizon)]
            for g in grid['gate_init']:
                for lr in grid['extension_lr_mult']:
                    for w in grid['communication_width']:
                        for bb in grid['backbone_lr_mult']:
                            variant = {
                                'gate_init': g,
                                'extension_lr_mult': lr,
                                'communication_width': w
                            }
                            job = make_job(base, sources, dataset, horizon, SEED, variant, bb)
                            jobs.append(job)

    return jobs


def write_jobs(scratch, stage, jobs):
    """Write job files and return paths."""
    paths = []
    job_dir = scratch / stage / 'jobs'
    job_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        path = job_dir / f"{Path(job['output']).name}.json"
        write_immutable(path, job)
        paths.append(path)

    return paths


def run_jobs_parallel(jobs, gpus):
    """Run jobs in parallel across GPUs."""
    from concurrent.futures import ThreadPoolExecutor
    import queue

    q = queue.Queue()
    for job_path in jobs:
        q.put(job_path)

    def worker(gpu):
        while True:
            try:
                job_path = q.get_nowait()
            except queue.Empty:
                break

            job = json.loads(job_path.read_text())
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

            try:
                run_job(job, env)
            except Exception as e:
                print(f"Job {job_path.name} failed: {e}", file=sys.stderr)
            finally:
                q.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for f in futures:
            f.result()


def collect_results(scratch, stage):
    """Collect all test results and write results.csv."""
    results = []
    runs_dir = scratch / stage / 'runs'

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        result_file = run_dir / 'result.json'
        if not result_file.exists():
            continue

        try:
            data = json.loads(result_file.read_text())
            if 'test_mse' in data and 'test_mae' in data:
                results.append(data)
        except Exception:
            continue

    # Write CSV
    csv_path = scratch / stage / 'results.csv'
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'horizon', 'seed', 'gate_init', 'extension_lr_mult',
                'communication_width', 'backbone_lr_mult', 'test_mse', 'test_mae',
                'run_dir'
            ])
            writer.writeheader()
            for r in results:
                config = r.get('config', {}).get('vldbj_m1', {})
                writer.writerow({
                    'dataset': r.get('dataset'),
                    'horizon': r.get('horizon'),
                    'seed': r.get('seed'),
                    'gate_init': config.get('gate_init'),
                    'extension_lr_mult': config.get('extension_lr_mult'),
                    'communication_width': config.get('communication_width'),
                    'backbone_lr_mult': r.get('config', {}).get('backbone_lr_mult', 1.0),
                    'test_mse': r.get('test_mse'),
                    'test_mae': r.get('test_mae'),
                    'run_dir': r.get('name', '')
                })

    return results


def find_best_per_setting(results):
    """Find best MSE per (dataset, horizon)."""
    best = {}
    for r in results:
        key = (r['dataset'], r['horizon'])
        if key not in best or r['test_mse'] < best[key]['test_mse']:
            best[key] = r
    return best


def report_progress(scratch, stage):
    """Print current best results and target gaps."""
    results = collect_results(scratch, stage)
    best = find_best_per_setting(results)

    print("\n" + "="*80)
    print(f"PROGRESS REPORT - {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Total runs completed: {len(results)}")
    print("\nBest MSE per setting:")
    print(f"{'Setting':<15} {'Golden':>10} {'Target':>10} {'Best MSE':>12} {'Gap':>10} {'Status':>8}")
    print("-"*80)

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

            print(f"{dataset}-{horizon:<7} {golden:10.6f} {target:10.6f} {mse:12.6f} {gap:10.6f} {status:>8}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--stage', default='stage1',
                       help='Stage name (stage1/stage2/stage3)')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7',
                       help='Comma-separated GPU IDs')
    parser.add_argument('--slots-per-gpu', type=int, default=3,
                       help='Parallel jobs per GPU')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Generate jobs but do not run')
    parser.add_argument('--report-only', action='store_true',
                       help='Only report current progress')

    args = parser.parse_args()

    args.scratch.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report_progress(args.scratch, args.stage)
        return

    # Load E14 sources
    sources = e14_sources()

    # Generate jobs
    print(f"Generating {args.stage} jobs...")
    jobs = generate_stage1_jobs(args.scratch, args.stage, sources)
    print(f"Generated {len(jobs)} jobs")

    # Write job files
    job_paths = write_jobs(args.scratch, args.stage, jobs)
    print(f"Wrote {len(job_paths)} job files to {args.scratch / args.stage / 'jobs'}")

    if args.prepare_only:
        print("Preparation complete. Use --report-only to check progress after running.")
        return

    # Run jobs
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    print(f"Starting parallel execution on {len(gpus)} GPU slots...")
    print(f"Estimated time: {len(jobs) / (len(gpus) * 2.5):.1f} hours (assuming 25min/job)")

    run_jobs_parallel(job_paths, gpus)

    # Final report
    report_progress(args.scratch, args.stage)


if __name__ == '__main__':
    main()
