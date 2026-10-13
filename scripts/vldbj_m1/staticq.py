#!/usr/bin/env python3
"""Static-q ablation of PhaseFormer-LA in the protocol configuration (VLDBJ extension).

Identical to the protocol-configuration PhaseFormer-LA (same E14 backbone protocol, head S2,
per-dataset gate sharing, the eight per-setting extension variants of
``protocol.FINAL_VARIANTS``, from scratch, same seeds, same paired backbone draw) except that
the modulation vector ``q = V2 V1 t`` no longer depends on the level-branch state ``t``: it is
a learned per-channel constant, zero-initialized, with the same slot-centering. It separates
the level-driven part of the modulation from its static per-channel part.

Stages:
  run      -- plan and train 28 settings x 3 seeds (84 jobs), slow datasets first
  collect  -- results.csv with the best-checkpoint test reads (native and modulation off)

Runs reuse ``proto.run_job`` (the worker of the protocol runs); the only difference to an LA
job is the ``static_q`` flag, which ``model.prepare_config`` writes into the registered config.
Test reads are recorded for audit; the per-setting hyperparameters were not re-tuned for
static-q, so any gap to LA is not hyperparameter-tuned on either side beyond the LA protocol.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vldbj_m1.proto import (  # noqa: E402
    SLOW_FIRST, make_job, manifest, name, provenance, run_jobs, settings_filter, write_immutable)
from scripts.vldbj_m1.run import SEEDS, commit, e14_sources  # noqa: E402


def plan(scratch, sources, settings, smoke):
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke, 'stage': 'staticq'}
    jobs = []
    for dataset, horizon in settings:
        for seed in SEEDS:
            job = make_job(base, sources, dataset=dataset, horizon=horizon, seed=seed, arm='m1',
                           proto='e14')
            job['static_q'] = True
            jobs.append(job)
    paths = []
    for job in jobs:
        job['output'] = str(scratch / 'staticq' / 'runs' / name(job))
        path = scratch / 'staticq' / 'jobs' / f'{name(job)}.json'
        write_immutable(path, job)
        paths.append(path)
    return sorted(paths, key=lambda p: (SLOW_FIRST.get(p.name.split('_')[0], 9), p.name))


def collect(scratch, smoke):
    rows = []
    for path in sorted((scratch / 'staticq' / 'runs').glob('*/result.json')):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            continue
        cfg = json.loads((path.parent / 'config.json').read_text())['vldbj_m1']
        if not cfg.get('static_q'):
            raise ValueError(f'{path.parent.name} is not a static-q run')
        best = r['best']
        rows.append({
            'dataset': r['dataset'], 'horizon': r['horizon'], 'seed': r['seed'],
            'variant_id': r['variant_id'], 'gate_init': cfg['gate_init'],
            'extension_lr_mult': cfg['extension_lr_mult'],
            'communication_width': cfg['communication_width'], 'shared_gate': cfg['shared_gate'],
            'test_mse': best['test_mse'], 'test_mae': best['test_mae'],
            'test_mse_modulation_off': best.get('test_mse_modulation_off'),
            'test_mae_modulation_off': best.get('test_mae_modulation_off'),
            'val_mse': best['val_mse'], 'epochs': r['epochs'],
            'modulation_params': r['parameters']['modulation'],
            'cycle_mean_passed': (r.get('cycle_mean_check') or {}).get('passed'),
            'elapsed_sec': r['elapsed_sec'], 'run': path.parent.name})
    out = scratch / ('results_smoke.csv' if smoke else 'results.csv')
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['run'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'results': len(rows), 'file': str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('run', 'collect'), required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--settings', default='', help='smoke only, e.g. ETTh1-96,Weather-192')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    if args.stage == 'collect':
        return collect(scratch, args.smoke)
    settings = settings_filter(args.settings, args.smoke)
    gpus = [g for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    with (scratch / 'staticq.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = plan(scratch, e14_sources(), settings, args.smoke)
        manifest(scratch, 'staticq', jobs, args.smoke)
        if not args.prepare_only:
            run_jobs(jobs, gpus)
            collect(scratch, args.smoke)


if __name__ == '__main__':
    main()
