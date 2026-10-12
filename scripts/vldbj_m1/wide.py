#!/usr/bin/env python3
"""Experiment 3: wide LA-only tuning search with test-set selection.

Plan: docs/PhaseFormer_VLDBJ_LA_wide_search_plan.md (user-approved 2026-10-11).
Budget is unlimited in runs and limited in time: after --launch-deadline the driver
refuses to start any job. Every choice below is made on the test split (seed 2021;
test-set selection by user decision 2026-10-10) and is recorded with all candidate
numbers; reports must label it as such.

Stages:
  r1       -- fixed 12-variant grid, seed 2021, all 28 settings
  r2       -- pre-registered adaptive refinement around each setting's r1 winner
  select   -- per-setting argmin over every candidate (incl. the current paper
              configuration and every modulation-off read) on seed-2021 test MSE
  confirm  -- rerun the provisional winners on seeds 2022/2023; switch only if
              the 3-seed mean beats the current 3-seed mean by more than 0.2%
  collect  -- results.csv + wide_selection.json with the full audit trail
  all      -- the stages above in order, guarded by the launch deadline
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_interaction_stage2 import digest, save_json  # noqa: E402
from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.run import (  # noqa: E402
    DATASETS, E14, E14_ROOT, HORIZONS, SEEDS, TUNE_GRID, commit, e14_sources,
    write_immutable)
from scripts.vldbj_m1.proto import provenance, run_job  # noqa: E402

SEED = 2021
CONFIRM_SEEDS = (2022, 2023)
HEAD = protocol.FINAL_HEAD
SWITCH_MARGIN = 0.002
GATE_MIN, GATE_MAX, LR_MIN, LR_MAX = 1e-4, 0.99, 0.1, 100.0
WIDTHS = (2, 3, 4)
# Round 1: fixed grid (gate_init, extension_lr_mult, communication_width).
R1_GRID = tuple(
    {'gate_init': g, 'extension_lr_mult': lr, 'communication_width': w}
    for g, lr, w in ((0.001, 0.1, 2), (0.001, 1.0, 2), (0.001, 10.0, 2),
                     (0.1, 0.1, 2), (0.1, 1.0, 2), (0.1, 10.0, 2),
                     (0.5, 0.1, 2), (0.5, 1.0, 2), (0.5, 10.0, 2),
                     (0.5, 30.0, 2), (0.9, 10.0, 2), (0.2, 10.0, 4)))
R2_CAP = 10
# Pre-registered r2 shrink: fewer variants per setting when r1 ran late.
R2_SHRINK = ((3.0, 10), (4.0, 6), (5.0, 3))
PILOT_ROOT = Path(os.environ.get('VLDBJ_PILOT_ROOT', E14_ROOT / 'research_runs/vldbj_m1_pilot'))


def name(dataset, horizon, seed, variant, backbone_mult):
    v = variant or {}
    tag = (f"g{v.get('gate_init', 0.2):g}_lr{v.get('extension_lr_mult', 1.0):g}"
           f"_w{v.get('communication_width', 2):g}_bb{backbone_mult:g}")
    return f"{dataset}_H{horizon}_S{seed}__m1__{tag}"


def snap(value, lo, hi):
    return min(hi, max(lo, round(value, 4)))


def variant_key(variant, backbone_mult=1.0, read='native'):
    v = variant or {'gate_init': 0.2, 'extension_lr_mult': 1.0, 'communication_width': 2}
    return (float(v['gate_init']), float(v['extension_lr_mult']), int(v['communication_width']),
            float(backbone_mult), read)


def make_job(base, sources, dataset, horizon, seed, variant, backbone_mult):
    job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='m1', protocol='e14',
               init='scratch', head=HEAD, shared_gate=protocol.FINAL_SHARED_GATE[dataset],
               variant_id=None, variant=dict(variant) if variant else None,
               extension_lr_mult=float(variant['extension_lr_mult']) if variant else 1.0,
               backbone_lr_mult=float(backbone_mult), warm_source=None, epoch_test=False,
               source=sources[(dataset, horizon, seed)])
    job['output'] = str(Path(base['scratch']) / base['stage'] / 'runs' /
                        name(dataset, horizon, seed, variant, backbone_mult))
    return job


def write_jobs(scratch, stage, jobs):
    paths, seen = [], set()
    for job in jobs:
        path = scratch / stage / 'jobs' / f"{Path(job['output']).name}.json"
        write_immutable(path, job)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    order = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 4}
    return sorted(paths, key=lambda p: (order.get(p.name.split('_')[0], 9), p.name))


def tune_runs(dataset, horizon, seed):
    """Seed-2021 test reads of the 2026-10-10 tune round, keyed by TUNE_GRID index."""
    if not PILOT_ROOT.exists():
        return {}
    reads = {}
    for vid, grid in enumerate(TUNE_GRID):
        for sub in ('tune', 'confirm'):
            pattern = (f"*_S{seed}__m1__*__t{vid}/tune_result.json" if sub == 'tune'
                       else f"{dataset}_H{horizon}_S{seed}__m1__*__t{vid}/tune_result.json")
            for path in (PILOT_ROOT / sub / 'runs').glob(pattern):
                record = json.loads(path.read_text())
                ds, rest = record['name'].split('_H', 1)
                if ds == dataset and int(rest.split('_')[0]) == horizon:
                    reads[vid] = record
    return reads


def main_test(dataset, horizon, seed):
    """Current paper-configuration test read (main run, or its tuned replacement)."""
    if not PILOT_ROOT.exists():
        return None
    shared = 'shared' if protocol.FINAL_SHARED_GATE[dataset] else 'perC'
    base = PILOT_ROOT / 'main' / 'runs' / f'{dataset}_H{horizon}_S{seed}__m1__S2__{shared}'
    chosen = None
    selection_path = PILOT_ROOT / 'tune_selection.json'
    if selection_path.exists():
        record = json.loads(selection_path.read_text())['settings'].get(f'{dataset}_H{horizon}')
        chosen = record['chosen'] if record else None
    if chosen is None:
        path = base / 'test_result.json'
        return json.loads(path.read_text()) if path.exists() else None
    reads = tune_runs(dataset, horizon, seed)
    return reads.get(chosen)


def current_records():
    """Per-setting current numbers: seed-2021 reads plus the 3-seed native means."""
    out = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            per_seed = {s: main_test(dataset, horizon, s) for s in SEEDS}
            if any(r is None for r in per_seed.values()):
                raise ValueError(f'Missing current read for {dataset}-{horizon}')
            native = [r['test_mse'] for r in per_seed.values()]
            mae = [r['test_mae'] for r in per_seed.values()]
            out[(dataset, horizon)] = {
                'per_seed': {str(s): {'native': r['test_mse'], 'native_mae': r['test_mae'],
                                      'off': r.get('test_mse_modulation_off'),
                                      'off_mae': r.get('test_mae_modulation_off')}
                             for s, r in per_seed.items()},
                'native_mean': statistics.fmean(native), 'mae_mean': statistics.fmean(mae),
                'config': 'current (paper final configuration)'}
    return out


def wide_reads(scratch, stage, dataset, horizon, seed, smoke=False):
    """Every finished wide run of one setting as candidate records."""
    out = []
    for path in sorted((scratch / stage / 'runs').glob(f'{dataset}_H{horizon}_S{seed}__m1__*/result.json')):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            continue
        variant = json.loads((path.parent / 'config.json').read_text())['vldbj_m1']
        record = {'run': f'{stage}/{path.parent.name}', 'stage': stage,
                  'variant': {'gate_init': variant['gate_init'],
                              'extension_lr_mult': variant['extension_lr_mult'],
                              'communication_width': variant['communication_width']},
                  'backbone_lr_mult': r['backbone_lr_mult'], 'seed': seed,
                  'native': r['best']['test_mse'], 'native_mae': r['best']['test_mae'],
                  'off': r['best'].get('test_mse_modulation_off'),
                  'off_mae': r['best'].get('test_mae_modulation_off')}
        out.append(record)
    return out


def r2_variants(winner):
    """Pre-registered adaptive neighbors of the r1 winner (plan section 4.1)."""
    g, lr, w = (float(winner['gate_init']), float(winner['extension_lr_mult']),
                int(winner['communication_width']))
    bb = float(winner.get('backbone_lr_mult', 1.0))
    grid = []
    for gate in (snap(g / 3, GATE_MIN, GATE_MAX), snap(g * 3, GATE_MIN, GATE_MAX)):
        grid.append({'gate_init': gate, 'extension_lr_mult': lr, 'communication_width': w, 'bb': bb})
    for ext in (snap(lr / 3, LR_MIN, LR_MAX), snap(lr * 3, LR_MIN, LR_MAX)):
        grid.append({'gate_init': g, 'extension_lr_mult': ext, 'communication_width': w, 'bb': bb})
    for width in WIDTHS:
        if width != w:
            grid.append({'gate_init': g, 'extension_lr_mult': lr, 'communication_width': width, 'bb': bb})
    for bb_mult in (0.3, 3.0):
        grid.append({'gate_init': g, 'extension_lr_mult': lr, 'communication_width': w,
                     'bb': snap(bb * bb_mult, 0.1, 10.0)})
    grid.append({'gate_init': g, 'extension_lr_mult': LR_MAX, 'communication_width': w, 'bb': bb})
    grid.append({'gate_init': GATE_MAX, 'extension_lr_mult': lr, 'communication_width': w, 'bb': bb})
    return grid[:R2_CAP]


def r2_cap(now, started):
    hours = (now - started).total_seconds() / 3600
    cap = R2_CAP
    for threshold, shrunk in R2_SHRINK:
        if hours > threshold:
            cap = shrunk
    return cap


def run_jobs(jobs, gpus, deadline):
    tasks = queue.Queue()
    for path in jobs:
        tasks.put(path)
    failures = []
    skipped = []

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            if dt.datetime.now() >= deadline:
                skipped.append(str(path))
                continue
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(ROOT / 'scripts/vldbj_m1/proto.py'),
                                        '--job', str(path)], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    print(json.dumps({'failed': len(failures), 'skipped_after_deadline': len(skipped)}), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:5]}')


def stage_jobs(scratch, stage, sources, smoke):
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke,
            'stage': stage, 'scratch': str(scratch)}
    settings = [(d, h) for d in DATASETS for h in HORIZONS]
    if stage == 'r1':
        jobs = []
        for dataset, horizon in settings:
            done = {variant_key(r['variant'], r['backbone_lr_mult'])
                    for r in wide_reads(scratch, 'r1', dataset, horizon, SEED, smoke)}
            if not smoke:
                done |= {variant_key(TUNE_GRID[v]) for v in tune_runs(dataset, horizon, SEED)}
            for variant in R1_GRID:
                if variant_key(variant) in done:
                    continue
                jobs.append(make_job(base, sources, dataset, horizon, SEED, variant, 1.0))
        return write_jobs(scratch, stage, jobs)
    if stage == 'r2':
        marker = scratch / 'all.started'
        if not marker.exists():
            marker = scratch / 'r1_manifest.json'
        cap = r2_cap(dt.datetime.now(), dt.datetime.fromtimestamp(marker.stat().st_mtime))
        jobs = []
        for dataset, horizon in settings:
            runs = wide_reads(scratch, 'r1', dataset, horizon, SEED, smoke)
            if not runs:
                continue  # r1 hit the deadline for this setting: keep the current configuration
            winner = min(runs, key=lambda r: min(r['native'], r['off'] or r['native']))
            done = {variant_key(r['variant'], r['backbone_lr_mult']) for r in
                    runs + wide_reads(scratch, 'r2', dataset, horizon, SEED, smoke)}
            planned = 0
            for candidate in r2_variants({**winner['variant'], 'backbone_lr_mult': winner['backbone_lr_mult']}):
                variant = {k: candidate[k] for k in ('gate_init', 'extension_lr_mult', 'communication_width')}
                if variant_key(variant, candidate['bb']) in done or planned >= cap:
                    continue
                jobs.append(make_job(base, sources, dataset, horizon, SEED, variant, candidate['bb']))
                planned += 1
        return write_jobs(scratch, stage, jobs)
    raise ValueError(stage)


def wide_candidates(scratch, stage, dataset, horizon, smoke):
    """Wide runs as selection candidates: one per (variant, read)."""
    out = []
    for r in wide_reads(scratch, stage, dataset, horizon, SEED, smoke):
        base = {'config': f"wide {r['variant']} bb{r['backbone_lr_mult']:g}", 'variant': r['variant'],
                'backbone_lr_mult': r['backbone_lr_mult'], 'stage': stage, 'run': r['run']}
        out.append({**base, 'read': 'native', 'seed2021_mse': r['native'],
                    'seed2021_mae': r['native_mae']})
        if r['off'] is not None:
            out.append({**base, 'read': 'off', 'seed2021_mse': r['off'], 'seed2021_mae': r['off_mae']})
    return out


def select(scratch, smoke):
    """Provisional per-setting winners on the seed-2021 test split (test-set selection)."""
    current = current_records()
    settings_out = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = f'{dataset}_H{horizon}'
            cur = current[(dataset, horizon)]
            candidates = [{
                'config': cur['config'], 'variant': None, 'backbone_lr_mult': 1.0,
                'read': 'native', 'seed2021_mse': cur['per_seed']['2021']['native'],
                'seed2021_mae': cur['per_seed']['2021']['native_mae'], 'stage': 'current'}]
            if cur['per_seed']['2021']['off'] is not None:
                candidates.append({**candidates[0], 'read': 'off',
                                  'seed2021_mse': cur['per_seed']['2021']['off'],
                                  'seed2021_mae': cur['per_seed']['2021']['off_mae']})
            for stage in ('r1', 'r2'):
                candidates += wide_candidates(scratch, stage, dataset, horizon, smoke)
            best = min(candidates, key=lambda r: r['seed2021_mse'])
            margin = (candidates[0]['seed2021_mse'] - best['seed2021_mse']) / candidates[0]['seed2021_mse']
            settings_out[key] = {
                'current_seed2021_mse': candidates[0]['seed2021_mse'],
                'best': best, 'margin_vs_current': margin,
                'needs_confirm': (best['stage'] != 'current') and margin > SWITCH_MARGIN,
                'candidate_count': len(candidates), 'candidates': candidates}
    record = {'selection_basis': 'seed-2021 test split; test-set selection (user decision '
                                '2026-10-10); every candidate number is recorded here',
              'rule': {'switch': f'seed-2021 margin > {SWITCH_MARGIN:.1%}, then 3-seed mean '
                                 f'margin > {SWITCH_MARGIN:.1%} against the current paper '
                                 'configuration', 'read': 'native and modulation-off reads are '
                                 'both candidates (architecture-internal off, user 2026-10-11)'},
              'settings': settings_out, 'smoke': smoke}
    write_immutable(scratch / 'wide_provisional.json', record)
    return record


def confirm_jobs(scratch, sources, smoke):
    provisional = json.loads((scratch / 'wide_provisional.json').read_text())
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke,
            'stage': 'confirm', 'scratch': str(scratch)}
    jobs = []
    for key, record in sorted(provisional['settings'].items()):
        if not record['needs_confirm']:
            continue
        dataset, horizon = key.split('_H')
        best = record['best']
        for seed in CONFIRM_SEEDS:
            jobs.append(make_job(base, sources, dataset, int(horizon), seed, best['variant'],
                                 best['backbone_lr_mult']))
    return write_jobs(scratch, 'confirm', jobs)


def collect(scratch, smoke):
    provisional = json.loads((scratch / 'wide_provisional.json').read_text())
    current = current_records()
    fields = ['stage', 'dataset', 'horizon', 'seed', 'gate_init', 'extension_lr_mult',
              'communication_width', 'backbone_lr_mult', 'test_mse', 'test_mae',
              'test_mse_modulation_off', 'test_mae_modulation_off', 'epochs', 'run']
    rows = []
    for stage in ('r1', 'r2', 'confirm'):
        for path in sorted((scratch / stage / 'runs').glob('*/result.json')):
            r = json.loads(path.read_text())
            if r['smoke'] != smoke:
                continue
            v = json.loads((path.parent / 'config.json').read_text())['vldbj_m1']
            rows.append({'stage': stage, 'dataset': r['dataset'], 'horizon': r['horizon'],
                         'seed': r['seed'], 'gate_init': v['gate_init'],
                         'extension_lr_mult': v['extension_lr_mult'],
                         'communication_width': v['communication_width'],
                         'backbone_lr_mult': r['backbone_lr_mult'],
                         'test_mse': r['best']['test_mse'], 'test_mae': r['best']['test_mae'],
                         'test_mse_modulation_off': r['best'].get('test_mse_modulation_off'),
                         'test_mae_modulation_off': r['best'].get('test_mae_modulation_off'),
                         'epochs': r['epochs'], 'run': f'{stage}/{path.parent.name}'})
    with (scratch / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def run_read(dataset, horizon, seed, variant, bb):
        """native/off test reads of one wide run, if it exists."""
        for stage in ('r1', 'r2', 'confirm'):
            path = (scratch / stage / 'runs' /
                    name(dataset, horizon, seed, variant, bb) / 'result.json')
            if path.exists():
                r = json.loads(path.read_text())
                return {'native': r['best']['test_mse'], 'native_mae': r['best']['test_mae'],
                        'off': r['best'].get('test_mse_modulation_off'),
                        'off_mae': r['best'].get('test_mae_modulation_off')}
        return None

    final = {}
    for key, record in sorted(provisional['settings'].items()):
        dataset, horizon = key.split('_H')
        horizon = int(horizon)
        cur = current[(dataset, horizon)]
        best = record['best']
        entry = {'provisional': best, 'margin_vs_current': record['margin_vs_current'],
                 'candidate_count': record['candidate_count'],
                 'switched': False, 'final_config': cur['config'],
                 'final_read': 'native',
                 'final_seed_mse': {s: cur['per_seed'][s]['native'] for s in ('2021', '2022', '2023')},
                 'final_seed_mae': {s: cur['per_seed'][s]['native_mae'] for s in ('2021', '2022', '2023')}}
        if record['needs_confirm']:
            reads = {str(seed): run_read(dataset, horizon, seed, best['variant'], best['backbone_lr_mult'])
                     for seed in SEEDS}
            read = best['read']
            if all(reads.values()):
                values = [reads[str(seed)][read] for seed in SEEDS]
                mae_values = [reads[str(seed)][f'{read}_mae'] for seed in SEEDS]
                mean = statistics.fmean(values)
                margin3 = (cur['native_mean'] - mean) / cur['native_mean']
                entry['confirm'] = {'seed_mse': {str(seed): reads[str(seed)][read] for seed in SEEDS},
                                    'mean': mean, 'mae_mean': statistics.fmean(mae_values),
                                    'margin_vs_current_3seed': margin3}
                if margin3 > SWITCH_MARGIN:
                    entry.update(switched=True,
                                 final_config=f"wide: {best['variant']} bb{best['backbone_lr_mult']:g}",
                                 final_read=read,
                                 final_seed_mse={str(seed): reads[str(seed)][read] for seed in SEEDS},
                                 final_seed_mae={str(seed): reads[str(seed)][f'{read}_mae'] for seed in SEEDS})
            else:
                entry['confirm'] = 'incomplete (deadline); current configuration kept'
        elif best['stage'] == 'current' and best['read'] == 'off':
            # Current configuration with the modulation-off read: no retraining needed.
            entry.update(switched=True, final_read='off',
                         final_seed_mse={s: cur['per_seed'][s]['off'] for s in ('2021', '2022', '2023')},
                         final_seed_mae={s: cur['per_seed'][s]['off_mae'] for s in ('2021', '2022', '2023')})
        final[key] = entry
    switched = sum(1 for e in final.values() if e['switched'])
    record = {'selection_basis': 'test split (seed 2021 screen, 3-seed confirm); test-set '
                                 'selection (user decisions 2026-10-10/11); all candidates '
                                 'in wide_provisional.json and results.csv',
              'switch_margin': SWITCH_MARGIN, 'switched_settings': switched,
              'settings': final, 'smoke': smoke}
    save_json(scratch / 'wide_selection.json', record)
    return record


def run_all(scratch, gpus, smoke, deadline):
    save_json(scratch / 'all.started', {'at': dt.datetime.now().isoformat(), 'deadline': str(deadline)})
    sources = e14_sources()
    for stage in ('r1', 'r2'):
        jobs = stage_jobs(scratch, stage, sources, smoke)
        write_immutable(scratch / f'{stage}_manifest.json', {
            'stage': stage, 'jobs': [str(p) for p in jobs], 'code_commit': commit(), 'smoke': smoke})
        print(json.dumps({'stage': stage, 'jobs': len(jobs)}), flush=True)
        run_jobs(jobs, gpus, deadline)
    provisional = select(scratch, smoke)
    winners = sum(1 for r in provisional['settings'].values() if r['needs_confirm'])
    print(json.dumps({'provisional_winners': winners}), flush=True)
    jobs = confirm_jobs(scratch, sources, smoke)
    write_immutable(scratch / 'confirm_manifest.json', {
        'jobs': [str(p) for p in jobs], 'code_commit': commit(), 'smoke': smoke})
    print(json.dumps({'stage': 'confirm', 'jobs': len(jobs)}), flush=True)
    if jobs:
        run_jobs(jobs, gpus, deadline)
    collect(scratch, smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('r1', 'r2', 'select', 'confirm', 'collect', 'all'))
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--launch-deadline', default='2026-10-12T08:30:00')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage is None or args.scratch_dir is None:
        parser.error('--stage and --scratch-dir are required')
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    gpus = [g for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    deadline = dt.datetime.fromisoformat(args.launch_deadline)
    with (scratch / f'{args.stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'all':
            return run_all(scratch, gpus, args.smoke, deadline)
        if args.stage == 'select':
            return print(json.dumps(select(scratch, args.smoke), indent=1))
        if args.stage == 'collect':
            return print(json.dumps(collect(scratch, args.smoke), indent=1))
        sources = e14_sources()
        if args.stage == 'confirm':
            jobs = confirm_jobs(scratch, sources, args.smoke)
        else:
            jobs = stage_jobs(scratch, args.stage, sources, args.smoke)
        write_immutable(scratch / f'{args.stage}_manifest.json', {
            'stage': args.stage, 'jobs': [str(p) for p in jobs], 'code_commit': commit(),
            'smoke': args.smoke})
        print(json.dumps({'stage': args.stage, 'jobs': len(jobs)}), flush=True)
        run_jobs(jobs, gpus, deadline)


if __name__ == '__main__':
    main()
