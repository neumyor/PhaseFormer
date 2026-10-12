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
GATE_MIN, GATE_MAX, LR_MIN, LR_MAX = 1e-4, 0.999, 0.03, 1000.0
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
# Round 3: interaction refinement, boundary extrapolation, and a head/gate-sharing
# probe on the three datasets still behind the Golden (head S1/S3 change the level
# branch size; adopting them would rewrite the paper's parameter claims).
R3_DATASETS = ('ETTm1', 'ETTm2', 'Traffic')
R3_CAP = 12
ROUND3_STAGES = ('r1', 'r2', 'r3', 'confirm')


def name(dataset, horizon, seed, variant, backbone_mult, head=None, shared=None):
    v = variant or {}
    tag = (f"g{v.get('gate_init', 0.2):g}_lr{v.get('extension_lr_mult', 1.0):g}"
           f"_w{v.get('communication_width', 2):g}_bb{backbone_mult:g}")
    default_shared = protocol.FINAL_SHARED_GATE[dataset]
    head = head or HEAD
    shared = default_shared if shared is None else shared
    if head != HEAD or shared != default_shared:
        tag += f"__{head}__{'shared' if shared else 'perC'}"
    return f"{dataset}_H{horizon}_S{seed}__m1__{tag}"


def snap(value, lo, hi):
    return min(hi, max(lo, round(value, 4)))


def variant_key(variant, backbone_mult=1.0, read='native'):
    v = variant or {'gate_init': 0.2, 'extension_lr_mult': 1.0, 'communication_width': 2}
    return (float(v['gate_init']), float(v['extension_lr_mult']), int(v['communication_width']),
            float(backbone_mult), read)


def make_job(base, sources, dataset, horizon, seed, variant, backbone_mult, head=None, shared=None):
    job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='m1', protocol='e14',
               init='scratch', head=head or HEAD,
               shared_gate=protocol.FINAL_SHARED_GATE[dataset] if shared is None else bool(shared),
               variant_id=None, variant=dict(variant) if variant else None,
               extension_lr_mult=float(variant['extension_lr_mult']) if variant else 1.0,
               backbone_lr_mult=float(backbone_mult), warm_source=None, epoch_test=False,
               source=sources[(dataset, horizon, seed)])
    job['output'] = str(Path(base['scratch']) / base['stage'] / 'runs' /
                        name(dataset, horizon, seed, variant, backbone_mult, head, shared))
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


def baseline_records(scratch):
    """Round-3 reference per setting: the exp3 final configuration and its
    3-seed test reads (wide_selection.json of the previous round)."""
    final = json.loads((scratch / 'wide_selection.json').read_text())['settings']
    out = {}
    for key, entry in final.items():
        dataset, horizon = key.split('_H')
        horizon = int(horizon)
        if entry['switched']:
            variant = entry['provisional']['variant']
            bb = float(entry['provisional']['backbone_lr_mult'])
            label = entry['final_config']
        else:
            vid = protocol.FINAL_VARIANTS.get((dataset, horizon))
            variant = dict(TUNE_GRID[vid]) if vid is not None else None
            bb = 1.0
            label = 'exp3-final: current (paper final configuration)'
        out[(dataset, horizon)] = {
            'seed_mse': {k: float(v) for k, v in entry['final_seed_mse'].items()},
            'seed_mae': {k: float(v) for k, v in entry['final_seed_mae'].items()},
            'read': entry['final_read'], 'variant': variant, 'backbone_lr_mult': bb,
            'config': label, 'head': HEAD, 'shared_gate': protocol.FINAL_SHARED_GATE[dataset]}
    if len(out) != 28:
        raise ValueError(f'Expected 28 baseline settings, found {len(out)}')
    return out


def wide_reads(scratch, stage, dataset, horizon, seed, smoke=False):
    """Every finished wide run of one setting as candidate records."""
    out = []
    for path in sorted((scratch / stage / 'runs').glob(f'{dataset}_H{horizon}_S{seed}__m1__*/result.json')):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            continue
        meta = json.loads((path.parent / 'config.json').read_text())['vldbj_m1']
        record = {'run': f'{stage}/{path.parent.name}', 'stage': stage,
                  'variant': {'gate_init': meta['gate_init'],
                              'extension_lr_mult': meta['extension_lr_mult'],
                              'communication_width': meta['communication_width']},
                  'backbone_lr_mult': r['backbone_lr_mult'], 'seed': seed,
                  'head': meta['head'], 'shared_gate': meta['shared_gate'],
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
    if stage == 'r3':
        baseline = baseline_records(scratch)
        jobs = []
        for dataset, horizon in settings:
            base_cfg = baseline[(dataset, horizon)]
            v = base_cfg['variant'] or {'gate_init': 0.2, 'extension_lr_mult': 1.0,
                                        'communication_width': 2}
            g0, lr0 = float(v['gate_init']), float(v['extension_lr_mult'])
            w0, bb0 = int(v['communication_width']), float(base_cfg['backbone_lr_mult'])
            done = {variant_key(r['variant'], r['backbone_lr_mult'])
                    for r in (wide_reads(scratch, 'r1', dataset, horizon, SEED, smoke)
                              + wide_reads(scratch, 'r2', dataset, horizon, SEED, smoke)
                              + wide_reads(scratch, 'r3', dataset, horizon, SEED, smoke))}
            done.add(variant_key(v, bb0))
            grid = []
            # Interaction refinement: 3x3 (gate x lr) around the final configuration.
            for gate in (snap(g0 / 3, GATE_MIN, GATE_MAX), g0, snap(g0 * 3, GATE_MIN, GATE_MAX)):
                for lr in (snap(lr0 / 3, LR_MIN, LR_MAX), lr0, snap(lr0 * 3, LR_MIN, LR_MAX)):
                    if (gate, lr) != (g0, lr0):
                        grid.append({'gate_init': gate, 'extension_lr_mult': lr,
                                     'communication_width': w0, 'bb': bb0})
            # Boundary extrapolation where the final configuration sits on a searched bound.
            if lr0 >= 100:
                grid += [{'gate_init': g0, 'extension_lr_mult': lr, 'communication_width': w0, 'bb': bb0}
                         for lr in (300.0, 1000.0)]
            if g0 >= 0.99:
                grid.append({'gate_init': 0.999, 'extension_lr_mult': lr0,
                             'communication_width': w0, 'bb': bb0})
            if bb0 >= 3.0:
                grid.append({'gate_init': g0, 'extension_lr_mult': lr0,
                             'communication_width': w0, 'bb': 10.0})
            if bb0 <= 0.3:
                grid.append({'gate_init': g0, 'extension_lr_mult': lr0,
                             'communication_width': w0, 'bb': 0.1})
            for width in WIDTHS:
                if width != w0:
                    grid.append({'gate_init': g0, 'extension_lr_mult': lr0,
                                 'communication_width': width, 'bb': bb0})
            planned = 0
            for candidate in grid:
                variant = {k: candidate[k] for k in ('gate_init', 'extension_lr_mult',
                                                     'communication_width')}
                if variant_key(variant, candidate['bb']) in done or planned >= R3_CAP:
                    continue
                jobs.append(make_job(base, sources, dataset, horizon, SEED, variant, candidate['bb']))
                planned += 1
            # Head / gate-sharing probe on the datasets still behind the Golden.
            if dataset in R3_DATASETS:
                for head in ('S1', 'S3'):
                    for shared in (True, False):
                        jobs.append(make_job(base, sources, dataset, horizon, SEED, v, bb0,
                                             head=head, shared=shared))
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
                'backbone_lr_mult': r['backbone_lr_mult'], 'stage': stage, 'run': r['run'],
                'head': r['head'], 'shared_gate': r['shared_gate']}
        out.append({**base, 'read': 'native', 'seed2021_mse': r['native'],
                    'seed2021_mae': r['native_mae']})
        if r['off'] is not None:
            out.append({**base, 'read': 'off', 'seed2021_mse': r['off'], 'seed2021_mae': r['off_mae']})
    return out


def select(scratch, smoke):
    """Round-3 provisional winners on the seed-2021 test split (test-set selection).

    The reference is the round-3 baseline: the previous round's final configuration
    (wide_selection.json), with its per-seed reads. Rounds accumulate on the same
    test split; every candidate number is recorded here.
    """
    baseline = baseline_records(scratch)
    settings_out = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = f'{dataset}_H{horizon}'
            base = baseline[(dataset, horizon)]
            candidates = [{
                'config': base['config'], 'variant': base['variant'],
                'backbone_lr_mult': base['backbone_lr_mult'], 'read': base['read'],
                'head': base['head'], 'shared_gate': base['shared_gate'],
                'seed2021_mse': base['seed_mse']['2021'], 'seed2021_mae': base['seed_mae']['2021'],
                'stage': 'baseline'}]
            for stage in ('r1', 'r2', 'r3'):
                candidates += wide_candidates(scratch, stage, dataset, horizon, smoke)
            best = min(candidates, key=lambda r: r['seed2021_mse'])
            margin = (candidates[0]['seed2021_mse'] - best['seed2021_mse']) / candidates[0]['seed2021_mse']
            settings_out[key] = {
                'baseline_seed2021_mse': candidates[0]['seed2021_mse'],
                'best': best, 'margin_vs_baseline': margin,
                'needs_confirm': (best['stage'] != 'baseline') and margin > SWITCH_MARGIN,
                'candidate_count': len(candidates), 'candidates': candidates}
    record = {'selection_basis': 'seed-2021 test split; test-set selection (user decisions '
                                '2026-10-10/11/12); every candidate number is recorded here',
              'rule': {'switch': f'seed-2021 margin > {SWITCH_MARGIN:.1%}, then 3-seed mean '
                                 f'margin > {SWITCH_MARGIN:.1%} against the round-3 baseline '
                                 '(the previous round final configuration)',
                       'read': 'native and modulation-off reads are both candidates '
                               '(architecture-internal off, user 2026-10-11)'},
              'settings': settings_out, 'smoke': smoke}
    write_immutable(scratch / 'wide_provisional_r3.json', record)
    return record


def confirm_result(scratch, dataset, horizon, seed, variant, bb, head, shared):
    """An existing confirm read of exactly this configuration, if one exists."""
    for stage in ('confirm', 'confirm3'):
        path = (scratch / stage / 'runs' /
                name(dataset, horizon, seed, variant, bb, head, shared) / 'result.json')
        if path.exists():
            r = json.loads(path.read_text())
            if not r['smoke']:
                return {'native': r['best']['test_mse'], 'native_mae': r['best']['test_mae'],
                        'off': r['best'].get('test_mse_modulation_off'),
                        'off_mae': r['best'].get('test_mae_modulation_off')}
    return None


def confirm_jobs(scratch, sources, smoke):
    provisional = json.loads((scratch / 'wide_provisional_r3.json').read_text())
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke,
            'stage': 'confirm3', 'scratch': str(scratch)}
    jobs = []
    for key, record in sorted(provisional['settings'].items()):
        if not record['needs_confirm']:
            continue
        dataset, horizon = key.split('_H')
        best = record['best']
        for seed in CONFIRM_SEEDS:
            # Round-2 confirm runs of the identical configuration are reusable.
            if confirm_result(scratch, dataset, int(horizon), seed, best['variant'],
                              float(best['backbone_lr_mult']), best.get('head'),
                              best.get('shared_gate')) is not None:
                continue
            jobs.append(make_job(base, sources, dataset, int(horizon), seed, best['variant'],
                                 best['backbone_lr_mult'], best.get('head'), best.get('shared_gate')))
    return write_jobs(scratch, 'confirm3', jobs)


def collect(scratch, smoke):
    provisional = json.loads((scratch / 'wide_provisional_r3.json').read_text())
    baseline = baseline_records(scratch)
    fields = ['stage', 'dataset', 'horizon', 'seed', 'gate_init', 'extension_lr_mult',
              'communication_width', 'backbone_lr_mult', 'head', 'shared_gate', 'test_mse', 'test_mae',
              'test_mse_modulation_off', 'test_mae_modulation_off', 'epochs', 'run']
    rows = []
    for stage in ('r1', 'r2', 'r3', 'confirm', 'confirm3'):
        for path in sorted((scratch / stage / 'runs').glob('*/result.json')):
            r = json.loads(path.read_text())
            if r['smoke'] != smoke:
                continue
            v = json.loads((path.parent / 'config.json').read_text())['vldbj_m1']
            rows.append({'stage': stage, 'dataset': r['dataset'], 'horizon': r['horizon'],
                         'seed': r['seed'], 'gate_init': v['gate_init'],
                         'extension_lr_mult': v['extension_lr_mult'],
                         'communication_width': v['communication_width'],
                         'backbone_lr_mult': r['backbone_lr_mult'], 'head': v['head'],
                         'shared_gate': v['shared_gate'],
                         'test_mse': r['best']['test_mse'], 'test_mae': r['best']['test_mae'],
                         'test_mse_modulation_off': r['best'].get('test_mse_modulation_off'),
                         'test_mae_modulation_off': r['best'].get('test_mae_modulation_off'),
                         'epochs': r['epochs'], 'run': f'{stage}/{path.parent.name}'})
    with (scratch / 'results_r3.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def run_read(dataset, horizon, seed, variant, bb, head=None, shared=None):
        """native/off test reads of one wide run, if it exists."""
        reads = confirm_result(scratch, dataset, horizon, seed, variant, bb, head, shared)
        if reads is not None:
            return reads
        for stage in ('r1', 'r2', 'r3'):
            path = (scratch / stage / 'runs' /
                    name(dataset, horizon, seed, variant, bb, head, shared) / 'result.json')
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
        cur = baseline[(dataset, horizon)]
        best = record['best']
        entry = {'provisional': best, 'margin_vs_baseline': record['margin_vs_baseline'],
                 'candidate_count': record['candidate_count'],
                 'switched': False, 'final_config': cur['config'],
                 'final_read': cur['read'], 'head': cur['head'], 'shared_gate': cur['shared_gate'],
                 'final_seed_mse': dict(cur['seed_mse']), 'final_seed_mae': dict(cur['seed_mae'])}
        if record['needs_confirm']:
            reads = {str(seed): run_read(dataset, horizon, seed, best['variant'], best['backbone_lr_mult'],
                                         best.get('head'), best.get('shared_gate'))
                     for seed in SEEDS}
            read = best['read']
            if all(reads.values()):
                values = [reads[str(seed)][read] for seed in SEEDS]
                mae_values = [reads[str(seed)][f'{read}_mae'] for seed in SEEDS]
                mean = statistics.fmean(values)
                base_mean = statistics.fmean(cur['seed_mse'].values())
                margin3 = (base_mean - mean) / base_mean
                entry['confirm'] = {'seed_mse': {str(seed): reads[str(seed)][read] for seed in SEEDS},
                                    'mean': mean, 'mae_mean': statistics.fmean(mae_values),
                                    'margin_vs_baseline_3seed': margin3}
                if margin3 > SWITCH_MARGIN:
                    head = best.get('head') or HEAD
                    shared = best.get('shared_gate', protocol.FINAL_SHARED_GATE[dataset])
                    entry.update(switched=True,
                                 final_config=f"wide r3: {best['variant']} bb{best['backbone_lr_mult']:g}"
                                              f" {head} {'shared' if shared else 'perC'}",
                                 final_read=read, head=head, shared_gate=shared,
                                 final_seed_mse={str(seed): reads[str(seed)][read] for seed in SEEDS},
                                 final_seed_mae={str(seed): reads[str(seed)][f'{read}_mae'] for seed in SEEDS})
            else:
                entry['confirm'] = 'incomplete (deadline); baseline kept'
        final[key] = entry
    switched = sum(1 for e in final.values() if e['switched'])
    record = {'selection_basis': 'test split (seed 2021 screen, 3-seed confirm); test-set '
                                 'selection (user decisions 2026-10-10/11/12); all candidates '
                                 'in wide_provisional.json and results.csv',
              'switch_margin': SWITCH_MARGIN, 'switched_settings': switched,
              'settings': final, 'smoke': smoke}
    save_json(scratch / 'wide_selection_r3.json', record)
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
    parser.add_argument('--stage', choices=('r1', 'r2', 'r3', 'select', 'confirm', 'collect', 'all'))
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
            # Round-3 confirm runs in confirm3/; its manifest must not collide with
            # the round-2 confirm manifest.
            write_immutable(scratch / 'confirm3_manifest.json', {
                'stage': 'confirm3', 'jobs': [str(p) for p in jobs], 'code_commit': commit(),
                'smoke': args.smoke})
        else:
            jobs = stage_jobs(scratch, args.stage, sources, args.smoke)
            write_immutable(scratch / f'{args.stage}_manifest.json', {
                'stage': args.stage, 'jobs': [str(p) for p in jobs], 'code_commit': commit(),
                'smoke': args.smoke})
        print(json.dumps({'stage': args.stage, 'jobs': len(jobs)}), flush=True)
        run_jobs(jobs, gpus, deadline)


if __name__ == '__main__':
    main()
