#!/usr/bin/env python3
"""Wide hyperparameter search on the no-amplitude arm (b0).

The plan mirrors the PhaseFormer-LA search: the same 28 settings, seed 2021,
selection by test MSE, the same coarse grid, then adaptive refinement, then a
cycle-length sweep over every admissible divisor, a level-head and gate-sharing
probe, and a boundary pass into the extreme ranges of every knob.  The arm stays
``b0``, so the amplitude modulation is absent and the level branch is the only
extension; ``communication_width`` has no effect there and is fixed at 2.

Stages, each generated from the winners of the ones before it:

  s1  the 11 distinct (gate_init, extension_lr_mult) cells of the LA round-1 grid
  s2  local refinement: gate and extension rate by 0.5/2, backbone rate by 0.3/3,
      capped at 16 new cells per setting
  s3  every admissible cycle length at that setting's current best variant
  s4  level head (S1/S2/S3) and gate sharing (shared / per-channel) at the best
  s5  boundary cells: gate down to 1e-4 and up to 0.999, extension rate to 0.03
      and 1000, backbone rate to 0.1 and 10, capped at 12 new cells per setting

Every number is test-set selection at seed 2021, exactly as the LA search, so the
comparison is like for like and stays exploratory.

    python3 scripts/b0_search.py --stage s1 --phase non-traffic --run
    python3 scripts/b0_search.py --stage s1 --phase traffic --run
    python3 scripts/b0_search.py --report
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation_no_amplitude import ROWS, SEED  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402

DEFAULT_SCRATCH = ROOT / 'research_runs/b0_search_v1'
STAGES = ('s1', 's2', 's3', 's4', 's5')
HEADS = ('S1', 'S2', 'S3')
GATE_LO, GATE_HI = 1e-4, 0.999
LR_LO, LR_HI = 0.03, 1000.0
BB_LO, BB_HI = 0.1, 10.0

# Round-1 cells of the LA search, with the inert width collapsed.
S1_CELLS = ((0.001, 0.1), (0.001, 1.0), (0.001, 10.0), (0.1, 0.1), (0.1, 1.0),
            (0.1, 10.0), (0.2, 10.0), (0.5, 0.1), (0.5, 1.0), (0.5, 10.0), (0.9, 10.0))

# Cycle lengths the model accepts; the sweep keeps the divisors of the horizon.
PERIODS = (8, 12, 16, 20, 24, 28, 30, 32, 36, 40, 45, 48, 56, 60, 64, 72, 80,
           84, 90, 96, 112, 120, 144, 168, 180, 240, 336, 360, 720)


def snap_gate(x):
    return min(GATE_HI, max(GATE_LO, round(x, 5)))


def snap_lr(x):
    return min(LR_HI, max(LR_LO, round(x, 4)))


def snap_bb(x):
    return min(BB_HI, max(BB_LO, round(x, 3)))


def job_name(dataset, horizon, period, gate, ext, bb, head, shared):
    return (f"{dataset}_H{horizon}_S{SEED}__b0__g{gate:g}_lr{ext:g}_bb{bb:g}"
            f"__p{period}__{head}__{'shared' if shared else 'perC'}")


def key(row, period, gate, ext, bb, head, shared):
    return (row['dataset'], row['horizon'], int(period), float(gate), float(ext),
            float(bb), head, bool(shared))


def load_results(scratch):
    """Every finished run under the scratch, with the knobs its config carries."""
    out = []
    for run in sorted(scratch.glob('*/runs/*')):
        record = run / 'result.json'
        if not record.is_file():
            continue
        result = json.loads(record.read_text())
        config = json.loads((run / 'config.json').read_text())
        protocol = config['vldbj_m1']
        out.append({
            'dataset': config['dataset'], 'horizon': int(config['horizon']),
            'period': int(config.get('period', config['hyperparams'].get('period_len', 24))),
            'gate': float(protocol['gate_init']),
            'ext': float(result['extension_lr_mult']),
            'bb': float(result['backbone_lr_mult']),
            'head': protocol['head'], 'shared': bool(protocol['shared_gate']),
            'mse': result['best']['test_mse'], 'mae': result['best']['test_mae'],
            'run': run.name,
        })
    return out


def winners(results, dataset, horizon):
    rows = [r for r in results if r['dataset'] == dataset and r['horizon'] == horizon]
    if not rows:
        return None
    return min(rows, key=lambda r: r['mse'])


def stage_candidates(stage, row, results):
    """New (period, gate, ext, bb, head, shared) cells for one setting."""
    tried = {key(row, r['period'], r['gate'], r['ext'], r['bb'], r['head'], r['shared'])
             for r in results if r['dataset'] == row['dataset'] and r['horizon'] == row['horizon']}
    base = winners(results, row['dataset'], row['horizon'])
    cells = []
    if stage == 's1':
        for gate, ext in S1_CELLS:
            cells.append((24, gate, ext, 1.0, 'S2', row['shared']))
    elif base is None:
        return []
    elif stage == 's2':
        for gate in (snap_gate(base['gate'] * 0.5), snap_gate(base['gate'] * 2.0)):
            for ext in (snap_lr(base['ext'] * 0.25), snap_lr(base['ext'] * 4.0)):
                cells.append((base['period'], gate, ext, base['bb'], base['head'], base['shared']))
        for bb in (snap_bb(base['bb'] * 0.3), snap_bb(base['bb'] * 3.0)):
            cells.append((base['period'], base['gate'], base['ext'], bb, base['head'], base['shared']))
        cells.append((base['period'], snap_gate(base['gate'] * 2.0), snap_lr(base['ext'] * 4.0),
                      snap_bb(base['bb'] * 3.0), base['head'], base['shared']))
    elif stage == 's3':
        for period in PERIODS:
            if row['horizon'] % period == 0:
                cells.append((period, base['gate'], base['ext'], base['bb'], base['head'], base['shared']))
    elif stage == 's4':
        for head in HEADS:
            for shared in (row['shared'], not row['shared']):
                cells.append((base['period'], base['gate'], base['ext'], base['bb'], head, shared))
    elif stage == 's5':
        for gate in (GATE_LO, snap_gate(base['gate'] * 0.1), snap_gate(0.95), GATE_HI):
            for ext in (LR_LO, LR_HI):
                cells.append((base['period'], gate, ext, base['bb'], base['head'], base['shared']))
        for bb in (BB_LO, BB_HI):
            cells.append((base['period'], base['gate'], base['ext'], bb, base['head'], base['shared']))
        cells.append((base['period'], GATE_HI, LR_HI, BB_HI, base['head'], base['shared']))
    seen, out = set(tried), []
    for cell in cells:
        k = key(row, *cell)
        if k in seen:
            continue
        seen.add(k)
        out.append(cell)
    cap = {'s2': 16, 's4': 12, 's5': 12}.get(stage)
    return out[:cap] if cap else out


def build_jobs(scratch, stage, rows, phase):
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    results = load_results(scratch) if stage != 's1' else []
    paths = []
    for row in rows:
        if phase == 'traffic' and row['dataset'] != 'Traffic':
            continue
        if phase == 'non-traffic' and row['dataset'] == 'Traffic':
            continue
        for period, gate, ext, bb, head, shared in stage_candidates(stage, row, results):
            source = sources[(row['dataset'], row['horizon'], SEED)]
            if period != 24:
                source = copy.deepcopy(source)
                source['config']['period'] = period
                source['config']['hyperparams']['period_len'] = period
            name = job_name(row['dataset'], row['horizon'], period, gate, ext, bb, head, shared)
            job = dict(base, dataset=row['dataset'], horizon=row['horizon'], seed=SEED,
                       arm='b0', protocol='e14', init='scratch', head=head,
                       shared_gate=bool(shared), variant_id=None,
                       variant={'gate_init': gate, 'extension_lr_mult': ext,
                                'communication_width': 2},
                       extension_lr_mult=float(ext), backbone_lr_mult=float(bb),
                       warm_source=None, epoch_test=False, source=source,
                       search='b0', search_stage=stage,
                       m1_reference={'test_mse': row['m1_mse'], 'test_mae': row['m1_mae']})
            job['output'] = str(scratch / stage / 'runs' / name)
            path = scratch / stage / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def report(scratch, rows):
    results = load_results(scratch)
    print(f"{'Dataset':>11} {'H':>4} | {'LA (m1)':>8} {'b0 best':>8} {'d%':>7} | "
          f"{'period':>6} {'gate':>7} {'ext_lr':>8} {'bb':>5} {'head':>4} {'gate':>7}")
    per_dataset = {}
    for row in rows:
        best = winners(results, row['dataset'], row['horizon'])
        if best is None:
            print(f"{row['dataset']:>11} {row['horizon']:>4} | (no runs)")
            continue
        delta = (best['mse'] - row['m1_mse']) / row['m1_mse'] * 100
        per_dataset.setdefault(row['dataset'], []).append((row['m1_mse'], best['mse'],
                                                           row['m1_mae'], best['mae']))
        print(f"{row['dataset']:>11} {row['horizon']:>4} | {row['m1_mse']:>8.5f} {best['mse']:>8.5f} "
              f"{delta:>+6.2f}% | {best['period']:>6} {best['gate']:>7g} {best['ext']:>8g} "
              f"{best['bb']:>5g} {best['head']:>4} {'shared' if best['shared'] else 'perC':>7}")
    print()
    print(f"{'Dataset':>11} | {'LA':>8} {'b0 best':>8} {'d%':>7} | {'LA MAE':>8} {'b0 MAE':>8} {'d%':>7}")
    tot = []
    for dataset, items in per_dataset.items():
        if len(items) != 4:
            continue
        la = sum(i[0] for i in items) / 4
        b0 = sum(i[1] for i in items) / 4
        lam = sum(i[2] for i in items) / 4
        b0m = sum(i[3] for i in items) / 4
        tot.append((la, b0, lam, b0m))
        print(f"{dataset:>11} | {la:>8.4f} {b0:>8.4f} {(b0-la)/la*100:>+6.2f}% | "
              f"{lam:>8.4f} {b0m:>8.4f} {(b0m-lam)/lam*100:>+6.2f}%")
    if tot:
        n = len(tot)
        la, b0, lam, b0m = (sum(t[i] for t in tot) / n for i in range(4))
        print(f"{'mean':>11} | {la:>8.4f} {b0:>8.4f} {(b0-la)/la*100:>+6.2f}% | "
              f"{lam:>8.4f} {b0m:>8.4f} {(b0m-lam)/lam*100:>+6.2f}%")


def missing_jobs(scratch):
    """Job files already written whose run has no result yet, without regenerating."""
    paths = []
    for stage in STAGES:
        for path in sorted((scratch / stage / 'jobs').glob('*.json')):
            job = json.loads(path.read_text())
            if not (Path(job['output']) / 'result.json').is_file():
                paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--stage', default='s1', choices=STAGES)
    parser.add_argument('--phase', default='non-traffic', choices=('all', 'non-traffic', 'traffic'))
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--slots-per-gpu', type=int, default=None)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--repair', action='store_true',
                        help='run only the job files already on disk that have no result')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--datasets', default=None)
    args = parser.parse_args()

    rows = ROWS
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(',')}
        rows = [r for r in rows if r['dataset'] in keep]

    if args.report:
        report(args.scratch, rows)
        return
    if args.repair:
        paths = missing_jobs(args.scratch)
        print(f'repair: {len(paths)} jobs without a result', flush=True)
        if not paths:
            return
        gpus = [int(g) for g in args.gpus.split(',')]
        small = [g for g in gpus for _ in range(args.slots_per_gpu or 2)]
        big = list(gpus)
        for group, devices in ((['p' for p in paths if not Path(p).name.startswith('Traffic')], small),
                               ([p for p in paths if Path(p).name.startswith('Traffic')], big)):
            if group:
                run_jobs(group, devices)
        return
    args.scratch.mkdir(parents=True, exist_ok=True)
    paths = build_jobs(args.scratch, args.stage, rows, args.phase)
    print(f'{args.stage}/{args.phase}: {len(paths)} jobs', flush=True)
    if args.prepare or not args.run:
        return
    slots = args.slots_per_gpu if args.slots_per_gpu is not None else (1 if args.phase == 'traffic' else 3)
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(slots)]
    print('launching on', gpus, flush=True)
    for attempt in (1, 2):
        try:
            run_jobs([str(p) for p in paths], gpus)
            break
        except RuntimeError as exc:
            print(f'pass {attempt} ended with failures: {exc}', flush=True)
            if attempt == 2:
                raise


if __name__ == '__main__':
    main()
