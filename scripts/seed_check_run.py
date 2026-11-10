#!/usr/bin/env python3
"""Is the ETTh1-336 best cell a stable good point, or one lucky seed-2021 run?

The b0 search found MSE 0.4110 at (gate_init 0.9, ext lr 10, backbone 1.0) while most
neighbouring cells sit at 0.43-0.44, and no gate-freeze or lr-split variant reproduced
it.  This reruns that cell and three neighbours on seeds 2021, 2022 and 2023 with
everything else as the b0 best cell (period 24, S2 head, per-channel gates).

    python3 scripts/seed_check_run.py --run --gpus 1,2,3,7 --slots-per-gpu 2
    python3 scripts/seed_check_run.py --report

Kept outside ``scripts/vldbj_m1``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET, HORIZON = 'ETTh1', 336
SEEDS = (2021, 2022, 2023)
CELLS = ((0.9, 10.0), (0.9, 3.0), (0.9, 30.0), (0.5, 10.0))
BACKBONE = 1.0
DEFAULT_SCRATCH = ROOT / 'research_runs/seed_check_v1'
STAGE = 'v1'


def name(seed, gate, ext):
    return f'{DATASET}_H{HORIZON}_S{seed}__b0__g{gate:g}_lr{ext:g}_bb{BACKBONE:g}__p24__S2__perC'


def build_jobs(scratch):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for seed in SEEDS:
        for gate, ext in CELLS:
            job = dict(base, dataset=DATASET, horizon=HORIZON, seed=seed, arm='b0', protocol='e14',
                       init='scratch', head='S2', shared_gate=False, variant_id=None,
                       variant={'gate_init': gate, 'extension_lr_mult': ext, 'communication_width': 2},
                       extension_lr_mult=float(ext), backbone_lr_mult=BACKBONE, warm_source=None,
                       epoch_test=False, source=sources[(DATASET, HORIZON, seed)])
            job['output'] = str(scratch / STAGE / 'runs' / name(seed, gate, ext))
            path = scratch / STAGE / 'jobs' / f'{name(seed, gate, ext)}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def report(scratch):
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results
    old = {(r['gate'], r['ext']): r['mse'] for r in load_results(B0)
           if (r['dataset'], r['horizon'], r['period'], r['bb'], r['head'], r['shared']) ==
           (DATASET, HORIZON, 24, BACKBONE, 'S2', False)}
    print(f"{'cell (gate, lr)':>16} | " + ' '.join(f'{f"seed {s}":>9}' for s in SEEDS) + f" | {'mean':>7} {'range':>7}")
    for gate, ext in CELLS:
        vals = []
        for seed in SEEDS:
            record = scratch / STAGE / 'runs' / name(seed, gate, ext) / 'result.json'
            vals.append(json.loads(record.read_text())['best']['test_mse'] if record.is_file() else None)
        done = [v for v in vals if v is not None]
        mean = f'{sum(done) / len(done):.4f}' if len(done) == len(SEEDS) else '-'
        spread = f'{max(done) - min(done):.4f}' if len(done) == len(SEEDS) else '-'
        seed_cells = ' '.join(f'{v:>9.4f}' if v is not None else f"{'-':>9}" for v in vals)
        print(f"{'g%g, lr%g' % (gate, ext):>16} | {seed_cells} | {mean:>7} {spread:>7}")
    print(f"\nseed-2021 values from the b0 search for the same cells: " +
          ', '.join(f'g{g:g}/lr{e:g}={old[(g, e)]:.4f}' for g, e in CELLS if (g, e) in old))


def main():
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
        from scripts.vldbj_m1.proto import run_jobs as run
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        run(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
