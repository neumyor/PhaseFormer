#!/usr/bin/env python3
"""Is rank three enough for the level branch?

The level head is a rank-3 encoder/decoder by construction, so the PCA evidence
that the level demand is low-dimensional needs one training-side check: vary the
rank and see where the test error stops improving.  Each run reuses its setting's
best b0 cell (period, gates, learning rates, head, gate sharing) and changes only
the rank of the level map.

The rank lives in ``scripts.vldbj_m1/model.py`` as a module constant, so the
worker patches ``model.RANK`` before building the model instead of editing the
frozen-protocol files.  Tiers that tie the best cell cannot justify a rank above
three; tiers below show what the third direction buys.

    python3 scripts/level_rank_sweep.py --prepare
    python3 scripts/level_rank_sweep.py --run --gpus 6,5 --slots-per-gpu 2
    python3 scripts/level_rank_sweep.py --report
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/level_rank_v1'
STAGE = 'r1'
SETTINGS = (('ETTh1', 720), ('ETTh2', 720), ('Weather', 720), ('Traffic', 720))
RANKS = (1, 2, 3, 4, 6)


def build_jobs(scratch):
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    sources = e14_sources()
    results = load_results(B0)
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for dataset, horizon in SETTINGS:
        best = winners(results, dataset, horizon)
        source = sources[(dataset, horizon, best.get('seed', 2021))] if False else \
            sources[(dataset, horizon, 2021)]
        import copy
        source = copy.deepcopy(source)
        source['config']['period'] = best['period']
        source['config']['hyperparams']['period_len'] = best['period']
        for rank in RANKS:
            job = dict(base, dataset=dataset, horizon=horizon, seed=2021, arm='b0',
                       protocol='e14', init='scratch', head=best['head'],
                       shared_gate=bool(best['shared']), variant_id=None,
                       variant={'gate_init': best['gate'], 'extension_lr_mult': best['ext'],
                                'communication_width': 2},
                       extension_lr_mult=float(best['ext']), backbone_lr_mult=float(best['bb']),
                       warm_source=None, epoch_test=False, source=source,
                       rank_sweep=int(rank),
                       b0_reference={'run': best['run'], 'mse': best['mse'], 'mae': best['mae']})
            name = f"{dataset}_H{horizon}_S2021__b0__r{rank}__p{best['period']}_{best['head']}"
            job['output'] = str(scratch / STAGE / 'runs' / name)
            path = scratch / STAGE / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def worker(job_path):
    import scripts.vldbj_m1.model as m
    from scripts.vldbj_m1 import proto
    job = json.loads(Path(job_path).read_text())
    m.RANK = int(job['rank_sweep'])
    proto.run_job(job_path)


def run_jobs(paths, gpus):
    tasks = queue.Queue()
    for path in paths:
        tasks.put(path)
    failures = []

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            out = Path(json.loads(Path(path).read_text())['output'])
            if (out / 'result.json').is_file():
                continue
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


def report(scratch):
    for dataset, horizon in SETTINGS:
        rows = []
        for rank in RANKS:
            hits = list((scratch / STAGE / 'runs').glob(f'{dataset}_H{horizon}_S2021__b0__r{rank}__*/result.json'))
            if not hits:
                rows.append((rank, None, None, None))
                continue
            r = json.loads(hits[0].read_text())['best']
            rows.append((rank, r['test_mse'], r['test_mae'], json.loads(hits[0].read_text())['epochs']))
        if all(x[1] is None for x in rows):
            continue
        base = next((x for x in rows if x[0] == 3 and x[1] is not None), None)
        out = ' '.join((f"r{r}:{m:.5f}" + (f"({(m - base[1]) / base[1] * 100:+.2f}%)" if base else ''))
                       if m is not None else f"r{r}:--" for r, m, a, e in rows)
        maes = ' '.join(f"r{r}:{a:.5f}" for r, m, a, e in rows if a is not None)
        print(f"{dataset}-{horizon}: {out}\n              MAE {maes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='6')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
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
