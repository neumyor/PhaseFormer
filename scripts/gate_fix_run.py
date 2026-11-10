#!/usr/bin/env python3
"""Does an extreme (gate_init, extension lr) only carry the fusion gate to where it belongs?

In the b0 search the best cell of the hard settings sits at gate 0.999 / lr 40 or
gate 0.0001 / lr 1000, while the trained gate ends somewhere ordinary.  This script
freezes the two fusion parameters (the level logit ``lowrank_interaction_gate`` and
the shape offset ``shape_bias_delta``) at the values the best cell ended with and
trains everything else.  If the extreme learning rate was only transporting the gate,
the frozen-gate runs should match the best cell; if it also speeds up the level
readout, the run with the default extension rate (``ext_lr = 1``) should fall behind
the run that keeps the best cell's rate.

Variants per setting (seed 2021, every other knob as the best b0 cell):

  fix_ch_keep   gates frozen at the per-channel trained values, best-cell ext lr
  fix_ch_one    same, ext lr = 1
  fix_mean_keep gates frozen at the channel mean (one level and one shape value)
  fix_mean_one  same, ext lr = 1

Lives outside ``scripts/vldbj_m1`` so the frozen-protocol hash stays valid while other
searches run.  The gate values are read from the best cell's checkpoint when the job
is written and stored in the job file, so the worker loads no checkpoint.

    python3 scripts/gate_fix_run.py --prepare
    python3 scripts/gate_fix_run.py --check
    python3 scripts/gate_fix_run.py --run --gpus 1,2,3,7 --slots-per-gpu 1
    python3 scripts/gate_fix_run.py --report
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 2021
DEFAULT_SCRATCH = ROOT / 'research_runs/gate_fix_v1'
STAGE = 'v1'
# setting -> why it is in the test
SETTINGS = {('Weather', 336): 'hardest for B0H+; best cell g0.999 lr40',
            ('ETTh2', 720): 'B0H+ collapse; best cell g0.9 lr10',
            ('ETTh1', 336): 'largest search spread; best cell g0.9 lr10',
            ('ETTm2', 96): 'gate stays ~0 but lr 1000; readout grows large'}
VARIANTS = (('fix_ch_keep', 'channel', True), ('fix_ch_one', 'channel', False),
            ('fix_mean_keep', 'mean', True), ('fix_mean_one', 'mean', False))


def run_dir_of(name):
    from scripts.b0_search import DEFAULT_SCRATCH as B0
    hits = list(B0.glob(f'*/runs/{name}'))
    if not hits:
        raise SystemExit(f'run directory not found: {name}')
    return hits[0]


def trained_gates(run_dir):
    """Level logit (C,) and shape offset (C,) of the best checkpoint of a run."""
    import torch
    record = json.loads((run_dir / 'result.json').read_text())
    state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    gate = state['lowrank_interaction_gate'].float().flatten()
    delta = state['lowrank_dynamic_fusion.shape_bias_delta'].float().flatten()
    return gate.tolist(), delta.tolist()


def build_jobs(scratch):
    from scripts.ablation_no_amplitude import ROWS
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    from scripts.b0h_run import job_for
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    results = load_results(B0)
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for (dataset, horizon), note in SETTINGS.items():
        best = winners(results, dataset, horizon)
        gate, delta = trained_gates(run_dir_of(best['run']))
        for tag, mode, keep in VARIANTS:
            ext = best['ext'] if keep else 1.0
            job = job_for(base, sources, best, 'history')
            job['extension_lr_mult'] = float(ext)
            job['variant'] = dict(job['variant'], extension_lr_mult=float(ext))
            if mode == 'mean':
                level = [sum(gate) / len(gate)]
                shape = [sum(delta) / len(delta)]
            else:
                level, shape = gate, delta
            job['gate_fix'] = {'mode': mode, 'level_logit': level, 'shape_delta': shape,
                               'source_run': best['run'], 'best_ext': best['ext'], 'best_mse': best['mse']}
            name = f"{dataset}_H{horizon}_S{SEED}__{tag}__g{best['gate']:g}_lr{ext:g}_bb{best['bb']:g}"
            job['output'] = str(scratch / STAGE / 'runs' / name)
            path = scratch / STAGE / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


# ---------------------------------------------------------------- worker
def apply_gate_fix(model, fix):
    import torch
    gate = model.lowrank_interaction_gate
    delta = model.lowrank_dynamic_fusion.shape_bias_delta
    for param, values in ((gate, fix['level_logit']), (delta, fix['shape_delta'])):
        tensor = torch.tensor(values, dtype=param.dtype)
        if tensor.numel() == 1:
            tensor = tensor.expand(param.numel())
        elif tensor.numel() != param.numel():
            raise ValueError(f'gate_fix size {tensor.numel()} != parameter size {param.numel()}')
        param.data.copy_(tensor.reshape(param.shape))
        param.requires_grad_(False)


def worker(job_path):
    from scripts.vldbj_m1 import proto
    original = proto.build_model

    def patched(job):
        model, args, cfg, common = original(job)
        if job.get('gate_fix'):
            apply_gate_fix(model, job['gate_fix'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.run_job(job_path)


# ---------------------------------------------------------------- driver
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
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
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


def check(paths):
    """Frozen, equal to the stored values, absent from the optimizer groups."""
    import torch
    from scripts.vldbj_m1.proto import build_model
    for path in paths:
        job = json.loads(Path(path).read_text())
        model, _, _, _ = build_model(job)
        apply_gate_fix(model, job['gate_fix'])
        gate = model.lowrank_interaction_gate
        delta = model.lowrank_dynamic_fusion.shape_bias_delta
        want_g = job['gate_fix']['level_logit']
        got_g = gate.flatten().tolist()
        ok = (not gate.requires_grad and not delta.requires_grad
              and all(abs(a - (want_g[i] if len(want_g) > 1 else want_g[0])) < 1e-6 for i, a in enumerate(got_g)))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{Path(job['output']).name:70} gate={tuple(gate.shape)} frozen={ok} "
              f"level_gate_mean={torch.sigmoid(gate).mean().item():.3f} trainable={trainable}")
        if not ok:
            raise SystemExit('gate freeze check failed')
    print('check passed')


def report(scratch):
    from scripts.b0_search import load_results, winners
    results = load_results(__import__('scripts.b0_search', fromlist=['x']).DEFAULT_SCRATCH)
    print(f"{'setting':>13} | {'best cell':>9} {'(g, lr)':>14} | " +
          ' '.join(f'{t:>14}' for t, _, _ in VARIANTS))
    for (dataset, horizon) in SETTINGS:
        best = winners(results, dataset, horizon)
        cells = []
        for tag, _, keep in VARIANTS:
            ext = best['ext'] if keep else 1.0
            name = f"{dataset}_H{horizon}_S{SEED}__{tag}__g{best['gate']:g}_lr{ext:g}_bb{best['bb']:g}"
            record = scratch / STAGE / 'runs' / name / 'result.json'
            if record.is_file():
                mse = json.loads(record.read_text())['best']['test_mse']
                cells.append(f"{mse:.4f} {(mse - best['mse']) / best['mse'] * 100:>+5.1f}%")
            else:
                cells.append('-')
        print(f"{dataset + '-' + str(horizon):>13} | {best['mse']:>9.4f} {'g%g lr%g' % (best['gate'], best['ext']):>14} | " +
              ' '.join(f'{c:>14}' for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='1,2,3,7')
    parser.add_argument('--slots-per-gpu', type=int, default=1)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
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
    if args.check:
        check(paths)
        return
    if args.run:
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        run_jobs(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
