#!/usr/bin/env python3
"""Split the extension learning rate into a gate rate and a level-readout rate.

The b0 search used one multiplier for every extension parameter.  The gate-freeze
test showed that this single knob does two different jobs: it carries the fusion
gate to its final value (Weather-336, ETTh2-720) and it trains the level readout
(ETTm2-96, whose gate stays near zero while the readout grows ~35x).  Here the two
get their own multiplier:

  gate group     lowrank_interaction_gate, lowrank_dynamic_fusion.shape_bias_delta
  readout group  lowrank_interaction_branch.* (level encoder and mix/decoder)

The gate starts at the value the best cell's gate ended with, so the question is
whether a correct starting gate still needs extreme rates once the two are separated.

Per setting (seed 2021, every other knob as the b0 best cell):
  grid     gate multiplier {1, 10, 100} x readout multiplier {1, 10, 100, 1000}
  control  both multipliers = the best cell's single rate, gate init = the best
           cell's own init; reproduces the best cell if the split is equivalent.

Kept outside ``scripts/vldbj_m1``; the optimizer is replaced by patching
``proto.attach_lr_groups`` in the worker process.

    python3 scripts/lr_split_run.py --check
    python3 scripts/lr_split_run.py --run --gpus 1,2,3,7 --slots-per-gpu 2
    python3 scripts/lr_split_run.py --report
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 2021
DEFAULT_SCRATCH = ROOT / 'research_runs/lr_split_v1'
STAGE = 'v1'
SETTINGS = (('ETTh1', 336), ('ETTm2', 96))
GATE_MULTS = (1.0, 10.0, 100.0)
READOUT_MULTS = (1.0, 10.0, 100.0, 1000.0)
GATE_PREFIXES = ('lowrank_interaction_gate', 'lowrank_dynamic_fusion.')
READOUT_PREFIX = 'lowrank_interaction_branch.'


def trained_level_gate(best):
    from scripts.gate_fix_run import run_dir_of, trained_gates
    logits, _ = trained_gates(run_dir_of(best['run']))
    mean = sum(1.0 / (1.0 + math.exp(-x)) for x in logits) / len(logits)
    return round(min(max(mean, 1e-4), 0.999), 3)


def cell_name(dataset, horizon, kind, gate_init, gate_mult, readout_mult, bb):
    return (f"{dataset}_H{horizon}_S{SEED}__{kind}__g{gate_init:g}_gm{gate_mult:g}_rm{readout_mult:g}_bb{bb:g}")


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
    for dataset, horizon in SETTINGS:
        best = winners(results, dataset, horizon)
        start = trained_level_gate(best)
        cells = [('control', best['gate'], best['ext'], best['ext'])]
        cells += [('grid', start, gm, rm) for gm in GATE_MULTS for rm in READOUT_MULTS]
        for kind, gate_init, gate_mult, readout_mult in cells:
            cell = dict(best, gate=gate_init, ext=gate_mult)
            job = job_for(base, sources, cell, 'history')
            job['lr_split'] = {'gate_mult': float(gate_mult), 'readout_mult': float(readout_mult),
                               'kind': kind, 'best_mse': best['mse'], 'best_run': best['run'],
                               'best_gate_init': best['gate'], 'best_ext': best['ext']}
            name = cell_name(dataset, horizon, kind, gate_init, gate_mult, readout_mult, best['bb'])
            job['output'] = str(scratch / STAGE / 'runs' / name)
            path = scratch / STAGE / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


# ---------------------------------------------------------------- optimizer
def make_attach(split):
    """Adam with three groups: backbone, gate, level readout."""
    def attach(model, backbone_mult, extension_mult):
        import types
        import torch
        base_lr = float(model.args.training_args.learning_rate)

        def configure_optimizers(self):
            groups = {'backbone': [], 'gate': [], 'readout': []}
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith(READOUT_PREFIX):
                    groups['readout'].append(p)
                elif name.startswith(GATE_PREFIXES):
                    groups['gate'].append(p)
                else:
                    groups['backbone'].append(p)
            mult = {'backbone': backbone_mult, 'gate': split['gate_mult'], 'readout': split['readout_mult']}
            optimizer = torch.optim.Adam([{'params': v, 'lr': base_lr * mult[k], 'name': k}
                                          for k, v in groups.items() if v])
            if self.args.training_args.lr_schedule_config.type == 'cos':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args.training_args.lr_schedule_config.tmax, eta_min=1e-8)
                return [optimizer], [scheduler]
            return optimizer

        model.configure_optimizers = types.MethodType(configure_optimizers, model)
    return attach


def worker(job_path):
    from scripts.vldbj_m1 import proto
    job = json.loads(Path(job_path).read_text())
    attach = make_attach(job['lr_split'])
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
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


def check(paths):
    """Each group exists with the intended learning rate and no parameter is lost."""
    from scripts.vldbj_m1.proto import build_model
    seen = set()
    for path in paths:
        job = json.loads(Path(path).read_text())
        key = (job['dataset'], job['horizon'], job['lr_split']['kind'])
        model, _, _, _ = build_model(job)
        make_attach(job['lr_split'])(model, job['backbone_lr_mult'], job['extension_lr_mult'])
        opt = model.configure_optimizers()
        opt = opt[0][0] if isinstance(opt, tuple) or isinstance(opt, list) else opt
        base = float(model.args.training_args.learning_rate)
        got = {g['name']: (sum(p.numel() for p in g['params']), g['lr'] / base) for g in opt.param_groups}
        total = sum(n for n, _ in got.values())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        want = job['lr_split']
        ok = (total == trainable and abs(got['gate'][1] - want['gate_mult']) < 1e-9
              and abs(got['readout'][1] - want['readout_mult']) < 1e-9
              and abs(got['backbone'][1] - job['backbone_lr_mult']) < 1e-9)
        if key not in seen or not ok:
            seen.add(key)
            print(f"{key[0]}-{key[1]} {key[2]:>7} {Path(job['output']).name[:62]:62} "
                  f"groups={ {k: v for k, v in got.items()} } {'' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit('optimizer group check failed')
    print('check passed')


def report(scratch):
    for dataset, horizon in SETTINGS:
        jobs = sorted((scratch / STAGE / 'jobs').glob(f'{dataset}_H{horizon}_*.json'))
        cells, best_mse, ctrl = {}, None, None
        for path in jobs:
            job = json.loads(path.read_text())
            record = Path(job['output']) / 'result.json'
            sp = job['lr_split']
            best_mse = sp['best_mse']
            if not record.is_file():
                continue
            mse = json.loads(record.read_text())['best']['test_mse']
            if sp['kind'] == 'control':
                ctrl = (mse, sp)
            else:
                cells[(sp['gate_mult'], sp['readout_mult'])] = mse
                start = job['variant']['gate_init']
        if best_mse is None:
            continue
        print(f"\n{dataset}-{horizon}: original best cell {best_mse:.4f}", end='')
        if ctrl:
            print(f" | control (split, both x{ctrl[1]['readout_mult']:g}, init g{ctrl[1]['best_gate_init']:g}) "
                  f"{ctrl[0]:.4f} ({(ctrl[0] - best_mse) / best_mse * 100:+.2f}%)")
        else:
            print()
        if cells:
            print(f"  gate init {start:g}; entries are MSE and change vs the original best cell")
            corner = 'gate x / readout x'
            print(f"  {corner:>20} " + ' '.join(f'{r:>14g}' for r in READOUT_MULTS))
            for gm in GATE_MULTS:
                row = []
                for rm in READOUT_MULTS:
                    mse = cells.get((gm, rm))
                    row.append('-' if mse is None else f"{mse:.4f} {(mse - best_mse) / best_mse * 100:+5.1f}%")
                print(f"  {gm:>20g} " + ' '.join(f'{c:>14}' for c in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='1,2,3,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
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
    elif args.run:
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        run_jobs(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
