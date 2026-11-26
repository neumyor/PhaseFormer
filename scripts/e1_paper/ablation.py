#!/usr/bin/env python3
"""Ablation of the trend pathway on E1 (paper replication, 05_ablation).

Arms, all on the selected E1 configuration (--selection test, the main tables, or val, the analysis) of each of the 28 settings, seed 2021, not retuned:
  single_gate  one scalar gate g = sigmoid(w) for the levels and the shapes, y = (1 - g) F + g F_b, started at the
               gate_init of the setting (the single-gate variant of the paper, scripts/b0hp_ablation.py)
  linear_last  the trend pathway replaced by one linear layer on the window, F_b = x_L 1 + W (x - x_L 1), W = 0 at the start
               (the linear variant of the paper: anchor = last value); the 720 H weights use the phase-pathway rate
  linear_mk    the same layer anchored on the mean of the last cycle, F_b = m_K 1 + W (x - m_K 1) (the anchor of E1)
  baseline     the selected configuration again (reproduction check, four settings)

    python3 scripts/e1_paper/ablation.py --prepare
    python3 scripts/e1_paper/ablation.py --run --gpus 4,5,6,7
    python3 scripts/e1_paper/ablation.py --report
    python3 scripts/e1_paper/ablation.py --job path/to/job.json         # worker
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE = ROOT / 'research_runs/e1_paper_v1'
OUT = BASE / '05_ablation'            # set per selection by set_selection()
SELECTION = 'test'
ARMS = ('single_gate', 'linear_last', 'linear_mk')
CHECK = (('ETTh1', 96), ('ETTm1', 96), ('Weather', 720), ('Traffic', 720))
HEAVY = ('Traffic', 'Electricity')
NEED_MIB = {'Traffic': 12000, 'Electricity': 10000, 'Weather': 6000}
MAX_PER_GPU, HEAVY_PER_GPU = 6, 5


def set_selection(selection):
    global OUT, SELECTION
    SELECTION = selection
    OUT = BASE / '05_ablation' / f'{selection}_selected'


# ------------------------------------------------------------------ worker
def linear_head(seq_len, pred_len, period, anchor):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LinearTrendHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len, self.pred_len = int(seq_len), int(pred_len)
            self.weight = nn.Parameter(torch.zeros(self.pred_len, self.seq_len))
            self.encoder = None

        def encode_latent(self, x):
            last = x[:, -1:, :] if anchor == 'last' else x[:, -period:, :].mean(1, keepdim=True)
            return x, last

        def decode_latent(self, latent, last):
            centered = (latent - last).permute(0, 2, 1).contiguous()
            return F.linear(centered, self.weight).permute(0, 2, 1).contiguous() + last.expand(-1, self.pred_len, -1)

        def forward(self, x):
            return self.decode_latent(*self.encode_latent(x))

    return LinearTrendHead()


def apply_variant(model, job):
    import scripts.b0hp_ablation as abl
    variant = job.get('e1_variant', 'baseline')
    if variant == 'single_gate':
        model.lowrank_dynamic_fusion = abl._simple_fusion(model.enc_in, float(job['variant']['gate_init']))
    elif variant in ('linear_last', 'linear_mk'):
        branch = model.lowrank_interaction_branch
        branch.lowrank_head = linear_head(model.seq_len, model.pred_len, model.period_len, variant.split('_')[1])
    elif variant != 'baseline':
        raise ValueError(variant)


def worker(job_path):
    import scripts.b0hp_ablation as abl
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e13 as pe13
    from scripts.vldbj_m1 import proto
    sa.register_heads()
    job = json.loads(Path(job_path).read_text())
    linear = job.get('e1_variant', '').startswith('linear')
    attach = abl.make_attach(dict(job, ablation_arm='linear_trend') if linear else job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        pe13.apply_structure(model, j['ablation_arm'], j)       # the E1 patches
        apply_variant(model, j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


# ------------------------------------------------------------------ jobs
def prepare():
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, write_immutable
    rows = [r for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open()) if r['selection'] == SELECTION]
    paths = []
    for r in rows:
        key = (r['dataset'], int(r['horizon']))
        arms = list(ARMS) + (['baseline'] if key in CHECK else [])
        for arm in arms:
            job = copy.deepcopy(json.loads(Path(r['job']).read_text()))
            job['e1_variant'] = arm
            job['stage'] = f'ablation_{arm}'
            job['scratch'] = str(OUT)
            job['source_hashes'] = provenance()
            job['code_commit'] = commit()
            name = f'{r["dataset"]}_H{r["horizon"]}_S2021__{arm}'
            job['output'] = str(OUT / 'runs' / arm / name)
            path = OUT / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    (OUT / 'manifest.json').write_text(json.dumps([str(p) for p in paths], indent=1))
    print(len(paths), 'jobs')
    return paths


def gpu_free():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    return {int(a): int(c) - int(b) for a, b, c in (line.split(',') for line in out.strip().splitlines())}


def run(paths, gpus):
    import scripts.b0hp_structure_ablation as sa
    jobs = []
    for path in paths:
        job = json.loads(Path(path).read_text())
        out = Path(job['output'])
        if (out / 'result.json').is_file() or sa.running(out):
            continue
        jobs.append((Path(path), job, out))
    jobs.sort(key=lambda t: (-NEED_MIB.get(t[1]['dataset'], 5000), t[0].name))
    lock = threading.Lock()
    alive = {g: [] for g in gpus}
    failures, attempts = [], {}

    def pick(gpu):
        with lock:
            alive[gpu] = [(p, d) for p, d in alive[gpu] if p.poll() is None]
            if len(alive[gpu]) >= MAX_PER_GPU:
                return None
            heavy = sum(1 for _, d in alive[gpu] if d in HEAVY)
            free = gpu_free().get(gpu, 0)
            for item in jobs:
                d = item[1]['dataset']
                if d in HEAVY and heavy >= HEAVY_PER_GPU:
                    continue
                if free < NEED_MIB.get(d, 5000) + 3000:
                    continue
                jobs.remove(item)
                return item
            return None

    def slot(gpu, number):
        import shutil
        while True:
            with lock:
                if not jobs:
                    return
            item = pick(gpu)
            if item is None:
                time.sleep(30 + 5 * number)
                continue
            path, job, out = item
            if (out / 'checkpoints').exists():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (out / 'job.log').open('a') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--job', str(path)], cwd=ROOT,
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                alive[gpu].append((process, job['dataset']))
            time.sleep(25)
            code = process.wait()
            if code == 0:
                print('done', out.name, flush=True)
                continue
            tail = (out / 'job.log').read_text()[-3000:]
            with lock:
                attempts[path] = attempts.get(path, 0) + 1
                if 'out of memory' in tail and attempts[path] < 4:
                    jobs.append(item)
                    print('oom, requeued', out.name, flush=True)
                else:
                    failures.append(str(path))
                    print('FAILED', out.name, flush=True)
            if (out / 'checkpoints').exists() and not (out / 'result.json').exists():
                shutil.rmtree(out, ignore_errors=True)

    threads = [threading.Thread(target=slot, args=(g, i)) for g in gpus for i in range(MAX_PER_GPU)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


def report():
    rows = []
    for arm in ARMS + ('baseline',):
        for path in sorted((OUT / 'runs' / arm).glob('*/result.json')):
            result = json.loads(path.read_text())
            best = result['best']
            rows.append(dict(dataset=result['dataset'], horizon=result['horizon'], arm=arm, val_mse=best['val_mse'],
                             test_mse=best['test_mse'], test_mae=best['test_mae'], epochs=result['epochs'],
                             params=result['parameters']['total'], run_dir=str(path.parent)))
    main = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open())
            if r['selection'] == SELECTION}
    for r in rows:
        m = main[(r['dataset'], int(r['horizon']))]
        r['e1_test_mse'], r['e1_test_mae'] = float(m['test_mse']), float(m['test_mae'])
    with (OUT / 'ablation_results_e1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), 'runs ->', OUT / 'ablation_results_e1.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--selection', choices=('test', 'val'), default='test',
                        help='which selected E1 configuration of 00_protocol/e1_final_configs_28.csv the variants start from')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    set_selection(args.selection)
    if args.job:
        worker(args.job)
    elif args.prepare:
        prepare()
    elif args.run:
        paths = [Path(p) for p in json.loads((OUT / 'manifest.json').read_text())]
        run(paths, [int(g) for g in args.gpus.split(',')])
        report()
    elif args.report:
        report()


if __name__ == '__main__':
    main()
