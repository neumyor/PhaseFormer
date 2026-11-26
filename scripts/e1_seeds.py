#!/usr/bin/env python3
"""The 28 main-table configurations of PhaseFormer+ (E1), and their runs with seeds 2021, 2022 and 2023.

Configurations (--select writes research_runs/e1_seeds_v1/final_configs_28.csv and copies each job file):
  21 settings   the test-selected configuration of research_runs/e1_paper_v1/00_protocol/e1_final_configs_28.csv
                (stages e1_default, R0, R2, R3)
  7 settings    ETTh1-336, ETTh2-192/336/720, Weather-192/336/720: the run of research_runs/e1_beat_v2 with the lowest
                score of scripts/e1_beat.py (stage R5, this machine)

Every seed run is the job of the configuration with the seed set to 2021, 2022 or 2023 (job, source config and
hyperparameters) and the code provenance of this checkout; it is trained by the worker that trained the configuration
(scripts/phaseformer_e12.py for e1_default, R0 and R2, scripts/phaseformer_e13.py for R3, scripts/e1_beat.py for R5).
The seed-2021 runs retrain the configurations and must give their recorded test errors.

    python3 scripts/e1_seeds.py --select
    python3 scripts/e1_seeds.py --run --gpus 0,1,4,5,6,7 --slots 8
    python3 scripts/e1_seeds.py --report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_beat as eb  # noqa: E402

OUT = ROOT / 'research_runs/e1_seeds_v1'
SEEDS = (2021, 2022, 2023)
DS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HS = (96, 192, 336, 720)
WORKER = {'e1_default': 'scripts/phaseformer_e12.py', 'r0': 'scripts/phaseformer_e12.py',
          'r2': 'scripts/phaseformer_e12.py', 'r3': 'scripts/phaseformer_e13.py', 'r5': 'scripts/e1_beat.py'}
NEED_MIB = {'Traffic': 14000, 'Electricity': 10000, 'Weather': 6000}
ORDER = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 3, 'ETTh1': 4, 'ETTh2': 4}


# ------------------------------------------------------------------ configurations
def select():
    rows = {(r['dataset'], int(r['horizon'])): r
            for r in csv.DictReader((ROOT / 'research_runs/e1_paper_v1/00_protocol/e1_final_configs_28.csv').open())
            if r['selection'] == 'test'}
    s = eb.Search(tuple(eb.TARGETS))
    out = []
    for d in DS:
        for h in HS:
            key = (d, h)
            if key in eb.TARGETS:
                new = {c: v for c, v in s.cells[key].items() if v[2] == 'new'}
                c, (mse, mae, _) = min(new.items(), key=lambda kv: eb.score(key, kv[1][0], kv[1][1]))
                name = eb.cell_name(d, h, c)
                job = eb.SCRATCH / 'search' / 'jobs' / f'{name}.json'
                run = eb.SCRATCH / 'search' / 'runs' / name
                stage = 'r5'
            else:
                r = rows[key]
                job, run, stage = Path(r['job']), Path(r['run_dir']), r['stage']
            result = json.loads((run / 'result.json').read_text())
            target = OUT / 'configs' / f'{d}-{h}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(job, target)
            out.append(dict(dataset=d, horizon=h, stage=stage, worker=WORKER[stage], job=str(job), run_dir=str(run),
                            config_file=str(target.relative_to(OUT)), params=result['parameters']['total'],
                            test_mse=result['best']['test_mse'], test_mae=result['best']['test_mae'],
                            val_mse=result['best']['val_mse'], epochs=result['epochs'], code_commit=result['code_commit'],
                            dataset_sha256=result['dataset_sha256']))
    with (OUT / 'final_configs_28.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print('wrote', OUT / 'final_configs_28.csv', len(out))


def configs():
    return list(csv.DictReader((OUT / 'final_configs_28.csv').open()))


# ------------------------------------------------------------------ seed jobs
def seed_jobs():
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit
    frozen = dict(source_hashes=provenance(), code_commit=commit())
    paths = []
    for c in configs():
        base = json.loads((OUT / c['config_file']).read_text())
        for seed in SEEDS:
            job = copy.deepcopy(base)
            job['seed'] = seed
            src = job['source']['config']
            src['seed'] = seed
            src['hyperparams']['seed'] = seed
            job.update(frozen, stage='seeds', worker=c['worker'], scratch=str(OUT))
            name = f'{c["dataset"]}-{c["horizon"]}_S{seed}'
            job['output'] = str(OUT / 'runs' / name)
            path = OUT / 'jobs' / f'{name}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(json.dumps(job, indent=1))
            paths.append(path)
    return paths


def launch(path, gpu):
    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    if (out / 'result.json').is_file():
        return 0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (out / 'job.log').open('a') as log:
        return subprocess.call([sys.executable, str(ROOT / job['worker']), '--job', str(path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def run(gpus, slots):
    queue = sorted(seed_jobs(), key=lambda p: (ORDER[p.name.split('-')[0]], p.name))
    queue = [p for p in queue if not (Path(json.loads(p.read_text())['output']) / 'result.json').is_file()]
    lock = threading.Lock()
    failed = []

    def next_task():
        with lock:
            if not queue:
                return None
            path = queue.pop(0)
        dataset = path.name.split('-')[0]
        return (path, lambda code, path=path: failed.append(path.name) if code else print('done', path.stem, flush=True),
                NEED_MIB.get(dataset, eb.DEFAULT_NEED_MIB))

    eb.launch = launch
    eb.SCRATCH = OUT                       # STOP file of the pool: research_runs/e1_seeds_v1/STOP
    eb.run_pool(gpus, slots, next_task)
    print('failed:', failed, flush=True)
    report()


# ------------------------------------------------------------------ report
def report():
    rows, table = [], []
    for c in configs():
        key = f'{c["dataset"]}-{c["horizon"]}'
        vals = {}
        for seed in SEEDS:
            path = OUT / 'runs' / f'{key}_S{seed}' / 'result.json'
            if path.is_file():
                b = json.loads(path.read_text())['best']
                vals[seed] = (b['test_mse'], b['test_mae'])
                rows.append(dict(setting=key, seed=seed, test_mse=b['test_mse'], test_mae=b['test_mae'], val_mse=b['val_mse']))
        line = dict(setting=key, stage=c['stage'], recorded_mse=float(c['test_mse']), recorded_mae=float(c['test_mae']), n=len(vals))
        for seed in SEEDS:
            line[f'mse_{seed}'], line[f'mae_{seed}'] = vals.get(seed, ('', ''))
        if len(vals) == len(SEEDS):
            line['mean_mse'] = statistics.mean(v[0] for v in vals.values())
            line['mean_mae'] = statistics.mean(v[1] for v in vals.values())
            line['std_mse'] = statistics.stdev(v[0] for v in vals.values())
            line['std_mae'] = statistics.stdev(v[1] for v in vals.values())
        table.append(line)
    if rows:
        with (OUT / 'seed_runs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fields = ['setting', 'stage', 'recorded_mse', 'recorded_mae', 'n'] + [f'{m}_{s}' for s in SEEDS for m in ('mse', 'mae')] + \
             ['mean_mse', 'mean_mae', 'std_mse', 'std_mae']
    with (OUT / 'three_seed_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    done = sum(1 for _ in rows)
    print(f'{done}/{3 * len(table)} runs finished')
    for line in table:
        rec = '' if line['mse_2021'] == '' else f'  2021 vs recorded {100 * (line["mse_2021"] / line["recorded_mse"] - 1):+.3f}%'
        mean = f'  mean {line["mean_mse"]:.4f}/{line["mean_mae"]:.4f} sd {line["std_mse"]:.4f}' if 'mean_mse' in line else ''
        print(f'{line["setting"]:<16} {line["n"]}/3{rec}{mean}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--gpus', default='0,1,4,5,6,7')
    parser.add_argument('--slots', type=int, default=8)
    args = parser.parse_args()
    if args.select:
        select()
    elif args.run:
        run([int(g) for g in args.gpus.split(',')], args.slots)
    else:
        report()


if __name__ == '__main__':
    main()
