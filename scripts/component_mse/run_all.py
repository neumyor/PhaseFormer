#!/usr/bin/env python3
"""Queue every run of the component-MSE experiment across the eight GPUs.

Seed 2021 of all six models and all four settings is queued first, so the first
comparison is complete before seeds 2022/2023 start; inside a seed the slow
Traffic runs go first so they do not form the tail. Each task is one subprocess
pinned to one GPU; finished tasks are skipped, so rerunning resumes.

    python -m scripts.component_mse.run_all --scratch <dir> --prepare
    python -m scripts.component_mse.run_all --scratch <dir> --run --gpus 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.component_mse import dump, models  # noqa: E402
from scripts.component_mse.phaseformer import (  # noqa: E402
    DATASETS, HORIZONS, SEEDS, job_name, prepare as prepare_phaseformer,
)

SEED_ORDER = {seed: index for index, seed in enumerate(SEEDS)}


def phaseformer_result(scratch, dataset, horizon, seed, arm):
    return Path(scratch) / 'component_mse' / 'runs' / job_name(dataset, horizon, seed, arm) / 'result.json'


PHASE_PRIORITY = {'Traffic': 0, 'Weather': 1, 'ETTm1': 2, 'ETTm2': 2, 'ETTh1': 3, 'ETTh2': 3}
BASE_PRIORITY = {'Traffic': 4, 'ETTm1': 5, 'ETTm2': 5, 'Weather': 6, 'ETTh1': 7, 'ETTh2': 7}


def task_key(task):
    """Slowest and most informative first: seed, then arm/dataset, then setting."""
    model, dataset, horizon, seed = task['model'], task['dataset'], task['horizon'], task['seed']
    stage = (PHASE_PRIORITY if model.startswith('PhaseFormer') else BASE_PRIORITY)[dataset]
    return (SEED_ORDER[seed], stage, model, dataset, horizon)


def build_tasks(scratch):
    tasks = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for model in ('PhaseFormer', 'PhaseFormerLA'):
                    arm = 'pf' if model == 'PhaseFormer' else 'm1'
                    tasks.append({'kind': 'phaseformer', 'model': model, 'arm': arm,
                                  'dataset': dataset, 'horizon': horizon, 'seed': seed})
                for model in models.BASELINES:
                    tasks.append({'kind': 'baseline', 'model': model, 'dataset': dataset,
                                  'horizon': horizon, 'seed': seed})
    return sorted(tasks, key=task_key)


def done(task, dump_root):
    return dump.result_path(dump_root, task['model'], task['dataset'],
                            task['horizon'], task['seed']).exists()


def command(task, scratch, dump_root, gpu):
    base = [sys.executable, '-m']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if task['kind'] == 'baseline':
        return base + ['scripts.component_mse.train_baseline',
                       '--model', task['model'], '--dataset', task['dataset'],
                       '--horizon', str(task['horizon']), '--seed', str(task['seed']),
                       '--gpu', str(gpu), '--dump-root', str(dump_root)], environment
    job = Path(scratch) / 'component_mse' / 'jobs' / f'{job_name(**{k: task[k] for k in ("dataset", "horizon", "seed", "arm")})}.json'
    return base + ['scripts.component_mse.phaseformer', '--job', str(job),
                   '--dump-root', str(dump_root)], environment


def run(tasks, scratch, dump_root, gpus):
    logs = Path(scratch) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    queue = list(tasks)
    failures = []

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                task = queue.pop(0)
            if done(task, dump_root):
                continue
            label = f"{task['model']}_{task['dataset']}_H{task['horizon']}_S{task['seed']}"
            log_path = logs / f'{label}.log'
            cmd, environment = command(task, scratch, dump_root, gpu)
            started = time.monotonic()
            with open(log_path, 'a') as log:
                log.write(f'$ {" ".join(cmd)}\n')
                log.flush()
                code = subprocess.call(cmd, cwd=ROOT, env=environment, stdout=log, stderr=log)
            elapsed = time.monotonic() - started
            with lock:
                if code == 0:
                    print(f'[gpu{gpu}] done {label} in {elapsed/60:.1f} min', flush=True)
                else:
                    print(f'[gpu{gpu}] FAILED {label} (exit {code}), see {log_path}', flush=True)
                    failures.append((label, code, str(log_path)))

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--dump-root', type=Path, default=None)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    dump_root = args.dump_root or args.scratch / 'predictions'
    if args.prepare:
        jobs = prepare_phaseformer(args.scratch, 'component_mse')
        tasks = build_tasks(args.scratch)
        (args.scratch / 'tasks.json').write_text(json.dumps(tasks, indent=1))
        print(f'prepared {len(jobs)} PhaseFormer jobs and {len(tasks)} tasks')
    if args.list:
        for task in build_tasks(args.scratch):
            print(task)
    if args.run:
        tasks = build_tasks(args.scratch)
        gpus = [int(g) for g in args.gpus.split(',')]
        failures = run(tasks, args.scratch, dump_root, gpus)
        print(json.dumps({'failures': failures}, indent=1))
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
