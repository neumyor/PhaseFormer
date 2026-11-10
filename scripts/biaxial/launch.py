"""Start detached workers for a job file, balancing the load over GPUs.

    python -m scripts.biaxial.launch jobs.jsonl results.jsonl --gpus 4,5 --per-gpu 6

Jobs are ordered by dataset cost, heaviest first, and dealt out round-robin so
every worker gets a similar mix.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COST = {'Traffic': 200, 'Electricity': 60, 'Weather': 20, 'ETTm1': 6, 'ETTm2': 6, 'ETTh1': 1, 'ETTh2': 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jobs')
    parser.add_argument('results')
    parser.add_argument('--gpus', default='4,5')
    parser.add_argument('--per-gpu', type=int, default=6)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    jobs.sort(key=lambda j: -COST.get(j['dataset'], 10))
    shards = len(gpus) * args.per_gpu
    out = Path(args.results)
    ordered = out.with_suffix('.jobs_ordered.jsonl')
    ordered.write_text(''.join(json.dumps(j) + '\n' for j in jobs))
    log_dir = out.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    for shard in range(shards):
        gpu = gpus[shard % len(gpus)]
        log = open(log_dir / f'{out.stem}_w{shard}.log', 'a')
        subprocess.Popen(
            [sys.executable, '-m', 'scripts.biaxial.worker', '--jobs', str(ordered), '--results', str(out),
             '--gpu', str(gpu), '--shard', str(shard), '--shards', str(shards)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'{len(jobs)} jobs, {shards} workers on GPUs {gpus}')


if __name__ == '__main__':
    main()
