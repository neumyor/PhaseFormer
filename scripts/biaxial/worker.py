"""Run the jobs of one shard on one GPU.

    python -m scripts.biaxial.worker --jobs jobs.jsonl --results results.jsonl \
        --gpu 4 --shard 0 --shards 6 [--data-root DIR]

A job already present in the results file is skipped, so an interrupted worker
resumes by being started again with the same arguments.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, row):
    with open(path, 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row) + '\n')
        fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()

    import torch
    from scripts.biaxial.train import load_splits, run_job
    device = torch.device(f'cuda:{args.gpu}')
    torch.set_num_threads(1)
    jobs = read_jsonl(args.jobs)[args.shard::args.shards]
    cache = {}
    for job in jobs:
        if any(r['id'] == job['id'] for r in read_jsonl(args.results)):
            continue
        key = (job['dataset'], job['horizon'])
        if key not in cache:
            cache = {key: load_splits(*key, data_root=args.data_root)}
        splits, batch = cache[key]
        try:
            row = run_job(job, splits, batch, device)
        except Exception:
            append_jsonl(str(Path(args.results).with_suffix('.errors.jsonl')),
                         {'id': job['id'], 'error': traceback.format_exc()})
            continue
        ckpt_dir = Path(args.results).parent / 'ckpt'
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({k: v.cpu() for k, v in row.pop('state').items()}, ckpt_dir / (job['id'].replace('/', '__') + '.pt'))
        append_jsonl(args.results, row)
        print(job['id'], round(row['val_mse'], 5), round(row['test_mse'], 5), flush=True)


if __name__ == '__main__':
    main()
