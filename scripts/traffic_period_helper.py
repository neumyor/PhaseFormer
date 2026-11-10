#!/usr/bin/env python3
"""Start only the untouched Traffic-period jobs on a set of GPUs.

Lives outside ``scripts/vldbj_m1`` on purpose: that directory is inside the
frozen-protocol hash, so a helper that edited it would invalidate every job
already on disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1.proto import run_jobs  # noqa: E402
from scripts.vldbj_m1.traffic_period_search import stage_t1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--gpus', default='2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    jobs = stage_t1(args.scratch, 't1')
    def idle(job):
        out = Path(job['output'])
        if (out / 'result.json').exists():
            return False          # finished
        if (out / 'checkpoints').exists():
            return False          # running or abandoned mid-training
        return True               # never started, or left a log by a failed attempt

    untouched = [job for job in jobs if idle(job)]
    paths = [str(args.scratch / 't1' / 'jobs' / f"{Path(job['output']).name}.json")
             for job in untouched]
    print(f'untouched {len(untouched)} of {len(jobs)} jobs', flush=True)
    if args.dry_run or not paths:
        return
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    run_jobs(paths, gpus)


if __name__ == '__main__':
    main()
