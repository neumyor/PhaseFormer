#!/usr/bin/env python3
"""Stage-3 refinement around the stage-2 winner on Traffic H336.

Stage t1 fixed the cycle length: period 12 is best (0.387659) and period 24 is
essentially tied (0.387744), against a Golden of 0.385. This grid keeps period
12 and searches the extension knobs around the settings that produced it.

Kept outside ``scripts/vldbj_m1`` so the frozen-protocol hash is untouched.

    python3 scripts/traffic_t2_h336.py --gpus 1,6 --slots-per-gpu 3
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402
from scripts.vldbj_m1.wide import make_job, name  # noqa: E402

DATASET, HORIZON, SEED, HEAD = 'Traffic', 336, 2021, protocol.FINAL_HEAD
PERIODS = (12,)   # period 24 is a near tie; add it only if this grid fails
GATES = (0.05, 0.1, 0.15, 0.2, 0.3)
LRS = (0.1, 0.2, 0.3, 0.5)
WIDTHS = (2,)
BB_MAIN = 3.0
BB_EXTRA = ()


def job_name(variant, backbone_mult, period):
    return f"{name(DATASET, HORIZON, SEED, variant, backbone_mult, HEAD, None)}__p{period}"


def build_jobs(scratch, stage):
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': stage, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    jobs = []
    for period in PERIODS:
        src = copy.deepcopy(sources[(DATASET, HORIZON, SEED)])
        src['config']['period'] = period
        src['config']['hyperparams']['period_len'] = period
        for gate in GATES:
            for lr in LRS:
                for width in WIDTHS:
                    jobs.append((period, src, {'gate_init': gate, 'extension_lr_mult': lr,
                                               'communication_width': width}, BB_MAIN))
        if period == PERIODS[0]:
            best = {'gate_init': 0.5, 'extension_lr_mult': 0.1, 'communication_width': 2}
            for bb in BB_EXTRA:
                jobs.append((period, src, dict(best), bb))
    out = []
    for period, src, variant, bb in jobs:
        job = make_job(base, {(DATASET, HORIZON, SEED): src}, DATASET, HORIZON, SEED,
                       variant, bb, head=HEAD, shared=None)
        job['output'] = str(Path(base['scratch']) / stage / 'runs' /
                            job_name(variant, bb, period))
        path = Path(base['scratch']) / stage / 'jobs' / f"{Path(job['output']).name}.json"
        write_immutable(path, job)
        out.append(str(path))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path,
                        default=ROOT / 'research_runs/traffic_period_v1')
    parser.add_argument('--gpus', default='1,6')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--only-missing', action='store_true')
    args = parser.parse_args()

    paths = build_jobs(args.scratch, 't3')
    print(f't3: {len(paths)} jobs')
    if args.only_missing:
        kept = []
        for path in paths:
            import json
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
            if not (out / 'result.json').exists() and not (out / 'checkpoints').exists():
                kept.append(path)
        paths = kept
        print(f'only-missing: {len(paths)}')
    if args.prepare_only or not paths:
        return
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    print('launching on', gpus)
    try:
        run_jobs(paths, gpus)
    except RuntimeError as exc:
        print('first pass failures:', exc, flush=True)
        run_jobs(paths, gpus)


if __name__ == '__main__':
    main()
