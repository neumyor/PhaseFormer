"""Write the job list of one experiment group.

    python -m scripts.biaxial.configs E1 > jobs_e1.jsonl

Screening set: ETTh1, ETTh2, ETTm1, Weather at horizons 96 and 720, seed 2021.
"""
from __future__ import annotations

import json
import sys

SCREEN = (('ETTh1', 24), ('ETTh2', 24), ('ETTm1', 48), ('Weather', 24))
HORIZONS = (96, 720)
SEED = 2021

M0 = {'arch': 'm0', 'd': 4, 'layers': 3}
BIAXIAL = {'d': 16, 'k': 8, 'p': 0, 'q': 1, 'layers': 3}


def screen_jobs(group, name, model, seeds=(SEED,), datasets=SCREEN, horizons=HORIZONS):
    for dataset, period in datasets:
        for horizon in horizons:
            for seed in seeds:
                yield {'id': f'{group}/{name}/{dataset}-{horizon}-s{seed}', 'group': f'{group}/{name}',
                       'dataset': dataset, 'horizon': horizon, 'seed': seed, 'period': period,
                       'model': model}


# Capacity controls: M0 widened to the parameter count (M0W-P) and to the FLOPs (M0W-F) of the bi-axial models.
M0W_P = {**M0, 'd': 24, 'heads': 2}
M0W_F = {**M0, 'd': 64, 'heads': 4}


def e0():
    yield from screen_jobs('E0', 'M0', M0)
    yield from screen_jobs('E0', 'M0W-P', M0W_P)
    yield from screen_jobs('E0', 'M0W-F', M0W_F)


def e1():
    for name, arch in (('B0', 'phase'), ('B1', 'temporal'), ('B2', 'dual'), ('B3', 'interleaved')):
        yield from screen_jobs('E1', name, {**BIAXIAL, 'arch': arch})


EXPERIMENTS = {'E0': e0, 'E1': e1}


def main():
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        for job in EXPERIMENTS[name]():
            print(json.dumps(job))


if __name__ == '__main__':
    main()
