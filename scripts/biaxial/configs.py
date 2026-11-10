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


def resolve(model, period):
    """Replace phase-length fractions such as 'P/2' by integers."""
    return {k: (period // int(v[2:]) if isinstance(v, str) and v.startswith('P/') else v)
            for k, v in model.items()}


def screen_jobs(group, name, model, seeds=(SEED,), datasets=SCREEN, horizons=HORIZONS):
    for dataset, period in datasets:
        model_p = resolve(model, period)
        for horizon in horizons:
            for seed in seeds:
                yield {'id': f'{group}/{name}/{dataset}-{horizon}-s{seed}', 'group': f'{group}/{name}',
                       'dataset': dataset, 'horizon': horizon, 'seed': seed, 'period': period,
                       'model': model_p}


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


def e2():
    """Depth 1-4 of the two layouts; interleaved d=20 matches the FLOPs of the independent branches."""
    for n in (1, 2, 3, 4):
        yield from screen_jobs('E2', f'dual-N{n}', {**BIAXIAL, 'arch': 'dual', 'layers': n})
        yield from screen_jobs('E2', f'interleaved-N{n}', {**BIAXIAL, 'arch': 'interleaved', 'layers': n})
        yield from screen_jobs('E2', f'interleaved-d20-N{n}',
                               {**BIAXIAL, 'arch': 'interleaved', 'layers': n, 'd': 20})
    for n in (2, 4):
        yield from screen_jobs('E2', f'hybrid-N{n}', {**BIAXIAL, 'arch': 'hybrid', 'layers': n})


E3_AXES = (
    ('k4', {'k': 4}), ('k16', {'k': 16}), ('d8', {'d': 8}), ('d32', {'d': 32}),
    ('p-half', {'p': 'P/2'}), ('p-quarter', {'p': 'P/4'}), ('q2', {'q': 2}), ('q4', {'q': 4}),
    ('wp-identity', {'wp': 'identity'}), ('wt-poly', {'w_init': 'poly'}),
)


def e3(base):
    """One-axis scan around the E2 backbone ``base`` (center included)."""
    yield from screen_jobs('E3', 'center', base)
    for name, change in E3_AXES:
        yield from screen_jobs('E3', name, {**base, **change})


E4_DECODERS = ('direct', 'basis_fixed', 'basis_learn', 'hybrid_fixed', 'hybrid_learn')
E4_ANCHORS = ('last_cycle', 'last_cycle_mean', 'none')


def e4(base):
    """Decoders on the E3 backbone ``base`` with the x_L anchor, then the other anchors on the direct
    decoder and on the decoder given as ``base['best_decoder']`` (removed before building the model)."""
    base = dict(base)
    best = base.pop('best_decoder', 'hybrid_learn')
    for decoder in E4_DECODERS:
        yield from screen_jobs('E4', decoder, {**base, 'decoder': decoder, 'anchor': 'x_last'})
    for decoder in sorted({'direct', best}):
        for anchor in E4_ANCHORS:
            yield from screen_jobs('E4', f'{decoder}+{anchor}', {**base, 'decoder': decoder, 'anchor': anchor})


def e5(base):
    """Mean-phase-token diagnostics: checkpoints of M0 and of the backbone, and the injected-token ablation."""
    yield from screen_jobs('E5', 'm0', M0)
    yield from screen_jobs('E5', 'backbone', base)
    yield from screen_jobs('E5', 'mean-token', {**base, 'mean_token': True})


ALL_DATASETS = (('ETTh1', 24), ('ETTh2', 24), ('ETTm1', 48), ('ETTm2', 24), ('Weather', 24),
                ('Electricity', 24), ('Traffic', 24))


def e7(models):
    """Final evaluation: ``models`` maps a name to its model dict; 7 datasets x 4 horizons x 3 seeds."""
    for name, model in models.items():
        yield from screen_jobs('E7', name, model, seeds=(2021, 2022, 2023), datasets=ALL_DATASETS,
                               horizons=(96, 192, 336, 720))


def e01_seeds():
    """Seeds 2022 and 2023 of every E0 and E1 group, under the same group names."""
    groups = [('E0', 'M0', M0), ('E0', 'M0W-P', M0W_P), ('E0', 'M0W-F', M0W_F)]
    groups += [('E1', n, {**BIAXIAL, 'arch': a}) for n, a in
               (('B0', 'phase'), ('B1', 'temporal'), ('B2', 'dual'), ('B3', 'interleaved'))]
    for group, name, model in groups:
        yield from screen_jobs(group, name, model, seeds=(2022, 2023))


EXPERIMENTS = {'E01S': e01_seeds, 'E0': e0, 'E1': e1, 'E2': e2, 'E3': e3, 'E4': e4, 'E5': e5, 'E7': e7}


def main():
    name = sys.argv[1]
    if name in ('E3', 'E4', 'E5', 'E7'):
        jobs = EXPERIMENTS[name](json.loads(sys.argv[2]))
    else:
        jobs = EXPERIMENTS[name]()
    for job in jobs:
        print(json.dumps(job))


if __name__ == '__main__':
    main()
