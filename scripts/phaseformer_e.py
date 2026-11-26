#!/usr/bin/env python3
"""PhaseFormer-E against B0H+ on the 28 settings of the main table, three seeds.

PhaseFormer-E is the structure selected by the structure ablation
(docs/PhaseFormer_E_design.md): no LayerNorm after the phase-token embedding, both
branches read the un-normalized tokens U, the trend branch reads only the mean of U
(no pooled history), absolute trend forecast, two static gates. The B0H+ arm is the
recorded model rerun with the same seeds and code, so the comparison is paired.

Every setting keeps the configuration recorded in
``docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`` (period, learning-rate groups, level
head, gate_init, shared gate, Huber delta); the model structure is the only difference.

    python3 scripts/phaseformer_e.py --check
    python3 scripts/phaseformer_e.py --run --gpus 6,6,6,6,7,7,7,7
    python3 scripts/phaseformer_e.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402

sa.SCRATCH = ROOT / 'research_runs/phaseformer_e_v1'
ARM_TO_PATCH = {'b0hp': 'rec', 'e': 'e'}
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)


def all_settings():
    rows = list(csv.DictReader(sa.settings_csv().open()))
    return tuple((r['dataset'], int(r['horizon'])) for r in rows)


def build_jobs(settings, seeds, arms):
    """The structure ablation builds jobs for its own arm names; map ours onto it."""
    sa.SETTINGS = settings
    renamed = tuple(ARM_TO_PATCH[a] for a in arms)
    paths = sa.build_jobs(settings, seeds, renamed)
    return paths


def check(settings):
    """Both arms build and run; E has no embedding LayerNorm and a token-only encoder."""
    import torch
    from scripts.vldbj_m1.proto import build_model
    sa.register_heads()
    for dataset, horizon in settings:
        models = {}
        for path in build_jobs(((dataset, horizon),), (2021,), ('b0hp', 'e')):
            job = json.loads(path.read_text())
            arm = job['ablation_arm']
            job = dict(job, smoke=True)
            job.pop('source_hashes', None)
            job.pop('code_commit', None)
            model, _, _, _ = build_model(job)
            sa.apply_structure(model, arm, job)
            out = sa.forward_once(model)
            if tuple(out.shape) != (2, model.pred_len, model.enc_in) or not torch.isfinite(out).all():
                raise SystemExit(f'{dataset}-{horizon} {arm}: bad output')
            models[arm] = model
        d = models['e'].latent_dim
        head_e = models['e'].lowrank_interaction_branch.lowrank_head
        head_b = models['rec'].lowrank_interaction_branch.lowrank_head
        ok = (type(models['e'].embedding.norm).__name__ == 'Identity' and head_e.encoder.in_features == d
              and head_b.encoder.in_features == head_b.pooled_len + d
              and type(models['rec'].embedding.norm).__name__ == 'LayerNorm')
        counts = {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}
        print(f'{dataset}-{horizon} ok={ok} params={counts}', flush=True)
        if not ok:
            raise SystemExit('check failed')
    print('check passed')


def report():
    rows = []
    for dataset, horizon in all_settings():
        for seed in sa.SEEDS:
            for arm, patch in ARM_TO_PATCH.items():
                for record in (sa.SCRATCH / patch / 'runs').glob(f'{dataset}_H{horizon}_S{seed}__*__{patch}/result.json'):
                    best = json.loads(record.read_text())['best']
                    rows.append({'dataset': dataset, 'horizon': horizon, 'seed': seed, 'arm': arm,
                                 'test_mse': best['test_mse'], 'test_mae': best['test_mae']})
    if not rows:
        raise SystemExit('no results')
    sa.SCRATCH.mkdir(parents=True, exist_ok=True)
    with (sa.SCRATCH / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value = {(r['dataset'], r['horizon'], r['seed'], r['arm']): r for r in rows}
    recorded = {(r['dataset'], int(r['horizon'])): (float(r['test_mse']), float(r['test_mae']))
                for r in csv.DictReader(sa.settings_csv().open())}
    lines = []
    for j, metric in ((0, 'test_mse'), (1, 'test_mae')):
        lines.append(f'\n== {metric}: three-seed mean, E against B0H+ rerun (paired), and the recorded seed-2021 B0H+')
        lines.append(f'{"setting":>16} {"recorded":>9} {"B0H+":>8} {"E":>8} {"E-B0H+":>8} {"E wins":>7}')
        per_dataset = {d: [] for d in ORDER}
        for dataset in ORDER:
            for horizon in HORIZONS:
                seeds = [s for s in sa.SEEDS if (dataset, horizon, s, 'b0hp') in value and (dataset, horizon, s, 'e') in value]
                if len(seeds) < len(sa.SEEDS):
                    lines.append(f'{dataset + "-" + str(horizon):>16}   incomplete ({len(seeds)}/3 seeds)')
                    continue
                b = statistics.mean(value[(dataset, horizon, s, 'b0hp')][metric] for s in seeds)
                e = statistics.mean(value[(dataset, horizon, s, 'e')][metric] for s in seeds)
                wins = sum(value[(dataset, horizon, s, 'e')][metric] < value[(dataset, horizon, s, 'b0hp')][metric] for s in seeds)
                change = 100 * (e / b - 1)
                per_dataset[dataset].append((b, e, change))
                lines.append(f'{dataset + "-" + str(horizon):>16} {recorded[(dataset, horizon)][j]:>9.4f} {b:>8.4f} {e:>8.4f} {change:>+7.2f}% {wins:>5}/3')
        lines.append(f'\n{"dataset mean":>16} {"B0H+":>8} {"E":>8} {"E-B0H+":>8}')
        done = [(d, v) for d, v in per_dataset.items() if len(v) == len(HORIZONS)]
        for dataset, vals in done:
            b, e = statistics.mean(v[0] for v in vals), statistics.mean(v[1] for v in vals)
            lines.append(f'{dataset:>16} {b:>8.4f} {e:>8.4f} {100 * (e / b - 1):>+7.2f}%')
        if len(done) == len(ORDER):
            b = statistics.mean(statistics.mean(v[0] for v in vals) for _, vals in done)
            e = statistics.mean(statistics.mean(v[1] for v in vals) for _, vals in done)
            lines.append(f'{"seven-dataset":>16} {b:>8.4f} {e:>8.4f} {100 * (e / b - 1):>+7.2f}%')
    text = '\n'.join(lines)
    print(text)
    (sa.SCRATCH / 'summary.txt').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='6,6,6,6,7,7,7,7', help='one entry per parallel slot')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    parser.add_argument('--seeds', default='')
    args = parser.parse_args()
    if args.job:
        sa.worker(args.job)
        return
    if args.report:
        report()
        return
    only = sa.parse_filter(args.only, str)
    settings = tuple(s for s in all_settings() if only is None or f'{s[0]}-{s[1]}' in only)
    seeds = tuple(sorted(sa.parse_filter(args.seeds, int) or sa.SEEDS))
    if args.check:
        check(settings)
        return
    paths = build_jobs(settings, seeds, ('b0hp', 'e'))
    print(f'{len(paths)} jobs under {sa.SCRATCH}', flush=True)
    if args.run:
        sa.run_jobs(paths, [int(g) for g in args.gpus.split(',')])
        report()


if __name__ == '__main__':
    main()
