#!/usr/bin/env python3
"""PhaseFormer-E1 and -E2: two cumulative structure changes on PhaseFormer-E.

Each version adds one change to the previous one (docs/PhaseFormer_E12_design.md):

  E1 = E  + the trend forecast is anchored on the mean of the last cycle instead of the
            last value, so the anchor and the trend encoder (slot mean of U, a linear map
            of the cycle means) read the same information
  E2 = E1 + Pre-LN routing layers: LayerNorm only on the inputs of the router attention
            and the MLP, an unnormalized residual stream from U to the predictor, and no
            LayerNorm in the in-projection of later layers

Every setting keeps the recorded B0H+ configuration of
``docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`` and the seeds 2021-2023, so each run
is paired with the B0H+ and E runs of ``scripts/phaseformer_e.py`` (research_runs/phaseformer_e_v1).

    python3 scripts/phaseformer_e12.py --check
    python3 scripts/phaseformer_e12.py --smoke --gpus 3
    python3 scripts/phaseformer_e12.py --run --gpus 6,6,6,6,7,7,7,7
    python3 scripts/phaseformer_e12.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402

sa.SCRATCH = ROOT / 'research_runs/phaseformer_e12_v1'
PAIRED_SCRATCH = ROOT / 'research_runs/phaseformer_e_v1'
SMOKE_SCRATCH = ROOT / 'research_runs/phaseformer_e12_smoke'
# learned decoder with P=24, Legendre decoder with P=48, Legendre decoder with P=36
SMOKE_SETTINGS = (('ETTh1', 96), ('ETTm1', 96), ('Weather', 720))
VARIANTS = ('e1', 'e2')
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)
# arm -> (scratch, directory); B0H+ and E come from the paired PhaseFormer-E runs
ARMS = {'b0hp': (PAIRED_SCRATCH, 'rec'), 'e': (PAIRED_SCRATCH, 'e'),
        'e1': (sa.SCRATCH, 'e1'), 'e2': (sa.SCRATCH, 'e2')}
_base_structure = sa.apply_structure


# ------------------------------------------------------------------ E1: cycle-mean anchor
def cycle_mean_anchor(model):
    head = model.lowrank_interaction_branch.lowrank_head
    period = model.period_len

    def encode(self, x):
        return self.encoder(self.phase_tokens.mean(2)), x[:, -period:, :].mean(1, keepdim=True)

    head.encode_latent = types.MethodType(encode, head)


# ------------------------------------------------------------------ E2: Pre-LN routing
def prenorm_routing(self, Z):
    """Router aggregate/distribute and MLP on LayerNorm'd inputs; the stream itself is not normalized."""
    B, C, L, D = Z.shape
    x = Z.reshape(B * C, L, D)
    if self.use_pos_embed:
        slots = torch.arange(L, device=Z.device) % self.period_len
        x = self.pos_dropout(x + self.pos_embedding[slots])
    h = self.norm1(x)
    routers = self.router.unsqueeze(0).expand(B * C, -1, -1)
    buffer, _ = self.router_sender(routers, h, h, attn_mask=None)
    received, _ = self.router_receiver(h, buffer, buffer, attn_mask=None)
    x = x + self.dropout_layer(received)
    x = x + self.dropout_layer(self.mlp(self.norm2(x)))
    return x.view(B, C, L, D)


def prenorm(model):
    from torch import nn
    for unit in model.routing_layers:
        layer = unit.interact
        if layer.use_circular_pos or layer.use_circular_attn_bias:
            raise ValueError('Pre-LN routing supports the learned position embedding only')
        layer.forward = types.MethodType(prenorm_routing, layer)
        if unit.in_proj is not None:
            unit.in_proj[1] = nn.Identity()


CHANGES = (cycle_mean_anchor, prenorm)


def apply_structure(model, arm, job):
    if arm not in VARIANTS:
        _base_structure(model, arm, job)
        return
    _base_structure(model, 'e', job)
    for change in CHANGES[:VARIANTS.index(arm) + 1]:
        change(model)


sa.apply_structure = apply_structure


# ------------------------------------------------------------------ jobs and check
def all_settings():
    return tuple((r['dataset'], int(r['horizon'])) for r in csv.DictReader(sa.settings_csv().open()))


def build_jobs(settings, seeds, arms=VARIANTS):
    sa.SETTINGS = settings
    return sa.build_jobs(settings, seeds, arms)


def check(settings):
    """Every version builds and runs, and each change is present exactly from its own version on."""
    from scripts.vldbj_m1.proto import build_model
    sa.register_heads()
    for dataset, horizon in settings:
        models = {}
        for path in build_jobs(((dataset, horizon),), (2021,)):
            job = json.loads(path.read_text())
            job = dict(job, smoke=True)
            job.pop('source_hashes', None)
            job.pop('code_commit', None)
            model, _, _, _ = build_model(job)
            apply_structure(model, job['ablation_arm'], job)
            out = sa.forward_once(model)
            if tuple(out.shape) != (2, model.pred_len, model.enc_in) or not torch.isfinite(out).all():
                raise SystemExit(f'{dataset}-{horizon} {job["ablation_arm"]}: bad output')
            models[job['ablation_arm']] = model
        x = torch.randn(2, models['e1'].seq_len, models['e1'].enc_in)
        period = models['e1'].period_len
        conditions = {}
        for index, arm in enumerate(VARIANTS):
            m = models[arm]
            _, anchor = m.lowrank_interaction_branch.lowrank_head.encode_latent(x)
            prenormed = all(u.interact.forward.__func__ is prenorm_routing for u in m.routing_layers)
            conditions[f'{arm} embedding norm is identity'] = type(m.embedding.norm).__name__ == 'Identity'
            conditions[f'{arm} cycle-mean anchor'] = torch.allclose(anchor, x[:, -period:].mean(1, keepdim=True))
            conditions[f'{arm} Pre-LN = {index >= 1}'] = prenormed == (index >= 1)
            if index >= 1:
                conditions[f'{arm} no in-projection LayerNorm'] = all(
                    u.in_proj is None or type(u.in_proj[1]).__name__ == 'Identity' for u in m.routing_layers)
        failed = [name for name, passed in conditions.items() if not passed]
        counts = {arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()}
        print(f'{dataset}-{horizon} ok={not failed} layers={len(models["e1"].routing_layers)} '
              f'params={counts}', flush=True)
        if failed:
            raise SystemExit(f'check failed: {failed}')
    print('check passed')


# ------------------------------------------------------------------ report
def read_results():
    value = {}
    for dataset, horizon in all_settings():
        for seed in sa.SEEDS:
            for arm, (scratch, directory) in ARMS.items():
                pattern = f'{dataset}_H{horizon}_S{seed}__*__{directory}/result.json'
                for record in (scratch / directory / 'runs').glob(pattern):
                    best = json.loads(record.read_text())['best']
                    value[(dataset, horizon, seed, arm)] = {'test_mse': best['test_mse'],
                                                            'test_mae': best['test_mae']}
    return value


def report():
    value = read_results()
    if not value:
        raise SystemExit('no results')
    sa.SCRATCH.mkdir(parents=True, exist_ok=True)
    with (sa.SCRATCH / 'results.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'horizon', 'seed', 'arm', 'test_mse', 'test_mae'])
        for (dataset, horizon, seed, arm), r in sorted(value.items()):
            writer.writerow([dataset, horizon, seed, arm, r['test_mse'], r['test_mae']])
    arms = tuple(ARMS)
    steps = (('e1', 'e'), ('e2', 'e1'), ('e2', 'b0hp'))
    lines = []
    for metric in ('test_mse', 'test_mae'):
        lines.append(f'\n== {metric}: three-seed mean; each step is the paired change (%) and wins out of 3 seeds')
        lines.append(f'{"setting":>16} ' + ' '.join(f'{a:>8}' for a in arms) + ' '
                     + ' '.join(f'{a + "/" + b:>14}' for a, b in steps))
        means = {a: {d: [] for d in ORDER} for a in arms}
        for dataset in ORDER:
            for horizon in HORIZONS:
                cells, mean = [], {}
                for arm in arms:
                    vals = [value[(dataset, horizon, s, arm)][metric] for s in sa.SEEDS
                            if (dataset, horizon, s, arm) in value]
                    if len(vals) == len(sa.SEEDS):
                        mean[arm] = statistics.mean(vals)
                        means[arm][dataset].append(mean[arm])
                    cells.append(f'{mean[arm]:>8.4f}' if arm in mean else f'{"--":>8}')
                for new, old in steps:
                    if new in mean and old in mean:
                        wins = sum(value[(dataset, horizon, s, new)][metric] < value[(dataset, horizon, s, old)][metric]
                                   for s in sa.SEEDS)
                        cells.append(f'{100 * (mean[new] / mean[old] - 1):>+9.2f}% {wins}/3')
                    else:
                        cells.append(f'{"--":>14}')
                lines.append(f'{dataset + "-" + str(horizon):>16} ' + ' '.join(cells))
        lines.append(f'\n{"dataset mean":>16} ' + ' '.join(f'{a:>8}' for a in arms))
        overall = {a: [] for a in arms}
        for dataset in ORDER:
            row = []
            for arm in arms:
                vals = means[arm][dataset]
                if len(vals) == len(HORIZONS):
                    overall[arm].append(statistics.mean(vals))
                    row.append(f'{statistics.mean(vals):>8.4f}')
                else:
                    row.append(f'{"--":>8}')
            lines.append(f'{dataset:>16} ' + ' '.join(row))
        lines.append(f'{"seven-dataset":>16} ' + ' '.join(
            f'{statistics.mean(overall[a]):>8.4f}' if len(overall[a]) == len(ORDER) else f'{"--":>8}' for a in arms))
    text = '\n'.join(lines)
    print(text)
    (sa.SCRATCH / 'summary.txt').write_text(text)


def smoke(settings, gpus):
    """One epoch of four training and two validation batches per arm and setting, seed 2021."""
    import shutil
    if sa.SCRATCH.exists():
        shutil.rmtree(sa.SCRATCH)
    paths = build_jobs(settings, (2021,))
    for path in paths:
        job = json.loads(path.read_text())
        path.write_text(json.dumps(dict(job, smoke=True), indent=2))
    sa.run_jobs(paths, gpus, script=Path(__file__))
    for path in paths:
        job = json.loads(path.read_text())
        best = json.loads((Path(job['output']) / 'result.json').read_text())['best']
        print(f'{job["dataset"]}-{job["horizon"]} {job["ablation_arm"]} '
              f'test_mse={best["test_mse"]:.4f} test_mae={best["test_mae"]:.4f}', flush=True)
    print('smoke passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='6,6,6,6,7,7,7,7', help='one entry per parallel slot')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='short training runs of e1 and e2')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    parser.add_argument('--seeds', default='')
    parser.add_argument('--arms', default='', help='comma list out of e1,e2')
    args = parser.parse_args()
    if args.job:
        sa.worker(args.job)
        return
    if args.report:
        report()
        return
    only = sa.parse_filter(args.only, str)
    settings = tuple(s for s in all_settings() if only is None or f'{s[0]}-{s[1]}' in only)
    if args.check or args.smoke:
        # job files are immutable once written, so test jobs stay out of the experiment scratch
        sa.SCRATCH = SMOKE_SCRATCH / ('check' if args.check else 'train')
    if args.check:
        check(settings)
        return
    if args.smoke:
        smoke(settings if only else SMOKE_SETTINGS, [int(g) for g in args.gpus.split(',')])
        return
    seeds = tuple(sorted(sa.parse_filter(args.seeds, int) or sa.SEEDS))
    arms = tuple(a for a in VARIANTS if not args.arms or a in args.arms.split(','))
    paths = build_jobs(settings, seeds, arms)
    print(f'{len(paths)} jobs under {sa.SCRATCH}', flush=True)
    if args.run:
        sa.run_jobs(paths, [int(g) for g in args.gpus.split(',')], script=Path(__file__))
        report()


if __name__ == '__main__':
    main()
