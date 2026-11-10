#!/usr/bin/env python3
"""Two fixes for the settings where B0H+ trails b0, on three seeds.

B0H+ feeds the level encoder the pooled cycle means plus the slot-mean phase token.
Two things about that input may be holding it back:

  token_source  the embedding output is LayerNorm'd over the D axis, so every token
                has zero mean and unit scale and its amplitude is discarded -- and
                amplitude is what a level readout wants. ``raw`` reads the linear
                projection before that normalization instead.
  zero init     the token columns of the encoder start from random weights, so B0H+
                begins somewhere different from b0. ``zero`` keeps b0's weight on the
                pooled columns and starts the token columns at zero, making the two
                models agree at step 0 and letting the token columns switch on only
                if they earn it.

Variants per setting: stock (norm, random), zero, raw, both, and b0 itself as the
reference. Every other knob is the setting's best b0 cell. Seeds 2021-2023, so the
comparison is a three-seed mean and not a single best run.

    python3 scripts/b0hp_variant_run.py --check
    python3 scripts/b0hp_variant_run.py --run --gpus 1,2,3,7 --slots-per-gpu 2
    python3 scripts/b0hp_variant_run.py --report

Kept outside ``scripts/vldbj_m1``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/b0hp_variants_v1'
STAGE = 'v1'
SETTINGS = (('ETTh2', 720), ('Weather', 720), ('ETTh1', 96))
SEEDS = (2021, 2022, 2023)
VARIANTS = {
    'stock': dict(level_input='history+token', token_source='norm', token_zero_init=False),
    'zero': dict(level_input='history+token', token_source='norm', token_zero_init=True),
    'raw': dict(level_input='history+token', token_source='raw', token_zero_init=False),
    'both': dict(level_input='history+token', token_source='raw', token_zero_init=True),
    'b0': dict(level_input='history', token_source='norm', token_zero_init=False),
}


def name(dataset, horizon, seed, tag, best):
    return (f"{dataset}_H{horizon}_S{seed}__{tag}__g{best['gate']:g}_lr{best['ext']:g}"
            f"_bb{best['bb']:g}__p{best['period']}__{best['head']}"
            f"__{'shared' if best['shared'] else 'perC'}")


def build_jobs(scratch, settings=SETTINGS, seeds=SEEDS, variants=VARIANTS):
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, e14_sources, write_immutable
    results = load_results(B0)
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    paths = []
    for dataset, horizon in settings:
        best = winners(results, dataset, horizon)
        for seed in seeds:
            source = sources[(dataset, horizon, seed)]
            if best['period'] != 24:
                import copy
                source = copy.deepcopy(source)
                source['config']['period'] = best['period']
                source['config']['hyperparams']['period_len'] = best['period']
            for tag, spec in variants.items():
                job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='b0',
                           protocol='e14', init='scratch', head=best['head'],
                           shared_gate=bool(best['shared']), variant_id=None,
                           variant={'gate_init': best['gate'], 'extension_lr_mult': best['ext'],
                                    'communication_width': 2},
                           extension_lr_mult=float(best['ext']),
                           backbone_lr_mult=float(best['bb']),
                           warm_source=None, epoch_test=False, source=source,
                           b0_reference={'run': best['run'], 'mse': best['mse']})
                job.update({k: v for k, v in spec.items() if k != 'level_input'})
                if spec['level_input'] != 'history':
                    job['level_input'] = spec['level_input']
                n = name(dataset, horizon, seed, tag, best)
                job['output'] = str(scratch / STAGE / 'runs' / n)
                path = scratch / STAGE / 'jobs' / f'{n}.json'
                write_immutable(path, job)
                paths.append(path)
    return paths


def check(paths):
    """Zero-init must match b0 exactly at step 0; raw tokens must differ from norm."""
    import torch
    from scripts.vldbj_m1.proto import build_model
    by_setting = {}
    for path in paths:
        job = json.loads(Path(path).read_text())
        key = (job['dataset'], job['horizon'], job['seed'])
        by_setting.setdefault(key, {})[Path(path).name.split('__')[1]] = job
    for key, group in sorted(by_setting.items()):
        models = {}
        for tag in ('b0', 'stock', 'zero', 'raw'):
            job = group[tag]
            job.pop('source_hashes', None)
            job.pop('code_commit', None)
            models[tag], _, _, _ = build_model(dict(job))
        base = models['b0'].state_dict()
        zero = models['zero'].state_dict()
        enc = 'lowrank_interaction_branch.lowrank_head.encoder.weight'
        pooled = base[enc].shape[1]
        same_out = torch.equal(zero[enc][:, :pooled], base[enc]) and bool((zero[enc][:, pooled:] == 0).all())
        head_z = models['zero'].lowrank_interaction_branch.lowrank_head
        head_s = models['stock'].lowrank_interaction_branch.lowrank_head
        head_r = models['raw'].lowrank_interaction_branch.lowrank_head
        x = torch.randn(1, models['stock'].seq_len, models['stock'].enc_in)
        mark = torch.zeros(1, models['stock'].seq_len, 4)
        y = torch.randn(1, models['stock'].seq_len + models['stock'].pred_len, models['stock'].enc_in)
        ymark = torch.zeros(1, models['stock'].seq_len + models['stock'].pred_len, 4)
        outs = {}
        for tag in ('b0', 'zero'):
            m = models[tag]
            m.eval()
            with torch.inference_mode():
                outs[tag] = m(x, mark, m._build_decoder_input(y), ymark)[0]
        identical = torch.allclose(outs['b0'], outs['zero'], atol=1e-6)
        # raw tokens differ from norm tokens: their slot means have different scale
        models['stock'].eval()
        with torch.inference_mode():
            models['stock'](x, mark, models['stock']._build_decoder_input(y), ymark)
            norm_mean = head_s.phase_tokens.mean(2).clone()
        models['raw'].eval()
        with torch.inference_mode():
            models['raw'](x, mark, models['raw']._build_decoder_input(y), ymark)
            raw_mean = head_r.phase_tokens.mean(2).clone()
        differ = not torch.allclose(norm_mean, raw_mean, atol=1e-4)
        print(f"{key[0]}-{key[1]}-S{key[2]}: zero_columns={same_out} b0==zero_at_init={identical} "
              f"raw_differs={differ}")
        if not (same_out and identical and differ):
            raise SystemExit('variant check failed')
    print('check passed')


def report(scratch):
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results, winners
    results = load_results(B0)
    header = ' '.join(f'{t:>9}' for t in VARIANTS)
    print(f"{'setting':>13} | {header} | best variant")
    means = {t: [] for t in VARIANTS}
    for dataset, horizon in SETTINGS:
        best = winners(results, dataset, horizon)
        row = {}
        for tag in VARIANTS:
            vals = []
            for seed in SEEDS:
                record = scratch / STAGE / 'runs' / name(dataset, horizon, seed, tag, best) / 'result.json'
                if record.is_file():
                    vals.append(json.loads(record.read_text())['best']['test_mse'])
            row[tag] = sum(vals) / len(vals) if len(vals) == len(SEEDS) else None
            if row[tag] is not None:
                means[tag].append(row[tag])
        cells = ' '.join(f'{row[t]:>9.4f}' if row[t] is not None else f"{'-':>9}" for t in VARIANTS)
        b0 = row['b0']
        pick = ' '.join(f'{t}({(row[t] - b0) / b0 * 100:+.2f}%)' for t in ('stock', 'zero', 'raw', 'both')
                        if row[t] is not None) if b0 else ''
        print(f"{dataset + '-' + str(horizon):>13} | {cells} | {pick}")
    if means['b0']:
        print(f"\n{'mean':>13} | " + ' '.join(f'{sum(means[t]) / len(means[t]):>9.4f}' if means[t] else f"{'-':>9}"
                                             for t in VARIANTS))
        b0m = sum(means['b0']) / len(means['b0'])
        print('change vs b0: ' + ', '.join(
            f"{t} {(sum(means[t]) / len(means[t]) - b0m) / b0m * 100:+.2f}%"
            for t in ('stock', 'zero', 'raw', 'both') if means[t]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='1,2,3,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report(args.scratch)
        return
    paths = [str(p) for p in build_jobs(args.scratch)]
    print(f'{len(paths)} jobs under {args.scratch / STAGE}', flush=True)
    if args.check:
        check(paths)
    elif args.run:
        from scripts.vldbj_m1 import proto
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        proto.run_jobs(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
