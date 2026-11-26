#!/usr/bin/env python3
"""Structure ablation of PhaseFormer+ on six settings, three seeds.

Settings: ETTh1, ETTm1 and Weather at horizons 96 and 336. Each setting keeps the
configuration recorded in ``docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`` (period,
learning-rate groups, level head, gate_init, shared gate, Huber delta); only the
structure below changes. Seeds 2021, 2022 and 2023 start from the same b0 parameters.

Every arm except ``rec`` drops the pooled history from the level encoder: the trend
pathway reads only the slot mean of the phase tokens (U = projection of the embedding,
Z = LayerNorm(U)).

  rec       the recorded B0H+ model (pooled history + token as recorded); context only
  ref       phase pathway reads Z, trend reads Z; absolute trend forecast; two static gates
  noln      embedding LayerNorm removed: both pathways read U
  raw       trend reads U, the phase pathway still reads Z
  both      trend reads [U; Z] (encoder input 2d)
  resgate   trend forecast is a residual correction DF (zero at start), y = F + g DF
            with the two static gates (level and shape) acting on DF
  resplain  y = F + DF, no gates

    python3 scripts/b0hp_structure_ablation.py --check
    python3 scripts/b0hp_structure_ablation.py --run --gpus 6,6,6,6,7,7,7,7
    python3 scripts/b0hp_structure_ablation.py --report
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_ablation as abl  # noqa: E402

CSV_CANDIDATES = (Path.home() / 'niuyiming/b0hp_best_settings_20261103.csv',
                  ROOT / 'docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv')
SCRATCH = ROOT / 'research_runs/b0hp_structure_v1'
ARMS = ('rec', 'ref', 'noln', 'raw', 'both', 'resgate', 'resplain')
SETTINGS = (('ETTh1', 96), ('ETTh1', 336), ('ETTm1', 96), ('ETTm1', 336),
            ('Weather', 96), ('Weather', 336))
SEEDS = (2021, 2022, 2023)
COST = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 3, 'ETTh1': 4, 'ETTh2': 4}   # heaviest first
EXTRA_HEADS = {'H12L': (12, 'learned'), 'H12G': (12, 'legendre'), 'H6L': (6, 'learned'),
               'H48L': (48, 'learned'), 'H8G': (8, 'legendre'), 'H48G': (48, 'legendre'),
               'H2L': (2, 'learned'), 'H3L': (3, 'learned'), 'H4L': (4, 'learned'),
               'H16L': (16, 'learned'), 'H2G': (2, 'legendre'), 'H4G': (4, 'legendre')}


def register_heads():
    from scripts.vldbj_m1 import model as m
    for tag, (pool, decoder) in EXTRA_HEADS.items():
        m.HEADS.setdefault(tag, {'pool_factor': pool, 'decoder': decoder})


def settings_csv():
    for path in CSV_CANDIDATES:
        if path.is_file():
            return path
    raise SystemExit('missing best-settings csv')


def parse_filter(text, cast):
    return None if not text else {cast(v) for v in text.split(',')}


# ------------------------------------------------------------------ jobs
def job_name(recorded, seed, arm):
    base = Path(recorded['output']).name.replace('_S2021__', f'_S{seed}__')
    return f'{base}__{arm}'


def build_jobs(settings=SETTINGS, seeds=SEEDS, arms=ARMS):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, write_immutable
    rows = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader(settings_csv().open())}
    paths = []
    for dataset, horizon in settings:
        recorded = abl.source_job(ROOT, rows[(dataset, horizon)]['run'])
        for seed in seeds:
            for arm in arms:
                job = copy.deepcopy(recorded)
                job['seed'] = seed
                job['source']['config']['seed'] = seed
                job['source_hashes'] = provenance()
                job['code_commit'] = commit()
                if arm != 'rec':
                    job['level_input'] = 'phase_token'
                    job['token_source'] = 'raw' if arm == 'raw' else 'norm'
                    job['token_zero_init'] = False
                    job['tune'] = dict(job.get('tune', {}), detach=0)
                job['ablation_arm'] = arm
                job['scratch'] = str(SCRATCH)
                job['stage'] = arm
                name = job_name(recorded, seed, arm)
                job['output'] = str(SCRATCH / arm / 'runs' / name)
                path = SCRATCH / arm / 'jobs' / f'{name}.json'
                write_immutable(path, job)
                paths.append(path)
    paths.sort(key=lambda p: (COST[json.loads(p.read_text())['dataset']], p.name))
    return paths


# ------------------------------------------------------------------ model patches
def detach_tokens(model, job):
    head = model.lowrank_interaction_branch.lowrank_head
    module = model.embedding if job.get('token_source', 'norm') == 'norm' else model.embedding.projection
    module._forward_hooks.clear()
    module.register_forward_hook(lambda m, inputs, output: setattr(head, 'phase_tokens', output.detach()))


def both_tokens(model, head):
    """The level encoder reads [slot mean of U; slot mean of Z]."""
    import torch
    from torch import nn
    embedding, projection = model.embedding, model.embedding.projection
    head.encoder = nn.Linear(2 * model.latent_dim, head.rank)
    embedding._forward_hooks.clear()
    projection._forward_hooks.clear()
    store = {}
    projection.register_forward_hook(lambda m, inputs, output: store.__setitem__('raw', output))
    embedding.register_forward_hook(
        lambda m, inputs, output: setattr(head, 'phase_tokens', torch.cat([store['raw'], output], -1)))


def residual_trend(model, head, gated):
    """The level branch returns DF = decode - last; the fusion adds it to the phase forecast."""
    decode = head.decode_latent
    pred_len = head.pred_len

    def decode_residual(latent, last):
        return decode(latent, last) - last.expand(-1, pred_len, -1)

    head.decode_latent = decode_residual
    fusion = model.lowrank_dynamic_fusion
    original = fusion.forward

    if gated:
        def forward(yp, yt, trend, phase, history, base_logit):
            return original(yp, yp + yt, trend, phase, history, base_logit)
    else:
        def forward(yp, yt, trend, phase, history, base_logit):
            fusion.last_gate = yt.new_ones(yp.shape[0], yp.shape[2], 2)
            fusion.last_reliability = yt.new_zeros(yp.shape[0], yp.shape[2])
            return yp + yt

    fusion.forward = forward


def apply_structure(model, arm, job):
    from torch import nn
    head = model.lowrank_interaction_branch.lowrank_head
    if arm == 'rec':
        if job.get('tune', {}).get('detach') and job.get('level_input') == 'history+token':
            detach_tokens(model, job)
    elif arm in ('noln', 'e'):
        model.embedding.norm = nn.Identity()
    elif arm == 'both':
        both_tokens(model, head)
    elif arm in ('resgate', 'resplain'):
        residual_trend(model, head, gated=arm == 'resgate')
    elif arm not in ('ref', 'raw'):
        raise ValueError(arm)


def worker(job_path):
    register_heads()
    from scripts.vldbj_m1 import proto
    job = json.loads(Path(job_path).read_text())
    attach = abl.make_attach(job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        apply_structure(model, j['ablation_arm'], j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


# ------------------------------------------------------------------ check
def forward_once(model):
    import torch
    model.eval()
    x = torch.randn(2, model.seq_len, model.enc_in)
    mark = torch.zeros(2, model.seq_len, 4)
    y = torch.randn(2, model.seq_len + model.pred_len, model.enc_in)
    ymark = torch.zeros(2, model.seq_len + model.pred_len, 4)
    with torch.inference_mode():
        return model(x, mark, model._build_decoder_input(y), ymark)[0][:, -model.pred_len:]


def check():
    import torch
    from scripts.vldbj_m1.proto import build_model
    register_heads()
    for dataset, horizon in SETTINGS:
        jobs = {}
        for path in build_jobs(((dataset, horizon),), (2021,)):
            job = json.loads(path.read_text())
            jobs[job['ablation_arm']] = job
        models = {}
        for arm, job in jobs.items():
            job = dict(job, smoke=True)
            job.pop('source_hashes', None)
            job.pop('code_commit', None)
            model, _, _, _ = build_model(job)
            apply_structure(model, arm, job)
            torch.manual_seed(0)
            branch_out = {}
            model.lowrank_interaction_branch.register_forward_hook(
                lambda m, inputs, output, store=branch_out: store.__setitem__('forecast', output[0].detach()))
            out = forward_once(model)
            model.branch_forecast = branch_out['forecast']
            if tuple(out.shape) != (2, model.pred_len, model.enc_in) or not torch.isfinite(out).all():
                raise SystemExit(f'{dataset}-{horizon} {arm}: bad output')
            models[arm] = model
        d = models['ref'].latent_dim
        tokens = {arm: m.lowrank_interaction_branch.lowrank_head.phase_tokens.detach().clone()
                  for arm, m in models.items()}
        # tokens: raw reads U, ref reads Z = LN(U), both reads [U; Z], noln reads U
        norm = models['ref'].embedding.norm
        head = models['rec'].lowrank_interaction_branch.lowrank_head
        conditions = {
            'raw tokens normalize to ref tokens': torch.allclose(norm(tokens['raw']), tokens['ref'], atol=1e-5),
            'both width 2d': tokens['both'].shape[-1] == 2 * d,
            'both first half is raw': torch.allclose(tokens['both'][..., :d], tokens['raw'], atol=1e-5),
            'both second half is norm': torch.allclose(tokens['both'][..., d:], tokens['ref'], atol=1e-5),
            'noln norm is identity': type(models['noln'].embedding.norm).__name__ == 'Identity',
            'both encoder width': models['both'].lowrank_interaction_branch.lowrank_head.encoder.in_features == 2 * d,
            'ref encoder width': models['ref'].lowrank_interaction_branch.lowrank_head.encoder.in_features == d,
            'rec encoder width': head.encoder.in_features == head.pooled_len + d,
        }
        # the residual arms start from the phase forecast: DF = 0 at initialization
        for arm in ('resgate', 'resplain'):
            delta = models[arm].branch_forecast
            conditions[f'{arm} DF is zero at init (max {float(delta.abs().max()):.2e})'] = bool(delta.abs().max() < 1e-6)
        failed = [name for name, passed in conditions.items() if not passed]
        ok = not failed
        counts = {arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()}
        print(f'{dataset}-{horizon} ok={ok} params={counts}', flush=True)
        if not ok:
            raise SystemExit(f'structure check failed: {failed}')
    print('check passed')


# ------------------------------------------------------------------ runner
def running(out):
    """True when another process holds the run's worker lock (the run is in progress)."""
    import fcntl
    lock = out / 'worker.lock'
    if not lock.is_file():
        return False
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def run_jobs(paths, gpus, script=None):
    """Run each job in its own process; ``script`` is the driver whose ``--job`` builds the arm."""
    script = Path(script or __file__).resolve()
    tasks = queue.Queue()
    for path in paths:
        tasks.put(path)
    failures = []

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            out = Path(json.loads(Path(path).read_text())['output'])
            if (out / 'result.json').is_file():
                continue
            if running(out):
                continue
            if (out / 'checkpoints').exists():
                import shutil
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(script), '--job', str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))
            else:
                print('done', out.name, flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


# ------------------------------------------------------------------ report
def read_results():
    rows = []
    for dataset, horizon in SETTINGS:
        for seed in SEEDS:
            for arm in ARMS:
                for record in (SCRATCH / arm / 'runs').glob(f'{dataset}_H{horizon}_S{seed}__*__{arm}/result.json'):
                    best = json.loads(record.read_text())['best']
                    rows.append({'dataset': dataset, 'horizon': horizon, 'seed': seed, 'arm': arm,
                                 'test_mse': best['test_mse'], 'test_mae': best['test_mae'],
                                 'val_mse': best.get('val_mse', ''), 'run': str(record.parent)})
    return rows


def report():
    rows = read_results()
    if not rows:
        raise SystemExit('no results')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with (SCRATCH / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value = {(r['dataset'], r['horizon'], r['seed'], r['arm']): r for r in rows}
    lines = []
    for metric in ('test_mse', 'test_mae'):
        lines.append(f'\n== {metric}: three-seed mean per setting, change against ref (%), wins out of 3 seeds')
        lines.append(f'{"setting":>12} ' + ' '.join(f'{a:>17}' for a in ARMS))
        overall = {a: [] for a in ARMS}
        wins = {a: [0, 0] for a in ARMS}
        for dataset, horizon in SETTINGS:
            cells, ref_mean = [], None
            for arm in ARMS:
                seeds = [s for s in SEEDS if (dataset, horizon, s, arm) in value]
                vals = [value[(dataset, horizon, s, arm)][metric] for s in seeds]
                if len(vals) < len(SEEDS):
                    cells.append(f'{"--":>17}')
                    continue
                mean = statistics.mean(vals)
                if arm == 'ref':
                    ref_mean = mean
                refs = [value.get((dataset, horizon, s, 'ref'), {}).get(metric) for s in SEEDS]
                if arm != 'ref' and None not in refs:
                    change = 100 * (mean / statistics.mean(refs) - 1)
                    win = sum(v < r for v, r in zip(vals, refs))
                    wins[arm][0] += win
                    wins[arm][1] += len(vals)
                    overall[arm].append(change)
                    cells.append(f'{mean:.4f} {change:+5.1f}% {win}/3')
                else:
                    cells.append(f'{mean:>17.4f}')
            lines.append(f'{dataset + "-" + str(horizon):>12} ' + ' '.join(f'{c:>17}' for c in cells))
        lines.append(f'{"mean change":>12} ' + ' '.join(
            f'{statistics.mean(overall[a]):+16.2f}%' if overall[a] else f'{"":>17}' for a in ARMS))
        lines.append(f'{"seed wins":>12} ' + ' '.join(
            f'{wins[a][0]:>12}/{wins[a][1]:<4}' if wins[a][1] else f'{"":>17}' for a in ARMS))
    text = '\n'.join(lines)
    print(text)
    (SCRATCH / 'summary.txt').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='6,6,6,6,7,7,7,7', help='one entry per parallel slot')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    parser.add_argument('--seeds', default='')
    parser.add_argument('--arms', default='')
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    if args.report:
        report()
        return
    if args.check:
        check()
        return
    only = parse_filter(args.only, str)
    settings = tuple(s for s in SETTINGS if only is None or f'{s[0]}-{s[1]}' in only)
    seeds = tuple(sorted(parse_filter(args.seeds, int) or SEEDS))
    arms = tuple(a for a in ARMS if not args.arms or a in args.arms.split(','))
    paths = build_jobs(settings, seeds, arms)
    print(f'{len(paths)} jobs under {SCRATCH}', flush=True)
    if args.run:
        gpus = [int(g) for g in args.gpus.split(',')]
        print('launching on', gpus, flush=True)
        run_jobs(paths, gpus)
        report()


if __name__ == '__main__':
    main()
