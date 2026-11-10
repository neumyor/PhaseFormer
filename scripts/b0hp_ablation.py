#!/usr/bin/env python3
"""Component ablations on the 28 recorded B0H+ configurations.

Each of the 28 settings keeps the exact configuration recorded in
``docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`` (source period, gates,
learning rates, level head, token variant) and changes one design choice:

  trend_only    the forecast is the level branch alone; the phase forecast is
                dropped from the fusion (the token input stays, since it feeds
                the level branch).
  simple_gate   the two-gate component fusion is replaced by one global scalar
                gate, out = (1-a)*phase + a*trend with a = sigmoid(w).
  linear_trend  the pooled rank-3 level head is replaced by one linear layer on
                the centered window, (B,C,L) -> (B,C,H), zero-initialized.

The first ablation of the set, "phase pathway only", is PhaseFormer itself and
uses the recorded PhaseFormer test numbers instead of a rerun.

All three trained arms start from scratch with the same seed and the same
optimizer split as the recorded run.  The linear head carries 720*H weights,
far more than the 105-parameter recorded head and far more than the tuned
readout rates assume, so its weights are trained with the backbone multiplier;
every other parameter keeps the recorded group and rate.

Kept outside ``scripts/vldbj_m1``; the model is patched in the worker.

    python3 scripts/b0hp_ablation.py --check
    python3 scripts/b0hp_ablation.py --run --gpus 2,7 --slots-per-gpu 4
    python3 scripts/b0hp_ablation.py --report
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/b0hp_ablation_v1'
ARMS = ('trend_only', 'simple_gate', 'linear_trend', 'trend_only_lr1')
# reproduction controls: the same recorded config, rerun under the current code
BASELINE = (('ETTh1', 192), ('ETTh2', 720), ('Weather', 720), ('Traffic', 720))
SETTINGS_CSV = ROOT / 'docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv'
GATE_PREFIXES = ('lowrank_interaction_gate', 'lowrank_dynamic_fusion.')
READOUT_PREFIX = 'lowrank_interaction_branch.'


# ------------------------------------------------------------------ jobs
def source_job(scratch, run):
    path = Path(run)
    job_path = scratch / Path(str(path).replace('/runs/', '/jobs/') + '.json')
    if not job_path.is_file():
        raise SystemExit(f'missing job file {job_path}')
    return json.loads(job_path.read_text())


def build_jobs(scratch):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, write_immutable
    rows = list(csv.DictReader(SETTINGS_CSV.open()))
    paths = []
    for row in rows:
        job = source_job(ROOT, row['run'])
        arms = list(ARMS)
        if (row['dataset'], int(row['horizon'])) in BASELINE:
            arms.append('baseline')
        for arm in arms:
            new = json.loads(json.dumps(job))
            # the recorded jobs froze an earlier revision; rerun under the current one
            new['source_hashes'] = provenance()
            new['code_commit'] = commit()
            if arm == 'trend_only_lr1':
                # alone the trend pathway replaces the whole forecast, so the
                # residual-tuned readout rate (down to 0.03x) cannot train it
                new['extension_lr_mult'] = 1.0
                new['variant']['extension_lr_mult'] = 1.0
                new['lr_split'] = {'gate_mult': 1.0, 'readout_mult': 1.0}
            new['ablation_arm'] = arm
            new['scratch'] = str(scratch)
            new['stage'] = arm
            base = Path(job['output']).name
            name = f'{base}__{arm}'
            new['output'] = str(scratch / arm / 'runs' / name)
            path = scratch / arm / 'jobs' / f'{name}.json'
            write_immutable(path, new)
            paths.append(path)
    return paths


# ------------------------------------------------------------------ modules
def _linear_head(seq_len, pred_len):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LinearTrendHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len, self.pred_len = int(seq_len), int(pred_len)
            self.weight = nn.Parameter(torch.zeros(self.pred_len, self.seq_len))
            self.encoder = None  # keeps the attribute the branch diagnostics look for

        def encode_latent(self, x):
            return x, x[:, -1:, :]

        def decode_latent(self, latent, last):
            centered = (latent - last).permute(0, 2, 1).contiguous()
            delta = F.linear(centered, self.weight)
            return delta.permute(0, 2, 1).contiguous() + last.expand(-1, self.pred_len, -1)

        def forward(self, x):
            return self.decode_latent(*self.encode_latent(x))

    return LinearTrendHead()


def _simple_fusion(channels, gate_init):
    import torch
    from torch import nn

    class SimpleFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(gate_init)))))
            self.last_gate = None
            self.last_reliability = None

        def forward(self, yp, yt, trend, phase, history, base_logit):
            a = torch.sigmoid(self.logit)
            self.last_gate = a.expand(yp.shape[0], yp.shape[2], 2).clone()
            return (1.0 - a) * yp + a * yt

    return SimpleFusion()


def apply_ablation(model, arm, job):
    if arm.startswith('trend_only'):
        fusion = model.lowrank_dynamic_fusion

        def forward(self, yp, yt, trend, phase, history, base_logit):
            self.last_gate = yt.new_ones(yp.shape[0], yp.shape[2], 2)
            self.last_reliability = yt.new_zeros(yp.shape[0], yp.shape[2])
            return yt

        fusion.forward = types.MethodType(forward, fusion)
    elif arm == 'simple_gate':
        init = float(job['variant']['gate_init'])
        model.lowrank_dynamic_fusion = _simple_fusion(model.enc_in, init)
    elif arm == 'linear_trend':
        branch = model.lowrank_interaction_branch
        branch.lowrank_head = _linear_head(model.seq_len, model.pred_len)
    elif arm == 'baseline':
        return
    else:
        raise ValueError(arm)


def make_attach(job):
    """Recorded three-group rates; the linear head joins the backbone group."""
    import torch

    arm = job.get('ablation_arm')
    split = job.get('lr_split')
    gate_mult = float(split['gate_mult']) if split else float(job['extension_lr_mult'])
    readout_mult = float(split['readout_mult']) if split else float(job['extension_lr_mult'])
    bb_mult = float(job['backbone_lr_mult'])

    def attach(model, backbone_mult, extension_mult):
        base_lr = float(model.args.training_args.learning_rate)

        def configure_optimizers(self):
            groups = {'backbone': [], 'gate': [], 'readout': []}
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if arm == 'linear_trend' and name.startswith(READOUT_PREFIX + 'lowrank_head.'):
                    groups['backbone'].append(p)
                elif name.startswith(READOUT_PREFIX):
                    groups['readout'].append(p)
                elif name.startswith(GATE_PREFIXES):
                    groups['gate'].append(p)
                else:
                    groups['backbone'].append(p)
            mult = {'backbone': bb_mult, 'gate': gate_mult, 'readout': readout_mult}
            optimizer = torch.optim.Adam([{'params': v, 'lr': base_lr * mult[k], 'name': k}
                                          for k, v in groups.items() if v])
            if self.args.training_args.lr_schedule_config.type == 'cos':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args.training_args.lr_schedule_config.tmax, eta_min=1e-8)
                return [optimizer], [scheduler]
            return optimizer

        model.configure_optimizers = types.MethodType(configure_optimizers, model)

    return attach


# ------------------------------------------------------------------ worker
def worker(job_path):
    from scripts.traffic_b0hp_search import register_heads
    from scripts.vldbj_m1 import proto
    register_heads()
    job = json.loads(Path(job_path).read_text())
    attach = make_attach(job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        apply_ablation(model, j['ablation_arm'], j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


# ------------------------------------------------------------------ checks
def check(scratch):
    import torch
    from scripts.traffic_b0hp_search import register_heads
    from scripts.vldbj_m1.proto import build_model
    register_heads()
    paths = build_jobs(scratch)
    seen = 0
    for path in paths:
        job = json.loads(path.read_text())
        if job['dataset'] not in ('Traffic', 'ETTh1') or int(job['horizon']) != 720:
            continue
        job = dict(job, smoke=True)
        model, _, cfg, _ = build_model(job)
        apply_ablation(model, job['ablation_arm'], job)
        attach = make_attach(job)
        attach(model, job['backbone_lr_mult'], job['extension_lr_mult'])
        opt = model.configure_optimizers()
        opt = opt[0][0] if isinstance(opt, (tuple, list)) else opt
        base_lr = float(model.args.training_args.learning_rate)
        got = {g['name']: round(g['lr'] / base_lr, 6) for g in opt.param_groups}
        model.train()
        x = torch.randn(2, model.seq_len, model.enc_in)
        mark = torch.zeros(2, model.seq_len, 4)
        y = torch.randn(2, model.seq_len + model.pred_len, model.enc_in)
        ymark = torch.zeros(2, model.seq_len + model.pred_len, 4)
        out = model(x, mark, model._build_decoder_input(y), ymark)[0][:, -model.pred_len:]
        ok = tuple(out.shape) == (2, model.pred_len, model.enc_in) and torch.isfinite(out).all()
        if job['ablation_arm'] == 'linear_trend':
            head = model.lowrank_interaction_branch.lowrank_head
            ok = ok and head.weight.requires_grad and got.get('backbone') == job['backbone_lr_mult']
        if job['ablation_arm'] == 'simple_gate':
            ok = ok and isinstance(model.lowrank_dynamic_fusion.logit, torch.nn.Parameter)
        print(f"{job['dataset']}-{job['horizon']} {job['ablation_arm']:12} groups={got} "
              f"out={tuple(out.shape)} {'' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit('ablation check failed')
        seen += 1
    if not seen:
        raise SystemExit('no check jobs')
    print('check passed')


# ------------------------------------------------------------------ runner
def run_jobs(paths, gpus):
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
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


def golden():
    """PhaseFormer per-setting results from the recorded gold standard."""
    import re
    text = (ROOT / 'docs/PhaseFormer_gold_standard.md').read_text()
    out = {}
    for m in re.finditer(r'^\| (\w+) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \|', text, re.M):
        out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


def report(scratch):
    rows = list(csv.DictReader(SETTINGS_CSV.open()))
    order = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
    gold = golden()
    arms = tuple(ARMS) + ('baseline',)
    out = {}
    for row in rows:
        job = source_job(ROOT, row['run'])
        base = Path(job['output']).name
        rec = {'phaseformer': gold[(row['dataset'], int(row['horizon']))],
               'b0hp': (float(row['test_mse']), float(row['test_mae']))}
        for arm in arms:
            record = scratch / arm / 'runs' / f'{base}__{arm}' / 'result.json'
            if record.is_file():
                best = json.loads(record.read_text())['best']
                rec[arm] = (best['test_mse'], best['test_mae'])
        out[(row['dataset'], int(row['horizon']))] = rec

    def pick(value, per_key):
        return value[per_key] if isinstance(value, tuple) else value

    def line(label, values, per_key):
        cells = ' '.join(f'{pick(values[k], per_key):>10.5f}' if k in values else f'{"--":>10}'
                         for k in ('b0hp',) + arms)
        pf = pick(values['phaseformer'], per_key)
        return f'{label:>13} | {pf:>10.3f} {cells}'

    for per_key, title in ((0, 'MSE'), (1, 'MAE')):
        print(f'\n== {title}: per setting (PhaseFormer is the recorded gold standard)')
        print(f'{"setting":>13} | {"PhaseFormer":>10} ' +
              ' '.join(f'{a:>10}' for a in ('PhaseFormer+',) + arms))
        means = {k: [] for k in ('phaseformer', 'b0hp') + arms}
        for d in order:
            for h in (96, 192, 336, 720):
                print(line(f'{d}-{h}', out[(d, h)], per_key))
            avg = {}
            for k in ('phaseformer',) + ('b0hp',) + arms:
                vals = [out[(d, h)][k][per_key] for h in (96, 192, 336, 720) if k in out[(d, h)]]
                if len(vals) == 4:
                    avg[k] = sum(vals) / 4
            print(line(d, avg, per_key))
            for k in means:
                if k in avg:
                    means[k].append(avg[k])
        overall = {k: sum(v) / len(v) for k, v in means.items() if len(v) == 7}
        print(line('mean', overall, per_key))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='2,7')
    parser.add_argument('--slots-per-gpu', type=int, default=4)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    if args.report:
        report(args.scratch)
        return
    if args.check:
        check(args.scratch)
        return
    paths = [str(p) for p in build_jobs(args.scratch)]
    print(f'{len(paths)} jobs under {args.scratch}', flush=True)
    if args.run:
        gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
        print('launching on', gpus, flush=True)
        run_jobs(paths, gpus)
        report(args.scratch)


if __name__ == '__main__':
    main()
