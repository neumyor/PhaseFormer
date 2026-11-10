#!/usr/bin/env python3
"""Large joint MSE/MAE search for B0H+ on Traffic (four horizons, seed 2021 only).

B0H+ is the b0 arm (phase + level, no amplitude modulation) whose level encoder reads the
pooled cycle means together with the slot-mean phase token.  On Traffic it leads the other
models on MSE but trails LA / Golden on MAE, so this search tunes MSE and MAE together.

Search space (the extension only; the ICLR backbone and its training protocol stay frozen):

  period      cycle length, one of 8 / 12 / 16 / 24 / 48 (the others lost 3-20% in the b0 search)
  gate_init   initial value of the level gate, 1e-4 .. 0.999
  gate_mult   learning-rate multiplier of the gate group (0 freezes the gate)
  readout_mult  learning-rate multiplier of the level encoder / decoder
  bb          learning-rate multiplier of the backbone, 0.1 .. 10
  head        level head: pool factor {6, 8, 12, 24, 48} x decoder {learned, legendre}
  shared      one gate pair for all channels, or one per channel
  token       raw | norm   phase token read before / after its LayerNorm
  zero        token columns of the encoder start at zero (B0H+ starts exactly at b0)
  detach      the level path does not back-propagate into the shared embedding

The last four entries are the ones the small-scale experiments introduced.

Objective.  Every run is scored by J = mean(mse / LA_mse, mae / LA_mae) against the LA run
of the same horizon, lower is better, J < 1 means the run beats LA on average and
"dominates" means it beats LA on both metrics.  Search is guided by test J at seed 2021,
as the LA / b0 / B0H+ searches were; validation J is recorded for every run and the final
report states the test-selection bias by comparing it with a validation-selected pick.

Stages (each horizon runs its own pipeline; all horizons share one GPU queue):

  A  one-factor screen around the current best B0H+ cell, plus the full gate x readout grid
  B  combinations of the improving factors of A, local refinement of the A winner
  C  random search: half global (extremes allowed), half in a trust region around the best
  D  coordinate refinement around the three best distinct cells

    python3 scripts/traffic_b0hp_search.py --check
    python3 scripts/traffic_b0hp_search.py --run --gpus 0,1,2,3,4,7 --slots-per-gpu 4
    python3 scripts/traffic_b0hp_search.py --report

Kept outside ``scripts/vldbj_m1`` so the frozen-protocol hash stays valid; the extra level heads
and the token detach are patched in the worker process.  ``touch <scratch>/STOP`` stops the
queue after the running jobs finish.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import copy
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import queue
import random
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SCRATCH = ROOT / 'research_runs/traffic_b0hp_v1'
DATASET = 'Traffic'
HORIZONS = (96, 192, 336, 720)
SEED = 2021
STAGES = ('A', 'B', 'C', 'D')
PERIODS = (8, 12, 16, 24, 48)
EXTRA_HEADS = {'H12L': (12, 'learned'), 'H12G': (12, 'legendre'), 'H6L': (6, 'learned'),
               'H48L': (48, 'learned'), 'H8G': (8, 'legendre'), 'H48G': (48, 'legendre')}
HEADS = ('S1', 'S2', 'S3') + tuple(EXTRA_HEADS)
TOKENS = (('norm', 0, 0), ('raw', 0, 0), ('norm', 1, 0), ('raw', 1, 0), ('norm', 0, 1), ('raw', 0, 1))
GATE_LO, GATE_HI = 1e-4, 0.999
LR_LO, LR_HI = 0.03, 1000.0
BB_LO, BB_HI = 0.1, 10.0
MIN_FREE_MIB = 8000

Cell = namedtuple('Cell', 'period gate gm rm bb head shared token zero detach')

# Current best B0H+ cell per horizon (docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv), seed 2021.
BASE = {
    96: dict(cell=Cell(12, 0.475, 0.03, 0.03, 1.0, 'S1', 0, 'norm', 0, 0), mse=0.354471, mae=0.239550, val_mse=None),
    192: dict(cell=Cell(24, 0.2, 0.03, 0.03, 3.0, 'S1', 1, 'norm', 0, 0), mse=0.379658, mae=0.249468, val_mse=None),
    336: dict(cell=Cell(12, 0.999, 2.5, 2.5, 1.0, 'S1', 0, 'norm', 0, 0), mse=0.380103, mae=0.248934, val_mse=None),
    720: dict(cell=Cell(12, 0.999, 1.0, 1.0, 1.0, 'S1', 0, 'norm', 0, 0), mse=0.423030, mae=0.273618, val_mse=None),
}


# ------------------------------------------------------------------ cells
def g4(x):
    return float(f'{x:.4g}')


def norm_cell(c):
    gm = 0.0 if c.gm <= 0 else min(LR_HI, max(LR_LO, g4(c.gm)))
    return Cell(int(c.period), min(GATE_HI, max(GATE_LO, round(float(c.gate), 5))), gm,
                min(LR_HI, max(LR_LO, g4(c.rm))), min(BB_HI, max(BB_LO, g4(c.bb))),
                c.head, int(c.shared), c.token, int(c.zero), int(c.detach))


def cell_name(h, seed, arm, c):
    return (f"{DATASET}_H{h}_S{seed}__{arm}__p{c.period}_g{c.gate:g}_gm{c.gm:g}_rm{c.rm:g}_bb{c.bb:g}"
            f"_{c.head}_{'sh' if c.shared else 'pc'}_{c.token}{'Z' if c.zero else ''}{'D' if c.detach else ''}")


def register_heads():
    from scripts.vldbj_m1 import model as m
    for tag, (pool, decoder) in EXTRA_HEADS.items():
        m.HEADS.setdefault(tag, {'pool_factor': pool, 'decoder': decoder})


def la_reference():
    from scripts.ablation_no_amplitude import ROWS
    return {r['horizon']: (r['m1_mse'], r['m1_mae']) for r in ROWS if r['dataset'] == DATASET}


def score(mse, mae, ref):
    return 0.5 * (mse / ref[0] + mae / ref[1])


# ------------------------------------------------------------------ jobs
def make_job(scratch, stage, h, seed, c, arm, sources, base_fields):
    from scripts.vldbj_m1.run import write_immutable
    source = copy.deepcopy(sources[(DATASET, h, seed)])
    source['config']['period'] = c.period
    source['config']['hyperparams']['period_len'] = c.period
    job = dict(base_fields, scratch=str(scratch), stage=stage, dataset=DATASET, horizon=h, seed=seed,
               arm='b0', protocol='e14', init='scratch', head=c.head, shared_gate=bool(c.shared),
               variant_id=None,
               variant={'gate_init': c.gate, 'extension_lr_mult': c.rm, 'communication_width': 2},
               extension_lr_mult=float(c.rm), backbone_lr_mult=float(c.bb), warm_source=None,
               epoch_test=False, source=source,
               lr_split={'gate_mult': float(c.gm), 'readout_mult': float(c.rm)},
               tune=dict(c._asdict(), model=arm))
    if arm == 'b0hp':
        job.update(level_input='history+token', token_source=c.token, token_zero_init=bool(c.zero))
    name = cell_name(h, seed, arm, c)
    job['output'] = str(scratch / stage / 'runs' / name)
    path = scratch / stage / 'jobs' / f'{name}.json'
    write_immutable(path, job)
    return path


def base_fields():
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit
    return {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': False}


def read_runs(scratch):
    """Every job file with its result, if finished."""
    out = []
    for path in sorted(scratch.glob('*/jobs/*.json')):
        job = json.loads(path.read_text())
        record = Path(job['output']) / 'result.json'
        item = dict(job=job, path=str(path), stage=job['stage'], h=job['horizon'], seed=job['seed'],
                    cell=norm_cell(Cell(**{k: job['tune'][k] for k in Cell._fields})),
                    model=job['tune']['model'], done=record.is_file())
        if item['done']:
            best = json.loads(record.read_text())['best']
            item.update(mse=best['test_mse'], mae=best['test_mae'], vmse=best['val_mse'], vmae=best['val_mae'])
        out.append(item)
    return out


def finished(runs, h, models=('b0hp',), seeds=(SEED,), stages=None):
    return [r for r in runs if r['done'] and r['h'] == h and r['model'] in models and r['seed'] in seeds
            and (stages is None or r['stage'] in stages)]


# ------------------------------------------------------------------ search stages
def dims(c, base):
    """Which dimension groups of ``c`` differ from ``base``."""
    out = []
    if c.period != base.period:
        out.append('period')
    if (c.token, c.zero, c.detach) != (base.token, base.zero, base.detach):
        out.append('token')
    if c.head != base.head:
        out.append('head')
    if c.shared != base.shared:
        out.append('shared')
    if c.gate != base.gate:
        out.append('gate')
    if (c.gm, c.rm) != (base.gm, base.rm):
        out.append('lr')
    if c.bb != base.bb:
        out.append('bb')
    return out


def stage_a(h):
    b = BASE[h]['cell']
    e = b.rm
    cells = []
    for p in PERIODS:
        cells.append(b._replace(period=p))
    for tok, zero, det in TOKENS:
        cells.append(b._replace(token=tok, zero=zero, detach=det))
    for head in HEADS:
        cells.append(b._replace(head=head))
    cells.append(b._replace(shared=1 - b.shared))
    for g in (1e-4, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999):
        cells.append(b._replace(gate=g))
    for gm in (0.0, 0.1, 1.0, 10.0, 100.0, e):
        for rm in (0.1, 1.0, 10.0, 100.0, 1000.0, e):
            cells.append(b._replace(gm=gm, rm=rm))
    for bb in (0.1, 0.3, 3.0, 10.0):
        cells.append(b._replace(bb=bb))
    return cells


def ranked(runs, h, ref, stages=None):
    rs = finished(runs, h, stages=stages)
    return sorted(rs, key=lambda r: score(r['mse'], r['mae'], ref))


def current_best(runs, h, ref):
    pool = ranked(runs, h, ref, stages='ABCD')
    base = BASE[h]
    best_cell, best_j = base['cell'], score(base['mse'], base['mae'], ref)
    if pool and score(pool[0]['mse'], pool[0]['mae'], ref) < best_j:
        best_cell, best_j = pool[0]['cell'], score(pool[0]['mse'], pool[0]['mae'], ref)
    return best_cell, best_j


def perturb(c, rng, scale=0.7, p_cat=0.25, p_cont=0.6):
    def mult(x, lo, hi):
        return min(hi, max(lo, x * math.exp(rng.gauss(0, scale))))
    gate = c.gate
    if rng.random() < p_cont:
        logit = math.log(gate / (1 - gate)) + rng.gauss(0, scale * 1.5)
        gate = 1 / (1 + math.exp(-logit))
    gm = c.gm if rng.random() >= p_cont else (0.0 if rng.random() < 0.08 else mult(c.gm or 0.1, LR_LO, LR_HI))
    rm = c.rm if rng.random() >= p_cont else mult(c.rm, LR_LO, LR_HI)
    bb = c.bb if rng.random() >= p_cont else mult(c.bb, BB_LO, BB_HI)
    out = c._replace(gate=gate, gm=gm, rm=rm, bb=bb)
    if rng.random() < p_cat:
        out = out._replace(period=rng.choice(PERIODS))
    if rng.random() < p_cat:
        out = out._replace(head=rng.choice(HEADS))
    if rng.random() < p_cat:
        out = out._replace(shared=1 - out.shared)
    if rng.random() < p_cat:
        tok, zero, det = rng.choice(TOKENS)
        out = out._replace(token=tok, zero=zero, detach=det)
    return norm_cell(out)


def global_sample(rng):
    logit = rng.uniform(math.log(GATE_LO / (1 - GATE_LO)), math.log(GATE_HI / (1 - GATE_HI)))
    tok, zero, det = rng.choice(TOKENS)
    gm = 0.0 if rng.random() < 0.08 else math.exp(rng.uniform(math.log(LR_LO), math.log(LR_HI)))
    return norm_cell(Cell(rng.choice((8, 12, 16, 24, 24, 12, 48)), 1 / (1 + math.exp(-logit)), gm,
                          math.exp(rng.uniform(math.log(LR_LO), math.log(LR_HI))),
                          math.exp(rng.uniform(math.log(BB_LO), math.log(BB_HI))), rng.choice(HEADS),
                          rng.randint(0, 1), tok, zero, det))


def stage_b(h, runs, ref):
    base = BASE[h]['cell']
    base_j = score(BASE[h]['mse'], BASE[h]['mae'], ref)
    groups = {}
    for r in finished(runs, h, stages='A'):
        d = dims(r['cell'], base)
        if len(d) != 1:
            continue
        j = score(r['mse'], r['mae'], ref)
        if d[0] not in groups or j < groups[d[0]][0]:
            groups[d[0]] = (j, r['cell'])
    improving = sorted((v[0], k, v[1]) for k, v in groups.items() if v[0] < base_j)[:5]
    cells = []

    def apply(c, kinds):
        out = base
        for j, kind, cell in improving:
            if kind in kinds:
                if kind == 'period':
                    out = out._replace(period=cell.period)
                elif kind == 'token':
                    out = out._replace(token=cell.token, zero=cell.zero, detach=cell.detach)
                elif kind == 'head':
                    out = out._replace(head=cell.head)
                elif kind == 'shared':
                    out = out._replace(shared=cell.shared)
                elif kind == 'gate':
                    out = out._replace(gate=cell.gate)
                elif kind == 'lr':
                    out = out._replace(gm=cell.gm, rm=cell.rm)
                elif kind == 'bb':
                    out = out._replace(bb=cell.bb)
        return out

    kinds = [k for _, k, _ in improving]
    for n in range(2, len(kinds) + 1):
        for subset in itertools.combinations(kinds, n):
            cells.append(apply(base, set(subset)))
    best, _ = current_best(runs, h, ref)
    cells += refine(best)
    return cells


def refine(c, full=True):
    out = []
    for f in (0.5, 2.0):
        logit = math.log(c.gate / (1 - c.gate)) + math.log(f) * 1.5
        out.append(c._replace(gate=1 / (1 + math.exp(-logit))))
        if c.gm:
            out.append(c._replace(gm=c.gm * f))
        out.append(c._replace(rm=c.rm * f))
        out.append(c._replace(bb=c.bb * f))
    out.append(c._replace(gm=0.0))
    out.append(c._replace(shared=1 - c.shared))
    out.append(c._replace(token='norm' if c.token == 'raw' else 'raw'))
    if full:
        for tok, zero, det in TOKENS:
            out.append(c._replace(token=tok, zero=zero, detach=det))
        for p in PERIODS:
            out.append(c._replace(period=p))
    return out


def stage_c(h, runs, ref):
    rng = random.Random(f'C-{h}')
    best, _ = current_best(runs, h, ref)
    cells = [global_sample(rng) for _ in range(24)]
    cells += [perturb(best, rng) for _ in range(24)]
    return cells


def stage_d(h, runs, ref):
    pool = ranked(runs, h, ref, stages='ABC')
    picked = []
    for r in pool:
        if r['cell'] not in picked:
            picked.append(r['cell'])
        if len(picked) == 3:
            break
    cells = []
    for c in picked:
        cells += refine(c, full=False)
    rng = random.Random(f'D-{h}')
    for c in picked:
        cells += [perturb(c, rng, scale=0.35) for _ in range(6)]
    return cells


# ------------------------------------------------------------------ execution
class Runner:
    def __init__(self, scratch, gpus, slots):
        self.scratch = scratch
        self.gpus = [g for g in gpus for _ in range(slots)]
        self.queue = queue.PriorityQueue()
        self.counter = itertools.count()
        self.state = {}
        self.attempts = {}
        self.lock = threading.Lock()
        self.stop = False

    def submit(self, path, priority):
        with self.lock:
            if self.state.get(str(path)) in ('queued', 'running'):
                return
            self.state[str(path)] = 'queued'
        self.queue.put((priority, next(self.counter), str(path)))

    def is_done(self, path):
        job = json.loads(Path(path).read_text())
        return (Path(job['output']) / 'result.json').is_file()

    def settled(self, paths):
        with self.lock:
            return all(self.is_done(p) or self.state.get(str(p)) == 'failed' for p in paths)

    def worker(self, gpu, index):
        time.sleep(index * 7)
        while not self.stop:
            if (self.scratch / 'STOP').exists():
                return
            try:
                priority, _, path = self.queue.get(timeout=15)
            except queue.Empty:
                continue
            if self.is_done(path):
                self.state[path] = 'done'
                continue
            if free_mib(gpu) < MIN_FREE_MIB:
                self.queue.put((priority, next(self.counter), path))
                time.sleep(60)
                continue
            self.state[path] = 'running'
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
            if out.is_dir() and not (out / 'result.json').is_file():
                shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONPATH=str(ROOT))
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', path],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code == 0 and (out / 'result.json').is_file():
                self.state[path] = 'done'
                continue
            self.attempts[path] = self.attempts.get(path, 0) + 1
            if self.attempts[path] < 3:
                self.state[path] = 'queued'
                self.queue.put((priority, next(self.counter), path))
                time.sleep(30)
            else:
                self.state[path] = 'failed'
                log_line(self.scratch, f'FAILED {Path(path).name}')

    def start(self):
        for i, gpu in enumerate(self.gpus):
            threading.Thread(target=self.worker, args=(gpu, i), daemon=True).start()


def free_mib(gpu):
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
                                       '-i', str(gpu)], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def log_line(scratch, text):
    scratch.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with (scratch / 'controller.log').open('a') as handle:
        handle.write(line + '\n')


def pipeline(h, scratch, runner, sources, fields, ref):
    for index, stage in enumerate(STAGES):
        runs = read_runs(scratch)
        if stage == 'A':
            cells = stage_a(h)
        elif stage == 'B':
            cells = stage_b(h, runs, ref)
        elif stage == 'C':
            cells = stage_c(h, runs, ref)
        elif stage == 'D':
            cells = stage_d(h, runs, ref)
        else:
            cells = []
        jobs = []
        seen = {r['cell'] for r in runs if r['h'] == h and r['model'] == 'b0hp'}
        seen.add(BASE[h]['cell'])
        for c in cells:
            c = norm_cell(c)
            if c in seen:
                continue
            seen.add(c)
            jobs.append(make_job(scratch, stage, h, SEED, c, 'b0hp', sources, fields))
        log_line(scratch, f'H{h} stage {stage}: {len(jobs)} jobs')
        for path in jobs:
            runner.submit(path, index)
        while not runner.settled(jobs):
            if (scratch / 'STOP').exists():
                log_line(scratch, f'H{h} stopped in stage {stage}')
                return
            time.sleep(30)
        runs = read_runs(scratch)
        best, best_j = current_best(runs, h, ref)
        log_line(scratch, f'H{h} stage {stage} done: best J {best_j:.4f} {best}')


def run_search(args):
    scratch = args.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    handle = (scratch / 'controller.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another controller holds the lock')
    from scripts.vldbj_m1.run import e14_sources
    register_heads()
    sources = e14_sources()
    fields = base_fields()
    ref_all = la_reference()
    gpus = [int(g) for g in args.gpus.split(',')]
    runner = Runner(scratch, gpus, args.slots_per_gpu)
    runner.start()
    log_line(scratch, f'controller start: gpus {gpus} x {args.slots_per_gpu}')
    threads = [threading.Thread(target=pipeline, args=(h, scratch, runner, sources, fields, ref_all[h]))
               for h in HORIZONS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log_line(scratch, 'controller finished')
    runner.stop = True


# ------------------------------------------------------------------ worker
def detach_tokens(model, job):
    head = model.lowrank_interaction_branch.lowrank_head
    module = model.embedding if job.get('token_source', 'norm') == 'norm' else model.embedding.projection
    if len(module._forward_hooks) != 1:
        raise AssertionError('unexpected forward hooks on the token source')
    module._forward_hooks.clear()
    module.register_forward_hook(lambda m, inputs, output: setattr(head, 'phase_tokens', output.detach()))


def worker(job_path):
    register_heads()
    from scripts.lr_split_run import make_attach
    from scripts.vldbj_m1 import proto
    job = json.loads(Path(job_path).read_text())
    attach = make_attach(job['lr_split'])
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        if j['tune'].get('detach') and j.get('level_input') == 'history+token':
            detach_tokens(model, j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


# ------------------------------------------------------------------ check / report
def check(args):
    """Heads register, optimizer groups carry the intended rates, detach cuts the embedding gradient."""
    import torch
    register_heads()
    from scripts.lr_split_run import make_attach
    from scripts.vldbj_m1.proto import build_model
    from scripts.vldbj_m1.run import e14_sources
    sources = e14_sources()
    fields = {'code_commit': '', 'source_hashes': {}, 'smoke': True}
    scratch = Path('/tmp/traffic_b0hp_check')
    h = 96
    b = BASE[h]['cell']
    tests = [b, b._replace(head='H12L'), b._replace(head='H48G', token='raw'),
             b._replace(token='raw', zero=1, gm=0.0, rm=3.0), b._replace(detach=1),
             b._replace(detach=1, token='raw')]
    for c in tests:
        c = norm_cell(c)
        path = make_job(scratch, 'check', h, SEED, c, 'b0hp', sources, fields)
        job = json.loads(path.read_text())
        job['source_hashes'] = {}
        model, _, _, _ = build_model(dict(job))
        if job['tune']['detach']:
            detach_tokens(model, job)
        make_attach(job['lr_split'])(model, job['backbone_lr_mult'], job['extension_lr_mult'])
        opt = model.configure_optimizers()
        opt = opt[0][0] if isinstance(opt, (tuple, list)) else opt
        base_lr = float(model.args.training_args.learning_rate)
        got = {g['name']: g['lr'] / base_lr for g in opt.param_groups}
        model.train()
        x = torch.randn(2, model.seq_len, model.enc_in)
        mark = torch.zeros(2, model.seq_len, 4)
        y = torch.randn(2, model.seq_len + model.pred_len, model.enc_in)
        ymark = torch.zeros(2, model.seq_len + model.pred_len, 4)
        out = model(x, mark, model._build_decoder_input(y), ymark)[0][:, -model.pred_len:]
        head = model.lowrank_interaction_branch.lowrank_head
        latent = head.encode_latent(x)[0]
        grad = torch.autograd.grad(latent.sum(), model.embedding.projection.weight, allow_unused=True)[0]
        reaches = grad is not None and bool(grad.abs().sum() > 0)
        ok = (abs(got['gate'] - c.gm) < 1e-9 and abs(got['readout'] - c.rm) < 1e-9
              and abs(got['backbone'] - c.bb) < 1e-9 and reaches == (not c.detach and not c.zero)
              and tuple(out.shape) == (2, model.pred_len, model.enc_in))
        print(f"{cell_name(h, SEED, 'b0hp', c)[-60:]:60} pooled={head.pooled_len:>3} groups="
              f"{ {k: round(v, 4) for k, v in got.items()} } embedding_grad={reaches} {'' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit('check failed')
    shutil.rmtree(scratch, ignore_errors=True)
    print('check passed')


def report(args):
    scratch = args.scratch
    ref_all = la_reference()
    runs = read_runs(scratch)
    for h in HORIZONS:
        ref = ref_all[h]
        base = BASE[h]
        done = finished(runs, h, stages='ABCD')
        total = [r for r in runs if r['h'] == h and r['stage'] in 'ABCD']
        print(f"\n== Traffic-{h}: {len(done)}/{len(total)} search runs finished; LA {ref[0]:.4f}/{ref[1]:.4f}; "
              f"B0H+ now {base['mse']:.4f}/{base['mae']:.4f} (J {score(base['mse'], base['mae'], ref):.4f})")
        ranked_runs = sorted(done, key=lambda r: score(r['mse'], r['mae'], ref))
        print(f"{'J':>7} {'mse':>7} {'mae':>7} {'vJ':>7} st  cell")
        for r in ranked_runs[:10]:
            c = r['cell']
            print(f"{score(r['mse'], r['mae'], ref):7.4f} {r['mse']:7.4f} {r['mae']:7.4f} "
                  f"{score(r['vmse'], r['vmae'], (1, 1)):7.4f}  {r['stage']}  p{c.period} g{c.gate:g} gm{c.gm:g} "
                  f"rm{c.rm:g} bb{c.bb:g} {c.head} {'sh' if c.shared else 'pc'} {c.token}"
                  f"{'+zero' if c.zero else ''}{'+detach' if c.detach else ''}")
        if done:
            bm = min(done, key=lambda r: r['mse'])
            ba = min(done, key=lambda r: r['mae'])
            dom = [r for r in done if r['mse'] < ref[0] and r['mae'] < ref[1]]
            print(f"best mse {bm['mse']:.4f} (mae {bm['mae']:.4f}); best mae {ba['mae']:.4f} (mse {ba['mse']:.4f}); "
                  f"cells beating LA on both: {len(dom)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--gpus', default='0,1,2,3,4,7')
    parser.add_argument('--slots-per-gpu', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
    elif args.check:
        check(args)
    elif args.report:
        report(args)
    elif args.run:
        run_search(args)


if __name__ == '__main__':
    main()
