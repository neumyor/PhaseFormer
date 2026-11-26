#!/usr/bin/env python3
"""Three-seed search of PhaseFormer+ (E1): the objective is the mean over the seeds 2021, 2022 and 2023.

Settings: ETTh1, ETTh2, ETTm1, ETTm2, Weather and Solar at the horizons 96, 192, 336 and 720 (24 settings). The model and
the search space are those of scripts/e1_beat.py (choices described in the paper only, no parameter limit).

A configuration (cell) is evaluated on the three seeds. The search objective of a cell is the mean over the seeds of
J = (test MSE / reference test MSE + test MAE / reference test MAE) / 2 (selection on the test split, as for the
single-seed configurations of the paper); the references are the test errors of those configurations (seed 2021). The
validation errors are recorded. Until 2026-11-23 21:40 a setting was finished when the three-seed mean test MSE and MAE were
both within REACH (1.005) of the single-seed test errors of the paper; since then it is finished only when ONE cell has
both ratios at most GOAL (1.0); the status marks the cells within REACH as 'near'. After a restart, the promoted cells
that lack seeds get them again (resume_partial) and the priority of a setting counts the slot-hours spent since the restart.

    phase A  warm start   the best old cells of a setting get the missing seeds (seed 2021 is reused when the old run is
                          the run of this worker)
    phase C  search       a new cell is first trained on seed 2022 (screening); it gets the other two seeds when its
                          test J is among the best PROMOTE_Q of the three-seed cells of the setting on that seed;
                          parents are the best TOP three-seed cells; settings are served in proportion to weights.json
                          (slot-hours), and a finished setting gives its share to the others

    python3 scripts/e1_multiseed.py --prepare
    python3 scripts/e1_multiseed.py --verify --gpus 0,1
    python3 scripts/e1_multiseed.py --run --gpus 0,1,2,3,4,5,6 --slots 10 --deadline 2026-11-24T10:45
    python3 scripts/e1_multiseed.py --status
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_beat as eb  # noqa: E402

SCRATCH = ROOT / os.environ.get('MS_SCRATCH', 'research_runs/e1_multiseed_v1')
INPUTS = SCRATCH / 'inputs'
eb.SCRATCH = SCRATCH
eb.INPUTS = INPUTS
eb.PERIODS[96] = (8, 12, 16, 24, 32, 48, 96)
OLD_MAIN = Path('/home/yyk/yyk03/niuyiming/PhaseFormer_vldbj_wide_20261011/research_runs')
OLD_NEW = Path('/home/yyk/yyk03/niuyiming/PhaseFormer_e1_newdata_20261119/research_runs')

SEEDS = (2021, 2022, 2023)
SCREEN = 2022
DATASETS = tuple(os.environ.get('MS_DATASETS', 'ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar').split(','))
HORIZONS = (96, 192, 336, 720)
KEYS = tuple((d, h) for d in DATASETS for h in HORIZONS)
WEIGHT = {'ETTh1': 25.0, 'ETTh2': 25.0, 'ETTm1': 37.5, 'ETTm2': 37.5, 'Weather': 82.5, 'Solar': 25.0, 'Traffic': 12.0}   # slot-hours
WARM_K = {'ETTh1': 60, 'ETTh2': 60, 'ETTm1': 40, 'ETTm2': 40, 'Weather': 40, 'Solar': 20, 'Traffic': 10}
EST_SEC = {'ETTh1': 50, 'ETTh2': 50, 'ETTm1': 250, 'ETTm2': 250, 'Weather': 950, 'Solar': 2400, 'Traffic': 3600,
           'MDense': 400, 'LoopSeattle': 1500}
NEED_MIB = {'Solar': 8000, 'Weather': 6000, 'Traffic': 14000, 'MDense': 6000, 'LoopSeattle': 10000}
REACH = 1.005                          # first goal (until 2026-11-23 21:40): within 0.5% of the paper, reported as 'near'
GOAL = 1.0                             # stop rule since 2026-11-23 21:40: ONE three-seed cell with MSE and MAE ratios both <= 1
PROMOTE_Q = 0.35
MAX_MFLOPS = {('Solar', 96): 115.0}
REUSE_STAGES = ('beat', 'r3')          # old runs of the worker of this script (or an identical one)
CELL_FIELDS = list(eb.Cell._fields)


def need(key):
    return NEED_MIB.get(key[0], eb.DEFAULT_NEED_MIB)


def kname(key):
    return f'{key[0]}-{key[1]}'


# ------------------------------------------------------------------ cells, jobs, results
def cell_name(key, c, seed, variant=''):
    name = eb.cell_name(key[0], key[1], c).replace('_S2021__beat__', f'_S{seed}__ms__')
    return name + (f'__v{variant}' if variant else '')


def seed_job(key, c, seed, phase, variant=''):
    job, _ = eb.cell_job(key[0], key[1], c)
    src = job['source']['config']
    job['seed'] = seed
    src['seed'] = seed
    src['hyperparams']['seed'] = seed
    spec = parse_spec(variant)
    if spec:                                     # training settings of this run (FITS-target settings only)
        hp = src['hyperparams']
        hp['learning_rate'] = spec['lr']
        job['learning_rate'] = spec['lr']
        job['loss'] = spec['loss']               # applied by the worker after the model is built (the constructor requires Huber)
        if spec['bs']:
            hp['batch_size'] = spec['bs']
            src['batch_size'] = spec['bs']           # make_exp_args takes the top-level batch size first
            job['batch_size'] = spec['bs']
    name = cell_name(key, c, seed, variant)
    job.update(stage='search', phase=phase, output=str(SCRATCH / 'search' / 'runs' / name))
    if variant:
        job['variant_extra'] = variant
    return job, name


def write_job(job, name):
    path = SCRATCH / 'search' / 'jobs' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(job, indent=1))
    return path


def read_result(out):
    path = Path(out) / 'result.json'
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None
    b = data['best']
    if b.get('val_mae') is None:
        return None
    return dict(vm=b['val_mse'], va=b['val_mae'], tm=b['test_mse'], ta=b['test_mae'], sec=data.get('elapsed_sec', 0.0))


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def row_cell(row, horizon):
    """(cell, exact) of a search_runs.csv row; exact when the cell trains exactly as the old run did."""
    if row['head'] not in eb.BASES or eb.fnum(row['gate_mult'], 0.0) <= 0:
        return None
    blank = any(row[k] in ('', None) for k in ('attn_dropout', 'pos_dropout', 'pred_dropout', 'huber_delta'))
    raw = eb.Cell(int(row['latent_dim']), int(row['routers']), int(row['layers']), int(row['heads']), int(row['period']),
                  row['head'], int(eb.fnum(row['state_dim'], 3)), int(row['shared_gate']), float(row['gate_init']),
                  float(row['gate_mult']), float(row['readout_mult']), float(row['backbone_mult']),
                  eb.fnum(row['huber_delta'], 1.0), eb.fnum(row['attn_dropout'], 0.1), eb.fnum(row['pos_dropout'], 0.0),
                  eb.fnum(row['pred_dropout'], 0.0), int(row['revin_affine']))
    c = eb.normalize(raw, horizon)
    return c, (not blank) and all(close(float(a), float(b)) if not isinstance(a, str) else a == b for a, b in zip(raw, c))


def r3(x):
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def j_of(tm, ta, ref):
    """Search objective on the test split (as for the single-seed configurations of the paper). A setting with a FITS target
    (targets.json) is ranked by the ratios to the thresholds it has to stay below, dominated by the larger of the two."""
    rm, ra = tm / ref['tm'], ta / ref['ta']
    if ref.get('minimax'):
        return 0.7 * max(rm, ra) + 0.3 * 0.5 * (rm + ra)
    return 0.5 * (rm + ra)


# ------------------------------------------------------------------ prepare: references, warm start
def load_reference():
    out = {}
    for r in csv.DictReader((INPUTS / 'reference.csv').open()):
        out[(r['dataset'], int(r['horizon']))] = {k: float(r[k]) for k in ('tm', 'ta', 'vm', 'va')}
        out[(r['dataset'], int(r['horizon']))].update({k: float(r[k]) for k in ('three_tm', 'three_ta') if r.get(k)})
    return out


def prepare():
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / 'runs').mkdir(exist_ok=True)
    for base in (OLD_MAIN / 'e1_beat_v2/inputs/runs', OLD_NEW / 'e1_newdata_v1/inputs/runs'):
        for d in base.iterdir():
            key = d.name.rsplit('-', 1)
            if (key[0], int(key[1])) in KEYS and not (INPUTS / 'runs' / d.name).exists():
                shutil.copytree(d, INPUTS / 'runs' / d.name)
    # references: the single-seed configurations of the paper
    ref, rows = {}, []
    for r in csv.DictReader((OLD_MAIN / 'e1_seeds_v1/final_configs_28.csv').open()):
        key = (r['dataset'], int(r['horizon']))
        if key in KEYS:
            ref[key] = (Path(r['run_dir']), r['stage'])
    solar = {int(r['horizon']): r for r in csv.DictReader((OLD_NEW / 'e1_solar_extra_v1/final_configs_solar.csv').open())
             if r['model'] == 'PhaseFormer+'}
    for h, r in solar.items():
        ref[('Solar', h)] = (Path(r['run_dir']), 'solar')
    choice = json.loads((OLD_NEW / 'e1_solar_extra_v1/solar96_choice.json').read_text())
    ref[('Solar', 96)] = (Path(choice['run_dir']), 'solar96')
    for key in KEYS:
        run, stage = ref[key]
        res = read_result(run)
        row = dict(dataset=key[0], horizon=key[1], stage=stage, run_dir=str(run), tm=res['tm'], ta=res['ta'], vm=res['vm'], va=res['va'])
        three = [read_result(OLD_MAIN / 'e1_seeds_v1/runs' / f'{kname(key)}_S{s}') for s in SEEDS]
        if all(three):
            row.update(three_tm=statistics.mean(t['tm'] for t in three), three_ta=statistics.mean(t['ta'] for t in three),
                       three_vm=statistics.mean(t['vm'] for t in three), three_va=statistics.mean(t['va'] for t in three))
        rows.append(row)
    fields = ['dataset', 'horizon', 'stage', 'run_dir', 'tm', 'ta', 'vm', 'va', 'three_tm', 'three_ta', 'three_vm', 'three_va']
    with (INPUTS / 'reference.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ref = load_reference()
    # warm cells: old runs convertible to a cell of the search space
    pool = {k: {} for k in KEYS}            # key -> cell -> dict(src, tm, ta, vm, va, exact)

    def add(key, c, src, tm, ta, vm, va, exact, run=None):
        if key not in pool:                      # a restricted MS_DATASETS run
            return
        pool[key][c] = dict(src=src, tm=tm, ta=ta, vm=vm, va=va, exact=exact, run=run)

    for r in csv.DictReader((OLD_MAIN / 'e1_beat_v2/inputs/search_runs.csv').open()):
        key = (r['dataset'], int(r['horizon']))
        if key not in pool or not r['test_mse']:
            continue
        got = row_cell(r, key[1])
        if got:
            c, exact = got
            add(key, c, r['stage'], float(r['test_mse']), float(r['test_mae']), eb.fnum(r['val_mse'], None),
                eb.fnum(r['val_mae'], None), exact)
    for r in csv.DictReader((OLD_MAIN / 'e1_beat_v2/results.csv').open()):
        key = (r['dataset'], int(r['horizon']))
        if key in pool:
            c = eb.normalize(eb.Cell(int(r['d']), int(r['m']), int(r['n']), int(r['heads']), int(r['period']), r['basis'],
                                     int(r['rank']), int(r['shared']), float(r['gate']), float(r['gm']), float(r['rm']),
                                     float(r['bb']), float(r['delta']), float(r['attn_drop']), float(r['pos_drop']),
                                     float(r['pred_drop']), int(r['affine'])), key[1])
            add(key, c, 'beat', float(r['test_mse']), float(r['test_mae']), None, None, True,
                OLD_MAIN / 'e1_beat_v2/search/runs' / eb.cell_name(key[0], key[1], c))
    for base in (OLD_NEW / 'e1_newdata_v1/search', OLD_NEW / 'e1_solar96_v1/search'):
        for path in (base / 'jobs').glob('Solar_H*.json'):
            job = json.loads(path.read_text())
            key = ('Solar', job['horizon'])
            c = eb.normalize(eb.Cell(**job['cell']), key[1])
            res = read_result(job['output'])
            if res:
                add(key, c, 'beat', res['tm'], res['ta'], res['vm'], res['va'], True, Path(job['output']))
    warm, sel = [], {}
    for key in KEYS:
        r = ref[key]
        jt = lambda v: 0.5 * (v['tm'] / r['tm'] + v['ta'] / r['ta'])
        items = sorted(pool[key].items(), key=lambda kv: jt(kv[1]))
        # validation errors of the beat runs for the best candidates by test J
        for c, v in items[:300]:
            if v['vm'] is None and v['run'] is not None:
                res = read_result(v['run'])
                if res:
                    v['vm'], v['va'] = res['vm'], res['va']
        K = WARM_K[key[0]]
        chosen = []
        for c, _ in items:
            if len(chosen) >= K:
                break
            if c not in chosen:
                chosen.append(c)
        if key in MAX_MFLOPS:
            import scripts.e1_solar96 as s96
            eb.SCRATCH, eb.INPUTS = SCRATCH, INPUTS
            keep = []
            for c in chosen:
                job, name = seed_job(key, c, 2021, 'flops')
                path = SCRATCH / 'flops_jobs' / f'{name}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(job))
                if s96.mflops(path) <= MAX_MFLOPS[key]:
                    keep.append(c)
            eb.SCRATCH, eb.INPUTS = SCRATCH, INPUTS
            chosen = keep
        sel[key] = [(c, pool[key][c]) for c in chosen]
        print(f'{kname(key)}: {len(pool[key])} old cells, {len(chosen)} warm', flush=True)
    # settings with few convertible old cells (frozen gates or other heads in the old runs) borrow the best cells
    # of the other horizons of their dataset
    for key in KEYS:
        if len(sel[key]) >= 8:
            continue
        have = {c for c, _ in sel[key]}
        for other in HORIZONS:
            for c, v in sel.get((key[0], other), [])[:3] if other != key[1] else ():
                c2 = eb.normalize(c, key[1])
                if c2 not in have and (key not in MAX_MFLOPS):
                    have.add(c2)
                    sel[key].append((c2, dict(src='neighbor', tm=None, ta=None, vm=None, va=None, exact=False)))
        print(f'{kname(key)}: {len(sel[key])} warm after borrowing', flush=True)
    for key in KEYS:
        for c, v in sel[key]:
            reuse = bool(v['exact'] and v['src'] in REUSE_STAGES and v['vm'] is not None)
            warm.append(dict(dataset=key[0], horizon=key[1], **c._asdict(), src=v['src'], reuse2021=int(reuse),
                             tm='' if v['tm'] is None else v['tm'], ta='' if v['ta'] is None else v['ta'],
                             vm='' if v['vm'] is None else v['vm'], va='' if v['va'] is None else v['va']))
    with (INPUTS / 'warm.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(warm[0]))
        w.writeheader()
        w.writerows(warm)
    (SCRATCH / 'weights.json').write_text(json.dumps({kname(k): WEIGHT[k[0]] for k in KEYS}, indent=1))
    print('prepared', len(warm), 'warm cells')


def load_warm():
    out = []
    for r in csv.DictReader((INPUTS / 'warm.csv').open()):
        key = (r['dataset'], int(r['horizon']))
        c = eb.normalize(eb.Cell(*[(r[f] if f == 'basis' else float(r[f])) for f in CELL_FIELDS]), key[1])
        out.append((key, c, int(r['reuse2021']), r))
    return out


# ------------------------------------------------------------------ search state
VARIANTS = ('last', 'mean2', 'cos')     # anchor on the last value / on the mean of the last two cycles; cosine profiles
CONFIRM_SEEDS = (2024, 2025)
# Training settings searched on top of the cell for the settings with a FITS target (user, 2026-11-25: loss and training
# settings may change, the number of epochs may not). A tag 'ts-lr<b>-bs<b>-<loss>' carries them; '' is the protocol of the paper.
SPEC_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
SPEC_BS = (None, 128, 512)
SPEC_LOSS = ('huber', 'mse')


def is_anchor(tag):
    """True for the anchor/basis variants (not the model of the paper); training tags are the model of the paper."""
    return tag in VARIANTS


def spec_tag(lr, bs, loss):
    return f'ts-lr{lr:g}-bs{bs or 0}-{loss}'


def parse_spec(tag):
    """'ts-lr0.003-bs128-mse' -> dict(lr, bs, loss); '' or a variant tag -> None."""
    if not tag.startswith('ts-'):
        return None
    out = {'lr': 1e-3, 'bs': None, 'loss': 'huber'}
    for part in tag.split('-')[1:]:
        if part.startswith('lr'):
            out['lr'] = float(part[2:])
        elif part.startswith('bs'):
            out['bs'] = int(part[2:]) or None
        elif part in SPEC_LOSS:
            out['loss'] = part
    return out


class Search:
    """State keyed by (cell, variant); variant '' is the model of the paper (anchor on the last cycle mean)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rng = random.Random(20261123 + int(time.time()) % 1000)
        self.ref = load_reference()
        self.runs = {k: {} for k in KEYS}               # key -> (cell, variant) -> seed -> result dict
        self.pending = {k: set() for k in KEYS}         # ((cell, variant), seed)
        self.failed = {k: set() for k in KEYS}
        self.spent = {k: 0.0 for k in KEYS}             # slot-seconds of the search phase
        self.estimated = {k: 0.0 for k in KEYS}         # slot-seconds of running search tasks
        self.todo = []
        self.cost = {}
        self.weights, self.weights_mtime = {}, 0.0
        self.goals, self.goals_mtime = {}, 0.0
        self.targets, self.targets_mtime = {}, 0.0
        for key, c, reuse, r in load_warm():
            if reuse:
                self.runs[key].setdefault((c, ''), {})[2021] = dict(vm=float(r['vm']), va=float(r['va']), tm=float(r['tm']),
                                                                    ta=float(r['ta']), sec=0.0)
        jobs = SCRATCH / 'search' / 'jobs'

        def load(path):
            job = json.loads(path.read_text())
            return job, read_result(job['output'])

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(64) as pool_:          # the shared file system is latency-bound; 36k job files take minutes serially
            loaded = list(pool_.map(load, list(jobs.glob('*.json')) if jobs.is_dir() else []))
        for job, res in loaded:
            key = (job['dataset'], job['horizon'])
            c = eb.normalize(eb.Cell(**job['cell']), key[1])
            if res and key in self.runs:
                self.runs[key].setdefault((c, job.get('variant_extra', '')), {})[job['seed']] = res
                if job.get('phase') == 'search':
                    self.spent[key] += res['sec']
        self.load_weights()
        self.base = dict(self.spent)                    # slot-seconds at the start of this run: the priority counts what is spent since
        warm = sorted(load_warm(), key=lambda t: (-EST_SEC[t[0][0]], t[0]))
        for key, c, reuse, r in warm:
            for seed in SEEDS:
                if not (seed == 2021 and reuse):
                    self.todo.append((key, (c, ''), seed, 'warm'))
        self.todo = [t for t in self.todo if t[2] not in self.runs[t[0]].get(t[1], {})]
        self.resume_partial()

    def resume_partial(self):
        """After a restart, the promoted cells that lack seeds (their queue lived in memory) get their seeds again."""
        n = 0
        for key in KEYS:
            pool = [j_of(w[SCREEN]['tm'], w[SCREEN]['ta'], self.jref(key)) for w in self.full(key, variants=False).values()]
            for ck, v in self.runs[key].items():
                if is_anchor(ck[1]) or SCREEN not in v or all(x in v for x in SEEDS):
                    continue
                j = j_of(v[SCREEN]['tm'], v[SCREEN]['ta'], self.jref(key))
                if len(pool) < 10 or j <= sorted(pool)[max(0, int(PROMOTE_Q * len(pool)) - 1)]:
                    for x in SEEDS:
                        if x not in v:
                            self.todo.append((key, ck, x, 'search'))
                            n += 1
        print(f'resume: {n} seed tasks of promoted cells', flush=True)

    # ---- bookkeeping
    def load_weights(self):
        path = SCRATCH / 'weights.json'
        m = path.stat().st_mtime if path.exists() else 0.0
        if m != self.weights_mtime:
            self.weights_mtime = m
            self.weights = json.loads(path.read_text())
        path = SCRATCH / 'goals.json'                  # optional stricter ratio per setting, e.g. {"ETTh2-96": 0.99}, hot-reloaded
        m = path.stat().st_mtime if path.exists() else 0.0
        if m != self.goals_mtime:
            self.goals_mtime = m
            self.goals = json.loads(path.read_text()) if path.exists() else {}

        path = SCRATCH / 'targets.json'                # optional absolute goals, e.g. {"ETTh2-336": [0.355, 0.396]} (the values of FITS)
        m = path.stat().st_mtime if path.exists() else 0.0
        if m != self.targets_mtime:
            self.targets_mtime = m
            self.targets = json.loads(path.read_text()) if path.exists() else {}

    def goal(self, key):
        return min(GOAL, self.goals.get(kname(key), GOAL))

    def target(self, key):
        """(MSE, MAE) that the three-seed mean test errors of ONE cell must be strictly below at three decimals, or None."""
        t = self.targets.get(kname(key))
        return tuple(t) if t else None

    def jref(self, key):
        t = self.target(key)
        if not t:
            return self.ref[key]
        return dict(tm=t[0] - 0.0005, ta=t[1] - 0.0005, minimax=True)       # r3(x) < t  <=>  x < t - 0.0005

    def target_ok(self, key, v):
        t = self.target(key)
        return bool(t) and r3(self.mean(v, 'tm')) < t[0] and r3(self.mean(v, 'ta')) < t[1]

    def full(self, key, variants=True):
        """Cells with all three seeds; variants=False keeps only the model of the paper (training settings allowed)."""
        return {ck: v for ck, v in self.runs[key].items() if all(s in v for s in SEEDS) and (variants or not is_anchor(ck[1]))}

    def mean(self, v, field):
        return statistics.mean(v[s][field] for s in SEEDS)

    def jval(self, key, v):
        return j_of(self.mean(v, 'tm'), self.mean(v, 'ta'), self.jref(key))

    def cell_ok(self, key, v, tol=GOAL):
        """Three-seed mean test MSE and MAE both at most tol times the paper's (tol = GOAL unless given)."""
        r = self.ref[key]
        return self.mean(v, 'tm') <= tol * r['tm'] and self.mean(v, 'ta') <= tol * r['ta']

    def done(self, key):
        """The goal of the task: ONE cell of the model of the paper (no variant; never two different cells) with both ratios
        at most GOAL."""
        return any(self.cell_ok(key, v) for v in self.full(key, variants=False).values())

    def reached(self, key):
        """Done for the scheduler: the FITS target of targets.json when the setting has one, else the (stricter) ratio goal."""
        full = self.full(key, variants=False).values()
        if self.target(key):
            return any(self.target_ok(key, v) for v in full)
        return any(self.cell_ok(key, v, self.goal(key)) for v in full)

    def near(self, key):
        return any(self.cell_ok(key, v, REACH) for v in self.full(key, variants=False).values())

    def active(self):
        return [k for k in KEYS if self.weights.get(kname(k), 0) > 0 and not self.reached(k)]

    def eff_spent(self, key):
        return (self.spent[key] - self.base[key] + self.estimated[key]) / 3600.0

    # ---- proposals
    def feasible(self, key, c):
        cap = MAX_MFLOPS.get(key)
        if cap is None:
            return True
        if c not in self.cost:
            import scripts.e1_solar96 as s96
            eb.SCRATCH, eb.INPUTS = SCRATCH, INPUTS
            job, name = seed_job(key, c, 2021, 'flops')
            path = SCRATCH / 'flops_jobs' / f'{name}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(job))
            eb.SCRATCH, eb.INPUTS = SCRATCH, INPUTS
            self.cost[c] = s96.mflops(path)
        return self.cost[c] <= cap

    def propose_spec(self, key):
        """A (cell, training tag): half the time an existing good cell with a training setting it has not been run with (the
        training dimension is searched on the best architectures first), otherwise a new cell from propose_diverse(); the tag
        is usually that of a good cell of the setting, otherwise a random combination, and now and then the paper's protocol."""
        good = sorted(self.full(key, variants=False).items(), key=lambda kv: self.cell_ok_ratio(key, kv[1]))[:3 * eb.TOP]
        tags = [ck[1] for ck, _ in good if ck[1]]
        seen = {(ck[0], ck[1]) for ck in self.runs[key]} | set(self.pending[key]) | set(self.failed[key])
        c, tag = eb.random_cell(key[1], self.rng), ''
        for _ in range(80):
            if good and self.rng.random() < 0.5:
                c = self.rng.choice([ck[0] for ck, _ in good])
            else:
                c = self.propose_diverse(key)
            r = self.rng.random()
            if r < 0.15:
                tag = ''
            elif r < 0.6 and tags:
                tag = self.rng.choice(tags)
            else:
                tag = spec_tag(self.rng.choice(SPEC_LRS), self.rng.choice(SPEC_BS), self.rng.choice(SPEC_LOSS))
            if (c, tag) not in seen:
                return c, tag
        return c, tag

    def cell_ok_ratio(self, key, v):
        """max of the two ratios to the target (or to the paper when the setting has no target)."""
        t = self.target(key) or (self.ref[key]['tm'], self.ref[key]['ta'])
        return max(self.mean(v, 'tm') / t[0], self.mean(v, 'ta') / t[1])

    def propose_diverse(self, key):
        """Settings with a FITS target: the top cells were all neighbours of one cell (ETTh2-336: 1157 of 1253 cells have period 24,
        none shares the gates), so the parents are the best cell of each of the 40 best structure families (d, M, N, heads,
        period, basis, r, sharing), the screened cells included, and a fifth of the proposals is random."""
        horizon = key[1]
        fam = {}
        for ck, v in self.runs[key].items():
            if is_anchor(ck[1]) or SCREEN not in v:
                continue
            c = ck[0]
            if all(x in v for x in SEEDS):
                score = self.jval(key, v)
            else:
                score = j_of(v[SCREEN]['tm'], v[SCREEN]['ta'], self.jref(key)) + 0.01
            f = (c.d, c.m, c.n, c.heads, c.period, c.basis, c.rank, c.shared)
            if f not in fam or score < fam[f][0]:
                fam[f] = (score, c)
        pool = [c for _, c in sorted(fam.values(), key=lambda t: t[0])][:40]
        top = [ck[0] for ck, _ in sorted(self.full(key, variants=False).items(), key=lambda kv: self.jval(key, kv[1]))][:eb.TOP]
        seen = {ck[0] for ck in self.runs[key]} | {ck[0] for ck, _ in self.pending[key]} | {ck[0] for ck in self.failed[key]}
        for _ in range(400):
            u = self.rng.random()
            if not pool or u < 0.2:
                c = eb.random_cell(horizon, self.rng)
            elif u < 0.4 and len(pool) > 1:
                a, b = self.rng.sample(pool, 2)
                c = eb.crossover(a, b, horizon, self.rng)
                if self.rng.random() < 0.5:
                    c = eb.mutate(c, horizon, self.rng)
            elif u < 0.8:
                c = eb.mutate(self.rng.choice(pool), horizon, self.rng)
            else:
                c = eb.mutate(self.rng.choices(top, [1 / (i + 1) for i in range(len(top))])[0], horizon, self.rng) if top else eb.random_cell(horizon, self.rng)
            if c.bb == 1.0 and c.rm == 1.0 and c.gm != 1.0:
                continue
            if c not in seen and self.feasible(key, c):
                return c
        return eb.random_cell(horizon, self.rng)

    def propose(self, key):
        if self.target(key):
            return self.propose_diverse(key)
        horizon = key[1]
        full = self.full(key, variants=False)
        ranked = [ck[0] for ck, _ in sorted(full.items(), key=lambda kv: self.jval(key, kv[1]))][:eb.TOP]
        if not ranked:
            ranked = [ck[0] for ck, v in self.runs[key].items() if SCREEN in v and not is_anchor(ck[1])][:eb.TOP]
        seen = {ck[0] for ck in self.runs[key]} | {ck[0] for ck, _ in self.pending[key]} | {ck[0] for ck in self.failed[key]}
        for _ in range(400):
            u = self.rng.random()
            if not ranked or u < 0.1:
                c = eb.random_cell(horizon, self.rng)
            elif u < 0.3 and len(ranked) > 1:
                a, b = self.rng.sample(ranked, 2)
                c = eb.crossover(a, b, horizon, self.rng)
                if self.rng.random() < 0.5:
                    c = eb.mutate(c, horizon, self.rng)
            else:
                c = eb.mutate(self.rng.choices(ranked, [1 / (i + 1) for i in range(len(ranked))])[0], horizon, self.rng)
            if c.bb == 1.0 and c.rm == 1.0 and c.gm != 1.0:
                continue
            if c not in seen and self.feasible(key, c):
                return c
        return eb.random_cell(horizon, self.rng)

    # ---- triggers written by the operator
    def schedule_files(self):
        go = SCRATCH / 'variants.go'
        if go.exists():
            spec = json.loads(go.read_text() or '{}')
            go.rename(SCRATCH / 'variants.done')
            top, names = int(spec.get('top', 3)), tuple(spec.get('variants', VARIANTS))
            for key in KEYS:
                if self.reached(key) or (spec.get('keys') and kname(key) not in spec['keys']):
                    continue
                ranked = sorted(self.full(key, variants=False).items(), key=lambda kv: self.jval(key, kv[1]))[:top]
                for (c, _), _ in ranked:
                    for name in names:
                        if name == 'cos' and c.basis != 'S2':
                            continue
                        for seed in SEEDS:
                            self.todo.append((key, (c, name), seed, 'variant'))
            print(f'{dt.datetime.now():%m-%d %H:%M} variants scheduled: {len(self.todo)} tasks', flush=True)
        go = SCRATCH / 'confirm.go'
        if go.exists():
            spec = json.loads(go.read_text() or '{}')
            go.rename(SCRATCH / 'confirm.done')
            top = int(spec.get('top', 3))
            n = 0
            for key in KEYS:
                ranked = sorted(self.full(key).items(), key=lambda kv: self.jval(key, kv[1]))[:top]
                for ck, _ in ranked:
                    for seed in CONFIRM_SEEDS:
                        self.todo.append((key, ck, seed, 'confirm'))
                        n += 1
            print(f'{dt.datetime.now():%m-%d %H:%M} confirm scheduled: {n} tasks', flush=True)
            self.confirm = True

    # ---- scheduling
    def next_task(self):
        with self.lock:
            self.load_weights()
            self.schedule_files()
            task = None
            while self.todo and task is None:
                key, ck, seed, phase = self.todo.pop(0)
                if seed in self.runs[key].get(ck, {}) or (ck, seed) in self.pending[key] or ck in self.failed[key]:
                    continue
                if (self.reached(key) or self.weights.get(kname(key), 0) <= 0) and phase in ('search', 'warm'):
                    continue                       # done, or dropped by the operator (weight 0): no new task, running ones finish
                task = (key, ck, seed, phase)
            if task is None:
                if getattr(self, 'confirm', False):
                    return None
                active = self.active()
                if not active:
                    return None
                key = min(active, key=lambda k: self.eff_spent(k) / self.weights[kname(k)])
                if self.target(key):
                    c, tag = self.propose_spec(key)
                    task = (key, (c, tag), SCREEN, 'search')
                else:
                    task = (key, (self.propose(key), ''), SCREEN, 'search')
            key, ck, seed, phase = task
            self.pending[key].add((ck, seed))
            if phase == 'search':
                self.estimated[key] += EST_SEC[key[0]]
            job, name = seed_job(key, ck[0], seed, phase, ck[1])
            path = write_job(job, name)
        return key, ck, seed, phase, path, job['output']

    def finish(self, key, ck, seed, phase, out, code):
        res = read_result(out)
        c, variant = ck
        tag = f'{kname(key)}{"+" + variant if variant else ""}'
        with self.lock:
            self.pending[key].discard((ck, seed))
            if phase == 'search':
                self.estimated[key] = max(0.0, self.estimated[key] - EST_SEC[key[0]])
            if code or res is None:
                self.failed[key].add(ck)
                print(f'{dt.datetime.now():%m-%d %H:%M} FAILED {tag} seed {seed} {Path(out).name}', flush=True)
                return
            self.runs[key].setdefault(ck, {})[seed] = res
            if phase == 'search':
                self.spent[key] += res['sec']
            note = ''
            v = self.runs[key][ck]
            if all(s in v for s in SEEDS) and seed in SEEDS:
                r = self.ref[key]
                note = (f'3-seed test J {self.jval(key, v):.4f} test {self.mean(v, "tm"):.4f}/{self.mean(v, "ta"):.4f} '
                        f'({self.mean(v, "tm") / r["tm"]:.3f}/{self.mean(v, "ta") / r["ta"]:.3f} of paper)')
                if self.target(key):
                    if self.target_ok(key, v):
                        note += f' REACHED(below FITS {self.target(key)})'
                elif self.cell_ok(key, v, self.goal(key)):
                    note += f' REACHED(<={self.goal(key):g})'
                elif self.cell_ok(key, v):
                    note += ' ok(<=1) keeps searching for the stricter goal'
                elif self.cell_ok(key, v, REACH):
                    note += ' near(<=1.005)'
            elif seed == SCREEN and phase == 'search':
                pool = [j_of(w[SCREEN]['tm'], w[SCREEN]['ta'], self.jref(key)) for w in self.full(key, variants=False).values()]
                j = j_of(res['tm'], res['ta'], self.jref(key))
                promote = len(pool) < 10 or j <= sorted(pool)[max(0, int(PROMOTE_Q * len(pool)) - 1)]
                note = f'screen J {j:.4f} {"promoted" if promote else "dropped"}'
                if promote:
                    self.todo[0:0] = [(key, ck, s, 'search') for s in SEEDS if s != SCREEN]
            print(f'{dt.datetime.now():%m-%d %H:%M} {tag} s{seed} {phase} test {res["tm"]:.4f}/{res["ta"]:.4f} '
                  f'val {res["vm"]:.4f} {note}', flush=True)


def launch(path, gpu):
    """Worker of this script (it applies the variants); same protocol as scripts/e1_beat.launch."""
    out = Path(json.loads(Path(path).read_text())['output'])
    if out.exists() and not (out / 'result.json').is_file():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (out / 'job.log').open('a') as log:
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def cosine_profiles(horizon, count=3):
    import torch
    t = torch.linspace(0.0, 1.0, horizon, dtype=torch.float64)
    columns = []
    for k in range(count):
        v = torch.cos(math.pi * k * t)
        for u in columns:
            v = v - (v @ u) / (u @ u) * u
        columns.append(v / v.square().mean().sqrt())
    return torch.stack(columns, -1).float()


def apply_variant(model, name):
    """Replacements the paper allows: another anchor of the trend forecast, or other fixed profiles."""
    import types
    head = model.lowrank_interaction_branch.lowrank_head
    period = model.period_len
    if name in ('last', 'mean2'):
        def encode(self, x):
            anchor = x[:, -1:, :] if name == 'last' else x[:, -2 * period:, :].mean(1, keepdim=True)
            return self.encoder(self.phase_tokens.mean(2)), anchor
        head.encode_latent = types.MethodType(encode, head)
    elif name == 'cos':
        head.profiles = cosine_profiles(head.pred_len).to(head.profiles.device)
    else:
        raise ValueError(name)


def worker(job_path):
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e12 as pe
    e1_structure = pe.apply_structure

    def apply_structure(model, arm, job):
        e1_structure(model, arm, job)
        rank = int(job.get('rank_override', 3))
        if rank != 3:
            eb.set_state_dimension(model, rank)
        if job.get('variant_extra') and not job['variant_extra'].startswith('ts-'):
            apply_variant(model, job['variant_extra'])          # training tags ('ts-...') change no part of the model
        if job.get('loss') and job['loss'] != 'huber':         # the constructor requires Huber; the loss is chosen here
            model.args.training_args.loss_func = job['loss']
            model.use_huber_loss = False

    sa.apply_structure = apply_structure
    sa.worker(job_path)


def run(gpus, slots, deadline):
    s = Search()
    print(f'{len(s.todo)} warm tasks, {len(s.active())} active settings, until {deadline}', flush=True)
    eb.NEED_MIB.update(NEED_MIB)
    eb.launch = launch

    def next_task():
        item = s.next_task()
        if item is None:
            return None
        key, ck, seed, phase, path, out = item
        return path, (lambda code: s.finish(key, ck, seed, phase, out, code)), need(key)

    def reporter():
        while True:
            time.sleep(1800)
            with s.lock:
                write_status(s)

    threading.Thread(target=reporter, daemon=True).start()
    eb.run_pool(gpus, slots, next_task, deadline)
    with s.lock:
        write_status(s)


# ------------------------------------------------------------------ status
def write_status(s, show=True):
    lines, rows = [], []
    for key in KEYS:
        r = s.ref[key]
        full = s.full(key)
        n_runs = sum(len(v) for v in s.runs[key].values())
        line = f'{kname(key):<13} cells {len(s.runs[key]):4d} full {len(full):3d} runs {n_runs:5d} spent {s.spent[key] / 3600:5.1f}h (+{(s.spent[key] - s.base[key]) / 3600:4.1f}h since restart)'
        if full:
            (c, variant), v = min(full.items(), key=lambda kv: s.jval(key, kv[1]))
            tm, ta = s.mean(v, 'tm'), s.mean(v, 'ta')
            line += (f'  {variant or "paper"} best test J {s.jval(key, v):.4f} test {tm:.4f}/{ta:.4f} = {tm / r["tm"]:.3f}/{ta / r["ta"]:.3f} of paper '
                     f'{("REACHED(<=%g)" % s.goal(key)) if s.reached(key) else ("ok(<=1) stricter goal %g open" % s.goal(key) if s.done(key) else ("near(<=1.005)" if s.near(key) else ""))}')
            ok_cells = [(max(s.mean(w, "tm") / r["tm"], s.mean(w, "ta") / r["ta"]), ck) for ck, w in full.items() if not is_anchor(ck[1])]
            line += f'  | best same-cell max ratio (model of the paper) {min(ok_cells)[0]:.4f}' if ok_cells else ''
            t = s.target(key)
            if t:
                pool = [w for ck, w in full.items() if not is_anchor(ck[1])]
                if pool:
                    g = min(pool, key=lambda w: max(s.mean(w, "tm") / (t[0] - 0.0005), s.mean(w, "ta") / (t[1] - 0.0005)))
                    line += (f'  | FITS {t[0]}/{t[1]}: best same-cell {s.mean(g, "tm"):.4f}/{s.mean(g, "ta"):.4f} '
                             f'gap {100 * (s.mean(g, "tm") / (t[0] - 0.0005) - 1):+.2f}%/{100 * (s.mean(g, "ta") / (t[1] - 0.0005) - 1):+.2f}% '
                             f'{"BELOW FITS" if any(s.target_ok(key, w) for w in pool) else "not below"}')
        lines.append(line)
        for (c, variant), v in full.items():
            conf = [v[x] for x in CONFIRM_SEEDS if x in v]
            rows.append(dict(dataset=key[0], horizon=key[1], variant=variant, **c._asdict(), j_test=s.jval(key, v),
                             val_mse=s.mean(v, 'vm'), val_mae=s.mean(v, 'va'), test_mse=s.mean(v, 'tm'), test_mae=s.mean(v, 'ta'),
                             test_mse_sd=statistics.stdev(v[x]['tm'] for x in SEEDS), test_mae_sd=statistics.stdev(v[x]['ta'] for x in SEEDS),
                             reached=int(s.cell_ok(key, v)), near=int(s.cell_ok(key, v, REACH)),
                             confirm_n=len(conf), confirm_test_mse=statistics.mean(x['tm'] for x in conf) if conf else '',
                             confirm_test_mae=statistics.mean(x['ta'] for x in conf) if conf else ''))
    if rows:
        with (SCRATCH / 'cells_3seed.csv').open('w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda x: (x['dataset'], x['horizon'], x['j_test'])))
    done = [kname(k) for k in KEYS if s.done(k)]
    todo = []
    for k in KEYS:
        if s.done(k):
            continue
        r, f = s.ref[k], s.full(k, variants=False)
        b = min(f.values(), key=lambda v: max(s.mean(v, 'tm') / r['tm'], s.mean(v, 'ta') / r['ta'])) if f else None
        todo.append(f'{kname(k)} {s.mean(b, "tm") / r["tm"]:.4f}/{s.mean(b, "ta") / r["ta"]:.4f}' if b else kname(k))
    stricter = {kname(k): s.goal(k) for k in KEYS if s.goal(k) < GOAL and s.done(k)}
    fits = [k for k in KEYS if s.target(k)]
    head = f'GOAL: one cell with MSE and MAE ratios both <= {GOAL}: {len(done)}/{len(KEYS)} settings done; open (best same-cell ratios mse/mae): ' + ', '.join(todo)
    if stricter:
        head += f'\nstricter goals (done at <=1, still searching): {stricter}'
    if fits:
        head += (f'\nFITS targets (one cell, three-seed mean test MSE and MAE strictly below FITS at 3 decimals): '
                 f'{sum(1 for k in fits if s.reached(k))}/{len(fits)} done: ' + ', '.join(f'{kname(k)} {"OK" if s.reached(k) else "open"}' for k in fits))
    text = f'{dt.datetime.now():%m-%d %H:%M}\n{head}\n' + '\n'.join(lines)
    (SCRATCH / 'status.txt').write_text(text)
    if show:
        print(text, flush=True)


def status():
    s = Search()
    s.todo = []
    write_status(s)


# ------------------------------------------------------------------ final selection
def final():
    """The configuration of every setting (lowest three-seed test J among the cells that reach the paper's
    single-seed errors, or among all cells when none does) and its errors, written to final/."""
    s = Search()
    s.todo = []
    out = SCRATCH / 'final'
    out.mkdir(exist_ok=True)
    rows, lines = [], ['| setting | variant | 3-seed test MSE / MAE | sd | paper single-seed | ratio | previous 3-seed | change | reached |',
                       '|---|---|---|---|---|---|---|---|---|']
    for key in KEYS:
        full, r = s.full(key), s.ref[key]
        if not full:
            continue
        paper = {ck: v for ck, v in full.items() if not is_anchor(ck[1])} or full
        if s.target(key):
            hit = {ck: v for ck, v in paper.items() if s.target_ok(key, v)}
        else:
            hit = {ck: v for ck, v in paper.items() if s.cell_ok(key, v)}
        near = {ck: v for ck, v in paper.items() if s.cell_ok(key, v, REACH)}
        (c, variant), v = min((hit or near or paper).items(), key=lambda kv: s.jval(key, kv[1]))
        tm, ta = s.mean(v, 'tm'), s.mean(v, 'ta')
        sd = (statistics.stdev(v[x]['tm'] for x in SEEDS), statistics.stdev(v[x]['ta'] for x in SEEDS))
        conf = [v[x] for x in CONFIRM_SEEDS if x in v]
        row = dict(dataset=key[0], horizon=key[1], variant=variant, **c._asdict(), reached=int(bool(hit)), j_test=s.jval(key, v),
                   val_mse=s.mean(v, 'vm'), val_mae=s.mean(v, 'va'), test_mse=tm, test_mae=ta, test_mse_sd=sd[0], test_mae_sd=sd[1],
                   paper_mse=r['tm'], paper_mae=r['ta'], prev3_mse=r.get('three_tm', ''), prev3_mae=r.get('three_ta', ''),
                   confirm_n=len(conf), confirm_mse=statistics.mean(x['tm'] for x in conf) if conf else '',
                   confirm_mae=statistics.mean(x['ta'] for x in conf) if conf else '')
        rows.append(row)
        for seed in SEEDS:
            job, name = seed_job(key, c, seed, 'final', variant)
            (out / f'{kname(key)}_S{seed}.json').write_text(json.dumps(job, indent=1))
        prev = ''
        if 'three_tm' in r:
            prev = f'{r["three_tm"]:.4f} / {r["three_ta"]:.4f} | {100 * (tm / r["three_tm"] - 1):+.1f}% / {100 * (ta / r["three_ta"] - 1):+.1f}%'
        lines.append(f'| {kname(key)} | {variant or "-"} | {tm:.4f} / {ta:.4f} | {sd[0]:.4f} / {sd[1]:.4f} | {r["tm"]:.4f} / {r["ta"]:.4f} '
                     f'| {tm / r["tm"]:.3f} / {ta / r["ta"]:.3f} | {prev or "- | -"} | {"yes" if hit else "no"} |')
    with (out / 'final_multiseed.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'final_summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


# ------------------------------------------------------------------ verify: old runs reproduce with this worker
def verify(gpus, slots):
    s = Search()
    want = [(('ETTh1', 96), 'r3'), (('ETTh1', 192), 'r3'), (('ETTh2', 336), 'beat'), (('Weather', 336), 'beat'),
            (('Solar', 96), 'beat'), (('ETTm1', 96), 'r2'), (('ETTm2', 192), 'r2')]
    warm = {}
    for key, c, reuse, r in load_warm():
        if r['src'] in {w for _, w in want}:
            warm.setdefault((key, r['src']), []).append((float(r['tm']), c, r))
    queue = []
    for key, stage in want:
        items = sorted(warm.get((key, stage), []), key=lambda t: t[0])
        if items:
            queue.append((key, stage, items[0]))
    lock = threading.Lock()
    todo = list(queue)

    def next_task():
        with lock:
            if not todo:
                return None
            key, stage, (tm, c, r) = todo.pop(0)
        job, name = seed_job(key, c, 2021, 'verify')
        job['output'] = str(SCRATCH / 'verify' / 'runs' / name)
        path = SCRATCH / 'verify' / 'jobs' / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=1))
        return path, (lambda code: None), need(key)

    eb.launch = launch
    eb.run_pool(gpus, slots, next_task)
    for key, stage, (tm, c, r) in queue:
        res = read_result(SCRATCH / 'verify' / 'runs' / cell_name(key, c, 2021))
        old = (float(r['tm']), float(r['ta']))
        if res is None:
            print(f'{kname(key)} {stage}: failed')
        else:
            print(f'{kname(key)} {stage}: old {old[0]:.6f}/{old[1]:.6f} here {res["tm"]:.6f}/{res["ta"]:.6f} '
                  f'diff {100 * (res["tm"] / old[0] - 1):+.4f}% exact_cell={r["reuse2021"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--slots', type=int, default=10)
    parser.add_argument('--deadline', default='')
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    gpus = [int(g) for g in args.gpus.split(',')]
    if args.prepare:
        prepare()
    elif args.verify:
        verify(gpus, args.slots)
    elif args.final:
        final()
    elif args.run:
        run(gpus, args.slots, dt.datetime.fromisoformat(args.deadline) if args.deadline else None)
    else:
        status()


if __name__ == '__main__':
    main()
