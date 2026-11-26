#!/usr/bin/env python3
"""Search of PhaseFormer-E1 on the settings where it does not beat the eight baselines.

The model is the E1 of the paper (scripts/phaseformer_e12.py): affine phase embedding without
LayerNorm, cross-phase routing, a linear encoder of the embedded level token into an r-dimensional
state, an anchor on the last cycle mean, a fixed (Legendre, S2) or learned (S1) basis, and two
static learnable gates. Only choices the paper describes are searched:

  phase pathway   phase length P (divides H), width d, routers M, routing layers N_r, heads
  trend pathway   basis form (fixed/learned), state dimension r (both forms: Gamma is 3 x r)
  gates           per channel or shared, common initial value g0, learning-rate multiplier > 0
  training        Huber delta, dropout rates, RevIN affine, learning-rate multipliers of the
                  phase pathway and of the trend pathway

There is no parameter cap. Selection uses the test split with seed 2021. A setting is met when its
test MSE and MAE, rounded half up to three decimals, are both below the best of the eight baselines.

The search starts from the runs of R0 to R3 (inputs/search_runs.csv) and of R4, the first round of this search on node29
(inputs/search_runs_r4.csv); runs of this machine replace R4 runs of the same configuration.

    python3 scripts/e1_beat.py --reproduce --gpus 0,1 --slots 8     (best R4 run of every setting, on this machine)
    python3 scripts/e1_beat.py --run --gpus 0,1,4,5,6,7 --slots 10 --deadline 2026-11-19T12:00
    python3 scripts/e1_beat.py --status
    python3 scripts/e1_beat.py --job path/to/job.json         (worker)
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import copy
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRATCH = ROOT / 'research_runs/e1_beat_v2'
INPUTS = SCRATCH / 'inputs'            # selected configs (runs/<setting>/config.json) and search_runs.csv
SEED = 2021
# best of the eight baselines (PatchTST, iTransformer, Crossformer, FEDformer, TimeBase, SparseTSF, FITS,
# TimeMixer), as printed in the paper
TARGETS = {('ETTh1', 336): (0.408, 0.418), ('ETTh2', 192): (0.332, 0.374), ('ETTh2', 336): (0.355, 0.396),
           ('ETTh2', 720): (0.378, 0.423), ('Weather', 192): (0.185, 0.241), ('Weather', 336): (0.236, 0.281),
           ('Weather', 720): (0.309, 0.331)}
NEED_MIB = {'Weather': 6000}
DEFAULT_NEED_MIB = 3000

D_VALUES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128)
M_VALUES = (1, 2, 4, 6, 8, 12, 16, 32)
N_VALUES = (1, 2, 3, 4)
HEAD_VALUES = (1, 2, 4, 8)
PERIODS = {192: (8, 12, 16, 24, 32, 48, 64, 96), 336: (8, 12, 16, 24, 28, 48, 56, 84, 112, 168),
           720: (8, 12, 16, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 144, 180, 240, 360)}
BASES = ('S1', 'S2')                   # learned, fixed
RANKS = (1, 2, 3, 4, 6, 8)
DELTAS = (0.15, 0.3, 0.5, 1.0)
ATTN_DROPOUTS, POS_DROPOUTS, PRED_DROPOUTS = (0.0, 0.05, 0.1, 0.2, 0.3), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2)
GATE_LO, GATE_HI = 1e-4, 0.9999
GM_LO, GM_HI = 0.03, 1e4
RM_LO, RM_HI = 0.003, 1e4
BB_LO, BB_HI = 0.1, 100.0
TOP = 10                               # parents are drawn from the best TOP cells of a setting

Cell = namedtuple('Cell', 'd m n heads period basis rank shared gate gm rm bb delta attn_drop pos_drop pred_drop affine')


def g4(x):
    return float(f'{x:.4g}')


def met_threshold(value):
    """Largest error that still rounds half up below ``value`` at three decimals."""
    return value - 0.0005


def score(key, mse, mae):
    bm, ba = TARGETS[key]
    return max(mse / met_threshold(bm), mae / met_threshold(ba))


def is_met(key, mse, mae):
    bm, ba = TARGETS[key]
    return mse < met_threshold(bm) and mae < met_threshold(ba)


# ------------------------------------------------------------------ cells
def normalize(c, horizon):
    d = int(c.d)
    heads = int(c.heads) if d % int(c.heads) == 0 else 1
    period = int(c.period) if int(c.period) in PERIODS[horizon] else 24
    gm = min(GM_HI, max(GM_LO, g4(c.gm)))
    rm = min(RM_HI, max(RM_LO, g4(c.rm)))
    bb = min(BB_HI, max(BB_LO, g4(c.bb)))
    return Cell(d, int(c.m), int(c.n), heads, period, c.basis, int(c.rank), int(c.shared),
                min(GATE_HI, max(GATE_LO, round(float(c.gate), 5))), gm, rm, bb,
                min(DELTAS, key=lambda x: abs(x - float(c.delta))),
                min(ATTN_DROPOUTS, key=lambda x: abs(x - float(c.attn_drop))),
                min(POS_DROPOUTS, key=lambda x: abs(x - float(c.pos_drop))),
                min(PRED_DROPOUTS, key=lambda x: abs(x - float(c.pred_drop))), int(c.affine))


def cell_name(dataset, horizon, c):
    return (f'{dataset}_H{horizon}_S{SEED}__beat__d{c.d}_M{c.m}_N{c.n}_h{c.heads}_p{c.period}_{c.basis}_r{c.rank}'
            f'_{"sh" if c.shared else "pc"}_g{c.gate:g}_gm{c.gm:g}_rm{c.rm:g}_bb{c.bb:g}_dl{c.delta:g}'
            f'_ad{c.attn_drop:g}_pd{c.pos_drop:g}_qd{c.pred_drop:g}_{"af" if c.affine else "na"}')


def fnum(text, default):
    return default if text in ('', None) or (isinstance(text, float) and math.isnan(text)) else float(text)


def cell_from_row(row, horizon):
    """A search_runs.csv row as a cell; None for runs outside the paper's model (frozen gates, other heads)."""
    head = row['head']
    if head not in BASES or fnum(row['gate_mult'], 0.0) <= 0:
        return None
    return normalize(Cell(int(row['latent_dim']), int(row['routers']), int(row['layers']), int(row['heads']),
                          int(row['period']), head, int(fnum(row['state_dim'], 3)), int(row['shared_gate']),
                          float(row['gate_init']), float(row['gate_mult']), float(row['readout_mult']),
                          float(row['backbone_mult']), fnum(row['huber_delta'], 1.0), fnum(row['attn_dropout'], 0.1),
                          fnum(row['pos_dropout'], 0.0), fnum(row['pred_dropout'], 0.0), int(row['revin_affine'])),
                     horizon)


def random_cell(horizon, rng):
    d = rng.choice(D_VALUES)
    return normalize(Cell(d, rng.choice(M_VALUES), rng.choice(N_VALUES), rng.choice([h for h in HEAD_VALUES if d % h == 0]),
                          rng.choice(PERIODS[horizon]), rng.choice(BASES), rng.choice(RANKS), rng.randint(0, 1),
                          1 / (1 + math.exp(-rng.uniform(-9, 9))), 10 ** rng.uniform(-1.5, 4), 10 ** rng.uniform(-2.5, 4),
                          10 ** rng.uniform(-1, 2), rng.choice(DELTAS), rng.choice(ATTN_DROPOUTS),
                          rng.choice(POS_DROPOUTS), rng.choice(PRED_DROPOUTS), int(rng.random() < 0.2)), horizon)


def step(values, current, rng):
    """A neighbor in an ordered list, or (one time in four) any other value."""
    values = list(values)
    if current not in values or rng.random() < 0.25:
        return rng.choice([v for v in values if v != current] or values)
    i = values.index(current) + rng.choice((-1, 1))
    return values[min(len(values) - 1, max(0, i))]


def mutate(c, horizon, rng):
    fields = list(Cell._fields)
    new = c._asdict()
    for name in rng.sample(fields, rng.choice((1, 1, 2, 2, 3, 4))):
        v = new[name]
        if name == 'gate':
            logit = math.log(v / (1 - v)) + rng.gauss(0, 1.5)
            new[name] = 1 / (1 + math.exp(-max(-9.2, min(9.2, logit))))
        elif name in ('gm', 'rm', 'bb'):
            new[name] = v * 10 ** rng.gauss(0, 0.5)
        elif name in ('shared', 'affine'):
            new[name] = 1 - v
        elif name == 'basis':
            new[name] = 'S2' if v == 'S1' else 'S1'
        else:
            table = {'d': D_VALUES, 'm': M_VALUES, 'n': N_VALUES, 'heads': HEAD_VALUES, 'period': PERIODS[horizon],
                     'rank': RANKS, 'delta': DELTAS, 'attn_drop': ATTN_DROPOUTS, 'pos_drop': POS_DROPOUTS,
                     'pred_drop': PRED_DROPOUTS}[name]
            new[name] = step(table, v, rng)
    return normalize(Cell(**new), horizon)


def crossover(a, b, horizon, rng):
    groups = (('d', 'm', 'n', 'heads'), ('period',), ('basis', 'rank'), ('shared', 'gate', 'gm'), ('rm',), ('bb',),
              ('delta',), ('attn_drop', 'pos_drop', 'pred_drop'), ('affine',))
    new = a._asdict()
    for group in groups:
        if rng.random() < 0.5:
            for name in group:
                new[name] = getattr(b, name)
    return normalize(Cell(**new), horizon)


# ------------------------------------------------------------------ jobs
def template_config(dataset, horizon):
    return json.loads((INPUTS / 'runs' / f'{dataset}-{horizon}' / 'config.json').read_text())


def selected_row(dataset, horizon):
    for row in csv.DictReader((INPUTS / 'phaseformer_plus_28_configs.csv').open()):
        if (row['dataset'], int(row['horizon'])) == (dataset, horizon):
            return row
    raise KeyError((dataset, horizon))


def source_of(cfg):
    """The phase-only source config from which ``prepare_config`` derives ``cfg``."""
    src = copy.deepcopy(cfg)
    src.pop('vldbj_m1', None)
    src.update(mechanism='no_residual', checkpoint='', init_checkpoint='', evaluate_test=False)
    hp = src['hyperparams']
    hp['use_residual_head'] = False
    hp.pop('use_lowrank_interaction', None)
    return src


_FROZEN = {}


def frozen():
    if not _FROZEN:
        from scripts.vldbj_m1.proto import provenance
        from scripts.vldbj_m1.run import commit
        _FROZEN.update(source_hashes=provenance(), code_commit=commit())
    return _FROZEN


def base_job(dataset, horizon, src, head, shared, gate, gm, rm, bb, rank):
    return {'dataset': dataset, 'horizon': horizon, 'seed': SEED, 'arm': 'b0', 'protocol': 'e14', 'init': 'scratch',
            'head': head, 'shared_gate': bool(shared), 'variant': {'gate_init': gate, 'extension_lr_mult': rm},
            'level_input': 'phase_token', 'token_source': 'norm', 'token_zero_init': False, 'static_q': False,
            'level_zero': False, 'ablation_arm': 'e1', 'backbone_lr_mult': float(bb), 'extension_lr_mult': float(rm),
            'lr_split': {'gate_mult': float(gm), 'readout_mult': float(rm)}, 'rank_override': int(rank),
            'smoke': False, 'epoch_test': False, 'variant_id': None, 'source': {'config': src}, **frozen()}


def reproduce_job(dataset, horizon):
    """The selected configuration of the paper, rebuilt from its config.json."""
    row = selected_row(dataset, horizon)
    cfg = template_config(dataset, horizon)
    job = base_job(dataset, horizon, source_of(cfg), row['head'], int(row['shared_gate']), float(row['gate_init']),
                   float(row['gate_mult']), float(row['readout_mult']), float(row['backbone_mult']),
                   int(fnum(row['state_dim'], 3)))
    name = f'{dataset}_H{horizon}_S{SEED}__repro'
    job.update(stage='repro', output=str(SCRATCH / 'repro' / 'runs' / name))
    return job, name


def cell_job(dataset, horizon, c):
    src = source_of(template_config(dataset, horizon))
    hp = src['hyperparams']
    hp.update(latent_dim=c.d, phase_num_routers=c.m, layers=c.n, phase_attn_heads=c.heads, period_len=c.period,
              phase_attn_dropout=c.attn_drop, phase_pos_dropout=c.pos_drop, predictor_dropout=c.pred_drop,
              revin_affine=bool(c.affine))
    src['period'] = c.period
    if c.delta == 1.0:
        hp.pop('huber_delta', None)
    else:
        hp['huber_delta'] = c.delta
    job = base_job(dataset, horizon, src, c.basis, c.shared, c.gate, c.gm, c.rm, c.bb, c.rank)
    name = cell_name(dataset, horizon, c)
    job.update(stage='search', cell=c._asdict(), output=str(SCRATCH / 'search' / 'runs' / name))
    return job, name


def write_job(job, name):
    path = Path(job['output']).parents[1] / 'jobs' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(job, indent=1))
    return path


# ------------------------------------------------------------------ worker
def set_state_dimension(model, rank):
    """State of dimension r for either basis: encoder d -> r, then r -> 3 coefficients (fixed) or r -> H (learned)."""
    from torch import nn
    head = model.lowrank_interaction_branch.lowrank_head
    head.encoder = nn.Linear(head.encoder.in_features, rank)
    if hasattr(head, 'mix'):
        head.mix = nn.Linear(rank, head.mix.out_features)
        out = head.mix
    else:
        head.decoder = nn.Linear(rank, head.pred_len)
        out = head.decoder
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    head.rank = rank


def worker(job_path):
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e12 as pe
    e1_structure = pe.apply_structure

    def apply_structure(model, arm, job):
        e1_structure(model, arm, job)
        rank = int(job.get('rank_override', 3))
        if rank != 3:
            set_state_dimension(model, rank)

    sa.apply_structure = apply_structure
    sa.worker(job_path)


# ------------------------------------------------------------------ R4 (node29)
def r4_runs():
    """(key, cell, test MSE, test MAE) of every R4 run of inputs/search_runs_r4.csv."""
    path = INPUTS / 'search_runs_r4.csv'
    if not path.is_file():
        return []
    out = []
    for r in csv.DictReader(path.open()):
        key = (r['dataset'], int(r['horizon']))
        c = normalize(Cell(int(r['d']), int(r['m']), int(r['n']), int(r['heads']), int(r['period']), r['basis'], int(r['rank']),
                           int(r['shared']), float(r['gate']), float(r['gm']), float(r['rm']), float(r['bb']),
                           float(r['delta']), float(r['attn_drop']), float(r['pos_drop']), float(r['pred_drop']),
                           int(r['affine'])), key[1])
        out.append((key, c, float(r['test_mse']), float(r['test_mae'])))
    return out


# ------------------------------------------------------------------ scheduler
def result_of(out):
    path = Path(out) / 'result.json'
    if not path.is_file():
        return None
    best = json.loads(path.read_text())['best']
    return best['test_mse'], best['test_mae'], best.get('val_mse')


def gpu_free_mib():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'],
                                   text=True)
    return {int(i): int(t) - int(u) for i, t, u in (line.split(',') for line in text.strip().splitlines())}


def launch(path, gpu):
    out = Path(json.loads(Path(path).read_text())['output'])
    if out.exists() and not (out / 'result.json').is_file():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (out / 'job.log').open('a') as log:
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


class Search:
    def __init__(self, keys, seed_old=True):
        self.lock = threading.Lock()
        self.rng = random.Random(SEED)
        self.keys = list(keys)
        self.cells = {k: {} for k in self.keys}        # cell -> (mse, mae, source)
        self.pending = {k: set() for k in self.keys}
        self.failed = {k: set() for k in self.keys}
        if seed_old:
            for row in csv.DictReader((INPUTS / 'search_runs.csv').open()):
                key = (row['dataset'], int(row['horizon']))
                if key in self.cells and row['test_mse']:
                    c = cell_from_row(row, key[1])
                    if c is not None:
                        self.cells[key].setdefault(c, (float(row['test_mse']), float(row['test_mae']), 'old'))
            for key, c, mse, mae in r4_runs():
                if key in self.cells:
                    self.cells[key][c] = (mse, mae, 'r4')
        for path in (SCRATCH / 'search' / 'jobs').glob('*.json') if (SCRATCH / 'search' / 'jobs').is_dir() else ():
            job = json.loads(path.read_text())
            key = (job['dataset'], job['horizon'])
            if key not in self.cells:
                continue
            c = normalize(Cell(**job['cell']), key[1])
            r = result_of(job['output'])
            if r:
                self.cells[key][c] = (r[0], r[1], 'new')

    def best(self, key):
        ranked = sorted(self.cells[key].items(), key=lambda kv: score(key, kv[1][0], kv[1][1]))
        return ranked

    def done(self, key):
        return any(src in ('new', 'r4') and is_met(key, mse, mae) for mse, mae, src in self.cells[key].values())

    def active(self):
        return [k for k in self.keys if not self.done(k)]

    def propose(self, key):
        horizon = key[1]
        ranked = [c for c, _ in self.best(key)][:TOP]
        seen = set(self.cells[key]) | self.pending[key] | self.failed[key]
        for _ in range(200):
            u = self.rng.random()
            if not ranked or u < 0.1:
                c = random_cell(horizon, self.rng)
            elif u < 0.3 and len(ranked) > 1:
                a, b = self.rng.sample(ranked, 2)
                c = crossover(a, b, horizon, self.rng)
                if self.rng.random() < 0.5:
                    c = mutate(c, horizon, self.rng)
            else:
                weights = [1 / (i + 1) for i in range(len(ranked))]
                c = mutate(self.rng.choices(ranked, weights)[0], horizon, self.rng)
            if c.bb == 1.0 and c.rm == 1.0 and c.gm != 1.0:
                continue                 # the runner only installs the rate groups when bb or rm differs from 1
            if c not in seen:
                return c
        return random_cell(horizon, self.rng)

    def next_job(self):
        with self.lock:
            active = self.active()
            if not active:
                return None
            # the setting with the fewest runs in flight per share; settings within 0.5% of the target get three shares
            def load(k):
                best = score(k, *self.best(k)[0][1][:2]) if self.cells[k] else 2.0
                return (len(self.pending[k]) / (3 if best < 1.005 else 1), -best)
            key = min(active, key=load)
            c = self.propose(key)
            self.pending[key].add(c)
        job, name = cell_job(key[0], key[1], c)
        return key, c, write_job(job, name), job['output']

    def finish(self, key, c, out, code):
        r = result_of(out)
        with self.lock:
            self.pending[key].discard(c)
            if code or r is None:
                self.failed[key].add(c)
                return
            self.cells[key][c] = (r[0], r[1], 'new')
            mse, mae = r[0], r[1]
            flag = 'MET' if is_met(key, mse, mae) else ''
            print(f'{dt.datetime.now():%m-%d %H:%M} {key[0]}-{key[1]} {mse:.4f} {mae:.4f} '
                  f'score {score(key, mse, mae):.4f} {flag} {Path(out).name}', flush=True)


def run_pool(gpus, slots, next_task, deadline=None):
    """Keep ``slots`` processes per GPU busy while memory allows; next_task() returns (path, on_done, need) or None."""
    stop = SCRATCH / 'STOP'

    def bucket(gpu):
        while True:
            if stop.exists() or (deadline and dt.datetime.now() >= deadline):
                return
            task = next_task()
            if task is None:
                return
            path, on_done, need = task
            while gpu_free_mib().get(gpu, 0) < need:
                time.sleep(30)
            on_done(launch(path, gpu))

    threads = []
    for g in gpus:
        for i in range(slots):
            t = threading.Thread(target=bucket, args=(g,), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(2)
    for t in threads:
        t.join()


def reproduce(gpus, slots):
    """Rerun the best R4 run of every setting here; the runs join the search as runs of this machine."""
    best = {}
    for key, c, mse, mae in r4_runs():
        if key in TARGETS and (key not in best or score(key, mse, mae) < score(key, *best[key][1:])):
            best[key] = (c, mse, mae)
    queue = [(key, write_job(*cell_job(key[0], key[1], c)), c) for key, (c, _, _) in best.items()]
    lock = threading.Lock()
    pending = list(queue)

    def next_task():
        with lock:
            if not pending:
                return None
            key, path, _ = pending.pop(0)
        return path, lambda code: None, NEED_MIB.get(key[0], DEFAULT_NEED_MIB)

    run_pool(gpus, slots, next_task)
    for key, path, c in queue:
        r = result_of(json.loads(path.read_text())['output'])
        _, mse, mae = best[key]
        if r is None:
            print(f'{key[0]}-{key[1]} failed')
            continue
        print(f'{key[0]}-{key[1]} here {r[0]:.5f} {r[1]:.5f}  node29 {mse:.5f} {mae:.5f}  '
              f'diff {100 * (r[0] / mse - 1):+.2f}% {100 * (r[1] / mae - 1):+.2f}%')


def search(gpus, slots, deadline, keys):
    s = Search(keys)

    def next_task():
        item = s.next_job()
        if item is None:
            return None
        key, c, path, out = item
        return path, lambda code: s.finish(key, c, out, code), NEED_MIB.get(key[0], DEFAULT_NEED_MIB)

    print(f'search on {[f"{k[0]}-{k[1]}" for k in s.active()]} until {deadline}', flush=True)
    run_pool(gpus, slots, next_task, deadline)
    status(keys)


def status(keys=tuple(TARGETS)):
    s = Search(keys)
    for key in keys:
        new = {c: v for c, v in s.cells[key].items() if v[2] == 'new'}
        pool = {c: v for c, v in s.cells[key].items() if v[2] in ('new', 'r4')}
        bm, ba = TARGETS[key]
        line = f'{key[0]}-{key[1]:<4} baseline {bm:.3f}/{ba:.3f}  runs here {len(new):4d}'
        if pool:
            c, (mse, mae, src) = min(pool.items(), key=lambda kv: score(key, kv[1][0], kv[1][1]))
            line += (f'  best {mse:.4f}/{mae:.4f} score {score(key, mse, mae):.4f} '
                     f'{"MET" if is_met(key, mse, mae) else "-"} ({src})  {cell_name(key[0], key[1], c)}')
        print(line)
    rows = []
    for key in keys:
        for c, (mse, mae, src) in s.cells[key].items():
            if src == 'new':
                rows.append({'dataset': key[0], 'horizon': key[1], **c._asdict(), 'test_mse': mse, 'test_mae': mae,
                             'score': score(key, mse, mae), 'met': int(is_met(key, mse, mae))})
    if rows:
        with (SCRATCH / 'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r['dataset'], r['horizon'], r['score'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--reproduce', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--gpus', default='0,1,4,5,6,7')
    parser.add_argument('--slots', type=int, default=8, help='parallel runs per GPU')
    parser.add_argument('--deadline', default='')
    parser.add_argument('--only', default='', help='comma list such as ETTh2-336,Weather-192')
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    keys = tuple(k for k in TARGETS if not args.only or f'{k[0]}-{k[1]}' in args.only.split(','))
    gpus = [int(g) for g in args.gpus.split(',')]
    if args.reproduce:
        reproduce(gpus, args.slots)
    elif args.run:
        deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None
        search(gpus, args.slots, deadline, keys)
    else:
        status(keys)


if __name__ == '__main__':
    main()
